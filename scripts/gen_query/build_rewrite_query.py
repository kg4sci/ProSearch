import json
import time
import threading
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import json_repair
import openai

SYSTEM_PROMPT = "You are a helpful scientific assistant."

HYDE_PROMPT = """
Given the query, write a short scientific paragraph that would likely answer it.
Preserve the key experimental constraints.

Query: {query}
"""

DECOMP_PROMPT = """
Given a retrieval query, decompose it into a small set of simpler retrieval sub-queries.

First, identify internally the query intent and the constraints in the query.
Then generate the smallest non-redundant set of sub-queries that together cover the original constraints.

Requirements:
(1) Each sub-query must remain a natural question.
(2) Each sub-query must preserve the original query intent.
    - For an exploration query, it must still ask about the same field as the original query.
    - For a reproduction query, it must still ask for a setup/protocol/paragraph satisfying constraints.
(3) Each sub-query should keep only part of the original constraints, so that it is simpler than the original query.
(4) Different sub-queries should reflect different retained constraints, not paraphrases of the same reduced query.
(5) Together, the sub-queries should cover all original constraints.
(6) Use as few sub-queries as possible.
(7) Do not introduce new constraints or change the meaning of the original query.

Examples:
Query: What applied potential is used (vs SCE) at 25 °C with an Au working electrode?
Decomposition:
[
  "What applied potential is used (vs SCE) with an Au working electrode?",
  "What applied potential is used (vs SCE) at 25 °C?"
]

Query: Find an electrochemistry experimental protocol describing cyclic voltammetry (CV) performed in acetate buffer using an AGCE working electrode.
Decomposition:
[
  "Find an electrochemistry experimental protocol describing cyclic voltammetry (CV) performed in acetate buffer.",
  "Find an electrochemistry experimental protocol describing cyclic voltammetry (CV) using an AGCE working electrode."
]

Return only a JSON-style list of plain question strings.

Query: {query}
"""

REASONING_FIRST_PROMPT = """
Given the query, generate a retrieval-oriented reasoning process for experimental setup retrieval.

The reasoning should briefly explain what kind of experimental setup, condition combination, or explored field is being sought, what information a relevant paragraph would explicitly contain, and which constraints must be preserved exactly. Focus on how the requested conditions would be expressed together in a relevant experimental description.

Return only the reasoning process.

Query: {query}
"""

REASONING_PROMPT = """"
(1) Identify the essential problem in the post. 
(2) Think step by step to reason about what should be included in the relevant documents. 
(3) Draft an answer.
"""


_thread_local = threading.local()


def get_client(api_key: str):
    clients = getattr(_thread_local, "clients", None)
    if clients is None:
        clients = {}
        _thread_local.clients = clients

    if api_key not in clients:
        clients[api_key] = openai.OpenAI(
            base_url=BASE_URL,
            api_key=api_key,
        )
    return clients[api_key]

def load_keys(path):
    with open(path, "r", encoding="utf-8") as f:
        keys = json.load(f)
    return keys


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(text):
    return " ".join(str(text).strip().split())


def build_messages(prompt_type, query):
    if prompt_type == "hyde":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": HYDE_PROMPT.format(query=query)},
        ]
    elif prompt_type == "decomp":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": DECOMP_PROMPT.format(query=query)},
        ]
    elif prompt_type == "reasoning_first":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": REASONING_FIRST_PROMPT.format(query=query)},
        ]
    elif prompt_type == "reasoning":
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": REASONING_PROMPT.format(query=query)},
        ]
    else:
        raise ValueError(f"Unsupported prompt_type: {prompt_type}")


def parse_decomp_output(raw_text):
    parsed = json_repair.loads(raw_text)

    if isinstance(parsed, list):
        subqueries = parsed
    else:
        raise ValueError(f"Unexpected decomp output type: {type(parsed)}")

    if not isinstance(subqueries, list):
        raise ValueError(f"decomp output does not contain a list: {parsed}")

    subqueries = [normalize_text(x) for x in subqueries if str(x).strip()]
    if not subqueries:
        raise ValueError("decomp output contains no valid subqueries")

    return subqueries


def postprocess_output(prompt_type, raw_text):
    raw_text = raw_text.strip()

    if prompt_type in {"hyde", "reasoning_first", "reasoning"}:
        return normalize_text(raw_text)

    if prompt_type == "decomp":
        return parse_decomp_output(raw_text)

    raise ValueError(f"Unsupported prompt_type: {prompt_type}")


def call_model(query_id, entry, prompt_type, worker_id, api_keys):
    last_error = None
    query = entry["query"]

    for attempt in range(MAX_RETRIES):
        api_key = api_keys[(worker_id + attempt) % len(api_keys)]

        try:
            client = get_client(api_key)
            messages = build_messages(prompt_type, query)

            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=messages,
            )

            raw_text = response.choices[0].message.content or ""
            parsed_output = postprocess_output(prompt_type, raw_text)

            return {
                "query_id": query_id,
                "raw_text": raw_text,
                "parsed_output": parsed_output,
            }

        except Exception as e:
            last_error = e
            print(
                f"[Retry {attempt + 1}/{MAX_RETRIES}] "
                f"query_id={query_id} prompt_type={prompt_type} worker={worker_id}: {e}"
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(SLEEP_SECONDS * (attempt + 1))

    raise RuntimeError(
        f"Failed after {MAX_RETRIES} retries for query_id={query_id}, "
        f"prompt_type={prompt_type}: {last_error}"
    )

def build_standard_entry(entry, generated_query):
    new_entry = deepcopy(entry)
    new_entry["query"] = generated_query
    return new_entry


def build_decomp_parent_entry(entry, subqueries):
    new_entry = deepcopy(entry)
    new_entry["subqueries"] = subqueries
    return new_entry


def build_decomp_expanded_entries(query_id, entry, subqueries):
    expanded = {}
    for i, subq in enumerate(subqueries):
        child_id = f"{query_id}__subq_{i:02d}"
        child_entry = deepcopy(entry)
        child_entry["query"] = subq
        child_entry["parent_id"] = query_id
        child_entry["subquery_id"] = i
        child_entry["decomp_group_size"] = len(subqueries)
        expanded[child_id] = child_entry
    return expanded


def order_standard_results(original_queries, results):
    return {
        qid: results[qid]
        for qid in original_queries.keys()
        if qid in results
    }


def order_decomp_parent_results(original_queries, parent_results):
    return {
        qid: parent_results[qid]
        for qid in original_queries.keys()
        if qid in parent_results
    }


def order_decomp_expanded_results(original_queries, parent_results, expanded_results):
    ordered = {}
    for qid in original_queries.keys():
        if qid not in parent_results:
            continue
        subqueries = parent_results[qid].get("subqueries", [])
        for i in range(len(subqueries)):
            child_id = f"{qid}__subq_{i:02d}"
            if child_id in expanded_results:
                ordered[child_id] = expanded_results[child_id]
    return ordered

def process_standard_prompt(queries, prompt_type, keys, output_path):
    items = list(queries.items())
    results = {}
    errors = {}

    max_workers = min(MAX_WORKERS, max(1, len(items)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_qid = {}

        for worker_id, (query_id, entry) in enumerate(items):
            future = executor.submit(
                call_model,
                query_id,
                entry,
                prompt_type,
                worker_id,
                keys,
            )
            future_to_qid[future] = query_id

        done = 0
        total = len(future_to_qid)

        for future in as_completed(future_to_qid):
            query_id = future_to_qid[future]
            entry = queries[query_id]

            try:
                out = future.result()
                results[query_id] = build_standard_entry(entry, out["parsed_output"])
            except Exception as e:
                errors[query_id] = str(e)
                if KEEP_FAILED_AS_ORIGINAL:
                    results[query_id] = deepcopy(entry)
                print(f"[ERROR] query_id={query_id}, prompt_type={prompt_type}: {e}")

            done += 1

            if done % SAVE_EVERY == 0 or done == total:
                save_json(order_standard_results(queries, results), output_path)

                print(f"[{prompt_type}] progress: {done}/{total}")

    save_json(order_standard_results(queries, results), output_path)



def process_decomp_prompt(queries, keys, parent_path, expanded_path):
    items = list(queries.items())
    parent_results = {}
    expanded_results = {}
    errors = {}

    max_workers = min(MAX_WORKERS, max(1, len(items)))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_qid = {}

        for worker_id, (query_id, entry) in enumerate(items):
            future = executor.submit(
                call_model,
                query_id,
                entry,
                "decomp",
                worker_id,
                keys,
            )
            future_to_qid[future] = query_id

        done = 0
        total = len(future_to_qid)

        for future in as_completed(future_to_qid):
            query_id = future_to_qid[future]
            entry = queries[query_id]

            try:
                out = future.result()
                subqueries = out["parsed_output"]

                parent_results[query_id] = build_decomp_parent_entry(entry, subqueries)
                expanded_results.update(
                    build_decomp_expanded_entries(query_id, entry, subqueries)
                )

            except Exception as e:
                errors[query_id] = str(e)
                if KEEP_FAILED_AS_ORIGINAL:
                    fallback_parent = deepcopy(entry)
                    fallback_parent["subqueries"] = []
                    parent_results[query_id] = fallback_parent
                print(f"[ERROR] query_id={query_id}, prompt_type=decomp: {e}")

            done += 1

            if done % SAVE_EVERY == 0 or done == total:
                save_json(
                    order_decomp_parent_results(queries, parent_results),
                    parent_path,
                )
                save_json(
                    order_decomp_expanded_results(queries, parent_results, expanded_results),
                    expanded_path,
                )

                print(f"[decomp] progress: {done}/{total}")

    save_json(
        order_decomp_parent_results(queries, parent_results),
        parent_path,
    )
    save_json(
        order_decomp_expanded_results(queries, parent_results, expanded_results),
        expanded_path,
    )

    print(f"Saved: {parent_path}")
    print(f"Saved: {expanded_path}")


if __name__ == "__main__":
    BASE_URL = "https://one-api.bltcy.top/v1"
    MODEL = "gpt-5.2"
    TEMPERATURE = 0
    MAX_TOKENS = 2048
    MAX_RETRIES = 5
    SLEEP_SECONDS = 2.0
    MAX_WORKERS = 40
    SAVE_EVERY = 50
    KEEP_FAILED_AS_ORIGINAL = True

    DOMAINS = [
        "electrochemistry",
        # "cell_culture",
        # "immunotechnology",
        # "inorganic_materials",
        # "organic_synthesis",
        # "polymer_materials"
    ]

    PROMPT_TYPES = [
        # "hyde",
        # "decomp",
        # "reasoning_first",
        "reasoning",
    ]
    keys_path = Path("scripts/keys.json")
    keys = load_keys(keys_path)

    for d in DOMAINS:
        print(f"\nProcessing domain: {d}")
        input_path = Path(f"data/{d}/gen_query/queries_raw.json")

        with open(input_path, "r", encoding="utf-8") as f:
            queries = json.load(f)
        
        # queries = dict(list(queries.items())[:2])

        for prompt_type in PROMPT_TYPES:
            print(f"\nProcessing prompt type: {prompt_type}")

            if prompt_type == "decomp":
                parent_path = Path(f"data/{d}/gen_query/queries_decomp_parent.json")
                expanded_path = Path(f"data/{d}/gen_query/queries_decomp_expanded.json")

                process_decomp_prompt(
                    queries=queries,
                    keys=keys,
                    parent_path=parent_path,
                    expanded_path=expanded_path,
                )
            else:
                output_path = Path(f"data/{d}/gen_query/queries_{prompt_type}.json")
                process_standard_prompt(
                    queries=queries,
                    prompt_type=prompt_type,
                    keys=keys,
                    output_path=output_path,
                )


# python scripts/gen_query/build_rewrite_query.py
