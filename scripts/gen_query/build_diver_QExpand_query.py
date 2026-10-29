import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_paragraphs(path: str) -> Dict[str, Dict[str, Any]]:
    data = load_json(path)
    return {str(x["id"]): x for x in data}


def approximate_truncate(text: str, max_tokens: int = 512) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def build_passage_text(item: Dict[str, Any], max_tokens: int = 512) -> str:
    paragraph = (item.get("paragraph") or "").strip()
    return approximate_truncate(paragraph, max_tokens=max_tokens)


def format_top_passages(passages: List[str]) -> str:
    return "\n\n".join([f"{i+1}. {p}" for i, p in enumerate(passages)])


def get_prompt(method, query, top_passages_str, last_round_answer=None):
    prompts = {
        "thinkqe_revise_0": f"""Given a query and the provided passages (most of which may be incorrect or irrelevant), identify helpful information from the passages and use it to write a correct answering passage. Use your own knowledge, not just the example passages!

Query:
{query}

Possible helpful passages:
{top_passages_str}
""",
        "thinkqe_revise": f"""Given a query, the provided passages (most of which may be incorrect or irrelevant), and the previous round's answer, identify helpful information from the passages and refine the prior answer.
Ensure the output directly addresses the original query. Use your own knowledge, not just the example passages!

Query:
{query}

Possible helpful passages:
{top_passages_str}

Prior generated answer:
{last_round_answer}
""",
    }
    return prompts[method]


def sort_run_entry(run_entry: Dict[str, float]) -> List[str]:
    return [
        str(doc_id)
        for doc_id, _ in sorted(run_entry.items(), key=lambda x: x[1], reverse=True)
    ]


def select_topk_doc_ids(
    run_entry: Dict[str, float],
    top_k: int = 5,
    exclude_ids: Optional[set] = None,
) -> List[str]:
    exclude_ids = exclude_ids or set()
    ranked_doc_ids = sort_run_entry(run_entry)

    selected = []
    for doc_id in ranked_doc_ids:
        if doc_id in exclude_ids:
            continue
        selected.append(doc_id)
        if len(selected) >= top_k:
            break
    return selected


class APIBasedGenerator:
    """
    统一处理两类后端:
    1) 本地 vLLM 服务
    2) 远程闭源 API / one-api

    两者本质上都是 OpenAI-compatible API 调用。
    """

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_keys: List[str],
        temperature: float = 0.7,
        max_new_tokens: int = 512,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
    ):
        from openai import OpenAI

        if not api_keys or not isinstance(api_keys, list):
            raise ValueError("api_keys must be a non-empty list")

        self.OpenAI = OpenAI
        self.model_name = model_name
        self.base_url = base_url
        self.api_keys = api_keys
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self._thread_local = threading.local()

    def _get_client(self, api_key: str):
        if not hasattr(self._thread_local, "clients"):
            self._thread_local.clients = {}

        if api_key not in self._thread_local.clients:
            self._thread_local.clients[api_key] = self.OpenAI(
                base_url=self.base_url,
                api_key=api_key,
            )
        return self._thread_local.clients[api_key]

    def generate(self, prompt: str, api_key: str) -> str:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                client = self._get_client(api_key)
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_sleep * (attempt + 1))
        raise last_err


def _build_round1_single(
    qid: str,
    qinfo: Dict[str, Any],
    run_data: Dict[str, Dict[str, float]],
    paragraphs: Dict[str, Dict[str, Any]],
    generator: APIBasedGenerator,
    top_k: int,
    max_passage_tokens: int,
    api_key: str,
):
    raw_query = qinfo["query"]

    if qid not in run_data:
        return qid, {
            **qinfo,
            "raw_query": raw_query,
            "diverqe_answer": "",
            "diverqe_doc_ids": [],
            "query": raw_query,
        }

    doc_ids = select_topk_doc_ids(run_data[qid], top_k=top_k)

    passages = []
    for doc_id in doc_ids:
        if doc_id in paragraphs:
            passages.append(
                build_passage_text(paragraphs[doc_id], max_tokens=max_passage_tokens)
            )

    prompt = get_prompt(
        method="thinkqe_revise_0",
        query=raw_query,
        top_passages_str=format_top_passages(passages),
    )
    answer = generator.generate(prompt, api_key=api_key).strip()
    expanded_query = f"{raw_query}\n\n{answer}".strip()

    return qid, {
        **qinfo,
        "raw_query": raw_query,
        "diverqe_answer": answer,
        "diverqe_doc_ids": doc_ids,
        "query": expanded_query,
    }


def _build_round2_single(
    qid: str,
    qinfo: Dict[str, Any],
    round1_queries: Dict[str, Dict[str, Any]],
    run_data_round2: Dict[str, Dict[str, float]],
    paragraphs: Dict[str, Dict[str, Any]],
    generator: APIBasedGenerator,
    top_k: int,
    max_passage_tokens: int,
    api_key: str,
):
    raw_query = qinfo["query"]

    prev_info = round1_queries.get(qid, {})
    prev_answer = prev_info.get("diverqe_answer", "")
    prev_doc_ids = set(prev_info.get("diverqe_doc_ids", []))

    if qid not in run_data_round2:
        return qid, {
            **qinfo,
            "raw_query": raw_query,
            "diverqe_answer_round1": prev_answer,
            "diverqe_doc_ids_round1": list(prev_doc_ids),
            "diverqe_answer_round2": "",
            "diverqe_doc_ids_round2": [],
            "query": f"{raw_query}\n\n{prev_answer}".strip() if prev_answer else raw_query,
        }

    doc_ids_round2 = select_topk_doc_ids(
        run_data_round2[qid],
        top_k=top_k,
        exclude_ids=prev_doc_ids,
    )

    passages = []
    for doc_id in doc_ids_round2:
        if doc_id in paragraphs:
            passages.append(
                build_passage_text(paragraphs[doc_id], max_tokens=max_passage_tokens)
            )

    prompt = get_prompt(
        method="thinkqe_revise",
        query=raw_query,
        top_passages_str=format_top_passages(passages),
        last_round_answer=prev_answer,
    )
    answer_round2 = generator.generate(prompt, api_key=api_key).strip()
    final_query = f"{raw_query}\n\n{answer_round2}".strip()

    return qid, {
        **qinfo,
        "raw_query": raw_query,
        "diverqe_answer_round1": prev_answer,
        "diverqe_doc_ids_round1": list(prev_doc_ids),
        "diverqe_answer_round2": answer_round2,
        "diverqe_doc_ids_round2": doc_ids_round2,
        "query": final_query,
    }


def build_round1_queries(
    queries: Dict[str, Dict[str, Any]],
    run_data: Dict[str, Dict[str, float]],
    paragraphs: Dict[str, Dict[str, Any]],
    generator: APIBasedGenerator,
    top_k: int = 5,
    max_passage_tokens: int = 256,
    max_workers: int = 8,
) -> Dict[str, Dict[str, Any]]:
    out = {}
    items = list(queries.items())
    if not items:
        return out

    worker_num = max(1, min(max_workers, len(generator.api_keys), len(items)))

    with ThreadPoolExecutor(max_workers=worker_num) as executor:
        futures = {}
        for idx, (qid, qinfo) in enumerate(items):
            api_key = generator.api_keys[idx % len(generator.api_keys)]
            fut = executor.submit(
                _build_round1_single,
                qid,
                qinfo,
                run_data,
                paragraphs,
                generator,
                top_k,
                max_passage_tokens,
                api_key,
            )
            futures[fut] = qid

        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            qid, result = fut.result()
            out[qid] = result
            done += 1
            if done % 20 == 0 or done == total:
                print(f"[Round1] Progress: {done}/{total}")

    return {qid: out[qid] for qid in queries.keys()}


def build_round2_queries(
    raw_queries: Dict[str, Dict[str, Any]],
    round1_queries: Dict[str, Dict[str, Any]],
    run_data_round2: Dict[str, Dict[str, float]],
    paragraphs: Dict[str, Dict[str, Any]],
    generator: APIBasedGenerator,
    top_k: int = 5,
    max_passage_tokens: int = 256,
    max_workers: int = 8,
) -> Dict[str, Dict[str, Any]]:
    out = {}
    items = list(raw_queries.items())
    if not items:
        return out

    worker_num = max(1, min(max_workers, len(generator.api_keys), len(items)))

    with ThreadPoolExecutor(max_workers=worker_num) as executor:
        futures = {}
        for idx, (qid, qinfo) in enumerate(items):
            api_key = generator.api_keys[idx % len(generator.api_keys)]
            fut = executor.submit(
                _build_round2_single,
                qid,
                qinfo,
                round1_queries,
                run_data_round2,
                paragraphs,
                generator,
                top_k,
                max_passage_tokens,
                api_key,
            )
            futures[fut] = qid

        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            qid, result = fut.result()
            out[qid] = result
            done += 1
            if done % 20 == 0 or done == total:
                print(f"[Round2] Progress: {done}/{total}")

    return {qid: out[qid] for qid in raw_queries.keys()}


def process_domain_round1(
    domain: str,
    generator: APIBasedGenerator,
    top_k: int = 5,
    max_passage_tokens: int = 256,
    max_workers: int = 8,
    output_tag: str = "vllm",
):
    query_file = f"data/{domain}/gen_query/queries_raw.json"
    run_file = f"data/{domain}/eval/diver_retriever/run_top100_raw.json"
    paragraph_file = f"seed_data/{domain}/paragraphs.json"
    output_file = f"data/{domain}/gen_query/queries_diverqe_round1_{output_tag}.json"

    print(f"[Round1] Processing domain: {domain}")

    queries = load_json(query_file)
    run_data = load_json(run_file)
    paragraphs = load_paragraphs(paragraph_file)

    out = build_round1_queries(
        queries=queries,
        run_data=run_data,
        paragraphs=paragraphs,
        generator=generator,
        top_k=top_k,
        max_passage_tokens=max_passage_tokens,
        max_workers=max_workers,
    )
    save_json(out, output_file)
    print(f"[Round1] Saved: {output_file}")


def process_domain_round2(
    domain: str,
    generator: APIBasedGenerator,
    top_k: int = 5,
    max_passage_tokens: int = 256,
    max_workers: int = 8,
    output_tag: str = "vllm",
):
    raw_query_file = f"data/{domain}/gen_query/queries_raw.json"
    round1_query_file = f"data/{domain}/gen_query/queries_diverqe_round1_{output_tag}.json"
    run_file = f"data/{domain}/eval/diver_retriever/run_top100_diverqe_round1_{output_tag}.json"
    paragraph_file = f"seed_data/{domain}/paragraphs.json"
    output_file = f"data/{domain}/gen_query/queries_diverqe_round2_{output_tag}.json"

    print(f"[Round2] Processing domain: {domain}")

    raw_queries = load_json(raw_query_file)
    round1_queries = load_json(round1_query_file)
    run_data_round2 = load_json(run_file)
    paragraphs = load_paragraphs(paragraph_file)

    out = build_round2_queries(
        raw_queries=raw_queries,
        round1_queries=round1_queries,
        run_data_round2=run_data_round2,
        paragraphs=paragraphs,
        generator=generator,
        top_k=top_k,
        max_passage_tokens=max_passage_tokens,
        max_workers=max_workers,
    )
    save_json(out, output_file)
    print(f"[Round2] Saved: {output_file}")


if __name__ == "__main__":
    MODE = "round2"
    DOMAINS = [
        "electrochemistry",
        # "cell_culture",
        # "immunotechnology",
        # "inorganic_materials",
        # "organic_synthesis",
        # "polymer_materials",
    ]

    # OUTPUT_TAG = "qwen"
    # PROVIDER = "vllm_local" 

    OUTPUT_TAG = "gpt"
    PROVIDER = "remote_api"

    VLLM_BASE_URL = "http://127.0.0.1:8002/v1"
    VLLM_MODEL_NAME = "QWEN-R1-Distill-14B"
    VLLM_API_KEYS = ["local-token"]

    REMOTE_BASE_URL = "https://one-api.bltcy.top/v1"
    REMOTE_MODEL_NAME = "gpt-5.2"
    USE_REMOTE_KEYS_JSON = True
    REMOTE_KEYS_PATH = "scripts/keys.json"

    TEMPERATURE = 0.7
    MAX_NEW_TOKENS = 1024
    TOP_K = 5
    MAX_PASSAGE_TOKENS = 512
    MAX_WORKERS = 8

    if PROVIDER == "vllm_local":
        base_url = VLLM_BASE_URL
        model_name = VLLM_MODEL_NAME
        api_keys = VLLM_API_KEYS
    elif PROVIDER == "remote_api":
        base_url = REMOTE_BASE_URL
        model_name = REMOTE_MODEL_NAME
        api_keys = load_json(REMOTE_KEYS_PATH)
    else:
        raise ValueError(f"Unsupported PROVIDER: {PROVIDER}")

    generator = APIBasedGenerator(
        model_name=model_name,
        base_url=base_url,
        api_keys=api_keys,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    for domain in DOMAINS:
        if MODE == "round1":
            process_domain_round1(
                domain=domain,
                generator=generator,
                top_k=TOP_K,
                max_passage_tokens=MAX_PASSAGE_TOKENS,
                max_workers=MAX_WORKERS,
                output_tag=OUTPUT_TAG,
            )
        elif MODE == "round2":
            process_domain_round2(
                domain=domain,
                generator=generator,
                top_k=TOP_K,
                max_passage_tokens=MAX_PASSAGE_TOKENS,
                max_workers=MAX_WORKERS,
                output_tag=OUTPUT_TAG,
            )
        else:
            raise ValueError(f"Unsupported MODE: {MODE}")


'''
conda activate vllm_env
CUDA_VISIBLE_DEVICES=4 vllm serve /home/liangwenliang/检索数据集/model/QWEN-R1-Distill-14B \
    --served-model-name QWEN-R1-Distill-14B \
    --host 0.0.0.0 --port 8002 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.80 \
    --api-key local-token


conda activate retrieve_dataset
cd 检索数据集
python scripts/gen_query/build_diver_QExpand_query.py
'''


