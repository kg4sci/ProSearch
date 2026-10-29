import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams


random.seed(42)


def avg_scores(did_scores):
    scores_by_doc = defaultdict(list)
    for doc_id, score in did_scores:
        scores_by_doc[str(doc_id)].append(float(score))
    return [(doc_id, sum(scores) / len(scores)) for doc_id, scores in scores_by_doc.items()]


class GroupReranker:
    def __init__(
        self,
        model_path: str,
        num_gpus: int = 1,
        dtype: str = "bfloat16",
        max_model_len: int = 32000,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0.3,
        top_p: float = 0.8,
        max_tokens: int = 8000,
    ) -> None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self.llm = LLM(
            model=model_path,
            dtype=dtype,
            tensor_parallel_size=num_gpus,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=10,
        )

        self.group_system_prompt = """Your task is to evaluate and rank documents based on how well they help answer the given query. Follow this evaluation priority:
1. PRIMARY: Usefulness & Helpfulness - Does the document provide actionable information, solutions, or direct answers that help address the user's needs?
2. SECONDARY: Relevance - Does the document contain information related to the query topic?

Evaluation Process:
1. First, identify the user's core intent and what kind of help they need from the query
2. For each document, assess:
   - How directly it addresses the user's intent
   - What actionable information or answers it provides
   - How much it helps solve the user's problem or need
3. Compare documents against each other to ensure proper ranking
4. Assign scores that reflect the relative usefulness ranking

Scoring Scale (0-10):
- 9-10: Extremely helpful, directly answers the query with actionable information
- 7-8: Very helpful, provides substantial useful information for the query
- 5-6: Moderately helpful, contains some useful information but incomplete
- 3-4: Minimally helpful, limited useful information despite topic relevance
- 1-2: Barely helpful, mentions related topics but provides little useful information
- 0: Not helpful at all, cannot assist with answering the query
"""

        self.group_user_prompt = """I will provide you {TOPK} documents, each indicated by a numerical identifier []. Score these documents based on their Usefulness and Relevance to the query.
Query:
{QUERY}

Documents:
{PASSAGES}

## Final Output Format
You must structure your response in exactly two parts: provide your brief reasoning process first, then output final scores in JSON format like below, with document IDs as string keys and integer scores as values for all {TOPK} documents.
The reasoning process and answer are enclosed within <reason> </reason> and <answer> </answer> tags, respectively. Do NOT output anything outside the specified tags. Follow this exact format:
<reason>
[Analyze each document's usefulness and relevance to the query, explaining your scoring rationale]
</reason>
<answer>
```json
{{"[1]": 5, "[2]": 3, "[3]": 8}}
```
</answer>
"""

    def _extract_rerank_output(self, output_str: str):
        if output_str is None:
            return None

        answer_match = re.search(r"<answer>(.*?)</answer>", output_str, re.DOTALL)
        if answer_match:
            output_str = answer_match.group(1).strip()

        try:
            json_matches = re.findall(r"(?:```json\s*)([\s\S]+?)(?:\s*```)", output_str)
            if json_matches:
                json_str = json_matches[-1].strip()
            else:
                json_str = re.findall(r"\{[\s\S]*?\}", output_str)[-1]

            json_str = json_str.strip()
            json_str = re.sub(r'(\[\d+\])\]', r'\1', json_str)
            if json_str.count("{") > json_str.count("}"):
                json_str += "}"
            json_str = re.sub(r"\s+", " ", json_str).replace("\n", " ").strip()

            return dict(json.loads(json_str))
        except Exception:
            print("=== Failed output ===")
            print(output_str)
            raise ValueError("No valid JSON found in model output.")

    def prepare_groups_one_epoch(
        self,
        qid_queries,
        qid_did_ctxs,
        group_size=10,
        interval=10,
        max_doc_lens=3000,
    ):
        batch_messages = []
        batch_info = []

        for item in qid_queries:
            qid = item["qid"]
            query = item["query"]
            docs = list(qid_did_ctxs[qid])
            random.shuffle(docs)

            for i in range(0, len(docs), interval):
                doc_group = docs[i : i + group_size]
                dids = []
                doc_texts = []

                for doc in doc_group:
                    dids.append(str(doc["id"]))
                    doc_text = re.sub(r"\n+", " ", doc["text"]).strip()
                    doc_texts.append(doc_text[:max_doc_lens])

                docs_str = "".join(
                    [f"[{idx + 1}]. {doc_text}\n\n" for idx, doc_text in enumerate(doc_texts)]
                )
                group_text = self.group_user_prompt.format(
                    QUERY=query,
                    PASSAGES=docs_str,
                    TOPK=len(dids),
                )

                message = self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": self.group_system_prompt},
                        {"role": "user", "content": group_text},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                batch_messages.append(message)
                batch_info.append((qid, dids))

        return batch_messages, batch_info

    def rerank_by_group(
        self,
        qid_queries,
        qid_did_ctxs,
        topk=100,
        group_size=10,
        interval=10,
        num_epoch=2,
        max_doc_lens=3000,
    ):
        total_batch_messages = []
        total_batch_info = []

        for _ in range(num_epoch):
            batch_messages, batch_info = self.prepare_groups_one_epoch(
                qid_queries=qid_queries,
                qid_did_ctxs=qid_did_ctxs,
                group_size=group_size,
                interval=interval,
                max_doc_lens=max_doc_lens,
            )
            total_batch_messages.extend(batch_messages)
            total_batch_info.extend(batch_info)

        print(f"Total {len(total_batch_messages)} groups to score.")
        outputs = self.llm.generate(total_batch_messages, self.sampling_params, use_tqdm=True)

        new_scores = {item["qid"]: [] for item in qid_queries}

        for out, (qid, dids) in tqdm(zip(outputs, total_batch_info), total=len(total_batch_info)):
            output_str = out.outputs[0].text
            try:
                idx_score = self._extract_rerank_output(output_str)
                did_score = []
                for idx, score in idx_score.items():
                    pos = int(str(idx).strip().strip("[]")) - 1
                    if 0 <= pos < len(dids):
                        did_score.append((dids[pos], float(score) / 10.0))

                scored = {did for did, _ in did_score}
                for did in dids:
                    if did not in scored:
                        did_score.append((did, 0.0))
            except Exception as e:
                print(output_str)
                print("-- group:", dids)
                print("Parse error:", e)
                did_score = [(did, 0.0) for did in dids]

            new_scores[qid].extend(did_score)

        final_scores = {}
        for qid, did_scores in new_scores.items():
            did_scores = avg_scores(did_scores)
            did_scores = sorted(did_scores, key=lambda x: x[1], reverse=True)[:topk]
            final_scores[qid] = {did: score for did, score in did_scores}

        return final_scores


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_queries(query_path: str):
    data = load_json(query_path)
    qid_to_query = {}
    for qid, item in data.items():
        qid_to_query[str(qid)] = item["query"]
    return qid_to_query


def load_paragraphs(paragraph_path: str):
    data = load_json(paragraph_path)
    did_to_text = {}
    for item in data:
        did_to_text[str(item["id"])] = item["paragraph"].strip()
    return did_to_text


def interpolate_scores(rerank_scores, retriever_scores, alpha_rerank=0.6, alpha_retriever=0.4):
    merged = {}
    for qid, score_dict in rerank_scores.items():
        merged[qid] = {}
        for did, rerank_score in score_dict.items():
            retr_score = float(retriever_scores[qid].get(did, 0.0))
            merged[qid][did] = alpha_rerank * float(rerank_score) + alpha_retriever * retr_score
        merged[qid] = dict(sorted(merged[qid].items(), key=lambda x: x[1], reverse=True))
    return merged


def rerank_with_diver_grouprank(
    input_path: str,
    query_path: str,
    paragraph_path: str,
    output_path: str,
    merged_output_path: str,
    model_path: str,
    input_k: int = 100,
    output_k: int = 100,
    group_size: int = 10,
    interval: int = 10,
    repeat_num: int = 2,
    max_doc_len: int = 3000,
    num_gpus: int = 1,
    dtype: str = "bfloat16",
):
    retriever_scores = load_json(input_path)
    qid_to_query = load_queries(query_path)
    did_to_text = load_paragraphs(paragraph_path)

    qid_queries = []
    qid_did_ctxs = {}

    for qid, scores in retriever_scores.items():
        qid = str(qid)
        qid_queries.append({"qid": qid, "query": qid_to_query[qid]})

        sorted_scores = sorted(
            [(str(did), float(score)) for did, score in scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:input_k]

        qid_did_ctxs[qid] = [
            {"id": did, "text": did_to_text.get(did, "")}
            for did, _ in sorted_scores
        ]

    model = GroupReranker(
        model_path=model_path,
        num_gpus=num_gpus,
        dtype=dtype,
    )

    rerank_scores = model.rerank_by_group(
        qid_queries=qid_queries,
        qid_did_ctxs=qid_did_ctxs,
        topk=output_k,
        group_size=group_size,
        interval=interval,
        num_epoch=repeat_num,
        max_doc_lens=max_doc_len,
    )

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(rerank_scores, f, ensure_ascii=False, indent=2)

    merged_scores = interpolate_scores(
        rerank_scores=rerank_scores,
        retriever_scores=retriever_scores,
        alpha_rerank=0.6,
        alpha_retriever=0.4,
    )
    with open(merged_output_path, "w", encoding="utf-8") as f:
        json.dump(merged_scores, f, ensure_ascii=False, indent=2)

    print(f"Saved pure rerank scores to: {output_path}")
    print(f"Saved interpolated scores to: {merged_output_path}")


if __name__ == "__main__":
    domains = [
        # "cell_culture",
        "electrochemistry",
        # "immunotechnology",
        # "inorganic_materials",
        # "organic_synthesis",
        # "polymer_materials",
    ]

    for d in domains:
        print(f"Processing {d} ...")
        input_path = f"data/{d}/eval/bge_reasoner/run_top100_raw.json"
        query_path = f"data/{d}/gen_query/queries_raw.json"
        paragraph_path = f"seed_data/{d}/paragraphs.json"

        output_path = f"data/{d}/eval/diver_grouprank/run_top100_raw.json"
        merged_output_path = f"data/{d}/eval/diver_grouprank/run_top100_interp.json"

        rerank_with_diver_grouprank(
            input_path=input_path,
            query_path=query_path,
            paragraph_path=paragraph_path,
            output_path=output_path,
            merged_output_path=merged_output_path,
            model_path="model/Diver-GroupRank-7B",
            input_k=100,
            output_k=100,
            group_size=10,
            interval=10,
            repeat_num=2,
            max_doc_len=3000,
            num_gpus=torch.cuda.device_count(),
            dtype="bfloat16",
        )

# conda activate reasonrank
# CUDA_VISIBLE_DEVICES=4,5 python scripts/reranker/run_diver_grouprank.py

