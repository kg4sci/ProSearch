import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_paragraph_index(paragraphs: List[dict]) -> Dict[str, str]:
    para_index = {}
    for item in paragraphs:
        para_index[str(item["id"])] = item.get("paragraph", "")
    return para_index


def format_instruction(instruction: str, query: str, doc: str) -> str:
    if instruction is None:
        instruction = "Given a web search query, retrieve relevant passages that answer the query"
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction,
        query=query,
        doc=doc,
    )


class Qwen3Reranker:
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-Reranker-8B",
        instruction: str = "Given a web search query, retrieve relevant passages that answer the query",
        max_length: int = 2048,
        device: str = "cuda",
        use_fp16: bool = True,
    ):
        self.instruction = instruction
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
        )

        model_kwargs = {}
        if use_fp16:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        ).eval().to(device)

        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        self.prefix = (
            "<|im_start|>system\n"
            'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
            'Note that the answer can only be "yes" or "no".'
            "<|im_end|>\n"
            "<|im_start|>user\n"
        )
        self.suffix = (
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )

        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def process_inputs(self, pairs: List[str]):
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )

        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens

        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )

        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(self, inputs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    @torch.no_grad()
    def score(self, query: str, docs: List[str], batch_size: int = 8) -> List[float]:
        all_scores = []

        for start in range(0, len(docs), batch_size):
            batch_docs = docs[start:start + batch_size]
            batch_pairs = [
                format_instruction(self.instruction, query, doc)
                for doc in batch_docs
            ]
            inputs = self.process_inputs(batch_pairs)
            batch_scores = self.compute_logits(inputs)
            all_scores.extend(batch_scores)

        return all_scores


def rerank_with_qwen3_reranker(
    input_path: str,
    query_path: str,
    paragraph_path: str,
    output_path: str,
    model_name_or_path: str = "Qwen/Qwen3-Reranker-8B",
    batch_size: int = 8,
):
    retrieval_runs = load_json(input_path)
    queries = load_json(query_path)
    paragraphs = load_json(paragraph_path)

    para_index = build_paragraph_index(paragraphs)

    reranker = Qwen3Reranker(
        model_name_or_path=model_name_or_path,
        instruction="Given a web search query, retrieve relevant passages that answer the query",
        max_length=8192,
        device="cuda",
        use_fp16=True,
    )

    reranked_results = {}
    debug_results = {}

    for qid, doc_score_dict in retrieval_runs.items():
        qid = str(qid)
        query_text = queries[qid]["query"]

        ranked_by_retriever = sorted(
            doc_score_dict.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        docids = []
        docs = []
        missing_docids = []

        for docid, _retrieval_score in ranked_by_retriever:
            docid = str(docid)
            passage_text = para_index.get(docid)

            if passage_text is None:
                missing_docids.append(docid)
                continue

            docids.append(docid)
            docs.append(passage_text)

        if not docs:
            reranked_results[qid] = {}
            debug_results[qid] = {
                "query": query_text,
                "num_candidates": 0,
                "missing_docids": missing_docids,
                "ranked_docids": [],
            }
            continue

        scores = reranker.score(query_text, docs, batch_size=batch_size)

        docid_score_pairs = list(zip(docids, scores))
        docid_score_pairs.sort(key=lambda x: x[1], reverse=True)

        reranked_results[qid] = {
            docid: float(score)
            for docid, score in docid_score_pairs
        }

        debug_results[qid] = {
            "query": query_text,
            "num_candidates": len(docids),
            "missing_docids": missing_docids,
            "ranked_docids": [docid for docid, _ in docid_score_pairs],
        }

    save_json(reranked_results, output_path)
    save_json(
        debug_results,
        output_path.replace(".json", ".debug.json"),
    )


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
        input_path = f"data/{d}/eval/reasonIR/run_top100_raw.json"
        query_path = f"data/{d}/gen_query/queries_raw.json"
        paragraph_path = f"seed_data/{d}/paragraphs.json"
        output_path = f"data/{d}/eval/qwen3_reranker/run_top100_raw.json"

        rerank_with_qwen3_reranker(
            input_path=input_path,
            query_path=query_path,
            paragraph_path=paragraph_path,
            output_path=output_path,
            model_name_or_path="model/Qwen3-Reranker-8B",
            batch_size=8
        )

# conda activate retrieve_dataset
# CUDA_VISIBLE_DEVICES=5 python scripts/reranker/run_qwen3_reranker.py

