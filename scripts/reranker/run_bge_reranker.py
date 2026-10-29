import json
from pathlib import Path
from typing import Dict, List

from FlagEmbedding import FlagReranker


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


def rerank_with_bge_reranker(
    input_path: str,
    query_path: str,
    paragraph_path: str,
    output_path: str,
    model_name_or_path: str = "BAAI/bge-reranker-v2-m3",
    max_length: int = 1024,
    batch_size: int = 32,
    use_fp16: bool = True,
    normalize: bool = False,
    max_doc_chars: int = None,
):
    retrieval_runs = load_json(input_path)
    queries = load_json(query_path)
    paragraphs = load_json(paragraph_path)

    para_index = build_paragraph_index(paragraphs)

    reranker = FlagReranker(
        model_name_or_path,
        use_fp16=use_fp16,
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
        pairs = []
        missing_docids = []

        for docid, _retrieval_score in ranked_by_retriever:
            docid = str(docid)
            passage_text = para_index.get(docid)

            if passage_text is None:
                missing_docids.append(docid)
                continue

            docids.append(docid)
            pairs.append([query_text, passage_text])

        if not pairs:
            reranked_results[qid] = {}
            debug_results[qid] = {
                "query": query_text,
                "num_candidates": 0,
                "missing_docids": missing_docids,
                "ranked_docids": [],
            }
            continue

        # FlagEmbedding 会按输入 pairs 顺序返回分数
        scores = reranker.compute_score(
            pairs,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
        )

        # 某些情况下单条输入会返回 float；这里统一成 list
        if isinstance(scores, float):
            scores = [scores]

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
        output_path = f"data/{d}/eval/bge_reranker/run_top100_raw.json"

        rerank_with_bge_reranker(
            input_path=input_path,
            query_path=query_path,
            paragraph_path=paragraph_path,
            output_path=output_path,
            model_name_or_path="/home/liangwenliang/术语对齐/entity_linking/model/bge-reranker-v2-m3",
            max_length=2048,
            batch_size=16,
            use_fp16=True,
            normalize=False
        )


# conda activate retrieve_dataset
# CUDA_VISIBLE_DEVICES=5 python scripts/reranker/run_bge_reranker.py
