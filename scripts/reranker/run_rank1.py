import json
from pathlib import Path
from rank1 import rank1

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_paragraph_index(paragraphs: list[dict]) -> dict[str, str]:
    para_index = {}
    for item in paragraphs:
        para_index[str(item["id"])] = item.get("paragraph", "")
    return para_index


def rerank_with_rank1(
    input_path: str,
    query_path: str,
    paragraph_path: str,
    output_path: str,
    model_name_or_path: str = "jhu-clsp/rank1-7B",
    num_gpus: int = 1,
    device: str = "cuda",
    context_size: int = 16000,
    max_output_tokens: int = 8192,
    fp_options: str = "float16",
):
    # 1) load files
    retrieval_runs = load_json(input_path)
    queries = load_json(query_path)
    paragraphs = load_json(paragraph_path)
    para_index = build_paragraph_index(paragraphs)

    # 2) init model once
    model = rank1(
        model_name_or_path=model_name_or_path,
        num_gpus=num_gpus,
        device=device,
        context_size=context_size,
        max_output_tokens=max_output_tokens,
        fp_options=fp_options,
    )

    reranked_results = {}
    debug_results = {}

    # 3) process each query
    for qid, doc_score_dict in retrieval_runs.items():
        qid = str(qid)
        query_text = queries[qid]["query"]

        # keep retrieval order as input order
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
            pairs.append((query_text, passage_text))

        if not pairs:
            reranked_results[qid] = {}
            debug_results[qid] = {
                "query": query_text,
                "num_candidates": 0,
                "missing_docids": missing_docids,
                "ranked_docids": [],
            }
            continue

        # 4) pointwise scoring
        # scores aligns with pairs/docids order
        scores = model.predict(pairs)

        # 5) sort by rank1 scores
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

    # 6) save
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
        output_path = f"data/{d}/eval/rank1/run_top100_raw.json"

        rerank_with_rank1(
            input_path=input_path,
            query_path=query_path,
            paragraph_path=paragraph_path,
            output_path=output_path,
            model_name_or_path="model/rank1-7b",
            num_gpus=1,
            device="cuda",
            context_size=16000,
            max_output_tokens=8192,
            fp_options="float16",
        )

# conda activate reasonrank
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
# CUDA_VISIBLE_DEVICES=4 python scripts/reranker/run_rank1.py

