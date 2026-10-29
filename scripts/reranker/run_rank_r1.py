import json
from pathlib import Path
from tqdm import tqdm

from llmrankers.setwise import RankR1SetwiseLlmRanker
from llmrankers.rankers import SearchResult


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_paragraph_index(paragraphs: list[dict]) -> dict[str, str]:
    para_index = {}
    for p in paragraphs:
        pid = str(p["id"])
        para_text = p.get("paragraph", "")
        para_index[pid] = para_text
    return para_index


def build_search_results(
    candidate_dict: dict,
    para_index: dict[str, str],
) -> tuple[list[SearchResult], list[str]]:
    docs = []
    missing_docids = []

    for pid, retriever_score in candidate_dict.items():
        pid = str(pid)
        para_text = para_index.get(pid)

        if para_text is None:
            missing_docids.append(pid)
            continue

        docs.append(
            SearchResult(
                docid=pid,
                text=para_text,
                score=float(retriever_score),
            )
        )

    return docs, missing_docids


def rerank_one_query(
    ranker: RankR1SetwiseLlmRanker,
    query_text: str,
    candidate_dict: dict,
    para_index: dict[str, str],
) -> tuple[dict[str, float], dict]:
    docs, missing_docids = build_search_results(candidate_dict, para_index)

    if len(docs) == 0:
        return {}, {
            "query": query_text,
            "num_candidates": 0,
            "num_missing": len(missing_docids),
            "missing_docids": missing_docids,
            "ranked_docids": [],
        }

    reranked_docs = ranker.rerank(query_text, docs)
    ranked_docids = [str(doc.docid) for doc in reranked_docs]

    n = len(ranked_docids)
    reranked_score_dict = {
        docid: float(n - rank_idx)
        for rank_idx, docid in enumerate(ranked_docids)
    }

    debug_info = {
        "query": query_text,
        "num_candidates": len(docs),
        "num_missing": len(missing_docids),
        "missing_docids": missing_docids,
        "ranked_docids": ranked_docids,
    }

    return reranked_score_dict, debug_info


if __name__ == "__main__":
    domains = [
        # "cell_culture",
        "electrochemistry",
        # "immunotechnology",
        # "inorganic_materials",
        # "organic_synthesis",
        # "polymer_materials",
    ]

    ranker = RankR1SetwiseLlmRanker(
        model_name_or_path="model/Qwen2.5-7B-Instruct",
        lora_name_or_path="model/Rank-R1-7B-v0.1",
        prompt_file="scripts/reranker/rank_r1_prompt.toml",
        num_child=9,   
        k=10,          
        verbose=True, 
    )

    for d in domains:
        input_path = f"data/{d}/eval/reasonIR/run_top100_raw.json"
        query_path = f"data/{d}/gen_query/queries_raw.json"
        paragraph_path = f"seed_data/{d}/paragraphs.json"

        output_path = f"data/{d}/eval/rank_r1/run_top100_raw.json"
        debug_output_path = f"data/{d}/eval/rank_r1/run_top100_raw.debug.json"

        queries = load_json(query_path)
        paragraphs = load_json(paragraph_path)
        input_data = load_json(input_path)

        para_index = build_paragraph_index(paragraphs)

        rerank_results = {}
        debug_results = {}

        for qid, candidate_dict in tqdm(input_data.items(), desc=f"Rank-R1 reranking [{d}]"):
            qid = str(qid)
            query_text = queries[qid]["query"]

            reranked_score_dict, debug_info = rerank_one_query(
                ranker=ranker,
                query_text=query_text,
                candidate_dict=candidate_dict,
                para_index=para_index,
            )

            rerank_results[qid] = reranked_score_dict
            debug_results[qid] = debug_info

            # if len(rerank_results) > 2:
            #     break

        save_json(rerank_results, output_path)
        save_json(debug_results, debug_output_path)

        print(f"[Done] {d}")
        print(f"rerank results saved to: {output_path}")
        print(f"debug info saved to:   {debug_output_path}")



# conda activate rankr1_vllm
# cd 检索数据集
# CUDA_VISIBLE_DEVICES=1,2  python scripts/reranker/run_rank_r1.py



