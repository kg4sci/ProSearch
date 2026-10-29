import os
import json
from tavily import TavilyClient

client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

def retrieve_evidence(query: str, max_results: int = 10, top_urls: int = 10, chunks_per_source: int = 2):
    search_resp = client.search(
        query=query,
        topic="general",
        search_depth="advanced",
        max_results=max_results,
        include_answer=False,
        include_raw_content=False,
        include_images=False,
    )

    results = search_resp.get("results", [])
    urls = [r["url"] for r in results[:top_urls] if r.get("url")]

    if not urls:
        return {"search_results": results, "evidence_items": []}

    extract_resp = client.extract(
        urls=urls,
        query=query,
        chunks_per_source=chunks_per_source,
        extract_depth="advanced",
        format="text",
    )

    evidence_items = []
    rank = 1
    for src_rank, item in enumerate(extract_resp.get("results", []), start=1):
        raw = (item.get("raw_content") or "").strip()
        url = item.get("url")

        # Tavily 会把多个 chunk 用 [...] 拼在一个 raw_content 里
        chunks = [c.strip() for c in raw.split("[...]") if c.strip()]

        for chunk_rank, chunk in enumerate(chunks, start=1):
            evidence_items.append({
                "rank": rank,
                "source_rank": src_rank,
                "chunk_rank_within_source": chunk_rank,
                "url": url,
                "evidence_text": chunk,
            })
            rank += 1

    # 只保留前10个 chunk
    evidence_items = evidence_items[:10]

    return {
        "search_results": [
            {
                "title": r.get("title"),
                "url": r.get("url"),
                "score": r.get("score"),
                "snippet": r.get("content"),
            }
            for r in results
        ],
        "evidence_items": evidence_items,
    }
def run_batch(input_path: str, output_path: str):
    queries = json.load(open(input_path, "r", encoding="utf-8"))
    outputs = []

    for ex in queries:
        out = retrieve_evidence(ex["query"])
        outputs.append({
            "query_id": ex["query_id"],
            "query": ex["query"],
            **out
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run_batch(
        "data/electrochemistry/gen_query/queries_list.json",
        "data/electrochemistry/eval/agentic/open_web_baseline.json",
    )


# cd /home/liangwenliang/检索数据集
# conda activate retrieve_dataset
# python scripts/agentic/open_web_baseline.py

# export TAVILY_API_KEY=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
# export OPENAI_API_KEY=sk-y8SLogNDFu0Y08wfvvR65naEsDyiuKK2mkEqvzmeGw0YLY92

