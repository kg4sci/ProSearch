import os
import json
from typing import List
from pydantic import BaseModel, Field
from tqdm import tqdm
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch, TavilyExtract


# ---------- 1) 定义最终输出 schema ----------
class EvidenceItem(BaseModel):
    rank: int = Field(description="Final rank of this evidence item, starting from 1")
    url: str = Field(description="Source URL")
    evidence_text: str = Field(description="A concise evidence passage copied or condensed from extracted content")

class EvidenceList(BaseModel):
    evidence_items: List[EvidenceItem] = Field(
        description="Top-10 ranked evidence passages for the query"
    )


# ---------- 2) 初始化工具 ----------
search_tool = TavilySearch(
    max_results=10,
    topic="general",
    include_answer=False,
    include_raw_content=False,
    search_depth="advanced",
)

extract_tool = TavilyExtract(
    extract_depth="advanced",
)


# ---------- 3) 初始化 agent ----------
agent = create_agent(
    model=ChatOpenAI(model="gpt-4o-mini", temperature=0,openai_api_base="https://one-api.bltcy.top/v1", openai_api_key='sk-y8SLogNDFu0Y08wfvvR65naEsDyiuKK2mkEqvzmeGw0YLY92'),
    tools=[search_tool, extract_tool],
    response_format=EvidenceList,
    system_prompt=(
        """
        You are an evidence retrieval agent for scientific experimental queries.
        Your goal is to find web evidence passages, not to answer directly.

        You may reformulate the user's query into better search-engine-friendly queries if needed, but you must preserve all original constraints and never weaken or drop them.
        Use search multiple times if the first results are generic background pages.
        Use extract only on promising URLs.
        Prefer protocol-like or experiment-like passages from papers, publisher pages, PMC, or experimental sections.
        Return only the final top-10 ranked evidence passages.
        """
    ),
)


# ---------- 4) 单条 query 运行 ----------
def run_one_query(query_id: str, query: str):
    result = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": query
            }
        ]
    })

    structured = result["structured_response"]

    return {
        "query_id": query_id,
        "query": query,
        "evidence_items": [item.model_dump() for item in structured.evidence_items]
    }


# ---------- 5) 批量运行 ----------
def run_batch(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    outputs = []
    for ex in tqdm(queries, desc="Processing Queries"):
        outputs.append(run_one_query(ex["query_id"], ex["query"]))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    run_batch(
        "data/electrochemistry/gen_query/queries_list.json",
        "data/electrochemistry/eval/agentic/open_world_agent.json"
    )

# python scripts/agentic/open_web_agentic.py

