import json

input_path = "data/electrochemistry/gen_query/queries_raw.json"
output_path = "data/electrochemistry/gen_query/queries_list.json"

with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

output = [
    {
        "query_id": qid,
        "query": info["query"]
    }
    for qid, info in data.items()
]

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Saved to {output_path}")

# python scripts/agentic/filter_query.py