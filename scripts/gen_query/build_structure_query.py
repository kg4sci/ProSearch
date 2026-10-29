import json


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest_jsonl(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data[item["candidate_id"]] = item
    return data


def render_task(type_name: str) -> str:
    return "reproduction" if type_name == "reproduction" else "exploration"


def render_constraint(c: dict) -> str:
    field = c["field"]
    op = c["op"]

    if op == "=":
        return f"{field}={c['value']}"
    elif op == "!=":
        return f"{field}!={c['value']}"
    elif op == "between":
        if c.get("surface"):
            return f"{field}={c['surface']}"
        r = c["range"]
        lower = r.get("lower")
        upper = r.get("upper")
        unit = r.get("unit", "")
        if unit:
            return f"{field}={lower}-{upper} {unit}"
        return f"{field}={lower}-{upper}"
    else:
        raise ValueError(f"Unsupported op: {op}")


def build_structured_query(item: dict) -> str:
    parts = [f"task={render_task(item['type'])}"]

    for c in item["constraints"]:
        parts.append(render_constraint(c))

    if item.get("explore_field"):
        parts.append(f"explore_field={item['explore_field']}")

    return "; ".join(parts)

DOMAINS = [
    "electrochemistry",
    # "cell_culture",
    # "immunotechnology",
    # "inorganic_materials",
    # "organic_synthesis",
    # "polymer_materials",
]

for domain in DOMAINS:
    raw_file = f"data/{domain}/gen_query/queries_raw.json"
    manifest_file = f"data/{domain}/stage1_relevance/sampled_constraints_stage1.boundary_manifest.jsonl"
    output_file = f"data/{domain}/gen_query/queries_structured.json"

    raw_queries = load_json(raw_file)
    manifest = load_manifest_jsonl(manifest_file)

    output = {}
    for qid, qinfo in raw_queries.items():
        item = manifest[qid]

        new_item = dict(qinfo)
        new_item["query"] = build_structured_query(item)
        output[qid] = new_item

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"完成: {output_file} | 共 {len(output)} 条")

# python scripts/gen_query/build_structure_query.py

