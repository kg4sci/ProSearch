import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

LEAF = "__LEAF__"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def is_field_node(obj: Any) -> bool:
    return isinstance(obj, dict) and "value" in obj and "evidence" in obj


def normalize_list(values: Union[str, List[Any], None]) -> List[str]:
    if values is None:
        return []
    if not isinstance(values, list):
        values = [values]

    out = []
    seen = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def original_value_key(v: Any) -> str:
    """
    原始抽取得到的 leaf value 一般是字符串。
    如果不是字符串，也统一转成稳定字符串键。
    """
    if isinstance(v, str):
        return v
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


def build_allowed_tree(field_paths: List[str]) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for path in field_paths:
        parts = path.split(".")
        cur = tree
        for i, p in enumerate(parts):
            if i == len(parts) - 1:
                cur[p] = LEAF
            else:
                cur = cur.setdefault(p, {})
    return tree

def load_normalization_results(
    normalization_path: Path,
    allowed_fields: set,
) -> Dict[str, Dict[str, List[str]]]:
    """
    只支持单个 json 文件，内容形如 {field_name: [ ... ]}
    """
    normalization_path = Path(normalization_path)
    mapping: Dict[str, Dict[str, List[str]]] = defaultdict(dict)

    data = load_json(normalization_path)
    if not isinstance(data, dict):
        return mapping

    for field_name, items in data.items():
        if field_name not in allowed_fields:
            continue
        if not isinstance(items, list):
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            if "original_value" not in item:
                continue

            orig_key = original_value_key(item["original_value"])
            norm_vals = normalize_list(item.get("value", []))
            mapping[field_name][orig_key] = norm_vals

    return mapping

def clean_leaf_node(
    field_path: str,
    src: Any,
    norm_map: Dict[str, Dict[str, List[str]]],
    drop_unmapped: bool = False,
) -> Optional[dict]:
    if not is_field_node(src):
        return None

    orig_value = src.get("value")
    evidence = src.get("evidence", "")
    orig_key = original_value_key(orig_value)

    if field_path in norm_map and orig_key in norm_map[field_path]:
        cleaned_values = normalize_list(norm_map[field_path][orig_key])
    else:
        if drop_unmapped:
            cleaned_values = []
        else:
            cleaned_values = normalize_list(orig_value)

    if not cleaned_values:
        return None

    return {
        "value": cleaned_values,
        "evidence": evidence,
    }


def is_empty_cleaned(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, list) and len(x) == 0:
        return True
    if isinstance(x, dict) and len(x) == 0:
        return True
    return False


def clean_by_tree(
    src: Any,
    tree: Any,
    current_path: str,
    norm_map: Dict[str, Dict[str, List[str]]],
    drop_unmapped: bool = False,
) -> Any:
    if tree == LEAF:
        # 单值字段
        if is_field_node(src):
            return clean_leaf_node(
                field_path=current_path,
                src=src,
                norm_map=norm_map,
                drop_unmapped=drop_unmapped,
            )

        # 多值字段：列表里的每个元素都是 {value, evidence}
        if isinstance(src, list):
            cleaned_list = []
            for item in src:
                cleaned_item = clean_leaf_node(
                    field_path=current_path,
                    src=item,
                    norm_map=norm_map,
                    drop_unmapped=drop_unmapped,
                )
                if not is_empty_cleaned(cleaned_item):
                    cleaned_list.append(cleaned_item)
            return cleaned_list if cleaned_list else None

        return None

    if isinstance(src, list):
        cleaned_list = []
        for item in src:
            cleaned_item = clean_by_tree(
                item, tree, current_path, norm_map, drop_unmapped
            )
            if not is_empty_cleaned(cleaned_item):
                cleaned_list.append(cleaned_item)
        return cleaned_list if cleaned_list else None

    if isinstance(src, dict):
        cleaned_dict = {}
        for key, sub_tree in tree.items():
            if key not in src:
                continue
            next_path = f"{current_path}.{key}" if current_path else key
            cleaned_val = clean_by_tree(
                src[key], sub_tree, next_path, norm_map, drop_unmapped
            )
            if not is_empty_cleaned(cleaned_val):
                cleaned_dict[key] = cleaned_val
        return cleaned_dict if cleaned_dict else None

    return None


def collect_unique_values_from_tree(
    obj: Any,
    tree: Any,
    current_path: str,
    out: Dict[str, set],
):
    if tree == LEAF:
        # 单值叶子：{"value": ..., "evidence": ...}
        if is_field_node(obj):
            vals = normalize_list(obj.get("value", []))
            for v in vals:
                out[current_path].add(v)
            return

        # 多值叶子：[{"value": ..., "evidence": ...}, ...]
        if isinstance(obj, list):
            for item in obj:
                if is_field_node(item):
                    vals = normalize_list(item.get("value", []))
                    for v in vals:
                        out[current_path].add(v)
            return

        return

    if isinstance(obj, list):
        for item in obj:
            collect_unique_values_from_tree(item, tree, current_path, out)
        return

    if isinstance(obj, dict):
        for key, sub_tree in tree.items():
            if key not in obj:
                continue
            next_path = f"{current_path}.{key}" if current_path else key
            collect_unique_values_from_tree(obj[key], sub_tree, next_path, out)
            

def clean_evidence_span(
    input_jsonl: Path,
    stats_json: Path,
    normalization_path: Path,
    output_jsonl: Path,
    output_stats_json: Path,
    drop_unmapped: bool = False,
):
    records = load_jsonl(input_jsonl)
    stats = load_json(stats_json)

    domain = stats.get("domain", "unknown")
    allowed_fields = set(stats.get("field_counts", {}).keys())
    allowed_tree = build_allowed_tree(sorted(allowed_fields))

    norm_map = load_normalization_results(
        normalization_path=normalization_path,
        allowed_fields=allowed_fields,
    )

    cleaned_records = []
    input_paragraphs = len(records)
    input_setups = 0
    kept_setups = 0

    for rec in records:
        result = rec.get("result", {})
        setups = result.get("setups", [])
        if not isinstance(setups, list):
            setups = []

        input_setups += len(setups)

        new_setups = []
        for setup in setups:
            if not isinstance(setup, dict):
                continue

            cleaned_setup_body = clean_by_tree(
                src=setup,
                tree=allowed_tree,
                current_path="",
                norm_map=norm_map,
                drop_unmapped=drop_unmapped,
            )

            if is_empty_cleaned(cleaned_setup_body):
                continue

            new_setup = {"setup_id": setup.get("setup_id", "setup_unknown")}
            new_setup.update(cleaned_setup_body)
            new_setups.append(new_setup)

        if not new_setups:
            continue

        kept_setups += len(new_setups)

        new_rec = {
            "domain": rec.get("domain", domain),
            "id": rec.get("id"),
            "result": {
                "is_valid": True,
                "invalid_reason": None,
                "setups": new_setups,
            },
        }
        cleaned_records.append(new_rec)

    write_jsonl(output_jsonl, cleaned_records)

    unique_values = defaultdict(set)
    for rec in cleaned_records:
        for setup in rec["result"]["setups"]:
            collect_unique_values_from_tree(
                obj=setup,
                tree=allowed_tree,
                current_path="",
                out=unique_values,
            )

    num_unique_values_per_field = {
        field: len(unique_values[field]) for field in sorted(allowed_fields)
    }

    summary = {
        "domain": domain,
        "input_paragraphs": input_paragraphs,
        "kept_paragraphs_with_evidence": len(cleaned_records),
        "dropped_paragraphs_after_cleaning": input_paragraphs - len(cleaned_records),
        "input_setups": input_setups,
        "kept_setups": kept_setups,
        "dropped_setups_after_cleaning": input_setups - kept_setups,
        "num_unique_values_per_field": num_unique_values_per_field,
    }

    write_json(output_stats_json, summary)


if __name__ == "__main__":
    domains = [
        "electrochemistry",
        "cell_culture",
        "immunotechnology",
        "inorganic_materials",
        "organic_synthesis",
        "polymer_materials"
    ]

    for d in domains:
        input_jsonl = f"seed_data/{d}/llm_extraction/evidence_span.jsonl"
        stats_json = f"seed_data/{d}/llm_extraction/stats.json"
        normalization_path = f"seed_data/{d}/llm_cleaning/field_values_stage1.json"
        output_stats_json = f"seed_data/{d}/llm_cleaning/stats.json"
        output_jsonl = f"seed_data/{d}/llm_cleaning/evidence_span.jsonl"
        drop_unmapped = False

        clean_evidence_span(
            input_jsonl=input_jsonl,
            stats_json=stats_json,
            normalization_path=normalization_path,
            output_jsonl=output_jsonl,
            output_stats_json=output_stats_json,
            drop_unmapped=drop_unmapped
        )


# conda activate retrieve_dataset
# python -m scripts.structured_extraction.mapping

