import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean


TYPE_ORDER = [
    "input_exploration",
    "reproduction",
    "condition_exploration",
    "method_exploration",
]

TYPE_PREFIX_ORDER = {
    "R": 0,
    "EI": 1,
    "EC": 2,
    "EM": 3,
}


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error in {path} at line {line_no}: {e}") from e
    return rows


def bool_to_str(x):
    return "True" if bool(x) else "False"


def sort_type_counts(counter: Counter):
    ordered = {}
    for t in TYPE_ORDER:
        if t in counter:
            ordered[t] = counter[t]
    for t in sorted(counter.keys()):
        if t not in ordered:
            ordered[t] = counter[t]
    return ordered


def parse_type_id_key(type_id: str):
    if not isinstance(type_id, str):
        return (999, str(type_id), 999999)

    for prefix in ["R", "EI", "EC", "EM"]:
        if type_id.startswith(prefix):
            suffix = type_id[len(prefix):]
            try:
                num = int(suffix)
            except ValueError:
                num = 999999
            return (TYPE_PREFIX_ORDER[prefix], prefix, num)

    return (999, type_id, 999999)


def sort_type_id_counts(counter: Counter):
    return {k: counter[k] for k in sorted(counter.keys(), key=parse_type_id_key)}


def sort_str_bool_counts(counter: Counter):
    out = {}
    for k in ["False", "True"]:
        if k in counter:
            out[k] = counter[k]
    for k in sorted(counter.keys()):
        if k not in out:
            out[k] = counter[k]
    return out


def sort_numeric_string_key_counts(counter: Counter):
    def key_fn(x):
        try:
            return int(x)
        except Exception:
            return float("inf")
    return {k: counter[k] for k in sorted(counter.keys(), key=key_fn)}


def nested_counter_to_dict(nested_counter, row_order=None, col_order=None, sort_row_fallback=True):
    result = {}

    row_keys = list(nested_counter.keys())
    if row_order:
        ordered_row_keys = [k for k in row_order if k in nested_counter]
        if sort_row_fallback:
            ordered_row_keys += sorted([k for k in row_keys if k not in ordered_row_keys])
        row_keys = ordered_row_keys
    elif sort_row_fallback:
        row_keys = sorted(row_keys)

    for row_key in row_keys:
        inner = nested_counter[row_key]
        inner_dict = {}

        if col_order:
            for col_key in col_order:
                if col_key in inner:
                    inner_dict[col_key] = inner[col_key]
            for col_key in sorted(inner.keys()):
                if col_key not in inner_dict:
                    inner_dict[col_key] = inner[col_key]
        else:
            for col_key in sorted(inner.keys()):
                inner_dict[col_key] = inner[col_key]

        result[row_key] = inner_dict

    return result


def build_stats(rows):
    type_counter = Counter()
    type_id_counter = Counter()
    negation_counter = Counter()
    numeric_counter = Counter()
    range_counter = Counter()
    source_counter = Counter()
    support_counter = Counter()
    constraint_count_counter = Counter()

    type_by_negation = defaultdict(Counter)
    type_by_numeric = defaultdict(Counter)
    type_by_range = defaultdict(Counter)
    type_by_constraint_count = defaultdict(Counter)

    support_values = []
    exact_support_values = []
    constraint_count_values = []

    for row in rows:
        type_name = row.get("type")
        type_id = row.get("type_id")
        has_negation = bool_to_str(row.get("has_negation", False))
        has_numeric = bool_to_str(row.get("has_numeric_constraint", False))
        has_range = bool_to_str(row.get("has_range_constraint", False))
        source = row.get("candidate_source")
        support = row.get("support")
        exact_support = row.get("exact_support_count")
        constraint_count = row.get("constraint_count")

        if type_name is not None:
            type_counter[type_name] += 1
        if type_id is not None:
            type_id_counter[type_id] += 1
        if source is not None:
            source_counter[source] += 1

        negation_counter[has_negation] += 1
        numeric_counter[has_numeric] += 1
        range_counter[has_range] += 1

        if type_name is not None:
            type_by_negation[type_name][has_negation] += 1
            type_by_numeric[type_name][has_numeric] += 1
            type_by_range[type_name][has_range] += 1

        if support is not None:
            support_counter[str(support)] += 1
            support_values.append(support)

        if exact_support is not None:
            exact_support_values.append(exact_support)

        if constraint_count is not None:
            constraint_count_counter[str(constraint_count)] += 1
            constraint_count_values.append(constraint_count)
            if type_name is not None:
                type_by_constraint_count[type_name][str(constraint_count)] += 1

    stats = {
        "total_count": len(rows),

        "type_counts": sort_type_counts(type_counter),
        "type_id_counts": sort_type_id_counts(type_id_counter),

        "negation_counts": sort_str_bool_counts(negation_counter),
        "numeric_counts": sort_str_bool_counts(numeric_counter),
        "range_counts": sort_str_bool_counts(range_counter),

        "source_counts": {k: source_counter[k] for k in sorted(source_counter.keys())},
        "support_counts": sort_numeric_string_key_counts(support_counter),
        "constraint_count_counts": sort_numeric_string_key_counts(constraint_count_counter),

        "type_by_negation_counts": nested_counter_to_dict(
            type_by_negation,
            row_order=TYPE_ORDER,
            col_order=["False", "True"],
        ),
        "type_by_numeric_counts": nested_counter_to_dict(
            type_by_numeric,
            row_order=TYPE_ORDER,
            col_order=["False", "True"],
        ),
        "type_by_range_counts": nested_counter_to_dict(
            type_by_range,
            row_order=TYPE_ORDER,
            col_order=["False", "True"],
        ),
        "type_by_constraint_count_counts": nested_counter_to_dict(
            type_by_constraint_count,
            row_order=TYPE_ORDER,
            col_order=sorted(
                {str(v) for v in constraint_count_values},
                key=lambda x: int(x)
            ) if constraint_count_values else None,
        ),
    }

    if support_values:
        stats["support_min"] = min(support_values)
        stats["support_max"] = max(support_values)
        stats["support_mean"] = round(mean(support_values), 4)
    else:
        stats["support_min"] = None
        stats["support_max"] = None
        stats["support_mean"] = None

    if exact_support_values:
        stats["exact_support_count_mean"] = round(mean(exact_support_values), 4)
        stats["exact_support_count_min"] = min(exact_support_values)
        stats["exact_support_count_max"] = max(exact_support_values)
    else:
        stats["exact_support_count_mean"] = None
        stats["exact_support_count_min"] = None
        stats["exact_support_count_max"] = None

    if constraint_count_values:
        stats["constraint_count_min"] = min(constraint_count_values)
        stats["constraint_count_max"] = max(constraint_count_values)
        stats["constraint_count_mean"] = round(mean(constraint_count_values), 4)
    else:
        stats["constraint_count_min"] = None
        stats["constraint_count_max"] = None
        stats["constraint_count_mean"] = None

    return stats


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
        input_path = Path(f'seed_data/{d}/constraints/sampled_constraints_stage1.jsonl')
        output_path = Path(f'seed_data/{d}/constraints/sampled_constraints_stage1.stats.json')

        rows = read_jsonl(input_path)
        stats = build_stats(rows)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"Loaded {len(rows)} rows from: {input_path}")
        print(f"Saved stats to: {output_path}")

# python scripts/constraint_set/get_sample_analysis.py
