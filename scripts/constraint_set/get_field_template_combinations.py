import json
from itertools import combinations, product
from collections import Counter
from pathlib import Path


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_bucket_fields(field_meta: dict, bucket: str, mode: str):
    assert mode in {"constraint", "explore"}
    flag = "constraint_enabled" if mode == "constraint" else "explore_enabled"

    results = []
    for field_name, meta in field_meta.items():
        if meta["bucket"] == bucket and meta.get(flag, False):
            results.append(field_name)
    return sorted(results)


def is_valid_under_dependency_rules(constraint_fields, explore_field, dependent_field_rules):
    """
    统一按 required_known_fields 校验：
    - 某个字段只要出现（作为约束字段或探索字段），
      它要求的 required_known_fields 必须都已经在 constraint_fields 中。
    """
    constraint_set = set(constraint_fields)
    all_fields = list(constraint_fields)
    if explore_field is not None:
        all_fields.append(explore_field)

    for field in all_fields:
        rule = dependent_field_rules.get(field, {})
        required_known_fields = rule.get("required_known_fields", [])
        if not set(required_known_fields).issubset(constraint_set):
            return False
    return True


def build_instance(
    domain: str,
    type_id: str,
    task_type: str,
    explore_bucket,
    skeleton_total: int,
    role_to_fields: dict,
    constraint_fields: list,
    explore_field,
    field_meta: dict,
):
    all_fields = list(constraint_fields)
    if explore_field is not None:
        all_fields.append(explore_field)

    numeric_fields_in_combo = [
        f for f in all_fields
        if field_meta[f]["value_type"] == "numeric"
    ]
    range_numeric_fields_in_combo = [
        f for f in all_fields
        if field_meta[f]["value_type"] == "numeric" and field_meta[f].get("range_enabled", False)
    ]
    negatable_fields_in_combo = [
        f for f in constraint_fields
        if field_meta[f].get("negation_enabled", False)
    ]

    return {
        "domain": domain,
        "type_id": type_id,
        "type": task_type,
        "explore_bucket": explore_bucket,
        "total": skeleton_total,
        "field_assignment": {
            "Goal": role_to_fields["Goal"],
            "Input": role_to_fields["Input"],
            "Condition": role_to_fields["Condition"],
            "Method": role_to_fields["Method"],
            "Explore": explore_field,
        },
        "constraint_fields": constraint_fields,
        "all_fields": all_fields,
        "constraint_count": skeleton_total,

        "numeric_field_count": len(numeric_fields_in_combo),
        "numeric_fields_in_combo": numeric_fields_in_combo,

        "has_range_numeric": len(range_numeric_fields_in_combo) > 0,
        "range_numeric_fields_in_combo": range_numeric_fields_in_combo,


        "negatable_field_count": len(negatable_fields_in_combo),
        "negatable_fields_in_combo": negatable_fields_in_combo,

        "has_numeric_field": len(numeric_fields_in_combo) > 0,
        "has_negation_field": len(negatable_fields_in_combo) > 0,
    }


def instantiate_templates(domain: str, field_meta: dict, template_registry: list, dependent_field_rules: dict):
    goal_fields = get_bucket_fields(field_meta, "goal", mode="constraint")
    input_fields = get_bucket_fields(field_meta, "input", mode="constraint")
    condition_fields = get_bucket_fields(field_meta, "condition", mode="constraint")
    method_fields = get_bucket_fields(field_meta, "method", mode="constraint")

    explore_bucket_to_fields = {
        "goal": get_bucket_fields(field_meta, "goal", mode="explore"),
        "input": get_bucket_fields(field_meta, "input", mode="explore"),
        "condition": get_bucket_fields(field_meta, "condition", mode="explore"),
        "method": get_bucket_fields(field_meta, "method", mode="explore"),
    }

    all_instances = []

    for skeleton in template_registry:
        type_id = skeleton["type_id"]
        task_type = skeleton["type"]
        explore_bucket = skeleton.get("explore", None)

        n_goal = skeleton.get("goal", 0)
        n_input = skeleton.get("input", 0)
        n_condition = skeleton.get("condition", 0)
        n_method = skeleton.get("method", 0)

        if (
            len(goal_fields) < n_goal or
            len(input_fields) < n_input or
            len(condition_fields) < n_condition or
            len(method_fields) < n_method
        ):
            continue

        goal_combos = list(combinations(goal_fields, n_goal)) if n_goal > 0 else [()]
        input_combos = list(combinations(input_fields, n_input)) if n_input > 0 else [()]
        condition_combos = list(combinations(condition_fields, n_condition)) if n_condition > 0 else [()]
        method_combos = list(combinations(method_fields, n_method)) if n_method > 0 else [()]

        for g_combo, i_combo, c_combo, m_combo in product(
            goal_combos, input_combos, condition_combos, method_combos
        ):
            role_to_fields = {
                "Goal": list(g_combo),
                "Input": list(i_combo),
                "Condition": list(c_combo),
                "Method": list(m_combo),
            }

            constraint_fields = (
                role_to_fields["Goal"] +
                role_to_fields["Input"] +
                role_to_fields["Condition"] +
                role_to_fields["Method"]
            )

            # reproduction
            if explore_bucket is None:
                if not is_valid_under_dependency_rules(
                    constraint_fields=constraint_fields,
                    explore_field=None,
                    dependent_field_rules=dependent_field_rules,
                ):
                    continue

                instance = build_instance(
                    domain=domain,
                    type_id=type_id,
                    task_type=task_type,
                    explore_bucket=None,
                    skeleton_total=skeleton["total"],
                    role_to_fields=role_to_fields,
                    constraint_fields=constraint_fields,
                    explore_field=None,
                    field_meta=field_meta,
                )
                all_instances.append(instance)
                continue

            # exploration
            explore_candidates = list(explore_bucket_to_fields.get(explore_bucket, []))
            explore_candidates = [f for f in explore_candidates if f not in constraint_fields]
            if not explore_candidates:
                continue

            for explore_field in explore_candidates:
                if not is_valid_under_dependency_rules(
                    constraint_fields=constraint_fields,
                    explore_field=explore_field,
                    dependent_field_rules=dependent_field_rules,
                ):
                    continue

                instance = build_instance(
                    domain=domain,
                    type_id=type_id,
                    task_type=task_type,
                    explore_bucket=explore_bucket,
                    skeleton_total=skeleton["total"],
                    role_to_fields=role_to_fields,
                    constraint_fields=constraint_fields,
                    explore_field=explore_field,
                    field_meta=field_meta,
                )
                all_instances.append(instance)

    return all_instances


def build_summary(domain: str, instances: list):
    by_type_id = Counter(x["type_id"] for x in instances)
    by_type = Counter(x["type"] for x in instances)
    by_constraint_count = Counter(x["constraint_count"] for x in instances)

    templates_with_numeric = sum(1 for x in instances if x["numeric_field_count"] > 0)
    templates_with_negation = sum(1 for x in instances if x["negatable_field_count"] > 0)
    templates_without_numeric_and_without_negation = sum(
        1 for x in instances
        if x["numeric_field_count"] == 0 and x["negatable_field_count"] == 0
    )

    total_numeric_field_slots = sum(x["numeric_field_count"] for x in instances)
    total_negatable_field_slots = sum(x["negatable_field_count"] for x in instances)

    return {
        "domain": domain,
        "total_instances": len(instances),
        "by_type_id": dict(sorted(by_type_id.items())),
        "by_type": dict(sorted(by_type.items())),
        "by_constraint_count": {
            str(k): v for k, v in sorted(by_constraint_count.items())
        },
        "templates_with_numeric": templates_with_numeric,
        "templates_with_negation": templates_with_negation,
        "templates_without_numeric_and_without_negation": templates_without_numeric_and_without_negation,
        "total_numeric_field_slots": total_numeric_field_slots,
        "total_negatable_field_slots": total_negatable_field_slots,
    }


def print_summary(summary: dict):
    domain = summary["domain"]
    print(f"\n[{domain}] summary")
    print(f"  total_instances: {summary['total_instances']}")

    print("  by_type_id:")
    for k, v in summary["by_type_id"].items():
        print(f"    {k}: {v}")

    print("  by_type:")
    for k, v in summary["by_type"].items():
        print(f"    {k}: {v}")

    print("  by_constraint_count:")
    for k, v in summary["by_constraint_count"].items():
        print(f"    {k}: {v}")

    print(f"  templates_with_numeric: {summary['templates_with_numeric']}")
    print(f"  templates_with_negation: {summary['templates_with_negation']}")
    print(
        "  templates_without_numeric_and_without_negation: "
        f"{summary['templates_without_numeric_and_without_negation']}"
    )
    print(f"  total_numeric_field_slots: {summary['total_numeric_field_slots']}")
    print(f"  total_negatable_field_slots: {summary['total_negatable_field_slots']}")


if __name__ == "__main__":
    domains = [
        "electrochemistry",
        "cell_culture",
        "immunotechnology",
        "inorganic_materials",
        "organic_synthesis",
        "polymer_materials",
    ]

    template_path = "seed_data/template_registry.json"
    template_registry = load_json(template_path)

    for domain in domains:
        registry_path = f"seed_data/{domain}/field_registry.json"
        dependency_path = f"seed_data/{domain}/dependent_field_rules.json"
        output_path = f"seed_data/{domain}/field_template_combinations.json"
        summary_output_path = f"seed_data/{domain}/field_template_stas.json"

        field_registry = load_json(registry_path)
        dependent_field_rules = load_json(dependency_path)

        instances = instantiate_templates(
            domain=domain,
            field_meta=field_registry,
            template_registry=template_registry,
            dependent_field_rules=dependent_field_rules,
        )
        save_json(instances, output_path)

        summary = build_summary(domain, instances)
        save_json(summary, summary_output_path)

        print(f"Saved {len(instances)} field template combinations to: {output_path}")
        print(f"Saved summary to: {summary_output_path}")
        print_summary(summary)

# python scripts/constraint_set/get_field_template_combinations.py
        
        