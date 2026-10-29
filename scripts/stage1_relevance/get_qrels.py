import os
import json
import argparse
from collections import defaultdict


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error in {path} at line {line_no}: {e}")


def safe_get_paragraph_verdict(record):
    """
    只在 ok=True 且 parsed_output 可解析时返回 paragraph_verdict，否则返回 None
    """
    if not record.get("ok", False):
        return None

    parsed_output = record.get("parsed_output")
    if parsed_output is None:
        return None

    if isinstance(parsed_output, str):
        try:
            parsed_output = json.loads(parsed_output)
        except Exception:
            return None

    if not isinstance(parsed_output, dict):
        return None

    return parsed_output.get("paragraph_verdict")


def build_manifest_index(boundary_manifest_path):
    """
    旧 manifest 只用于补充统计标签，不参与正样本判定。
    """
    manifest_index = {}

    for rec in load_jsonl(boundary_manifest_path):
        candidate_id = rec.get("candidate_id")
        if not candidate_id:
            continue

        manifest_index[candidate_id] = {
            "domain": rec.get("domain"),
            "type": rec.get("type"),
            "type_id": rec.get("type_id"),
            "candidate_source": rec.get("candidate_source"),
            "has_negation": rec.get("has_negation"),
            "has_numeric_constraint": rec.get("has_numeric_constraint"),
            "has_range_constraint": rec.get("has_range_constraint"),
            "constraint_count": len(rec.get("constraints", []) or []),
            "explore_field": rec.get("explore_field"),
        }

    return manifest_index


def sort_str_ids(ids):
    """
    如果是纯数字字符串，按数字排序；否则按字符串排序。
    """
    def key_fn(x):
        s = str(x)
        return (0, int(s)) if s.isdigit() else (1, s)

    return sorted(ids, key=key_fn)


def init_bucket():
    return {
        "query_count": 0,
        "query_count_with_positive": 0,
        "query_count_without_positive": 0,
        "total_positive_support": 0,
        "avg_positive_support_per_query": 0.0,
        "avg_positive_support_per_nonzero_query": 0.0,
    }


def finalize_bucket(bucket):
    q = bucket["query_count"]
    q_pos = bucket["query_count_with_positive"]
    total = bucket["total_positive_support"]

    bucket["avg_positive_support_per_query"] = round(total / q, 4) if q > 0 else 0.0
    bucket["avg_positive_support_per_nonzero_query"] = round(total / q_pos, 4) if q_pos > 0 else 0.0
    return bucket


def main(judge_results_path, boundary_manifest_path, output_qres_path, output_stats_path):
    # 1) 读取旧 manifest，只做标签补充
    manifest_index = build_manifest_index(boundary_manifest_path)

    # 2) 从 judge_results 构建 qres
    # qres[candidate_id][paragraph_id] = 1
    qres = defaultdict(dict)

    # 保存 judge_results 中实际出现过的 query 元信息
    judge_query_meta = {}

    total_records = 0
    satisfy_records = 0
    skipped_not_ok_or_unparsed = 0

    for rec in load_jsonl(judge_results_path):
        total_records += 1

        candidate_id = rec.get("candidate_id")
        paragraph_id = rec.get("paragraph_id")

        if candidate_id is None or paragraph_id is None:
            continue

        paragraph_id = str(paragraph_id)

        # 记录 judge_results 自身携带的 query 基本信息
        if candidate_id not in judge_query_meta:
            judge_query_meta[candidate_id] = {
                "domain": rec.get("domain"),
                "type": rec.get("type"),
                "type_id": rec.get("type_id"),
                "task_family": rec.get("task_family"),
                "candidate_source": rec.get("candidate_source"),
            }

        verdict = safe_get_paragraph_verdict(rec)

        if verdict is None:
            skipped_not_ok_or_unparsed += 1
            continue

        if verdict == "satisfy":
            qres[candidate_id][paragraph_id] = 1
            satisfy_records += 1

    # 3) 输出 qres_stage1.json
    # 只保留至少有一个正样本的 query
    qres_out = {}
    for candidate_id in sort_str_ids(qres.keys()):
        paragraph_map = qres[candidate_id]
        qres_out[candidate_id] = {
            pid: 1 for pid in sort_str_ids(paragraph_map.keys())
        }

    os.makedirs(os.path.dirname(output_qres_path), exist_ok=True)
    with open(output_qres_path, "w", encoding="utf-8") as f:
        json.dump(qres_out, f, ensure_ascii=False, indent=2)

    # 4) 统计
    # 注意：统计的 query 集合以 judge_results 中实际出现过的 candidate_id 为准
    all_query_ids = sort_str_ids(judge_query_meta.keys())

    overall = init_bucket()
    by_type = defaultdict(init_bucket)
    by_type_has_negation = defaultdict(init_bucket)
    by_type_has_numeric_constraint = defaultdict(init_bucket)
    by_type_has_range_constraint = defaultdict(init_bucket)
    by_type_flag_combo = defaultdict(init_bucket)

    missing_in_manifest = []

    for candidate_id in all_query_ids:
        meta_from_judge = judge_query_meta.get(candidate_id, {})
        meta_from_manifest = manifest_index.get(candidate_id, {})

        if candidate_id not in manifest_index:
            missing_in_manifest.append(candidate_id)

        query_type = meta_from_manifest.get("type", meta_from_judge.get("type", "UNKNOWN"))
        pos_support = len(qres.get(candidate_id, {}))

        has_negation = meta_from_manifest.get("has_negation", None)
        has_numeric = meta_from_manifest.get("has_numeric_constraint", None)
        has_range = meta_from_manifest.get("has_range_constraint", None)

        # overall
        overall["query_count"] += 1
        overall["total_positive_support"] += pos_support
        if pos_support > 0:
            overall["query_count_with_positive"] += 1
        else:
            overall["query_count_without_positive"] += 1

        # by type
        by_type[query_type]["query_count"] += 1
        by_type[query_type]["total_positive_support"] += pos_support
        if pos_support > 0:
            by_type[query_type]["query_count_with_positive"] += 1
        else:
            by_type[query_type]["query_count_without_positive"] += 1

        # by type + has_negation
        key_neg = f"{query_type}__has_negation={has_negation}"
        by_type_has_negation[key_neg]["query_count"] += 1
        by_type_has_negation[key_neg]["total_positive_support"] += pos_support
        if pos_support > 0:
            by_type_has_negation[key_neg]["query_count_with_positive"] += 1
        else:
            by_type_has_negation[key_neg]["query_count_without_positive"] += 1

        # by type + has_numeric_constraint
        key_num = f"{query_type}__has_numeric_constraint={has_numeric}"
        by_type_has_numeric_constraint[key_num]["query_count"] += 1
        by_type_has_numeric_constraint[key_num]["total_positive_support"] += pos_support
        if pos_support > 0:
            by_type_has_numeric_constraint[key_num]["query_count_with_positive"] += 1
        else:
            by_type_has_numeric_constraint[key_num]["query_count_without_positive"] += 1

        # by type + has_range_constraint
        key_rng = f"{query_type}__has_range_constraint={has_range}"
        by_type_has_range_constraint[key_rng]["query_count"] += 1
        by_type_has_range_constraint[key_rng]["total_positive_support"] += pos_support
        if pos_support > 0:
            by_type_has_range_constraint[key_rng]["query_count_with_positive"] += 1
        else:
            by_type_has_range_constraint[key_rng]["query_count_without_positive"] += 1

        # by type + 三个标志组合
        combo_key = (
            f"{query_type}"
            f"__has_negation={has_negation}"
            f"__has_numeric_constraint={has_numeric}"
            f"__has_range_constraint={has_range}"
        )
        by_type_flag_combo[combo_key]["query_count"] += 1
        by_type_flag_combo[combo_key]["total_positive_support"] += pos_support
        if pos_support > 0:
            by_type_flag_combo[combo_key]["query_count_with_positive"] += 1
        else:
            by_type_flag_combo[combo_key]["query_count_without_positive"] += 1

    overall = finalize_bucket(overall)

    by_type = {
        k: finalize_bucket(v)
        for k, v in sorted(by_type.items(), key=lambda x: x[0])
    }
    by_type_has_negation = {
        k: finalize_bucket(v)
        for k, v in sorted(by_type_has_negation.items(), key=lambda x: x[0])
    }
    by_type_has_numeric_constraint = {
        k: finalize_bucket(v)
        for k, v in sorted(by_type_has_numeric_constraint.items(), key=lambda x: x[0])
    }
    by_type_has_range_constraint = {
        k: finalize_bucket(v)
        for k, v in sorted(by_type_has_range_constraint.items(), key=lambda x: x[0])
    }
    by_type_flag_combo = {
        k: finalize_bucket(v)
        for k, v in sorted(by_type_flag_combo.items(), key=lambda x: x[0])
    }

    stats = {
        "input_files": {
            "judge_results_path": judge_results_path,
            "boundary_manifest_path": boundary_manifest_path,
        },
        "summary": {
            "total_judge_records": total_records,
            "satisfy_records": satisfy_records,
            "skipped_not_ok_or_unparsed": skipped_not_ok_or_unparsed,
            "total_unique_queries_in_judge_results": len(all_query_ids),
            "queries_with_positive": sum(1 for qid in all_query_ids if len(qres.get(qid, {})) > 0),
            "queries_without_positive": sum(1 for qid in all_query_ids if len(qres.get(qid, {})) == 0),
            "queries_missing_in_manifest": len(missing_in_manifest),
        },
        "overall": overall,
        "by_type": by_type,
        "by_type_has_negation": by_type_has_negation,
        "by_type_has_numeric_constraint": by_type_has_numeric_constraint,
        "by_type_has_range_constraint": by_type_has_range_constraint,
        "by_type_flag_combo": by_type_flag_combo,
        "queries_missing_in_manifest": missing_in_manifest,
    }

    os.makedirs(os.path.dirname(output_stats_path), exist_ok=True)
    with open(output_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"[Done] qres saved to: {output_qres_path}")
    print(f"[Done] stats saved to: {output_stats_path}")


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
        judge_results_path = f'data/{d}/stage1_relevance/judge_results_stage1_gpt-5.4.jsonl'
        boundary_manifest_path = f'data/{d}/stage1_relevance/sampled_constraints_stage1.boundary_manifest.jsonl'
        output_qres_path = f'data/{d}/stage1_relevance/qrels_stage1.json'
        output_stats_path =f'data/{d}/stage1_relevance/qrels_stage1.stats.json'
        main(judge_results_path, boundary_manifest_path, output_qres_path, output_stats_path)

# python scripts/stage1_relevance/get_qrels.py