import json
import re
from pathlib import Path
from collections import Counter, defaultdict
import json_repair



INVALID_VALUES = {
    "", "null", "none", "unknown",
    "not specified", "not stated", "not explicitly stated",
    "not reported", "n/a", "na",  "-", "?", "...", 
}

invalid_fields = {
    "culture_substrate_or_coating.cultured_cells.cell_type": 1,
    "culture_substrate_or_coating.culture_substrate_or_coating": 2,
    "culture_medium.culture_mode": 1,
    "culture_medium.seeding_density": 1,
    "culture_medium.confluence": 1,
    "serum.identity": 1,
    "serum.concentration": 1,
    "culture_conditions.temperature": 1,
    "culture_conditions.co2": 1,
    "culture_conditions.duration": 1,
    "culture_medium.identity": 1,
    "culture_medium.supplements": 1,
    "culture_medium.serum": 1,
    "": 13,
    "basal_medium": 16,
    "culture_conditions": 1,
    "treatments.agent": 2,
    "treatments.dose": 2,
    "cultured_cells.culture_mode": 1,
    "cultured_cells.culture_medium.basal_medium": 1,
    "concentration": 5,
    "electrolytes_additional.chemical": 1,
    "electrolytes_additional.concentration": 1,
    "counter_electrode": 6,
    "electrolytes.measurement_method": 3,
    "electrolytes.working_electrode": 1,
    "electrolytes.electrolytes.chemical": 6,
    "electrolytes.electrolytes.concentration": 4,
    "electrolytes.electrolytes.measurement_method": 1,
    "electrolytes.electrolytes.working_electrode": 1,
    "electrolytes.electrolytes.electrolytes.chemical": 2,
    "electrolytes.electrolytes.electrolytes.concentration": 2,
    "electrolytes.electrolytes.electrolytes.measurement_method": 1,
    "electrolytes.electrolytes.electrolytes.working_electrode": 1,
    "electrolytes.electrolytes.electrolytes.electrolytes.chemical": 2,
    "electrolytes.electrolytes.electrolytes.electrolytes.concentration": 2,
    "electrolytes.electrolytes.electrolytes.electrolytes.measurement_method": 1,
    "electrolytes.electrolytes.electrolytes.electrolytes.working_electrode": 1,
    "current_density": 1,
    " },\"working_electrode": 1,
    "blocking_solution": 9,
    "protein_targets.normalization_control": 1,
     "post_treatment.atmosphere": 24,
    "post_treatment.heating_rate": 2,
    "post_treatment_2.type": 1,
    "post_treatment_2.temperature": 1,
    "post_treatment_2.time": 1,
    "sintering.pre_sintering_steps.type": 3,
    "sintering.pre_sintering_steps.temperature": 3,
    "sintering.pre_sintering_steps.time": 3,
    "dopant_or_additive_2.identity": 1,
    "dopant_or_additive_2.amount": 1,
    "sintering.atmosphere": 2,
    "type": 1,
    "temperature": 1,
    "time": 2,
    "sintering.post_treatment.type": 1,
    "sintering.post_treatment.temperature": 1,
    "sintering.post_treatment.time": 1,
    "sintering.},\"post_treatment.type": 2,
    "sintering.},\"post_treatment.temperature": 2,
    "sintering.},\"post_treatment.time": 2,
    "pressing_pressure_pre": 2,
    "pressing_die": 1,
    "pressing_type": 1,
    "pressing_pressure_high": 1,
     "activation_mode.ouche": 1,
    "additives_loading": 1,
    "additives_role": 1,
    "electrochemical_conditions.potential": 1,
    "electrochemical_conditions.electrolyte": 2,
    "electrochemical_conditions.atmosphere": 1,
    "electrochemical_conditions.duration": 1,
    "electrochemical_conditions.working_electrode_substrate": 1,
    "electrolyte": 2,
    "electrolytes": 2,
    "cell_type": 3,
    "membrane": 3,
    "loading": 1,
    "products": 2,
    "yield": 2,
    "electrochemical_conditions.applied_potential": 2,
    "electrochemical_conditions.electrochemical_method": 1,
    "electrodes.working_electrode": 1,
    "electrodes.counter_electrode": 1,
    "electrodes.reference_electrode": 1,
    "instrumentation.potentiostat": 1,
    "gas_flow": 1,
    "analysis": 3,
    "notes": 3,
    "invalid_reason_note": 2,
    "protein_targets.sample_type": 15,
    "protein_targets.lysis_buffer": 2,
    "protein_targets.blocking_solution.identity": 1,
    "protein_targets.blocking_solution.concentration": 1,
    "protein_targets.protein_targets.target": 2,
    "protein_targets.protein_targets.sample_type": 1,
    "target": 4,
    "protein_targets.gel_percentage": 2,
    "protein_targets": 1,
}

def normalize_text(x):
    if x is None:
        return None
    if not isinstance(x, str):
        return x
    return re.sub(r"\s+", " ", x).strip()


def is_invalid_scalar(v):
    if v is None:
        return True
    if isinstance(v, str):
        s = normalize_text(v)
        if s is None:
            return True
        return s.lower() in INVALID_VALUES
    return False


def is_leaf_field(obj):
    return isinstance(obj, dict) and ("value" in obj) and ("evidence" in obj)


def recursive_extract_fields(
    node,
    prefix,
    record_meta,
    field_counts,
    field_values  
):
    """
    递归遍历 setup，把所有叶子字段提取出来。
    返回当前 setup 中抽到的有效字段数量。
    """
    extracted_count = 0

    if is_leaf_field(node):
        val = node.get("value")
        ev = node.get("evidence")

        if is_invalid_scalar(val):
            return 0
        if is_invalid_scalar(ev):
            return 0

        val = normalize_text(val)
        ev = normalize_text(ev)

        field_path = prefix
        if field_path in invalid_fields:
            return 0

        # 统计出现次数
        field_counts[field_path] += 1

        # 保存唯一值及其 evidence（仅首次出现）
        if str(val) not in field_values[field_path]:
            field_values[field_path][str(val)] = ev

        return 1

    if isinstance(node, dict):
        for k, v in node.items():
            if k == "setup_id":
                continue
            new_prefix = f"{prefix}.{k}" if prefix else k
            extracted_count += recursive_extract_fields(
                v,
                new_prefix,
                record_meta,
                field_counts,
                field_values
            )
        return extracted_count

    if isinstance(node, list):
        # list 本身不写字段，把其内部元素继续展开
        for idx, item in enumerate(node, start=1):
            new_meta = dict(record_meta)
            new_meta["list_field"] = prefix
            new_meta["list_index"] = idx
            extracted_count += recursive_extract_fields(
                item,
                prefix,
                new_meta,
                field_counts,
                field_values
            )
        return extracted_count

    return 0


def process_new_extraction_jsonl(
    in_path: str,
    out_valid_jsonl: str,
    out_dir: str,
    domain: str = None
):
    in_path = Path(in_path)
    out_valid_jsonl = Path(out_valid_jsonl)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    repaired = 0
    bad_raw = 0
    invalid_paragraphs = 0
    valid_paragraphs = 0
    kept_paragraphs = 0
    kept_setups_total = 0

    invalid_reason_counts = Counter()
    field_counts = Counter()
    field_values = defaultdict(dict)   # key: field_path, value: dict {value: evidence}

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_valid_jsonl, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            total += 1

            try:
                rec = json.loads(line)
            except Exception:
                bad_raw += 1
                continue
            
            # print(rec.get("id"))


            # 如果没能解析成功
            if "result" not in rec and "raw" in rec and json_repair is not None:
                try:
                    obj = json_repair.repair_json(rec["raw"], return_objects=True)
                    if isinstance(obj, dict):
                        rec["result"] = obj
                        repaired += 1
                    else:
                        bad_raw += 1
                        continue
                except Exception:
                    bad_raw += 1
                    continue

            result = rec.get("result")
            if not isinstance(result, dict):
                bad_raw += 1
                continue

            is_valid = result.get("is_valid")
            if is_valid is not True:
                invalid_paragraphs += 1
                invalid_reason = result.get("invalid_reason")
                if invalid_reason:
                    invalid_reason_counts[invalid_reason] += 1
                continue

            valid_paragraphs += 1

            setups = result.get("setups", [])
            if not isinstance(setups, list):
                continue

            kept_setups = []

            for setup in setups:
                if not isinstance(setup, dict):
                    continue

                setup_id = setup.get("setup_id", f"setup_{len(kept_setups)+1}")

                record_meta = {
                    "id": rec.get("id"),
                    "setup_id": setup_id,
                    "list_field": None,
                    "list_index": None,
                }

                extracted_num = recursive_extract_fields(
                    node=setup,
                    prefix="",
                    record_meta=record_meta,
                    field_counts=field_counts,
                    field_values=field_values
                )

                if extracted_num > 0:
                    kept_setups.append(setup)

            if kept_setups:
                kept_paragraphs += 1
                kept_setups_total += len(kept_setups)

                cleaned_rec = {
                    "domain": rec.get("domain", domain),
                    "id": rec.get("id"),
                    "result": {
                        "is_valid": True,
                        "invalid_reason": None,
                        "setups": kept_setups
                    }
                }
                fout.write(json.dumps(cleaned_rec, ensure_ascii=False) + "\n")

    # 构建 field_values.json 结构：每个字段包含唯一值列表，每个值带一个 evidence
    field_values_json = {}
    for field_path, value_evidence_dict in field_values.items():
        # 将字典转换为列表，每个元素是 {"value": v, "evidence": e}
        value_list = [{"original_value": v, "value": [v], "evidence": e} for v, e in value_evidence_dict.items()]
        # 按 value 的字母顺序排序（不区分大小写，若需区分可调整）
        value_list.sort(key=lambda x: x["value"][0].lower())
        field_values_json[field_path] = value_list

    stats = {
        "domain": domain,
        "total_paragraphs": total,
        "repaired_records": repaired,
        "bad_raw_records": bad_raw,
        "invalid_paragraphs": invalid_paragraphs,
        "valid_paragraphs_model_labeled": valid_paragraphs,
        "kept_paragraphs_with_evidence": kept_paragraphs,
        "kept_setups_total": kept_setups_total,
        "invalid_reason_counts": dict(invalid_reason_counts),
        "field_counts": dict(field_counts),
        "num_unique_values_per_field": {
            k: len(v) for k, v in field_values_json.items()
        }
    }

    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    (out_dir / "field_values.json").write_text(
        json.dumps(field_values_json, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"[{domain}] total={total}")
    print(f"[{domain}] repaired={repaired}, bad_raw={bad_raw}")
    print(f"[{domain}] invalid_paragraphs={invalid_paragraphs}")
    print(f"[{domain}] valid_paragraphs_model_labeled={valid_paragraphs}")
    print(f"[{domain}] kept_paragraphs_with_evidence={kept_paragraphs}")
    print(f"[{domain}] kept_setups_total={kept_setups_total}")
    print(f"[{domain}] out_valid_jsonl -> {out_valid_jsonl}")
    print(f"[{domain}] stats -> {out_dir / 'stats.json'}")
    print(f"[{domain}] field_values -> {out_dir / 'field_values.json'}")


if __name__ == "__main__":
    domains = [
        # "electrochemistry",
        # "cell_culture",
        "immunotechnology",
        # "inorganic_materials",
        # "organic_synthesis",
        "polymer_materials",
    ]

    for d in domains:
        process_new_extraction_jsonl(
            in_path=f"seed_data/{d}/llm_extraction/raw_evidence_span.jsonl",
            out_valid_jsonl=f"seed_data/{d}/llm_extraction/evidence_span.jsonl",
            out_dir=f"seed_data/{d}/llm_extraction",
            domain=d
        )

# python scripts/structured_extraction/filter_valid_paragraph.py
