import json
import os
from collections import defaultdict


AUDIT_SAMPLE_PATH = "data/manual_audit/audit_sample_data.jsonl"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    audit_rows = load_jsonl(AUDIT_SAMPLE_PATH)

    # domain -> set(candidate_id)
    domain_to_candidate_ids = defaultdict(set)

    # domain -> candidate_id -> ordered unique paragraph_ids
    domain_candidate_to_paragraph_ids = defaultdict(lambda: defaultdict(list))
    domain_candidate_to_seen_pids = defaultdict(lambda: defaultdict(set))

    for row in audit_rows:
        domain = row["domain"]
        candidate_id = row["candidate_id"]
        paragraph_id = str(row["paragraph_id"])

        domain_to_candidate_ids[domain].add(candidate_id)

        if paragraph_id not in domain_candidate_to_seen_pids[domain][candidate_id]:
            domain_candidate_to_seen_pids[domain][candidate_id].add(paragraph_id)
            domain_candidate_to_paragraph_ids[domain][candidate_id].append(paragraph_id)

    for domain in sorted(domain_to_candidate_ids.keys()):
        candidate_ids = domain_to_candidate_ids[domain]

        input_manifest_path = f"data/{domain}/stage1_relevance/sampled_constraints_stage1.boundary_manifest.jsonl"
        output_dir = f"data/{domain}/manual_audit"
        output_manifest_path = os.path.join(output_dir, "filtered_boundary_manifest.jsonl")
        output_candidate_paragraph_ids_path = os.path.join(output_dir, "candidate_paragraph_ids.json")

        ensure_dir(output_dir)

        # 1) 写该领域 candidate_paragraph_ids.json
        candidate_paragraph_ids = {
            cid: domain_candidate_to_paragraph_ids[domain][cid]
            for cid in sorted(domain_candidate_to_paragraph_ids[domain].keys())
        }
        with open(output_candidate_paragraph_ids_path, "w", encoding="utf-8") as f:
            json.dump(candidate_paragraph_ids, f, ensure_ascii=False, indent=2)

        print(f"[OK] 写出: {output_candidate_paragraph_ids_path}")
        print(f"     candidate 数: {len(candidate_paragraph_ids)}")

        # 2) 过滤 boundary manifest
        if not os.path.exists(input_manifest_path):
            print(f"[WARN] 不存在，跳过 manifest 过滤: {input_manifest_path}")
            continue

        total = 0
        kept = 0
        found_candidate_ids = set()
        
        with open(input_manifest_path, "r", encoding="utf-8") as fin, \
             open(output_manifest_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                total += 1
                obj = json.loads(line)
                cid = obj["candidate_id"]
                if cid in candidate_ids:
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1
                    found_candidate_ids.add(cid)

        missing = sorted(candidate_ids - found_candidate_ids)

        print(f"[OK] {domain}")
        print(f"     输入: {input_manifest_path}")
        print(f"     输出: {output_manifest_path}")
        print(f"     保留 {kept}/{total} 条 manifest")
        if missing:
            print(f"     [WARN] 有 {len(missing)} 个 candidate_id 未在 manifest 中找到")
            print(f"     示例: {missing[:20]}")

    print("\n全部完成。")


if __name__ == "__main__":
    main()

# python scripts/manual_audit/test.py