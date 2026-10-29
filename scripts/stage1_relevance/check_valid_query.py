import json
from pathlib import Path


def main(input_path):
    total_candidates = 0
    bad_candidates = 0
    total_overlap_pairs = 0

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            total_candidates += 1

            candidate_id = obj.get("candidate_id", f"<line_{line_no}>")
            support_ids = set(map(str, obj.get("support_paragraph_ids", [])))
            boundary_dict = obj.get("sampled_boundary_paragraph_ids", {})

            candidate_has_overlap = False

            for boundary_type, boundary_ids in boundary_dict.items():
                boundary_ids = set(map(str, boundary_ids))
                overlap = sorted(support_ids & boundary_ids, key=lambda x: (len(x), x))

                if overlap:
                    if not candidate_has_overlap:
                        bad_candidates += 1
                        candidate_has_overlap = True

                    total_overlap_pairs += len(overlap)

                    print(f"[OVERLAP] candidate_id={candidate_id}")
                    print(f"  boundary_type: {boundary_type}")
                    print(f"  overlap_count: {len(overlap)}")
                    print(f"  overlap_ids: {overlap}")
                    print()

    print("==== Summary ====")
    print(f"total_candidates: {total_candidates}")
    print(f"bad_candidates: {bad_candidates}")
    print(f"total_overlap_pairs: {total_overlap_pairs}")

if __name__ == "__main__":
    domains = [
        # "cell_culture",
        # "electrochemistry",
        # "immunotechnology",
        # "inorganic_materials",
        # "organic_synthesis",
        "polymer_materials",
    ]

    for domain in domains:
        input_path = Path(f"data/{domain}/stage1_relevance/sampled_constraints_stage1.boundary_manifest.jsonl")

        main(input_path)

    # python scripts/stage1_relevance/check_valid_query.py