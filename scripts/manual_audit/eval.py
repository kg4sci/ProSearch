import json
import pandas as pd

domains = [
        # "cell_culture",
        # "electrochemistry",
        # "immunotechnology",
        "inorganic_materials",
        # "organic_synthesis",
        # "polymer_materials",
    ]

for d in domains:
    human_path = f"data/{d}/manual_audit/{d}_paragraph_human_labels.json"
    judge_path = f"data/{d}/manual_audit/judge_results_stage1_gpt-5.4.jsonl"
    output_json_path = f"data/{d}/manual_audit/judge_eval_summary.json"

    LABELS = ["satisfy", "uncertain", "violate"]

    # 1) 读取人工标注
    with open(human_path, "r", encoding="utf-8") as f:
        human_data = json.load(f)

    human_df = pd.DataFrame(human_data)[["candidate_id", "paragraph_id", "human_label"]].copy()
    human_df["paragraph_id"] = human_df["paragraph_id"].astype(str)

    # 2) 读取模型结果
    judge_rows = []
    with open(judge_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            judge_rows.append({
                "candidate_id": obj["candidate_id"],
                "paragraph_id": str(obj["paragraph_id"]),
                "pred_label": obj.get("parsed_output", {}).get("paragraph_verdict")
            })

    judge_df = pd.DataFrame(judge_rows)

    # 3) 对齐
    df = human_df.merge(
        judge_df,
        on=["candidate_id", "paragraph_id"],
        how="inner"
    )

    print(f"对齐后的样本数: {len(df)}")

    # 4) 混淆表：行=模型预测，列=人工标签
    conf_mat = pd.crosstab(
        df["pred_label"],
        df["human_label"],
        rownames=["pred_label"],
        colnames=["human_label"],
        dropna=False
    ).reindex(index=LABELS, columns=LABELS, fill_value=0)

    print("\n=== 混淆表（行=模型预测，列=人工标签） ===")
    print(conf_mat.to_string())

    # 5) 构造你要的 JSON
    result = {
        "by_label": {},
        "total": {},
        "overall_accuracy": float((df["pred_label"] == df["human_label"]).mean()) if len(df) > 0 else 0.0,
        "n_samples": int(len(df)),
    }

    # by_label: 按模型预测标签分组，统计人工标签分布
    for pred_label in LABELS:
        row_counts = {label: int(conf_mat.loc[pred_label, label]) for label in LABELS}
        row_total = sum(row_counts.values())
        row_ratio = {
            label: (row_counts[label] / row_total if row_total > 0 else 0.0)
            for label in LABELS
        }

        result["by_label"][pred_label] = {
            "counts": row_counts,
            "ratio": row_ratio,
            "total": row_total,
            "accuracy": row_ratio[pred_label]  # 该预测标签下的准确率
        }

    # total: 全部样本的人工标签分布
    total_counts = {label: int((df["human_label"] == label).sum()) for label in LABELS}
    total_n = sum(total_counts.values())
    total_ratio = {
        label: (total_counts[label] / total_n if total_n > 0 else 0.0)
        for label in LABELS
    }
    result["total"] = {
        "counts": total_counts,
        "ratio": total_ratio,
        "total": total_n
    }

    # 6) 打印表格
    print("\n=== 按模型预测标签分组的人工标签分布 ===")
    rows = []
    for pred_label in LABELS:
        item = result["by_label"][pred_label]
        rows.append({
            "pred_label": pred_label,
            "human_satisfy": item["counts"]["satisfy"],
            "human_uncertain": item["counts"]["uncertain"],
            "human_violate": item["counts"]["violate"],
            "group_total": item["total"],
            "group_accuracy": item["accuracy"],
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n=== 全部样本的人工标签分布 ===")
    total_df = pd.DataFrame([{
        "human_satisfy": result["total"]["counts"]["satisfy"],
        "human_uncertain": result["total"]["counts"]["uncertain"],
        "human_violate": result["total"]["counts"]["violate"],
        "total": result["total"]["total"],
        "overall_accuracy": result["overall_accuracy"],
    }])
    print(total_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n=== JSON 结果 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))


    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 统计
    uncertain_but_satisfy = df[(df["pred_label"] == "violate") & (df["human_label"] == "satisfy")]
    if len(uncertain_but_satisfy) > 0:
        print("\n=== 模型预测  但人工标签  的样本 ===")
        # 打印 candidate_id 和 paragraph_id
        for _, row in uncertain_but_satisfy.iterrows():
            print(f"candidate_id: {row['candidate_id']}, paragraph_id: {row['paragraph_id']}")
        # 也可以单独提取列表
        uncertain_satisfy_ids = uncertain_but_satisfy["candidate_id"].tolist()
        print(f"\n共 {len(uncertain_satisfy_ids)} 条")
        print("candidate_id 列表:", uncertain_satisfy_ids)

# python scripts/manual_audit/eval.py


