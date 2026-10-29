import json
import pandas as pd

domains = [
    # "cell_culture",
    "electrochemistry",
    # "immunotechnology",
    # "inorganic_materials",
    # "organic_synthesis",
    # "polymer_materials",
]

for d in domains:
    base_dir = f"data/{d}/manual_audit"
    human_path = f"{base_dir}/{d}_query_human_labels.json"
    gen_path = f"{base_dir}/generated_queries.jsonl"
    eval_path = f"{base_dir}/evaluated_queries.jsonl"


    def load_jsonl(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


    # 1) 读取人工标签
    with open(human_path, "r", encoding="utf-8") as f:
        human_labels = json.load(f)

    human_df = pd.DataFrame(
        [{"candidate_id": k, "human_label": int(v)} for k, v in human_labels.items()]
    )

    # 2) 读取生成阶段结果
    gen_rows = load_jsonl(gen_path)
    gen_df = pd.DataFrame(gen_rows)[["candidate_id", "skip"]].copy()
    gen_df["skip"] = gen_df["skip"].astype(bool)

    # 生成阶段自动决策：1=keep, 0=drop
    gen_df["pred_keep_gen"] = (~gen_df["skip"]).astype(int)

    # 3) 读取评估阶段结果
    eval_rows = load_jsonl(eval_path)
    eval_df = pd.DataFrame(eval_rows)[["candidate_id", "retrieval_usefulness"]].copy()
    eval_df["retrieval_usefulness"] = pd.to_numeric(eval_df["retrieval_usefulness"], errors="coerce")

    # 4) 合并
    df = human_df.merge(gen_df, on="candidate_id", how="inner")
    df = df.merge(eval_df, on="candidate_id", how="left")

    print(f"人工标签数: {len(human_df)}")
    print(f"与生成结果成功对齐数: {len(df)}")
    print(f"其中有评估结果的数目: {df['retrieval_usefulness'].notna().sum()}")

    # 5) 生成+评估阶段自动决策
    # skip=true => drop
    # skip=false and retrieval_usefulness=1 => keep
    # skip=false and retrieval_usefulness=0 => drop
    # 如果 skip=false 但没有 eval 结果，这里先记为缺失
    df["pred_keep_gen_eval"] = pd.NA
    df.loc[df["skip"] == True, "pred_keep_gen_eval"] = 0
    df.loc[(df["skip"] == False) & (df["retrieval_usefulness"] == 1), "pred_keep_gen_eval"] = 1
    df.loc[(df["skip"] == False) & (df["retrieval_usefulness"] == 0), "pred_keep_gen_eval"] = 0

    # 6) 统计函数
    def summarize_binary(sub_df, pred_col, stage_name):
        valid_df = sub_df[sub_df[pred_col].notna()].copy()
        valid_df[pred_col] = valid_df[pred_col].astype(int)

        n = len(valid_df)
        if n == 0:
            return {
                "stage": stage_name,
                "n_eval": 0,
                "kept": 0,
                "dropped": 0,
                "drop_ratio": 0.0,
                "accuracy": 0.0,
                "tp_keep": 0,
                "tn_drop": 0,
                "fp_keep": 0,
                "fn_drop": 0,
            }, pd.DataFrame()

        kept = int((valid_df[pred_col] == 1).sum())
        dropped = int((valid_df[pred_col] == 0).sum())
        drop_ratio = dropped / n

        acc = float((valid_df[pred_col] == valid_df["human_label"]).mean())

        tp_keep = int(((valid_df[pred_col] == 1) & (valid_df["human_label"] == 1)).sum())
        tn_drop = int(((valid_df[pred_col] == 0) & (valid_df["human_label"] == 0)).sum())
        fp_keep = int(((valid_df[pred_col] == 1) & (valid_df["human_label"] == 0)).sum())
        fn_drop = int(((valid_df[pred_col] == 0) & (valid_df["human_label"] == 1)).sum())

        conf = pd.crosstab(
            valid_df[pred_col],
            valid_df["human_label"],
            rownames=["pred_keep"],
            colnames=["human_label"],
            dropna=False
        ).reindex(index=[0, 1], columns=[0, 1], fill_value=0)

        summary = {
            "stage": stage_name,
            "n_eval": n,
            "kept": kept,
            "dropped": dropped,
            "drop_ratio": drop_ratio,
            "accuracy": acc,
            "tp_keep": tp_keep,
            "tn_drop": tn_drop,
            "fp_keep": fp_keep,
            "fn_drop": fn_drop,
        }
        return summary, conf


    gen_summary, gen_conf = summarize_binary(df, "pred_keep_gen", "generation_only")
    gen_eval_summary, gen_eval_conf = summarize_binary(df, "pred_keep_gen_eval", "generation_plus_evaluation")

    # 7) 打印总表
    summary_df = pd.DataFrame([gen_summary, gen_eval_summary])
    print("\n=== 总体统计 ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\n=== 生成阶段混淆表 ===")
    print("行: pred_keep (0=drop, 1=keep), 列: human_label (0=drop, 1=keep)")
    print(gen_conf.to_string())

    print("\n=== 生成+评估阶段混淆表 ===")
    print("行: pred_keep (0=drop, 1=keep), 列: human_label (0=drop, 1=keep)")
    print(gen_eval_conf.to_string())

    # 8) 再单独打印你最关心的几个数
    print("\n=== 关键结果 ===")
    print(f"生成阶段: 删除 {gen_summary['dropped']} / {gen_summary['n_eval']} = {gen_summary['drop_ratio']:.6f}, accuracy = {gen_summary['accuracy']:.6f}")
    print(f"生成+评估阶段: 删除 {gen_eval_summary['dropped']} / {gen_eval_summary['n_eval']} = {gen_eval_summary['drop_ratio']:.6f}, accuracy = {gen_eval_summary['accuracy']:.6f}")

    fp_gen_eval = df[(df['pred_keep_gen_eval'] == 1) & (df['human_label'] == 0)]
    print(f"\n生成+评估阶段误报 (pred_keep_gen_eval=1, human_label=0) 数量: {len(fp_gen_eval)}")
    print("candidate_id 列表:", fp_gen_eval['candidate_id'].tolist())
    print(fp_gen_eval[['candidate_id', 'skip', 'retrieval_usefulness', 'human_label']].to_string())

    fp_gen_eval = df[(df['pred_keep_gen_eval'] == 0) & (df['human_label'] == 1)]
    print(f"\n生成+评估阶段误报 (pred_keep_gen_eval=0, human_label=1) 数量: {len(fp_gen_eval)}")
    print("candidate_id 列表:", fp_gen_eval['candidate_id'].tolist())
    print(fp_gen_eval[['candidate_id', 'skip', 'retrieval_usefulness', 'human_label']].to_string())

    missing_eval = int(((df['skip'] == False) & (df['pred_keep_gen_eval'].isna())).sum())
    if missing_eval > 0:
        print(f"\n[WARN] 有 {missing_eval} 条 skip=false 但没有评估结果，因此未计入“生成+评估阶段”的 accuracy。")

# python scripts/manual_audit/eval_query.py

