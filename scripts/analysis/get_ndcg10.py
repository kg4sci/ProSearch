import pandas as pd

# 1. 加载数据
df = pd.read_csv("result/tables/all_metrics_raw.csv")

# 2. 筛选出 global 指标
# 3. 进行透视表操作
# index 为行（模型），columns 为列（领域），values 为我们需要查看的指标
global_df = df[df['group_field'] == '__global__']
pivot_table = global_df.pivot(index='model', columns='domain', values='ndcg_cut_10')
pivot_table.to_csv("result/tables/global_ndcg_report.csv")



# filtered_df = df[
#     (df['domain'] == 'cell_culture') & 
#     (df['group_field'] == '__global__')
# ]
# metrics = ['model', 'ndcg_cut_10', 'P_10', 'recall_10', 'recip_rank']
# result_table = filtered_df[metrics]
# result_table.to_csv("result/tables/domain_metric_report.csv")

# python scripts/analysis/get_ndcg10.py

