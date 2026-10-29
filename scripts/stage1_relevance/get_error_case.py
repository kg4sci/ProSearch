from scripts.relevance.prompt import *
import json
from typing import List, Dict, Any, Optional
from collections import defaultdict

domains = [
        # "electrochemistry",
        "cell_culture",
        # "immunotechnology",
        # "inorganic_materials",
        # "organic_synthesis",
        # "polymer_materials",
    ]

task_family = "reproduction"
error_type = 'exact_support'
# error_type = "fixed_constraint_missing_1"
# error_type = "fixed_constraint_mismatch_1"
# error_type = "negated_field_mismatch"
# error_type = "negated_field_missing"
# error_type = "explore_field_missing"


paragraph_verdict = 'uncertain'
# paragraph_verdict = "violate"
choose_i = 0

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error in {path} at line {line_no}: {e}") from e
    return rows


def get_paragraph_by_id(data: list[dict], target_id: str) -> Optional[str]:
    target_id = str(target_id)
    for item in data:
        if str(item.get("id")) == target_id:
            return item["paragraph"]
    return None


def get_extraction_by_id(data: list[dict], target_id: str):
    target_id = str(target_id)
    for item in data:
        if str(item.get("id")) == target_id:
            return item.get("result", {}).get("setups")
    return None


# ======================
# 构建映射
# ======================
def build_support_map(boundary_data):
    """
    (candidate_id, paragraph_id) -> True
    """
    support = set()

    for row in boundary_data:
        cid = row["candidate_id"]
        for pid in row.get("support_paragraph_ids", []):
            support.add((cid, str(pid)))

    return support


def build_boundary_map(boundary_data):
    """
    (candidate_id, paragraph_id) -> [boundary_types]
    """
    mapping = defaultdict(list)

    for row in boundary_data:
        cid = row["candidate_id"]
        pools = row.get("sampled_boundary_paragraph_ids", {})

        for btype, pids in pools.items():
            for pid in pids:
                mapping[(cid, str(pid))].append(btype)

    return mapping


# ======================
# 主逻辑
# ======================
error_cases = []

for domain in domains:

    print(f"\nProcessing domain: {domain}")

    paragraph_path = f"seed_data/{domain}/paragraphs.json"
    extraction_path = f"seed_data/{domain}/llm_extraction/evidence_span.jsonl"
    normalized_path = f"seed_data/{domain}/llm_cleaning/evidence_span.jsonl"
    judge_path = f"data/{domain}/stage1_relevance/judge_results_stage1.jsonl"
    boundary_path = f"data/{domain}/stage1_relevance/sampled_constraints_stage1.boundary_manifest.jsonl"

    # 加载数据
    with open(paragraph_path, "r", encoding="utf-8") as f:
        paragraphs = json.load(f)

    extractions = read_jsonl(extraction_path)
    normalized = read_jsonl(normalized_path)
    judges = read_jsonl(judge_path)
    boundary_data = read_jsonl(boundary_path)

    # 构建映射
    support_map = build_support_map(boundary_data)
    boundary_map = build_boundary_map(boundary_data)

    domain_cases = []

    for judge in judges:
        cid = judge["candidate_id"]
        pid = str(judge["paragraph_id"])

        # 基础过滤
        if judge.get("task_family") != task_family:
            continue

        verdict = judge.get("parsed_output", {}).get("paragraph_verdict")
        if verdict != paragraph_verdict:
            continue

        # ======================
        # 判断 error_type
        # ======================
        is_match = False

        if error_type == "exact_support":
            is_match = (cid, pid) in support_map

        else:
            btypes = boundary_map.get((cid, pid), [])
            is_match = error_type in btypes

        if not is_match:
            continue

        # ======================
        # 收集 case
        # ======================
        case = {
            "domain": domain,
            "candidate_id": cid,
            "paragraph_id": pid,
            "constraints": judge["constraints"],
            "explore_field": judge["explore_field"],
            "field_definitions": get_relevant_field_definitions(
                domain, judge["constraints"], judge["explore_field"]
            ),
            "paragraph": get_paragraph_by_id(paragraphs, pid),
            "extraction_raw": get_extraction_by_id(extractions, pid),
            "extraction_normalized": get_extraction_by_id(normalized, pid),
            "judge_output": judge["parsed_output"],
            "boundary_types": boundary_map.get((cid, pid), []),
            "is_support": (cid, pid) in support_map,
        }

        domain_cases.append(case)

    print(f"Found {len(domain_cases)} cases in {domain}")
    error_cases.extend(domain_cases)

print(domain_cases[3]["candidate_id"], domain_cases[3]["paragraph_id"])
print(domain_cases[4]["candidate_id"], domain_cases[4]["paragraph_id"])
print(domain_cases[5]["candidate_id"], domain_cases[5]["paragraph_id"])

#     # 选择 case
    # choose_i = range(5)

#     with open('error_analysis_prompts.txt', 'w', encoding='utf-8') as outfile:
#         for i in choose_i:
#             case = error_case[i]
#             prompt = f'''
#     请你做一次“错误归因分析”，判断这个 case 的主要问题最可能出在哪个阶段。

#     你必须只基于我提供的内容进行判断，不要猜测未提供的信息，不要补充不存在的字段、值或证据。

#     请注意各阶段的职责边界：

#     1. 抽取阶段问题
#     指原始抽取结果本身就有问题，例如：
#     - 漏抽了段落中明确提到的信息
#     - 错抽了值或字段（包括将否定语境中的内容当作存在项抽取）
#     - 多抽了不属于该 setup 的内容
#     - setup 划分不合理
#     - 同一 setup 内字段对应关系错误

#     判断抽取阶段是否有问题时，只看原始抽取结果中的“取值”是否正确反映了段落原文的语义，不依赖 evidence 字段来判断抽取是否正确。

#     2. 标准化阶段问题
#     这里的“标准化”只是对抽取结果做简单清洗和规范化，不负责重新理解段落，也不负责校验抽取是否正确。
#     它只应处理这类问题：
#     - 术语归一错误
#     - 单位统一错误
#     - 数值/格式规范化错误
#     - 低风险字段映射错误
#     - 清洗过程中误删、误改、误合并原本已正确抽取的值

#     如果原始抽取本来就漏了、错了、setup 搞错了，这不属于标准化阶段问题。

#     3. 判读阶段问题
#     判读阶段的输入是**段落原文**和**约束条件**，输出是对每个固定字段的判定（match / uncertain / violate）以及最终 verdict。
#     判读可以且应当基于段落原文判断，不受抽取结果或标准化结果的错误影响（即它可以覆盖或修正中间结果的错误）。
#     判读阶段问题指：在抽取结果和标准化结果即使存在错误的情况下，判读对段落原文的理解、对约束与字段定义的应用、对 satisfy / uncertain / violate 的判定本身有误。

#     4. 约束/字段定义/任务规则问题
#     指不是某个阶段单独出错，而是约束本身、字段定义、字段依赖关系或任务规则存在歧义，导致这个 case 很难稳定判定。

#     请完成以下任务：

#     A. 判断这个 case 的“主要问题阶段”是哪一个。
#     只能选一个最主要的阶段。

#     B. 判断是否存在“次要问题阶段”。
#     如果有，请按可能性从高到低列出。

#     C. 给出详细证据链。
#     请结合：
#     - 段落内容
#     - 原始抽取结果
#     - 标准化后结果
#     - 约束与探索字段
#     - 当前 judge verdict
#     逐项说明你的判断依据。

#     D. 特别说明标准化阶段有没有问题。
#     请明确回答：
#     - 原始抽取结果中哪些内容只是需要规范化？
#     - 标准化是否只是做了简单清洗？
#     - 标准化有没有做了它本不该做的事情（例如把值改错、删错、映射错）？

#     E. 如果你认为当前 judge verdict 不合理，请明确说明：
#     - 更合理的 verdict 是什么
#     - 为什么

#     F. 最后请输出一个适合和专家讨论的简短总结，格式如下：
#     - 主要问题阶段：
#     - 次要问题阶段：
#     - 关键证据：
#     - 更合理的判定：
#     - 建议优先检查：

#     请尽量输出具体、可核查、面向讨论，不要泛泛而谈。

#     【任务信息】
#     任务类型：{task_family}
#     约束：{case[0]}
#     探索字段：{case[1]}
#     字段定义：{case[2]}

#     【样本信息】
#     段落 ID：{case[3]}
#     段落内容：{case[4]}

#     【中间结果】
#     原始抽取结果：{case[5]}
#     标准化后结果：{case[6]}

#     【判读信息】
#     判读规则：{case[8]}
#     当前段落判断：{case[7]}
# '''
#             outfile.write(prompt)
#             outfile.write('\n\n' + '=' * 80 + '\n\n')  







# cd 检索数据集
# python -m scripts.stage1_relevance.get_error_case
