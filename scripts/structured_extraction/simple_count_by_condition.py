import json

class JSONLScanner:
    def __init__(self, file_path):
        self.file_path = file_path

    def _get_nested_value(self, data, path_list):
        """
        递归获取嵌套字段值，支持列表自动展开。
        """
        if not path_list or data is None:
            return [data]
        
        current_key = path_list[0]
        remaining_path = path_list[1:]
        
        if isinstance(data, dict):
            value = data.get(current_key)
            return self._get_nested_value(value, remaining_path)
        
        elif isinstance(data, list):
            results = []
            for item in data:
                results.extend(self._get_nested_value(item, path_list))
            return results
        
        return []

    def analyze_condition(self, field_path, target_value):
        """
        分析条件并返回详细结果。
        """
        path_list = field_path.split('.')
        success_ids = []
        failed_ids = []
        total_lines = 0

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                total_lines += 1
                
                try:
                    record = json.loads(line)
                    record_id = str(record.get("id", "Unknown"))
                    
                    # 提取路径下的所有可能取值
                    extracted_values = self._get_nested_value(record, path_list)
                    
                    # 检查是否有任何一个值匹配目标值
                    if any(val and target_value in val for val in extracted_values):
                        success_ids.append(record_id)
                    else:
                        failed_ids.append(record_id)
                        
                except json.JSONDecodeError:
                    continue

        return {
            "total": total_lines,
            "match_count": len(success_ids),
            "fail_count": len(failed_ids),
            "matched_ids": success_ids,
            "failed_ids": failed_ids
        }

# ================= 配置与运行区 =================
scanner = JSONLScanner('seed_data/electrochemistry/llm_extraction/raw_evidence_span.jsonl')

# 定义你想检查的条件
field_to_check = "result.setups.task.value"
target_value = "ORR"

result = scanner.analyze_condition(field_to_check, target_value)

print(f"检查字段: {field_to_check}")
print(f"目标数值: {target_value}")
print("-" * 30)
print(f"总计行数: {result['total']}")
print(f"符合条件: {result['match_count']} 行")
print(f"不符合条件: {result['fail_count']} 行")

if result['failed_ids']:
    print(f"\n以下 ID 不符合条件 (共 {len(result['failed_ids'])} 个):")
    # 为了防止 ID 太多刷屏，这里只打印前 10 个，你可以根据需要修改
    print(", ".join(result['failed_ids'][:50]) + ("..." if len(result['failed_ids']) > 50 else ""))

    

# python scripts/structured_extraction/simple_count_by_condition.py