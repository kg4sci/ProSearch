import json
from typing import Optional, Dict, Any


def load_candidates(json_path: str) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_paragraph_by_id(data: list[dict], target_id: str) -> Optional[Dict[str, Any]]:
    target_id = str(target_id)
    for item in data:
        if str(item.get("id")) == target_id:
            return item
    return None


if __name__ == "__main__":
    json_path = "seed_data/inorganic_materials/paragraphs.json"   
    target_id = "28115"   

    data = load_candidates(json_path)
    result = get_paragraph_by_id(data, target_id)

    if result is None:
        print(f"未找到 id={target_id} 的段落")
    else:
        print("找到的结果：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("\nparagraph 内容：")
        print(result["paragraph"])
    
# python scripts/relevance/get_paragraph_example.py
        
