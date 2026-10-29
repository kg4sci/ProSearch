import json

domains = [
    "electrochemistry", 
    "cell_culture", 
    "immunotechnology",
    "inorganic_materials",
    "organic_synthesis", 
    "polymer_materials"
]

for d in domains:
    input_path = f"seed_data/{d}/paragraphs.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        text = item["paragraph"]
        text = text.replace(" H 2 SO 4", " H2SO4")
        text = text.replace("(N 2 )", "(N2)")
        item["paragraph"] = text

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[DONE] {d} -> {input_path}")



# cd /home/liangwenliang/检索数据集/
# python scripts/process_pdf/cleaning.py