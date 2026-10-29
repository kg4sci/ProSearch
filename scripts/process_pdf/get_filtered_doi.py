# -*- coding: utf-8 -*-
import json
from pathlib import Path

def main():
    # ===== 路径在 main 里定义（可自行修改）=====
    JSON_PATH = Path("/home/liangwenliang/检索数据集/seed_data/electrochemistry/gold_tree_data.json")            # 你的 JSON 文件
    PDF_DIR   = Path("/home/liangwenliang/检索数据集/pdf/electrochemistry")                 # 存放 PDF 的文件夹
    OUT_TXT   = Path("/home/liangwenliang/检索数据集/seed_data/electrochemistry/needed_doi.txt")     # 只在 JSON 中的 DOI 输出

    # ===== 读取 JSON，提取 DOI =====
    with JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    json_dois = set()
    json_dois.update(d['doi'] for d in data if d["operations"] != [])
    print(len(json_dois))

    # ===== 扫描 PDF，按“首个下划线变斜杠”规则还原 DOI =====
    # fs_dois = set()
    # for pdf in sorted(PDF_DIR.glob("*.pdf")):
    #     stem = pdf.stem  # 例如 10.1002_adfm.201910118
    #     doi = stem.replace("_", "/")  # 只替换第一个下划线
    #     fs_dois.add(doi)
    # print(len(fs_dois))
    # # ===== 统计 =====
    # both        = json_dois & fs_dois
    # only_json   = json_dois - fs_dois
    # only_folder = fs_dois  - json_dois

    # print(f"一致（同时出现在 JSON 和文件夹）: {len(both)}")
    # print(f"只出现在 JSON 中: {len(only_json)}")
    # print(f"只出现在文件夹中: {len(only_folder)}")

    # # ===== 保存只在 JSON 中的 DOI =====
    # with OUT_TXT.open("w", encoding="utf-8") as f:
    #     for doi in sorted(only_json):
    #         f.write(doi + "\n")

    # print(f"已写出只在 JSON 中的 DOI 到: {OUT_TXT.resolve()}")

if __name__ == "__main__":
    main()


# python scripts/process_pdf/get_filtered_doi.py
