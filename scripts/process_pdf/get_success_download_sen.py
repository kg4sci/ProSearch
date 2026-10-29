import json
from pathlib import Path

def load_json(p: Path):
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, p: Path):
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_doi_set(p: Path):
    if not p.exists():
        return set()
    with p.open('r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def write_doi_list(dois, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(dois)))

def norm_doi(x):
    return None if x is None else str(x).strip()

def has_any_real_doi(rec: dict) -> bool:
    for cref in rec.get("cited_refs", []):
        for doi in (cref.get("dois") or []):
            if doi is None:
                return False
    return True

def main():
    JSON_PATH        = Path('seed_data/electrochemistry/filtered_cited_sentence.json')
    NEEDED_TXT_PATH  = Path('seed_data/electrochemistry/needed_doi.txt')
    APPENDIX_TXT_PATH= Path('seed_data/electrochemistry/appendix_doi.txt')

    # ============== 1) 读取 needed_doi 并在 JSON 中置 None ==============
    recs = load_json(JSON_PATH)

    needed = {norm_doi(d) for d in load_doi_set(NEEDED_TXT_PATH)}
    needed.discard(None)

    replaced_cnt = 0
    touched_recs = 0
    for rec in recs:
        touched = False
        for cref in rec.get("cited_refs", []):
            dois = cref.get("dois")
            if not isinstance(dois, list):
                continue
            for i, d in enumerate(dois):
                if norm_doi(d) in needed:
                    dois[i] = None
                    replaced_cnt += 1
                    touched = True
        if touched:
            touched_recs += 1

    print(f"[JSON] 已将 {touched_recs} 条记录中的 {replaced_cnt} 个 DOI 设置为 None（因下载失败）")

    # ============== 2) 删除“dois 全为空/None”的记录 ==============
    before_n = len(recs)
    recs = [rec for rec in recs if has_any_real_doi(rec)]
    after_n = len(recs)
    removed_n = before_n - after_n
    print(f"[JSON] 删除 'dois 全为空/None' 的记录：{removed_n} 条；剩余：{after_n} 条")
    print(len(set([rec['doi'] for rec in recs])))

    # 覆盖写回
    save_json(recs, JSON_PATH)
    print(f"[JSON] 已覆盖写入：{JSON_PATH}")

    # ============== 3) 更新 appendix_doi.txt（移除 needed_doi 中的项） ==============
    appendix = load_doi_set(APPENDIX_TXT_PATH)
    old_app_size = len(appendix)
    if appendix:
        appendix = {d for d in appendix if d not in needed}
    # 去重后写回
    write_doi_list(appendix, APPENDIX_TXT_PATH)
    print(f"[APPENDIX] 从 appendix_doi.txt 中移除失败 DOI 共 {old_app_size - len(appendix)} 个；当前剩余 {len(appendix)} 个")
    print(f"[输出] 已更新：{APPENDIX_TXT_PATH}")

if __name__ == "__main__":
    main()

# python scripts/process_pdf/get_success_download_sen.py