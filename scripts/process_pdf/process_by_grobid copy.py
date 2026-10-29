# -*- coding: utf-8 -*-
import time, requests, json, re
from pathlib import Path
from xml.etree import ElementTree as ET
from collections import OrderedDict

HOST        = "http://localhost:8070"
INPUT_DIR   = Path("pdf/electrochemistry")
OUTPUT_DIR  = Path("tei/electrochemistry")
OUT_JSON    = Path("scripts/heads_first.json")
TIMEOUT_S   = 300
TOKEN_LIMIT = 100

def out_path_for(pdf: Path) -> Path:
    rel = pdf.relative_to(INPUT_DIR)
    out = OUTPUT_DIR / rel.with_suffix(".tei.xml")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

def process_pdf():
    pdfs = sorted(list(INPUT_DIR.rglob("*.pdf")) + list(INPUT_DIR.rglob("*.PDF")))
    print(f"Found {len(pdfs)} file(s) to process")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, pdf in enumerate(pdfs, 1):
        outp = out_path_for(pdf)
        if outp.exists():
            print(f"[{i}/{len(pdfs)}] skip exists {pdf}")
            continue
        print(f"[{i}/{len(pdfs)}] -> {pdf}")
        with pdf.open("rb") as f:
            files = {"input": (pdf.name, f, "application/pdf")}
            data  = {"consolidateHeader": 0, "consolidateCitations": 0, "teiCoordinates": 1}
            r = requests.post(f"{HOST}/api/processFulltextDocument", files=files, data=data, timeout=TIMEOUT_S)
            if r.status_code == 200:
                outp.write_bytes(r.content)
                print("OK ->", outp)
            else:
                print("FAIL HTTP", r.status_code)

def get_ns(root):
    """推断 TEI 命名空间（兼容无命名空间的 TEI）"""
    return {"tei": root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}

def norm_head(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    s = s.rstrip(".").strip()
    return s

def tokenize(text: str):
    # 简单按空白切分为“token”
    return re.findall(r"\S+", text or "")

def first_100_tokens(text: str) -> str:
    toks = tokenize(text)
    return " ".join(toks[:TOKEN_LIMIT])

def _iter_top_divs_with_head(root, ns, where="body"):
    """
    遍历 text/body 或 text/back 下的顶层 div（只取直接子 div），并返回 (div_elem, head_text)
    为避免 xml.etree.ElementTree 无 getparent，这里直接找 div[head]。
    """
    if ns:
        base = f".//tei:text/tei:{where}/tei:div[tei:head]"
        divs = root.findall(base, ns)
        # 兼容无命名空间
        if not divs:
            base = f".//text/{where}/div[head]"
            divs = root.findall(base, ns)
    else:
        base = f".//text/{where}/div[head]"
        divs = root.findall(base)
    for div in divs:
        # 取 head 文本（选择第一个 head）
        if ns:
            heads = div.findall("tei:head", ns) or div.findall("head", ns)
        else:
            heads = div.findall("head")
        if not heads:
            continue
        head_txt = norm_head("".join(heads[0].itertext()))
        if head_txt:
            yield div, head_txt

def _div_body_text(div, ns):
    """
    返回 div 去掉 head 后的正文文本。
    优先拼接 div 下的 <p>；若无 <p>，则取 div 的全部文本并尝试去掉开头的 head 文本。
    """
    # 1) 优先取 <p>
    if ns:
        ps = div.findall(".//tei:p", ns) or div.findall(".//p", ns)
    else:
        ps = div.findall(".//p")
    parts = []
    for p in ps:
        txt = " ".join(p.itertext()) if p is not None else ""
        txt = re.sub(r"\s+", " ", txt).strip()
        if txt:
            parts.append(txt)
    body = " ".join(parts).strip()

    if body:
        return body

    # 2) 若无 <p>，则取 div 所有文本，并去掉 head 的文本前缀
    raw = " ".join(div.itertext())
    raw = re.sub(r"\s+", " ", raw).strip()

    # 尝试识别 head 文本并剔除（粗略处理）
    # 找第一个 head
    if ns:
        heads = div.findall("tei:head", ns) or div.findall("head", ns)
    else:
        heads = div.findall("head")
    if heads:
        htxt = re.sub(r"\s+", " ", "".join(heads[0].itertext())).strip()
        if raw.startswith(htxt):
            raw = raw[len(htxt):].strip()
    return raw

def extract_heads_with_snippets(tei_path: Path):
    """
    从单个 TEI 中提取 (head, snippet) 列表。
    snippet：对应 div 的正文（去 head），按空白切词截断到前 100 tokens。
    """
    try:
        root = ET.parse(tei_path).getroot()
    except ET.ParseError:
        return []

    ns = get_ns(root)
    results = []

    # for where in ("body", "back"):
    for where in ("body",):
        for div, head_txt in _iter_top_divs_with_head(root, ns, where=where):
            body_text = _div_body_text(div, ns)
            snippet = first_100_tokens(body_text)
            results.append((head_txt, snippet))
    return results

def tei_to_pdf_name(tei_path: Path) -> str:
    """把 xxx.tei.xml 映射回 xxx.pdf 的文件名（仅用于输出展示）"""
    name = tei_path.name
    if name.endswith(".tei.xml"):
        return name[:-8] + ".pdf"
    # 兜底
    return Path(name).with_suffix(".pdf").name

def get_heads_as_json():
    tei_files = sorted(OUTPUT_DIR.rglob("*.tei.xml"))
    print(f"Found {len(tei_files)} TEI file(s) in {OUTPUT_DIR}")

    seen = set()           # 已出现的 head（去重，全局首次）
    records = []           # 要写入 JSON 的列表

    for i, tei in enumerate(tei_files, 1):
        triplets = extract_heads_with_snippets(tei)  # [(head, snippet)]
        if not triplets:
            print(f"[{i}/{len(tei_files)}] {tei.name}: no heads")
            continue

        added_this_file = 0
        pdf_name = tei_to_pdf_name(tei)

        for head_txt, snippet in triplets:
            if not head_txt:
                continue
            if head_txt in seen:
                continue  # 只保留“首次出现”的子标题
            seen.add(head_txt)
            records.append({
                "head": head_txt,
                "file": pdf_name,
                "snippet": snippet
            })
            added_this_file += 1

        print(f"[{i}/{len(tei_files)}] {tei.name}: +{added_this_file} new")

    # 可按标题排序，便于检索
    records.sort(key=lambda x: x["head"].casefold())
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAdded: {len(records)}, Saved -> {OUT_JSON.resolve()}")

if __name__ == "__main__":
    # process_pdf()
    get_heads_as_json()


# conda activate grobid17        
# cd /home/liangwenliang/grobid-0.8.2
# ./gradlew -Dorg.gradle.java.home="$CONDA_PREFIX" run


# cd /home/liangwenliang/检索数据集
# conda activate retrieve_dataset
# python scripts/process_pdf.py
