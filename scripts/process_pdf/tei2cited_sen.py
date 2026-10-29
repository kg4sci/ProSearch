# -*- coding: utf-8 -*-
import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

FOOTNOTE_TYPE_SET = {"footnote", "endnote", "fn", "foot"}
DISALLOWED_ANCESTORS = {"refList", "listBibl", "caption", "figure", "table"}

# —— 实验类小节关键词（可按需扩展）——
EXPERIMENT_HEAD_PAT = re.compile(
    r"""
    \b(
        experimental(\s+section)?|
        experiments?|
        material?|
        materials?\s*&?\s*methods?|
        methods?|
        methodology|
        experimental\s+procedures?|
        characteri[sz]ation(s)?|      
        preparations?|                
        synthes(?:is|es)         
    )\b
    """,
    re.IGNORECASE | re.X
)


# 仍保留原来的严格等值排除（可留可去）
EXCLUDED_HEAD_RE = re.compile(
    r"^\s*(abstract|introduction|general introduction|background|"
    r"conclusion[s]?|concluding remarks|summary|summary\s*&\s*conclusions|"
    r"summary\s+and\s+conclusions|summary\s+and\s+outlook)\s*$",
    re.IGNORECASE
)

BRACKET_CIT_RE = re.compile(r"\[(?:\d+|\d+\s*[-–]\s*\d+|[\d,\s–-]+)\]")

def get_ns(root: ET.Element) -> dict:
    if root.tag.startswith("{"):
        uri = root.tag.split("}", 1)[0][1:]
        return {"tei": uri}
    return {}

# NEW: 取 xml:id 的通用辅助
_XML_NS = "http://www.w3.org/XML/1998/namespace"
def get_xml_id(elem: ET.Element) -> Optional[str]:
    return elem.get(f"{{{_XML_NS}}}id") or elem.get("xml:id") or elem.get("id")

# NEW: 建索引——biblStruct xml:id -> DOI 或 None
def build_bibl_doi_map(root: ET.Element, ns: dict) -> dict[str, Optional[str]]:
    m = {}
    path = ".//tei:listBibl//tei:biblStruct" if ns else ".//listBibl//biblStruct"
    for b in root.findall(path, ns):
        bid = get_xml_id(b)
        if not bid:
            continue
        doi = None
        # 常见位置：analytic/idno[@type='DOI']；有些在 monogr/idno
        candidates = []
        if ns:
            candidates += b.findall(".//tei:analytic/tei:idno[@type='DOI']", ns)
            candidates += b.findall(".//tei:monogr/tei:idno[@type='DOI']", ns)
        else:
            candidates += b.findall(".//analytic/idno[@type='DOI']")
            candidates += b.findall(".//monogr/idno[@type='DOI']")
        for c in candidates:
            t = (c.text or "").strip()
            if t:
                doi = t
                break
        m[bid] = doi  # 可能为 None
    return m

def is_footnote(elem: ET.Element) -> bool:
    tag = elem.tag.split("}")[-1]
    if tag != "note":
        return False
    place = (elem.get("place") or "").lower()
    typ = (elem.get("type") or "").lower()
    return place == "foot" or typ in FOOTNOTE_TYPE_SET

def any_ancestor(elem: ET.Element, parent_map: dict, names: set[str]) -> bool:
    cur = elem
    while cur in parent_map:
        cur = parent_map[cur]
        if cur.tag.split("}")[-1] in names:
            return True
    return False

def any_ancestor_is_footnote(elem: ET.Element, parent_map: dict) -> bool:
    cur = elem
    while cur in parent_map:
        cur = parent_map[cur]
        if is_footnote(cur):
            return True
    return False

def find_body(root: ET.Element, ns: dict) -> Optional[ET.Element]:
    body = root.find(".//tei:text/tei:body", ns) if ns else None
    return body or root.find(".//text/body")

def clean_text(e: ET.Element) -> str:
    return " ".join("".join(e.itertext()).split())

def get_immediate_head_text(div_elem: ET.Element, ns: dict) -> str:
    head = div_elem.find("./tei:head", ns) if ns else div_elem.find("./head")
    return clean_text(head) if head is not None else ""

def is_intro_or_conclusion_head(head_text: str) -> bool:
    if not head_text:
        return False
    t = head_text.lower()
    return ("introduct" in t) or ("conclusion" in t) or ('computational' in t) or ('theory' in t) or ('calculation' in t)

def collect_inner_heads_until_container(
    node: ET.Element, container: ET.Element, parent_map: dict, ns: dict
) -> list[str]:
    tmp = []
    cur = node
    while cur in parent_map and cur is not container:
        cur = parent_map[cur]
        if cur.tag.split("}")[-1] == "div":
            txt = get_immediate_head_text(cur, ns)
            if txt:
                tmp.append(txt)
    heads = list(reversed(tmp))
    if heads and heads[0] == get_immediate_head_text(container, ns):
        heads = heads[1:]
    return heads

_NUM_TOKEN_RE = re.compile(r"\d+")

def _expand_range(a: int, b: int) -> list[int]:
    return list(range(min(a, b), max(a, b) + 1))

def _extract_numbers_from_bracket_token(tok: str) -> list[int]:
    tok = tok.strip()
    if not tok:
        return []
    if "-" in tok or "–" in tok:
        parts = re.split(r"[-–]", tok)
        if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
            return _expand_range(int(parts[0]), int(parts[1]))
    if tok.isdigit():
        return [int(tok)]
    return [int(m.group()) for m in _NUM_TOKEN_RE.finditer(tok)]

def extract_display_labels_from_text(text: str) -> list[str]:
    seen, out = set(), []
    for m in BRACKET_CIT_RE.finditer(text):
        inner = m.group()[1:-1]
        tokens = re.split(r"[,\s]+", inner)
        for tok in tokens:
            for n in _extract_numbers_from_bracket_token(tok):
                s = str(n)
                if s not in seen:
                    seen.add(s)
                    out.append(s)
    return out

def extract_labels_and_mapping_from_refs(s_elem: ET.Element, ns: dict) -> tuple[list[str], dict]:
    label_list, label_map, seen = [], {}, set()
    xpath = ".//tei:ref" if ns else ".//ref"
    for r in s_elem.findall(xpath, ns):
        if (r.get("type") or "").lower() != "bibr":
            continue
        label = (r.text or "").strip()
        if not label:
            continue
        nums = _NUM_TOKEN_RE.findall(label)
        if len(nums) != 1:
            continue
        label = nums[0]
        if label not in seen:
            seen.add(label)
            label_list.append(label)
        tgt = (r.get("target") or "").lstrip("#").strip()
        if tgt:
            if label not in label_map:
                label_map[label] = tgt
            else:
                cur = label_map[label]
                if isinstance(cur, list):
                    if tgt not in cur:
                        cur.append(tgt)
                else:
                    if tgt != cur:
                        label_map[label] = [cur, tgt]
    return label_list, label_map

def extract_cited_records(tei_path: Path, top_level_only: bool = True, dedup: bool = True, min_chars: int = 15) -> list[dict]:
    try:
        root = ET.parse(tei_path).getroot()
    except ET.ParseError:
        return []

    ns = get_ns(root)
    body = find_body(root, ns)
    if body is None:
        return []

    any_s = body.find(".//tei:s" if ns else ".//s", ns)
    if any_s is None:
        return []

    parent_map = {c: p for p in root.iter() for c in p}

    # NEW: 为整篇文档构建 biblStruct xml:id -> DOI 的索引
    bibl_doi_map = build_bibl_doi_map(root, ns)

    div_path = "./tei:div" if ns else "./div"
    top_divs = body.findall(div_path, ns)
    containers = top_divs if (top_level_only and top_divs) else [body]

    # 仅在“实验类”顶层小节内检索
    experiment_divs = [d for d in top_divs if EXPERIMENT_HEAD_PAT.search(get_immediate_head_text(d, ns) or "")]
    has_experiment = len(experiment_divs) > 0
    if has_experiment:
        containers = experiment_divs
    else:
        return []

    records, seen = [], set()
    doi = tei_path.stem

    for container in containers:
        container_head = get_immediate_head_text(container, ns)
        if container_head and EXCLUDED_HEAD_RE.match(container_head):
            continue
        if is_intro_or_conclusion_head(container_head):
            continue

        for p in container.iter():
            if p.tag.split("}")[-1] != "p":
                continue
            if any_ancestor_is_footnote(p, parent_map):
                continue
            if any_ancestor(p, parent_map, DISALLOWED_ANCESTORS):
                continue

            paragraph_text = clean_text(p)

            s_path = ".//tei:s" if ns else ".//s"
            for s in p.findall(s_path, ns):
                inner_heads = collect_inner_heads_until_container(s, container, parent_map, ns)
                if any(is_intro_or_conclusion_head(h) for h in ([container_head] + inner_heads)):
                    continue

                txt = " ".join("".join(s.itertext()).split())
                if len(txt) < min_chars:
                    continue

                ref_labels, label_map = extract_labels_and_mapping_from_refs(s, ns)
                bracket_labels = extract_display_labels_from_text(txt)
                if not (ref_labels or bracket_labels):
                    continue

                # 合并显式编号（ref 标签）与仅从方括号解析到的编号
                seen_labels, cited_label_list = set(), []
                for lab in ref_labels + [b for b in bracket_labels if b not in set(ref_labels)]:
                    if lab not in seen_labels:
                        seen_labels.add(lab)
                        cited_label_list.append(lab)

                cited_refs = []  # 顺序保真列表

                for lab in cited_label_list:
                    tgt = label_map.get(lab, None)

                    # 统一成列表形态，保持一一对齐
                    if tgt is None:
                        xml_ids = [None]
                    elif isinstance(tgt, list):
                        xml_ids = tgt
                    else:
                        xml_ids = [tgt]

                    dois = []
                    for xid in xml_ids:
                        if xid is None:
                            dois.append(None)
                        else:
                            dois.append(bibl_doi_map.get(xid))  # 可能为 None

                    # 顺序保真列表
                    cited_refs.append({
                        "label": lab,
                        "xml_ids": xml_ids,
                        "dois": dois,
                    })

                # 放回记录
                record = {
                    "doi": doi,
                    "has_experiment": has_experiment,
                    "top_level_head": container_head,
                    "inner_heads": inner_heads,
                    "sentence": txt,
                    "paragraph": paragraph_text,
                    "cited_refs": cited_refs,
                }
                records.append(record)

    return records

def write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

# 处理单个文件
def process_one_file(xml_path: Path, out_dir: Path) -> Optional[Path]:
    recs = extract_cited_records(xml_path, top_level_only=True)
    if not recs:
        print(f"[SKIP] no numbered citations found: {xml_path}")
        return None
    out_path = out_dir / (xml_path.stem + ".cited.sentences.json")
    write_json(out_path, recs)
    print(f"[OK] {xml_path} -> {out_path} ({len(recs)} records)")
    return out_path

# 处理文件夹文件
def process_folder(input_dir: Path, output_path: Path) -> None:
    xml_files = sorted(input_dir.rglob("*.xml"))
    if not xml_files:
        print(f"[WARN] No .xml under {input_dir}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_recs = []
    total_files = used_files = total_records = 0
    for x in xml_files:
        total_files += 1
        try:
            recs = extract_cited_records(x, top_level_only=True)
            if not recs:
                print(f"[SKIP] no numbered citations found: {x}")
                continue
            all_recs.extend(recs)
            used_files += 1
            total_records += len(recs)
            print(f"[OK] {x} ({len(recs)} records)")
        except Exception as e:
            print(f"[ERR] {x}: {e}")

    if not all_recs:
        print(f"[WARN] No numbered-citation sentences collected from {total_files} files.")
        return

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_recs, f, ensure_ascii=False, indent=2)

    kept = len(all_recs)
    print(f"[DONE] Aggregated {kept} records across {used_files}/{total_files} files -> {output_path}")

if __name__ == "__main__":
    process_folder(
        input_dir=Path("tei/nanoparticle"),
        output_path=Path("seed_data/nanoparticle/cited_sentence.json"),
    )


# python scripts/process_pdf/tei2cited_sen.py
