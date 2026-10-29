import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from transformers import GPT2TokenizerFast

_gpt2_tok = GPT2TokenizerFast.from_pretrained("model/gpt2")

def get_len(text: str) -> int:
    return len(_gpt2_tok.encode(text))


FOOTNOTE_TYPE_SET = {"footnote", "endnote", "fn", "foot"}
DISALLOWED_ANCESTORS = {"refList", "listBibl", "caption", "figure", "table"}

# EXPERIMENT_HEAD_PAT = re.compile(
#     r"""
#     \b(
#         experimental(\s+section)?|
#         experiments?|
#         material?|
#         materials?\s*&?\s*methods?|
#         methods?|
#         methodology|
#         experimental\s+procedures?|
#         characteri[sz]ation(s)?|
#         preparations?|
#         synthes(?:is|es)
#     )\b
#     """,
#     re.IGNORECASE | re.X
# )

EXPERIMENT_HEAD_PAT = re.compile(
    r"""
    \b(
        # -------------------------
        # Generic "methods/experimental" headings
        # -------------------------
        experimental(\s+(section|part|details|methods?|procedures?|setup|design|conditions?))?|
        experiments?|
        (materials?\s*(and|&)\s*methods?)|
        (methods?\s*(and|&)\s*materials?)|
        methods?|
        methodology|
        experimental\s+procedures?|
        experimental\s+methods?|
        experimental\s+details|
        experimental\s+setup|
        experimental\s+conditions?|
        materials?\s+preparation|
        sample\s+preparation|
        preparation(\s+of\s+(samples?|materials?|compounds?|polymers?))?|
        synthes(?:is|es)\s+(and|&)\s+characteri[sz]ation|
        characteri[sz]ation(s)?|
        characterization\s+methods?|
        measurements?|
        instrumentation|
        analytical\s+methods?|

        # -------------------------
        # Organic synthesis–style headings
        # -------------------------
        general\s+procedure(s)?|
        typical\s+procedure(s)?|
        synthetic\s+procedure(s)?|
        synthetic\s+details|
        synthesis\s+of\b|
        preparation\s+of\b|
        reaction\s+conditions?|
        catalytic\s+conditions?|
        substrate\s+scope|
        optimization\s+of\s+(reaction|conditions?)|

        # -------------------------
        # Inorganic materials / ceramics–style headings
        # -------------------------
        solid[-\s]?state\s+(synthesis|reaction)|
        calcination|
        sinter(?:ing|ed)?|
        anneal(?:ing|ed)?|
        firing|
        pellet\s+preparation|
        powder\s+preparation|
        ceramic\s+processing|
        fabrication|
        materials?\s+fabrication|

        # -------------------------
        # Electrochemistry–style headings
        # -------------------------
        electrochemical\s+(methods?|measurements?|characteri[sz]ation|testing|evaluation)|
        electrode\s+preparation|
        electrolyte\s+preparation|
        cell\s+assembly|
        electrochemical\s+setup|
        cyclic\s+voltammetry|
        linear\s+sweep\s+voltammetry|
        chrono(amperometry|potentiometry)|
        impedance\s+spectroscopy|
        e i s|eis|

        # -------------------------
        # Cell culture–style headings
        # -------------------------
        cell\s+culture(\s+(methods?|conditions?))?|
        culture\s+conditions?|
        cell\s+lines?|
        in\s+vitro(\s+experiments?)?|
        cell\s+treatment(s)?|
        incubation\s+conditions?|

        # -------------------------
        # Immunotechnology–style headings
        # -------------------------
        western\s+blot(ting)?|
        immunoblot(ting)?|
        elisa|
        flow\s+cytometry|
        immunofluorescence|
        immunostaining|
        immunohistochemistry|
        antibodies|
        sds[-\s]?page|
        protein\s+extraction|
        sample\s+processing
    )\b
    """,
    re.IGNORECASE | re.X
)


# —— 仅用于“排除”的顶层小节（原逻辑保留，用于 method 抽取）——
EXCLUDED_HEAD_RE = re.compile(
    r"^\s*(abstract|introduction|general introduction|background|troduction|"
    r"conclusion[s]?|concluding remarks|summary|summary\s*&\s*conclusions|"
    r"summary\s+and\s+conclusions|summary\s+and\s+outlook)\s*$",
    re.IGNORECASE
)


def get_ns(root: ET.Element) -> dict:
    if root.tag.startswith("{"):
        uri = root.tag.split("}", 1)[0][1:]
        return {"tei": uri}
    return {}

def clean_text(e: ET.Element) -> str:
    return " ".join("".join(e.itertext()).split())

def find_body(root: ET.Element, ns: dict) -> Optional[ET.Element]:
    body = root.find(".//tei:text/tei:body", ns) if ns else None
    return body or root.find(".//text/body")

def get_immediate_head_text(div_elem: ET.Element, ns: dict) -> str:
    head = div_elem.find("./tei:head", ns) if ns else div_elem.find("./head")
    return clean_text(head) if head is not None else ""

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

def _iter_paragraphs_in_containers(
    root: ET.Element,
    body: ET.Element,
    ns: dict,
    containers: list[ET.Element],
    paper_doi: str,
    parent_map: dict,
    *,
    exclude_top_heads_for_method: bool,
) -> list[dict]:
    """
    给定 containers（顶层div列表，或某些筛选后的div），按顺序抽取<p>段落。
    """
    records = []
    for container in containers:
        top_head = get_immediate_head_text(container, ns) if container is not body else ""

        # 仅 method 抽取时沿用你的“排除顶层 head”逻辑
        if exclude_top_heads_for_method and top_head and EXCLUDED_HEAD_RE.match(top_head):
            continue

        for p in container.iter():
            if p.tag.split("}")[-1] != "p":
                continue
            if any_ancestor_is_footnote(p, parent_map):
                continue
            if any_ancestor(p, parent_map, DISALLOWED_ANCESTORS):
                continue

            paragraph_text = clean_text(p)
            para_len = get_len(paragraph_text)
            if para_len < 50 or para_len > 2048:
                continue

            records.append({
                "doi": paper_doi,
                "head": top_head,
                "paragraph": paragraph_text,
                "source": "pdf",
            })
    return records

def extract_method_paragraph_records(tei_path: Path) -> list[dict]:
    """
    原先逻辑：只抽取“实验类”顶层小节中的段落；没有匹配实验小节则返回空。
    """
    try:
        root = ET.parse(tei_path).getroot()
    except ET.ParseError:
        return []

    ns = get_ns(root)
    body = find_body(root, ns)
    if body is None:
        return []

    paper_doi = tei_path.stem
    parent_map = {c: p for p in root.iter() for c in p}

    div_path = "./tei:div" if ns else "./div"
    top_divs = body.findall(div_path, ns)
    if not top_divs:
        return []

    experiment_divs = [
        d for d in top_divs
        if EXPERIMENT_HEAD_PAT.search(get_immediate_head_text(d, ns) or "")
    ]
    if not experiment_divs:
        return []

    return _iter_paragraphs_in_containers(
        root=root,
        body=body,
        ns=ns,
        containers=experiment_divs,
        paper_doi=paper_doi,
        parent_map=parent_map,
        exclude_top_heads_for_method=True,
    )

def extract_all_body_paragraph_records(tei_path: Path) -> list[dict]:
    """
    新增：抽取正文 body 内的全部段落（按顶层div顺序）。
    仍然排除：脚注、参考文献列表、图表caption/figure/table 等。
    """
    try:
        root = ET.parse(tei_path).getroot()
    except ET.ParseError:
        return []

    ns = get_ns(root)
    body = find_body(root, ns)
    if body is None:
        return []

    paper_doi = tei_path.stem
    parent_map = {c: p for p in root.iter() for c in p}

    div_path = "./tei:div" if ns else "./div"
    top_divs = body.findall(div_path, ns)
    containers = top_divs if top_divs else [body]

    return _iter_paragraphs_in_containers(
        root=root,
        body=body,
        ns=ns,
        containers=containers,
        paper_doi=paper_doi,
        parent_map=parent_map,
        exclude_top_heads_for_method=False,  # 全文不排除 introduction/conclusion 等
    )

def write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for i, r in enumerate(records, 1):
        item = {"id": str(i)}
        item.update(r)
        out.append(item)

    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

def process_folder(input_dir: Path, out_method_path: Path, out_all_path: Path) -> None:
    xml_files = sorted(input_dir.rglob("*.xml"))
    if not xml_files:
        print(f"[WARN] No .xml under {input_dir}")
        return

    # 1) 先按旧逻辑收集 method 段落（保证顺序一致 => id 不变）
    method_recs: list[dict] = []
    method_keys: set[tuple] = set()
    total_files = used_files_method = 0

    for x in xml_files:
        total_files += 1
        try:
            recs = extract_method_paragraph_records(x)
            if not recs:
                continue
            method_recs.extend(recs)
            used_files_method += 1

            # 用于从全文段落中剔除 method 段落，避免重复
            # key 里包含 doi + head + paragraph，尽量精确
            for r in recs:
                method_keys.add((r["doi"], r.get("head", ""), r["paragraph"]))

        except Exception as e:
            print(f"[ERR][METHOD] {x}: {e}")

    # 2) 再收集全文段落，过滤掉已在 method 中出现的段落；并“追加”到 method 后面
    other_recs: list[dict] = []
    used_files_all = 0
    total_all_paras = 0

    for x in xml_files:
        try:
            all_recs = extract_all_body_paragraph_records(x)
            if not all_recs:
                continue
            used_files_all += 1
            total_all_paras += len(all_recs)

            for r in all_recs:
                key = (r["doi"], r.get("head", ""), r["paragraph"])
                if key in method_keys:
                    continue
                other_recs.append(r)

        except Exception as e:
            print(f"[ERR][ALL] {x}: {e}")

    # 3) 写出：method-only + all(paragraphs with method first)
    if method_recs:
        write_json(out_method_path, method_recs)
        print(f"[DONE] Method pool: {len(method_recs)} paragraphs across {used_files_method}/{total_files} files -> {out_method_path}")
    else:
        print(f"[WARN] No method paragraphs collected from {total_files} files. (paragraphs.json not written)")

    combined = method_recs + other_recs
    if combined:
        write_json(out_all_path, combined)
        print(
            f"[DONE] Full pool: {len(combined)} paragraphs "
            f"(method={len(method_recs)} first, other={len(other_recs)}) "
            f"from {used_files_all}/{total_files} files -> {out_all_path}"
        )
    else:
        print(f"[WARN] No paragraphs collected from {total_files} files. (all_paragraphs.json not written)")


if __name__ == "__main__":
    for d in [
        # "cell_culture",
        # "electrochemistry",
        # "immunotechnology",
        "inorganic_materials",
        "organic_synthesis",
        # "polymer_materials"
    ]:
        process_folder(
            input_dir=Path(f"tei/{d}"),
            out_method_path=Path(f"seed_data/{d}/paragraphs.json"),
            out_all_path=Path(f"seed_data/{d}/all_paragraphs.json"),
        )



# cd 检索数据集
# python scripts/process_pdf/tei2cited_para.py

