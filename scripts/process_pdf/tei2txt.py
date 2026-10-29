import xml.etree.ElementTree as ET
from pathlib import Path

FOOTNOTE_TYPE_SET = {"footnote", "endnote", "fn", "foot"}

def get_ns(root: ET.Element) -> dict:
    if root.tag.startswith("{"):
        uri = root.tag.split("}", 1)[0][1:]
        return {"tei": uri}
    return {}

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

def find_body(root: ET.Element, ns: dict) -> ET.Element | None:
    body = root.find(".//tei:text/tei:body", ns) if ns else None
    return body or root.find(".//text/body")

def extract_body_text(tei_path: Path,
                      top_level_only: bool = False) -> str:
    """
    以“标题 -> 本段正文 -> 下一标题 …”的顺序导出正文。
    - 过滤 figure/table/refList/listBibl/caption 内的内容
    - 排除脚注 <note place="foot"> 等
    - top_level_only=True 时，仅遍历 <body>/<div> 顶层分节
    """
    try:
        root = ET.parse(tei_path).getroot()
    except ET.ParseError:
        return ""

    ns = get_ns(root)
    body = find_body(root, ns)
    if body is None:
        return ""

    parent_map = {c: p for p in root.iter() for c in p}

    DISALLOWED_ANCESTORS = {
        "refList", "listBibl", "caption",
        "figure", "table", 
    }

    def clean_text(e: ET.Element) -> str:
        return " ".join("".join(e.itertext()).split())

    sections: list[str] = []

    # 选择遍历的“分节容器”
    if top_level_only:
        div_path = "./tei:div" if ns else "./div"
        divs = body.findall(div_path, ns)
        # 如果有些文档没有用 <div> 包裹，就回退到直接遍历 body
        containers = divs if divs else [body]
    else:
        containers = [body]

    for container in containers:
        # 在该容器内按文档顺序遍历，遇到 head 开启新节，收集随后的 p
        current_section_lines: list[str] = []
        def flush_section():
            if current_section_lines:
                sections.append("\n".join(current_section_lines).strip())
                current_section_lines.clear()

        for elem in container.iter():
            tag = elem.tag.split("}")[-1]

            # 只考虑当前 container 的后代；跳过更深层的分节（当 top_level_only=True 时）
            if top_level_only and tag in {"div"} and elem is not container:
                # 跳过子小节，避免把次级结构并进来
                continue

            if tag not in {"head", "p"}:
                continue

            # 过滤脚注/不需要的容器祖先
            if any_ancestor_is_footnote(elem, parent_map):
                continue
            if any_ancestor(elem, parent_map, DISALLOWED_ANCESTORS):
                continue

            txt = clean_text(elem)
            if not txt:
                continue

            if tag == "head":
                # 新标题来了，先把上一节落盘，再开新节
                flush_section()
                current_section_lines.append(txt)
            else:  # tag == "p"
                # 若还没遇到 head，但直接来了段落，也收进当前节
                current_section_lines.append(txt)

        # 收尾
        flush_section()

    # 用空行分隔各节
    return "\n\n".join(s for s in sections if s).strip()


def write_txt(output_dir: Path, xml_path: Path, content: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = xml_path.with_suffix(".txt").name
    out_path = output_dir / out_name
    out_path.write_text(content, encoding="utf-8")
    return out_path

def process_one(xml_path: Path, out_dir: Path) -> None:
    content = extract_body_text(xml_path)
    out_path = write_txt(out_dir, xml_path, content)
    print(f"[OK] {xml_path} -> {out_path} ({len(content)} chars)")

def process_input(input_path: Path, output_dir: Path | None) -> None:
    if input_path.is_file():
        out_dir = output_dir or input_path.parent
        process_one(input_path, out_dir)
    else:
        # 批量：遍历目录下的 .xml（包含 .tei.xml）
        xml_files = sorted([p for p in input_path.rglob("*.xml")])
        if not xml_files:
            print(f"[WARN] No .xml files found under {input_path}")
            return
        out_dir = output_dir or input_path / "txt_out"
        for x in xml_files:
            try:
                process_one(x, out_dir)
            except Exception as e:
                print(f"[ERR] {x}: {e}")

def main():
    process_input(
        Path("tei/electrochemistry/10.1021_jacs.8b11237.tei.xml"),
        Path("scripts")
    )

if __name__ == "__main__":
    main()

# cd /home/liangwenliang/检索数据集
# conda activate retrieve_dataset
# python scripts/process_pdf/tei2txt.py
