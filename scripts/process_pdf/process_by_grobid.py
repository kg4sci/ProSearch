import time
import requests
from pathlib import Path

HOST        = "http://localhost:8070"
TIMEOUT_S   = 100           # 300
PER_FILE_PAUSE_S = 1
CONSOLIDATE_HEADER     = 0 
CONSOLIDATE_CITATIONS  = 0 # 取值为2表示获取参考文献的doi , 为0表示不获取
SEGMENT_SENTENCES      = 1  
MAX_RETRIES            = 4
BACKOFF_BASE_S         = 2
BACKOFF_MAX_S          = 60
HEADERS = {
    "Accept": "application/xml",
    # "User-Agent": "grobid-client-lite/0.1"
}

def out_path_for(pdf: Path) -> Path:
    rel = pdf.relative_to(INPUT_DIR)
    out = OUTPUT_DIR / rel.with_suffix(".tei.xml")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out

def request_grobid(pdf_path: Path) -> bytes | None:
    url = f"{HOST}/api/processFulltextDocument"
    files = {"input": (pdf_path.name, pdf_path.open("rb"), "application/pdf")}
    data = {
        "consolidateHeader":     str(CONSOLIDATE_HEADER),
        "consolidateCitations":  str(CONSOLIDATE_CITATIONS),
        "segmentSentences":      str(SEGMENT_SENTENCES),   
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, files=files, data=data, headers=HEADERS, timeout=TIMEOUT_S)
            if r.status_code == 200 and r.content:
                return r.content
            else:
                print(f"[WARN] HTTP {r.status_code} for {pdf_path.name}")
        except requests.RequestException as e:
            print(f"[WARN] Request error on {pdf_path.name}: {e}")

        sleep_s = min(BACKOFF_BASE_S ** attempt, BACKOFF_MAX_S)
        time.sleep(sleep_s)

    print(f"[FAIL] Exhausted retries for {pdf_path.name}")
    return None

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
        content = request_grobid(pdf)
        if content:
            outp.write_bytes(content)
            print("OK ->", outp, f"({len(content)} bytes)")
        else:
            print("FAIL ->", pdf)

        time.sleep(PER_FILE_PAUSE_S)


if __name__ == "__main__":
    domain = [
        # 'electrochemistry',
        # 'cell_culture', 
        # 'immunotechnology',
        'inorganic_materials',
        'organic_synthesis',
        # 'polymer_materials'
    ]
    for d in domain:
        INPUT_DIR   = Path(f"pdf/{d}")
        OUTPUT_DIR  = Path(f"tei/{d}")
        process_pdf()




# conda activate grobid17        
# cd /home/liangwenliang/grobid-0.8.2
# ./gradlew -Dorg.gradle.java.home="$CONDA_PREFIX" run


# cd /home/liangwenliang/检索数据集
# conda activate retrieve_dataset
# python scripts/process_pdf/process_by_grobid.py

