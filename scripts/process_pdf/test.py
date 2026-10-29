import asyncio
import aiohttp
import aiofiles
import os
import io
import random
import pandas as pd
from pypdf import PdfReader

# ================= 配置区 =================
CSV_FILE_PATH = "pdf/csv_from_openalex/organic_synthesis.csv"
SAVE_DIR = "pdf/organic_synthesis/"
FAILED_LOG = "pdf/failed_dois.txt"

# 必须提供的 Email (Unpaywall/CORE 建议)
USER_EMAIL = "3407182028@qq.com" 
# CORE API KEY (可选，去 core.ac.uk 免费申请可大幅提升限速，没有也能跑)
CORE_API_KEY = "" 

CONCURRENCY_LIMIT = 5 

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
# ==========================================

def is_valid_pdf(pdf_content):
    """验证 PDF 是否完整"""
    try:
        if len(pdf_content) < 10000: # 有机合成 PDF 通常大于 10KB
            return False
        pdf_file = io.BytesIO(pdf_content)
        reader = PdfReader(pdf_file, strict=False)
        return len(reader.pages) > 0
    except:
        return False

def get_safe_path(doi):
    """统一文件名生成逻辑，方便查重"""
    clean_doi = doi.replace("https://doi.org/", "").strip()
    safe_name = clean_doi.replace('/', '_').replace(':', '_').replace('\\', '_') + ".pdf"
    return os.path.join(SAVE_DIR, safe_name), clean_doi

async def get_core_urls(doi, session):
    """从 CORE API 获取 OA 链接"""
    clean_doi = doi.replace("https://doi.org/", "").strip()
    url = f"https://api.core.ac.uk/v3/works/doi:{clean_doi}"
    params = {"apiKey": CORE_API_KEY} if CORE_API_KEY else {}
    try:
        async with session.get(url, params=params, timeout=10) as resp:
            if resp.status == 200:
                data = await resp.json()
                # CORE 返回的 downloadUrl 通常是直链
                if data.get('downloadUrl'):
                    return [data['downloadUrl']]
    except:
        pass
    return []

async def get_unpaywall_urls(doi, session):
    """从 Unpaywall 获取 OA 链接"""
    clean_doi = doi.replace("https://doi.org/", "").strip()
    api_url = f"https://api.unpaywall.org/v2/{clean_doi}?email={USER_EMAIL}"
    try:
        async with session.get(api_url, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                return [loc['url_for_pdf'] for loc in data.get('oa_locations', []) if loc.get('url_for_pdf')]
    except:
        pass
    return []

async def download_pdf(url, session):
    """下载并验证 PDF"""
    try:
        async with session.get(url, headers=HEADERS, timeout=45, allow_redirects=True) as response:
            if response.status == 200 and 'html' not in response.headers.get('Content-Type', '').lower():
                content = await response.read()
                if is_valid_pdf(content):
                    return content
    except:
        pass
    return None

async def get_scihub_url(doi, session):
    """保底方案：从 Sci-Hub 尝试获取"""
    # 常用镜像站列表
    mirrors = ["https://www.sci-hub.se", "https://sci-hub.st", "https://sci-hub.ru"]
    clean_doi = doi.replace("https://doi.org/", "").strip()
    
    for base_url in mirrors:
        try:
            target = f"{base_url}/{clean_doi}"
            async with session.get(target, timeout=15) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    # 寻找 PDF 嵌入链接或下载按钮链接
                    if 'id="pdf"' in html or 'location.href=' in html:
                        # 简单的正则或字符串查找（Sci-Hub 结构较固定）
                        if "src=\"" in html:
                            pdf_url = html.split('src="')[1].split('"')[0]
                            if pdf_url.startswith('//'):
                                pdf_url = "https:" + pdf_url
                            return [pdf_url]
        except:
            continue
    return []

async def process_doi(doi, session, semaphore):
    """处理单个 DOI：多源查找 -> 下载 -> 验证 -> 保存"""
    async with semaphore:
        filepath, clean_doi = get_safe_path(doi)
        
        # 1. 汇总所有可能的下载源
        sources = []
        sources.extend(await get_unpaywall_urls(clean_doi, session))
        sources.extend(await get_core_urls(clean_doi, session))
        sources.extend(await get_scihub_url(clean_doi, session))
        
        # 去重并保持顺序
        pdf_urls = list(dict.fromkeys(sources))

        if not pdf_urls:
            return "NO_LINK", clean_doi

        # 2. 迭代尝试下载
        for url in pdf_urls:
            await asyncio.sleep(random.uniform(0.5, 1.5)) # 礼貌延迟
            pdf_content = await download_pdf(url, session)
            if pdf_content:
                async with aiofiles.open(filepath, mode='wb') as f:
                    await f.write(pdf_content)
                return "SUCCESS", None
        
        return "DOWNLOAD_FAIL", clean_doi

async def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 读取数据
    try:
        df = pd.read_csv(CSV_FILE_PATH)
        dois = df['doi'].dropna().unique().tolist()
    except Exception as e:
        print(f"❌ 读取 CSV 出错: {e}")
        return

    # --- 核心改进：预过滤已存在的文件 ---
    to_process = []
    already_count = 0
    for doi in dois:
        path, _ = get_safe_path(doi)
        if os.path.exists(path):
            already_count += 1
        else:
            to_process.append(doi)
    
    total = len(to_process)
    print(f"📊 任务统计：")
    print(f"   - 总 DOI 数: {len(dois)}")
    print(f"   - 已下载跳过: {already_count}")
    print(f"   - 本次需下载: {total}")
    
    if total == 0:
        print("✅ 所有文献已就绪。")
        return

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    async with aiohttp.ClientSession() as session:
        tasks = [process_doi(doi, session, semaphore) for doi in to_process]
        
        success_count = 0
        fail_list = []
        done = 0

        for coro in asyncio.as_completed(tasks):
            result, error_doi = await coro
            done += 1
            if result == "SUCCESS":
                success_count += 1
            else:
                fail_list.append(f"{error_doi} | {result}")

            if done % 5 == 0 or done == total:
                print(f"⏳ 进度: {done}/{total} | 成功: {success_count} | 失败: {len(fail_list)}")

    # 记录失败者以便后续针对性重试
    if fail_list:
        with open(FAILED_LOG, 'w') as f:
            f.write("\n".join(fail_list))
        print(f"📝 失败记录已存至: {FAILED_LOG}")

    print(f"\n🎉 运行结束！本次新成功下载: {success_count}")

if __name__ == "__main__":
    asyncio.run(main())

# python scripts/process_pdf/test.py