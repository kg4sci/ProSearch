import asyncio
import aiohttp
import aiofiles
import os
import io
import random
import pandas as pd
from pypdf import PdfReader

# ================= 配置区 =================
# 1. 输入和输出路径
CSV_FILE_PATH = "pdf/csv_from_openalex/inorganic_materials.csv"
SAVE_DIR = "pdf/inorganic_materials/"
FAILED_LOG = "pdf/unpaywall_failed_dois.txt"

# 2. 用户信息（Unpaywall 要求提供 Email）
USER_EMAIL = "wlliang@cnic.cn" 

# 3. 爬虫伪装请求头
# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
#     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
#     "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
# }

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://doi.org/",
}

# 4. 并发控制（建议 3-5，有机合成类文献出版社反爬较严，不宜过快）
CONCURRENCY_LIMIT = 5 
# ==========================================

def is_valid_pdf(pdf_content):
    """验证 PDF 是否完整且有效"""
    try:
        if len(pdf_content) < 1000: # 文件太小肯定不是 PDF
            return False
        pdf_file = io.BytesIO(pdf_content)
        reader = PdfReader(pdf_file, strict=False)
        return len(reader.pages) > 0
    except:
        return False

async def get_urls_from_unpaywall(doi, session):
    """从 Unpaywall 获取所有可能的 OA 链接"""
    clean_doi = doi.replace("https://doi.org/", "").strip()
    api_url = f"https://api.unpaywall.org/v2/{clean_doi}?email={USER_EMAIL}"
    
    try:
        async with session.get(api_url, timeout=15) as response:
            if response.status == 200:
                data = await response.json()
                urls = []
                if data.get('is_oa') and data.get('oa_locations'):
                    for loc in data['oa_locations']:
                        url = loc.get('url_for_pdf')
                        if url:
                            urls.append(url)
                return urls
    except Exception as e:
        pass
    return []

async def get_url_from_semantic_scholar(doi, session):
    """备选方案：从 Semantic Scholar 获取链接"""
    clean_doi = doi.replace("https://doi.org/", "").strip()
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{clean_doi}?fields=isOpenAccess,openAccessPdf"
    
    try:
        async with session.get(api_url, timeout=15) as response:
            if response.status == 200:
                data = await response.json()
                if data.get('isOpenAccess') and data.get('openAccessPdf'):
                    return data['openAccessPdf'].get('url')
    except:
        pass
    return None

async def download_pdf(doi, url, session):
    """尝试下载单个链接"""
    try:
        async with session.get(url, headers=HEADERS, timeout=60, allow_redirects=True) as response:
            if response.status == 200:
                # 检查内容类型，如果返回的是 text/html 说明被拦截或重定向到了预览页
                content_type = response.headers.get('Content-Type', '').lower()
                if 'html' in content_type:
                    return None
                
                content = await response.read()
                if is_valid_pdf(content):
                    return content
    except Exception:
        pass
    return None

async def process_doi(doi, session, semaphore):
    """处理单个 DOI 的完整逻辑"""
    async with semaphore:
        clean_doi = doi.replace("https://doi.org/", "").strip()
        safe_filename = clean_doi.replace('/', '_').replace(':', '_') + ".pdf"
        filepath = os.path.join(SAVE_DIR, safe_filename)
        
        # 1. 查重
        if os.path.exists(filepath):
            return "ALREADY_EXISTS", None

        # 2. 收集所有可能的 URL
        # 先找 Unpaywall
        pdf_urls = await get_urls_from_unpaywall(clean_doi, session)
        # 补充 Semantic Scholar
        ss_url = await get_url_from_semantic_scholar(clean_doi, session)
        if ss_url and ss_url not in pdf_urls:
            pdf_urls.append(ss_url)

        if not pdf_urls:
            return "NO_LINK", clean_doi

        # 3. 逐个尝试下载
        for url in pdf_urls:
            # 随机延迟，防止频率过快
            await asyncio.sleep(random.uniform(1.0, 2.5))
            
            pdf_content = await download_pdf(clean_doi, url, session)
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
        if 'doi' not in df.columns:
            print("错误：CSV 文件中未找到 'doi' 列")
            return
        dois = df['doi'].dropna().unique().tolist()
    except Exception as e:
        print(f"读取 CSV 出错: {e}")
        return

    total = len(dois)
    print(f"🚀 任务开始：共有 {total} 个 DOI 需要处理")

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_doi(doi, session, semaphore) for doi in dois]
        
        done_count = 0
        success_count = 0
        exists_count = 0
        fail_list = []

        for coro in asyncio.as_completed(tasks):
            result, error_doi = await coro
            done_count += 1
            
            if result == "SUCCESS":
                success_count += 1
            elif result == "ALREADY_EXISTS":
                exists_count += 1
            else:
                fail_list.append(f"{error_doi} | Reason: {result}")

            if done_count % 10 == 0 or done_count == total:
                print(f"进度: {done_count}/{total} | 成功: {success_count} | 跳过: {exists_count} | 失败: {len(fail_list)}")

    # 写入失败日志
    if fail_list:
        async with aiofiles.open(FAILED_LOG, mode='w') as f:
            await f.write("\n".join(fail_list))
        print(f"❌ 失败详情已记录至: {FAILED_LOG}")

    print(f"\n✅ 全部任务完成！")
    print(f"最终成功下载: {success_count}")
    print(f"总计无法获取: {len(fail_list)}")

if __name__ == "__main__":
    asyncio.run(main())

# cd /home/liangwenliang/检索数据集
# conda activate retrieve_dataset
# python scripts/process_pdf/download_pdf.py
