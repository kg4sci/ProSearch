import asyncio
import json
import itertools
import httpx
from scripts.structured_extraction.llm_extract_prompt import render_prompt
import time

ENDPOINTS = [
    "http://localhost:8002/v1/chat/completions",
    "http://localhost:8003/v1/chat/completions",
]

MODEL_NAME = "/home/liangwenliang/Model/Qwen3-8B"

def build_request(prompt: str, max_tokens: int = 1024):
    return {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "response_format": { "type": "json_object" },
    }

async def one_call(client: httpx.AsyncClient, url: str, payload: dict, timeout=120):
    r = await client.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

async def run_batch(domain, items, out_path: str, concurrency: int = 64):
    rr = itertools.cycle(ENDPOINTS)

    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    async with httpx.AsyncClient() as client:
        with open(out_path, "w", encoding="utf-8") as f:
            async def worker(i, item):
                async with sem:
                    paragraph = item["paragraph"] 

                    prompt = render_prompt(domain, paragraph)  
                    payload = build_request(prompt)
    
                    url = next(rr)
                    try:
                        content = await one_call(client, url, payload)
                        obj = json.loads(content)
                        async with write_lock:
                            f.write(json.dumps({
                                "domain": domain,
                                "id": item.get("id"),
                                "result": obj
                            }, ensure_ascii=False) + "\n")
                    except Exception as e:
                        async with write_lock:
                            f.write(json.dumps({
                                "domain": domain,
                                "id": item.get("id"),
                                "error": str(e),
                                "raw": content if "content" in locals() else None
                            }, ensure_ascii=False) + "\n")

            await asyncio.gather(*(worker(i, it) for i, it in enumerate(items)))

if __name__ == "__main__":
    domains = [
        'electrochemistry',
        'cell_culture', 
        'immunotechnology',
        'inorganic_materials',
        'organic_synthesis',
        'polymer_materials'
    ]
    for domain in domains:
        start = time.time()
        in_path = f"seed_data/{domain}/paragraphs.json"
        out_path = f"seed_data/{domain}/open_llm_extraction/raw_evidence_span.jsonl"

        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            print(len(data))
            # data = data[:24]
            
        asyncio.run(run_batch(domain, data, out_path, concurrency=32))
        print("done:", out_path)
        print('domain:', domain, 'time:', time.time() - start)

'''
conda activate vllm_env
CUDA_VISIBLE_DEVICES=4 vllm serve /home/liangwenliang/Model/Qwen3-8B \
  --host 0.0.0.0 --port 8002 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80


CUDA_VISIBLE_DEVICES=5 vllm serve /home/liangwenliang/Model/Qwen3-8B \
  --host 0.0.0.0 --port 8004 \
  --dtype bfloat16 \
  --max-model-len 3072 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.80 

conda activate retrieve_dataset
cd 检索数据集
python -m scripts.structured_extraction.open_llm_extract
'''
