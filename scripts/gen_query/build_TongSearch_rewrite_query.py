import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "model/TongSearch-QR-7B"

DOMAINS = [
    "electrochemistry",
    # "cell_culture",
    # "immunotechnology",
    # "inorganic_materials",
    # "organic_synthesis",
    # "polymer_materials",
]

INPUT_FILE_NAME = "queries_raw.json"
OUTPUT_FILE_NAME = "queries_tongsearch.json"

BATCH_SIZE = 8
MAX_INPUT_LENGTH = 1024
MAX_NEW_TOKENS = 2048

prompt =  (
    "Instructions:\n"
    "Rewrite the following user query into a search query for retrieval.\n"
    "Preserve all important constraints faithfully, including entities, conditions, negation, numbers, and ranges.\n"
    "Do not add new constraints. Do not remove important constraints.\n"
    "Return only the rewritten query, without explanation.\n\n"
    "Query: {query}\n\n"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)



def build_input_text(tokenizer, query: str) -> str:
    messages = [{"role": "user", "content": prompt.replace("{query}", query)}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


@torch.no_grad()
def rewrite_queries_batch(model, tokenizer, query_texts):
    input_texts = [build_input_text(tokenizer, q) for q in query_texts]

    model_inputs = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    ).to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_len = model_inputs.input_ids.shape[1]
    generated_only = generated_ids[:, prompt_len:]

    outputs = tokenizer.batch_decode(
        generated_only,
        skip_special_tokens=True
    )

    outputs = [x.strip() for x in outputs]
    return outputs


def process_one_file(model, tokenizer, input_path, output_path):
    data = load_json(input_path)

    qids = list(data.keys())
    queries = [data[qid]["query"] for qid in qids]

    rewritten_queries = []

    for start in tqdm(range(0, len(queries), BATCH_SIZE), desc=f"Processing {input_path}"):
        batch_queries = queries[start:start + BATCH_SIZE]
        batch_outputs = rewrite_queries_batch(
            model,
            tokenizer,
            batch_queries,
        )
        rewritten_queries.extend(batch_outputs)

    assert len(rewritten_queries) == len(qids)

    output = {}
    for qid, rewritten in zip(qids, rewritten_queries):
        item = dict(data[qid])
        item["query"] = rewritten
        output[qid] = item

    save_json(output, output_path)
    print(f"完成: {output_path} | 共 {len(output)} 条")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map="auto"
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for domain in DOMAINS:
        input_path = f"data/{domain}/gen_query/{INPUT_FILE_NAME}"
        output_path = f"data/{domain}/gen_query/{OUTPUT_FILE_NAME}"

        process_one_file(
            model,
            tokenizer,
            input_path,
            output_path,
        )


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1,2 python scripts/gen_query/build_TongSearch_rewrite_query.py
