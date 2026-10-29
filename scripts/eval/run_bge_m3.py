import json
import os
import pickle
from pathlib import Path
from collections import defaultdict

import faiss
import numpy as np
import scipy.sparse as sp
import torch
from FlagEmbedding import BGEM3FlagModel  

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_pickle(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_trec(path, run, run_name="run"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, doc_score_map in run.items():
            ranked = sorted(doc_score_map.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.8f} {run_name}\n")


class BGEM3Encoder:
    def __init__(self, model_name_or_path="model/bge-m3", device="cuda", use_fp16=True, max_length=2048):
        self.model = BGEM3FlagModel(
            model_name_or_path,
            use_fp16=use_fp16,
            device=device,
        )
        self.max_length = max_length

    def encode_hybrid(self, texts, batch_size):
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        
        # 处理 Dense
        dense_vecs = np.asarray(output["dense_vecs"], dtype=np.float32)
        
        # 处理 Sparse
        lexical_weights = output["lexical_weights"]
        sparse_list = []
        for item in lexical_weights:
            sparse_list.append({int(tid): float(w) for tid, w in item.items()})
            
        return dense_vecs, sparse_list

def get_or_build_hybrid_corpus(
    model, corpus, batch_size, 
    dense_path, sparse_weights_path, sparse_matrix_path, token_to_col_path, doc_ids_path, 
    overwrite=False
):

    all_exists = all(os.path.exists(p) for p in [dense_path, sparse_weights_path, sparse_matrix_path, token_to_col_path, doc_ids_path])
    
    if (not overwrite) and all_exists:
        print("[Hybrid] loading all caches...")
        dense = np.load(dense_path)
        sparse_list = load_pickle(sparse_weights_path)
        sparse_matrix = sp.load_npz(sparse_matrix_path)
        token_to_col = load_pickle(token_to_col_path)
        doc_ids = load_json(doc_ids_path)
        return dense, sparse_list, sparse_matrix, token_to_col, doc_ids

    print("[Hybrid] encoding corpus (Dense + Sparse)...")
    dense_chunks = []
    sparse_chunks = []
    
    # 按照 Batch 循环
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i : i + batch_size]
        texts = [x["text"] for x in batch]
        
        # 一次编码获取两者
        b_dense, b_sparse = model.encode_hybrid(texts, batch_size=len(texts))
        dense_chunks.append(b_dense)
        sparse_chunks.extend(b_sparse)

    # 合并结果
    dense = np.concatenate(dense_chunks, axis=0).astype(np.float32)
    sparse_matrix, token_to_col = build_sparse_csr(sparse_chunks)
    doc_ids = [x["id"] for x in corpus]

    # 保存缓存
    np.save(dense_path, dense)
    save_pickle(sparse_weights_path, sparse_chunks)
    sp.save_npz(sparse_matrix_path, sparse_matrix)
    save_pickle(token_to_col_path, token_to_col)
    save_json(doc_ids_path, doc_ids)

    return dense, sparse_chunks, sparse_matrix, token_to_col, doc_ids


def build_sparse_csr(sparse_list):
    token_to_col = {}
    rows, cols, data = [], [], []
    for row_idx, weights in enumerate(sparse_list):
        for token_id, weight in weights.items():
            if token_id not in token_to_col:
                token_to_col[token_id] = len(token_to_col)
            col_idx = token_to_col[token_id]
            rows.append(row_idx)
            cols.append(col_idx)
            data.append(float(weight))
    matrix = sp.csr_matrix((np.array(data), (rows, cols)), shape=(len(sparse_list), len(token_to_col)), dtype=np.float32)
    return matrix, token_to_col

def dense_search_topk(query_dense, doc_dense, top_k):
    index = faiss.IndexFlatIP(doc_dense.shape[1])
    index.add(doc_dense)
    return index.search(query_dense, top_k)

def sparse_search_topk(query_sparse_list, doc_sparse_csr, token_to_col, top_k):
    num_docs, vocab_size = doc_sparse_csr.shape
    all_scores, all_indices = [], []
    for q_dict in query_sparse_list:
        # 转换为 CSR
        cols, data = [], []
        for tid, w in q_dict.items():
            if tid in token_to_col:
                cols.append(token_to_col[tid])
                data.append(float(w))
        if not cols:
            all_scores.append(np.array([], dtype=np.float32))
            all_indices.append(np.array([], dtype=np.int64))
            continue
        q_csr = sp.csr_matrix((data, (np.zeros(len(cols)), cols)), shape=(1, vocab_size))
        score_vec = (doc_sparse_csr @ q_csr.T).toarray().ravel()
        
        k = min(top_k, num_docs)
        if k == 0:
            all_scores.append(np.array([], dtype=np.float32))
            all_indices.append(np.array([], dtype=np.int64))
            continue
        top_idx = np.argpartition(-score_vec, kth=k - 1)[:k]
        top_scores = score_vec[top_idx]

        order = np.argsort(-top_scores)
        top_scores = top_scores[order]
        top_idx = top_idx[order]

        positive_mask = top_scores > 0
        top_scores = top_scores[positive_mask]
        top_idx = top_idx[positive_mask]

        all_scores.append(top_scores.astype(np.float32))
        all_indices.append(top_idx.astype(np.int64))
    return all_scores, all_indices

def rrf_fuse(dense_run, sparse_run, query_ids, top_k, rrf_k=60):
    final_run = {}
    for qid in query_ids:
        fused_scores = defaultdict(float)
        d_ranked = sorted(dense_run.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
        s_ranked = sorted(sparse_run.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
        for rank, (doc_id, _) in enumerate(d_ranked, start=1):
            fused_scores[doc_id] += 1.0 / (rrf_k + rank)
        for rank, (doc_id, _) in enumerate(s_ranked, start=1):
            fused_scores[doc_id] += 1.0 / (rrf_k + rank)
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        final_run[qid] = {doc_id: float(score) for doc_id, score in ranked}
    return final_run



def run_one_domain(model, domain, queries_path, corpus_path, corpus_text_field, 
                   dense_cache_path, sparse_weights_path, sparse_matrix_path, token_to_col_path, doc_ids_path,
                   dense_pool_json, sparse_pool_json, hybrid_run_json, hybrid_run_trec,
                   batch_size, top_k, candidate_pool_k, rrf_k, overwrite=False):
    
    print(f"\nProcessing domain: {domain}")
    queries_json = load_json(queries_path)
    paragraphs_json = load_json(corpus_path)
    query_ids, query_texts = [], []
    for qid, item in queries_json.items():
        query_ids.append(qid); query_texts.append(item["query"])
    corpus = [{"id": str(item["id"]), "text": item.get(corpus_text_field, "")} for item in paragraphs_json]

    # 1. 构建语料库 (一次处理两路)
    doc_dense, _, doc_sparse_csr, token_to_col, doc_ids = get_or_build_hybrid_corpus(
        model, corpus, batch_size, dense_cache_path, sparse_weights_path, 
        sparse_matrix_path, token_to_col_path, doc_ids_path, overwrite=overwrite
    )

    # 2. 查询编码 (一次编码两路)
    print("[Query] hybrid encoding...")
    query_dense, query_sparse = model.encode_hybrid(query_texts, batch_size=batch_size)

    # 3. 检索
    pool_k = max(top_k, candidate_pool_k)
    print("[Dense] searching...")
    d_scores, d_indices = dense_search_topk(query_dense, doc_dense, pool_k)
    print("[Sparse] searching...")
    s_scores, s_indices = sparse_search_topk(query_sparse, doc_sparse_csr, token_to_col, pool_k)

    # 4. 构建 Run 对象并保存中间结果
    dense_run = {qid: {doc_ids[int(idx)]: float(s) for s, idx in zip(d_scores[i], d_indices[i]) if idx >=0} for i, qid in enumerate(query_ids)}
    sparse_run = {qid: {doc_ids[int(idx)]: float(s) for s, idx in zip(s_scores[i], s_indices[i])} for i, qid in enumerate(query_ids)}
    save_json(dense_pool_json, dense_run)
    save_json(sparse_pool_json, sparse_run)

    # 5. RRF 融合
    print("[Hybrid] RRF fusing...")
    final_run = rrf_fuse(dense_run, sparse_run, query_ids, top_k, rrf_k)
    save_json(hybrid_run_json, final_run)
    save_trec(hybrid_run_trec, final_run)

if __name__ == "__main__":
    DOMAINS = [
        "electrochemistry",
        "cell_culture",
        "immunotechnology",
        "inorganic_materials",
        "organic_synthesis",
        "polymer_materials",
    ]

    TYPE = "raw"

    MODEL_NAME_OR_PATH = "model/bge-m3"
    MODEL_TAG = "bge_m3"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_FP16 = True
    MAX_LENGTH = 2048

    CORPUS_TEXT_FIELD = "paragraph"
    TOP_K = 100
    CANDIDATE_POOL_K = 1000
    BATCH_SIZE = 16

    RRF_K = 60

    OVERWRITE_DENSE = False
    OVERWRITE_SPARSE = False

    RUN_NAME = "bge_m3"

    print("[Config]")
    print(f"DOMAINS={DOMAINS}")
    print(f"TYPE={TYPE}")
    print(f"MODEL_NAME_OR_PATH={MODEL_NAME_OR_PATH}")
    print(f"DEVICE={DEVICE}")
    print(f"TOP_K={TOP_K}")
    print(f"CANDIDATE_POOL_K={CANDIDATE_POOL_K}")
    print(f"BATCH_SIZE={BATCH_SIZE}")
    print(f"RRF_K={RRF_K}")
    print()

    model = BGEM3Encoder(
        model_name_or_path=MODEL_NAME_OR_PATH,
        device=DEVICE,
        use_fp16=USE_FP16,
        max_length=MAX_LENGTH,
    )

    for domain in DOMAINS:
        queries_path = f"data/{domain}/gen_query/queries_{TYPE}.json"
        corpus_path = f"seed_data/{domain}/paragraphs.json"

        cache_dir = f"data/{domain}/eval/cache/{MODEL_TAG}"
        output_dir = f"data/{domain}/eval/{MODEL_TAG}"

        dense_cache_path = f"{cache_dir}/paragraph_dense.npy"

        sparse_weights_path = f"{cache_dir}/paragraph_sparse_weights.pkl"
        sparse_matrix_path = f"{cache_dir}/paragraph_sparse_matrix.npz"
        token_to_col_path = f"{cache_dir}/token_to_col.pkl"

        doc_ids_path = f"{cache_dir}/paragraph_doc_ids.json"

        dense_pool_json = f"{output_dir}/dense_top{CANDIDATE_POOL_K}_{TYPE}.json"
        sparse_pool_json = f"{output_dir}/sparse_top{CANDIDATE_POOL_K}_{TYPE}.json"

        hybrid_run_json = f"{output_dir}/hybrid_top{TOP_K}_{TYPE}.json"
        hybrid_run_trec = f"{output_dir}/hybrid_top{TOP_K}_{TYPE}.trec"

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        run_one_domain(
            model=model,
            domain=domain,
            queries_path=queries_path,
            corpus_path=corpus_path,
            corpus_text_field=CORPUS_TEXT_FIELD,
            dense_cache_path=dense_cache_path,
            sparse_weights_path=sparse_weights_path,
            sparse_matrix_path=sparse_matrix_path,
            token_to_col_path=token_to_col_path,
            doc_ids_path=doc_ids_path,
            dense_pool_json=dense_pool_json,
            sparse_pool_json=sparse_pool_json,
            hybrid_run_json=hybrid_run_json,
            hybrid_run_trec=hybrid_run_trec,
            batch_size=BATCH_SIZE,
            top_k=TOP_K,
            candidate_pool_k=CANDIDATE_POOL_K,
            rrf_k=RRF_K
        )

    print("All tasks completed!")


# conda activate retrieve_dataset
# CUDA_VISIBLE_DEVICES=5,6 python scripts/eval/run_bge_m3.py

