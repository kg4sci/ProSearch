import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn.functional as F
import faiss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from model_class import (SPECTER2Encoder, NV_Embed2Encoder, SBERTEncoder, BGEReasonerEncoder,
    GTELargeEncoder, Qwen3EmbeddingEncoder, NVEmbedReasoningEncoder, ReasonIREncoder, 
    INFRetrieverEncoder, ReasonEmbedQwen3Encoder, SentenceTransformerEncoder,
    MatSciBertEncoder, 
)



class CustomDenseEncoder:
    def __init__(self, model_name_or_path=None, device="cuda", **kwargs):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.model = None

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        return self._batch_encode_texts(
            texts=queries,
            batch_size=batch_size,
        )

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        texts = []
        for x in corpus:
            text = x["text"]
            texts.append(text)

        return self._batch_encode_texts(
            texts=texts,
            batch_size=batch_size,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Dense/BM25 retrieval for experimental paragraph search.")

    parser.add_argument("--domain", type=str, required=True)

    parser.add_argument("--retrieval_mode", type=str, choices=["dense", "bm25", "both"], required=True)

    parser.add_argument("--model_family", type=str, default="sentence_transformers")
    parser.add_argument("--model_name_or_path", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--queries_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--corpus_text_field", type=str, default="paragraph")

    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--dense_run_json", type=str, default="")
    parser.add_argument("--dense_run_trec", type=str, default="")
    parser.add_argument("--bm25_run_json", type=str, default="")
    parser.add_argument("--bm25_run_trec", type=str, default="")

    parser.add_argument("--corpus_emb_path", type=str, required=True)
    parser.add_argument("--corpus_doc_ids_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=20)

    # 注意：这里建议 dense retrieval 用 dot
    # 如果要 cosine，则在 Faiss 前手动归一化
    parser.add_argument("--score_function", type=str, choices=["dot", "cosine"], default="dot")

    parser.add_argument("--overwrite_corpus_embeddings", action="store_true")
    parser.add_argument("--run_name", type=str, default="retrieval_run")

    return parser.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_trec(path: str, run: Dict[str, Dict[str, float]], run_name="run"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, doc_score_map in run.items():
            ranked = sorted(doc_score_map.items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {doc_id} {rank} {score:.8f} {run_name}\n")


def build_query_list(queries_json: Dict[str, Dict[str, Any]]):
    query_ids = []
    query_texts = []
    query_meta = {}
    for qid, item in queries_json.items():
        query_ids.append(qid)
        query_texts.append(item["query"])
        query_meta[qid] = item
    return query_ids, query_texts, query_meta


def build_corpus_list(paragraphs_json: List[Dict[str, Any]], corpus_text_field="paragraph"):
    corpus = []
    doc_ids = []
    for item in paragraphs_json:
        doc_id = str(item["id"])
        text = item.get(corpus_text_field, "")
        corpus.append({
            "id": doc_id,
            "text": text,
            "doi": item.get("doi", ""),
            "head": item.get("head", ""),
            "source": item.get("source", ""),
        })
        doc_ids.append(doc_id)
    return corpus, doc_ids


def get_dense_encoder(model_family, model_name_or_path, device, score_function):
    normalize_embeddings = (score_function == "dot")

    if model_family == "specter2":
        return SPECTER2Encoder(
            model_name_or_path=model_name_or_path,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )
    elif model_family == "nv_embed":
        return NV_Embed2Encoder(
            model_name_or_path=model_name_or_path,
            device=device,
        )
    elif model_family == "sbert":
        return SBERTEncoder(
            model_name_or_path=model_name_or_path,
            device=device,
            normalize_embeddings=normalize_embeddings,
        )
    elif model_family == "bge_reasoner":
        return BGEReasonerEncoder(
            model_name_or_path=model_name_or_path,
            device=device,
            normalize_embeddings=True,
        )
    elif model_family == "gte_large":
        return GTELargeEncoder(
            model_name_or_path=model_name_or_path,
            device=device,
            normalize_embeddings=True,
        )
    elif model_family in ["qwen3-8b", "diver_retriever", "harrier-oss-v1"]:
        return Qwen3EmbeddingEncoder(
            model_name_or_path=model_name_or_path,
            device=device,
        )
    elif model_family == "llama-nv-embed":
        return NVEmbedReasoningEncoder(
            model_name_or_path=model_name_or_path,
            device=device,
        )
    elif model_family == "reasonIR":
        return ReasonIREncoder(
            model_name_or_path=model_name_or_path,
            device=device
        )
    elif model_family == "inf-retriever-v1":
        return INFRetrieverEncoder(
            model_name_or_path=model_name_or_path,
            device=device
        )
    elif model_family == "reasonEmbed":
        return ReasonEmbedQwen3Encoder(
            model_name_or_path=model_name_or_path,
            device=device
        )
    elif model_family in ['scincl','pubmedbert-base','bioclinical-modernbert','ChEmbed-prog','chemical-bert']:
        return SentenceTransformerEncoder(
            model_name_or_path=model_name_or_path,
            device=device
        )
    elif model_family == 'matscibert':
        return MatSciBertEncoder(
            model_name_or_path=model_name_or_path,
            device=device
        )
    else:
        raise ValueError(f"Unsupported model_family: {model_family}")


def get_or_build_corpus_embeddings(
    model,
    corpus,
    batch_size,
    corpus_emb_path,
    doc_ids_path,
    overwrite_corpus_embeddings,
):
    if (
        (not overwrite_corpus_embeddings)
        and os.path.exists(corpus_emb_path)
        and os.path.exists(doc_ids_path)
    ):
        print(f"[Dense] Loading cached corpus embeddings from: {corpus_emb_path}")
        doc_embs = np.load(corpus_emb_path)
        saved_doc_ids = load_json(doc_ids_path)
        return doc_embs, saved_doc_ids

    print("[Dense] Encoding corpus...")
    doc_embs = model.encode_corpus(corpus, batch_size=batch_size)
    doc_embs = np.asarray(doc_embs, dtype=np.float32)

    Path(corpus_emb_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(corpus_emb_path, doc_embs)
    save_json(doc_ids_path, [x["id"] for x in corpus])

    print(f"[Dense] Saved corpus embeddings to: {corpus_emb_path}")
    return doc_embs, [x["id"] for x in corpus]


def build_faiss_index(doc_embs: np.ndarray, score_function: str):
    """
    Exact Faiss search:
    - dot    -> IndexFlatIP directly
    - cosine -> normalize doc vectors, then IndexFlatIP
    """
    doc_embs = np.asarray(doc_embs, dtype=np.float32)

    if doc_embs.ndim != 2:
        raise ValueError(f"doc_embs must be 2D, got shape={doc_embs.shape}")

    dim = doc_embs.shape[1]

    if score_function == "cosine":
        doc_embs = doc_embs.copy()
        faiss.normalize_L2(doc_embs)
        index = faiss.IndexFlatIP(dim)
    elif score_function == "dot":
        index = faiss.IndexFlatIP(dim)
    else:
        raise ValueError(f"Unsupported score_function: {score_function}")

    index.add(doc_embs)
    return index


def faiss_search_topk(
    query_embs: np.ndarray,
    doc_embs: np.ndarray,
    top_k: int,
    score_function: str,
):
    query_embs = np.asarray(query_embs, dtype=np.float32)

    if query_embs.ndim != 2:
        raise ValueError(f"query_embs must be 2D, got shape={query_embs.shape}")

    index = build_faiss_index(doc_embs, score_function)

    if score_function == "cosine":
        query_embs = query_embs.copy()
        faiss.normalize_L2(query_embs)

    scores, indices = index.search(query_embs, top_k)
    return scores, indices


def build_dense_run_from_faiss(query_ids, scores, indices, doc_ids):
    run = {}
    for i, qid in enumerate(query_ids):
        run[qid] = {}
        for score, idx in zip(scores[i], indices[i]):
            if idx < 0:
                continue
            run[qid][doc_ids[int(idx)]] = float(score)
    return run


def tokenize_for_bm25(text: str):
    return text.lower().split()


def run_bm25_retrieval(query_ids, query_texts, corpus, top_k):
    from rank_bm25 import BM25Okapi

    print("[BM25] Building tokenized corpus...")
    tokenized_corpus = [tokenize_for_bm25(x["text"]) for x in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    doc_ids = [x["id"] for x in corpus]

    run = {}
    for qid, query in zip(query_ids, query_texts):
        query_tokens = tokenize_for_bm25(query)
        scores = bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        run[qid] = {}
        for idx in top_indices:
            run[qid][doc_ids[int(idx)]] = float(scores[idx])

    return run


def main():
    args = parse_args()

    print(f"[Info] domain={args.domain}")
    print(f"[Info] retrieval_mode={args.retrieval_mode}")
    print(f"[Info] model_family={args.model_family}")
    print(f"[Info] model_name_or_path={args.model_name_or_path}")
    print(f"[Info] queries_path={args.queries_path}")
    print(f"[Info] corpus_path={args.corpus_path}")
    print(f"[Info] batch_size={args.batch_size}")
    print(f"[Info] top_k={args.top_k}")
    print(f"[Info] device={args.device}")
    print(f"[Info] score_function={args.score_function}")

    queries_json = load_json(args.queries_path)
    paragraphs_json = load_json(args.corpus_path)

    query_ids, query_texts, _ = build_query_list(queries_json)
    corpus, doc_ids = build_corpus_list(paragraphs_json, corpus_text_field=args.corpus_text_field)

    print(f"[Info] #queries={len(query_ids)}")
    print(f"[Info] #docs={len(doc_ids)}")

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.retrieval_mode in {"dense", "both"}:
        model = get_dense_encoder(
            model_family=args.model_family,
            model_name_or_path=args.model_name_or_path,
            device=args.device,
            score_function=args.score_function,
        )

        doc_embs, saved_doc_ids = get_or_build_corpus_embeddings(
            model=model,
            corpus=corpus,
            batch_size=args.batch_size,
            corpus_emb_path=args.corpus_emb_path,
            doc_ids_path=args.corpus_doc_ids_path,
            overwrite_corpus_embeddings=args.overwrite_corpus_embeddings,
        )

        print("[Dense] Encoding queries...")
        query_embs = model.encode_queries(query_texts, batch_size=args.batch_size)
        query_embs = np.asarray(query_embs, dtype=np.float32)

        print("[Dense] Searching with Faiss IndexFlatIP...")
        scores, indices = faiss_search_topk(
            query_embs=query_embs,
            doc_embs=doc_embs,
            top_k=args.top_k,
            score_function=args.score_function,
        )

        dense_run = build_dense_run_from_faiss(
            query_ids=query_ids,
            scores=scores,
            indices=indices,
            doc_ids=saved_doc_ids,
        )

        if args.dense_run_json:
            save_json(args.dense_run_json, dense_run)
            print(f"[Dense] Saved run json to: {args.dense_run_json}")

        if args.dense_run_trec:
            save_trec(args.dense_run_trec, dense_run, run_name=args.run_name)
            print(f"[Dense] Saved run trec to: {args.dense_run_trec}")

    if args.retrieval_mode in {"bm25", "both"}:
        bm25_run = run_bm25_retrieval(
            query_ids=query_ids,
            query_texts=query_texts,
            corpus=corpus,
            top_k=args.top_k,
        )

        if args.bm25_run_json:
            save_json(args.bm25_run_json, bm25_run)
            print(f"[BM25] Saved run json to: {args.bm25_run_json}")

        if args.bm25_run_trec:
            save_trec(args.bm25_run_trec, bm25_run, run_name="bm25_run")
            print(f"[BM25] Saved run trec to: {args.bm25_run_trec}")


if __name__ == "__main__":
    main()


