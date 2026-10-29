import os
from pathlib import Path
from typing import Dict, List, Any,  Optional,Union, Tuple
from torch import Tensor
import numpy as np
import torch
import torch.nn.functional as F
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from transformers import AutoTokenizer, AutoModel
from adapters import AutoAdapterModel
from sentence_transformers import SentenceTransformer

class SPECTER2Encoder:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        normalize_embeddings: bool = True,
        query_adapter_name: str = "model/specter2_adhoc_query",
        corpus_adapter_name: str = "model/specter2_adapter",
        max_length: int = 512,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.query_adapter_name = query_adapter_name
        self.corpus_adapter_name = corpus_adapter_name
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoAdapterModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        self.loaded_adapters = {}

        if self.query_adapter_name:
            self.model.load_adapter(
                self.query_adapter_name,
                source="hf",
                load_as="query_adapter",
                set_active=False,
            )
            self.loaded_adapters["query"] = "query_adapter"

        if self.corpus_adapter_name:
            self.model.load_adapter(
                self.corpus_adapter_name,
                source="hf",
                load_as="corpus_adapter",
                set_active=False,
            )
            self.loaded_adapters["corpus"] = "corpus_adapter"

        self.model.to(self.device)
        self.model.eval()

    def _set_active_adapter(self, mode: str):
        if mode == "query":
            if "query" in self.loaded_adapters:
                self.model.set_active_adapters("query_adapter")
            else:
                self.model.set_active_adapters(None)
        elif mode == "corpus":
            if "corpus" in self.loaded_adapters:
                self.model.set_active_adapters("corpus_adapter")
            else:
                self.model.set_active_adapters(None)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _batch_encode_texts(self, texts: List[str], batch_size: int, mode: str) -> np.ndarray:
        self._set_active_adapter(mode)
        all_embeddings = []

        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start : start + batch_size]

                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)
                
                embeddings = outputs.last_hidden_state[:, 0, :]

                if self.normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

                all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        return self._batch_encode_texts(
            texts=queries,
            batch_size=batch_size,
            mode="query",
        )

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        texts = []
        for x in corpus:
            text = x["text"]
            texts.append(text)

        return self._batch_encode_texts(
            texts=texts,
            batch_size=batch_size,
            mode="corpus",
        )

class NV_Embed2Encoder:
    def __init__(
        self,
        model_name_or_path,
        device: str = "cuda",
        query_instruction: str = "Given a web search query, retrieve paragraphs that answer the question",
        max_length: int = 2048,  # 32768
        normalize: bool = True,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_length = max_length
        self.normalize = normalize
        self.default_batch_size = batch_size
        self.num_workers = num_workers

        self.query_prefix = f"Instruct: {query_instruction}\nQuery: "
        self.passage_prefix = ""

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16, 
        )

        try:
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            pass

    def _encode_texts(
        self,
        texts: List[str],
        instruction: str,
        batch_size: int,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            with torch.no_grad():
                # 用官方支持 batch 的接口
                batch_emb = self.model._do_encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    instruction=instruction,
                    max_length=self.max_length,
                    num_workers=0,
                    return_numpy=True,
                )

            batch_emb = np.asarray(batch_emb, dtype=np.float32)

            if self.normalize:
                norms = np.linalg.norm(batch_emb, axis=1, keepdims=True)
                norms = np.clip(norms, 1e-12, None)
                batch_emb = batch_emb / norms

            all_embeddings.append(batch_emb)

            # 对超大 corpus 更稳一些
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(all_embeddings, axis=0)

    def encode_queries(
        self,
        queries: List[str],
        batch_size: int = 16,
        **kwargs,
    ) -> np.ndarray:
        actual_batch_size = batch_size or self.default_batch_size or 32
        return self._encode_texts(
            texts=queries,
            instruction=self.query_prefix,
            batch_size=actual_batch_size,
        )

    def encode_corpus(
        self,
        corpus: List[Dict[str, str]],
        batch_size: int = 16,
        **kwargs,
    ) -> np.ndarray:
        actual_batch_size = batch_size or self.default_batch_size or 32

        texts = []
        for doc in corpus:
            text = doc.get("text", "")
            texts.append(text)

        return self._encode_texts(
            texts=texts,
            instruction=self.passage_prefix,
            batch_size=actual_batch_size,
        )

class SBERTEncoder:
    def __init__(
        self,
        model_name_or_path: str = "model/all-mpnet-base-v2",
        device: str = "cuda",
        max_length: int = 512,
        normalize_embeddings: bool = True,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        self.model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / torch.clamp(
            input_mask_expanded.sum(dim=1), min=1e-9
        )

    def _batch_encode_texts(
        self,
        texts: List[str],
        batch_size: int,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            encoded_input = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])

            if self.normalize_embeddings:
                sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            all_embeddings.append(sentence_embeddings.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(all_embeddings, axis=0)

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

class BGEReasonerEncoder:
    def __init__(
        self,
        model_name_or_path: str = "model/bge-reasoner",
        device: str = "cuda",
        query_instruction: str = "Given a web search query, retrieve paragraphs that answer the question",
        max_length: int = 1024,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.query_instruction = query_instruction
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        self.model = AutoModel.from_pretrained(model_name_or_path, **kwargs)
        self.model.to(device)
        self.model.eval()

        if self.use_fp16 and "cuda" in device:
            self.model.half()

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"

    def _tokenize_texts(self, texts: List[str]):
        batch_dict = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        return batch_dict

    def _batch_encode_texts(
        self,
        texts: List[str],
        batch_size: int,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            batch_dict = self._tokenize_texts(batch_texts)

            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = self.last_token_pool(
                    outputs.last_hidden_state,
                    batch_dict["attention_mask"],
                )

                if self.normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(
                embeddings.detach().cpu().float().numpy().astype(np.float32)
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(all_embeddings, axis=0)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        instructed_queries = [
            self.get_detailed_instruct(self.query_instruction, q) for q in queries
        ]
        return self._batch_encode_texts(
            texts=instructed_queries,
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

class GTELargeEncoder:
    def __init__(
        self,
        model_name_or_path: str = "model/gte-large-en-v1.5",
        device: str = "cuda",
        max_length: int = 1024,
        normalize_embeddings: bool = True,
        use_fp16: bool = False,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            **kwargs,
        )
        self.model.to(device)
        self.model.eval()

        if self.use_fp16 and "cuda" in device:
            self.model.half()

    def _batch_encode_texts(
        self,
        texts: List[str],
        batch_size: int,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            batch_dict = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}

            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = outputs.last_hidden_state[:, 0]

                if self.normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(
                embeddings.detach().cpu().float().numpy().astype(np.float32)
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(all_embeddings, axis=0)

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

class Qwen3EmbeddingEncoder:
    def __init__(
        self,
        model_name_or_path: str = "model/Qwen3-Embedding-8B",
        device: str = "cuda",
        query_instruction: str = "Given a question, retrieve paragraphs that answer the question",
        max_length: int = 2048,
        normalize_embeddings: bool = True,
        use_fp16: bool = False,
        use_bf16: bool = True,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.query_instruction = query_instruction
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            **kwargs,
        )

        model_kwargs = dict(kwargs)
        if self.use_bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif self.use_fp16:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

        self.model.to(device)
        self.model.eval()

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery:{query}"

    def _tokenize_texts(self, texts: List[str]):
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        return batch_dict

    def _batch_encode_texts(
        self,
        texts: List[str],
        batch_size: int,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            batch_dict = self._tokenize_texts(batch_texts)

            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = self.last_token_pool(
                    outputs.last_hidden_state,
                    batch_dict["attention_mask"],
                )

                if self.normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(
                embeddings.detach().cpu().float().numpy().astype(np.float32)
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(all_embeddings, axis=0)

    def encode_queries(
        self,
        queries: List[str],
        batch_size: int = 16,
        **kwargs,
    ) -> np.ndarray:
        instructed_queries = [
            self.get_detailed_instruct(self.query_instruction, q)
            for q in queries
        ]
        return self._batch_encode_texts(
            texts=instructed_queries,
            batch_size=batch_size,
        )

    def encode_corpus(
        self,
        corpus: List[Dict[str, str]],
        batch_size: int = 16,
        **kwargs,
    ) -> np.ndarray:
        texts = []
        for x in corpus:
            text = x["text"]
            texts.append(text)

        return self._batch_encode_texts(
            texts=texts,
            batch_size=batch_size,
        )

class NVEmbedReasoningEncoder:
    def __init__(
        self,
        model_name_or_path: str = "model/llama-nv-embed-reasoning-3b",
        device: str = "cuda",
        query_prefix: str = "query:",
        document_prefix: str = "passage:",
        max_length: int = 2048,
        normalize_embeddings: bool = True,
        trust_remote_code: bool = True,
        use_fp16: bool = False,
        use_bf16: bool = True,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.trust_remote_code = trust_remote_code
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        model_kwargs = dict(kwargs)
        model_kwargs["trust_remote_code"] = trust_remote_code

        if self.use_bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif self.use_fp16:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

        self.model.to(device)
        self.model.eval()

    @staticmethod
    def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden_states_masked = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(),
            0.0,
        )
        embeddings = last_hidden_states_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        return embeddings

    def _tokenize_texts(self, texts: List[str]):
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        return batch_dict

    def _batch_encode_texts(
        self,
        texts: List[str],
        batch_size: int,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            batch_dict = self._tokenize_texts(batch_texts)

            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = self.average_pool(
                    outputs.last_hidden_state,
                    batch_dict["attention_mask"],
                )

                if self.normalize_embeddings:
                    embeddings = F.normalize(embeddings, dim=-1)

            all_embeddings.append(
                embeddings.detach().cpu().float().numpy().astype(np.float32)
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(all_embeddings, axis=0)

    def encode_queries(
        self,
        queries: List[str],
        batch_size: int = 16,
        **kwargs,
    ) -> np.ndarray:
        prefixed_queries = [f"{self.query_prefix} {q}" for q in queries]
        return self._batch_encode_texts(
            texts=prefixed_queries,
            batch_size=batch_size,
        )

    def encode_corpus(
        self,
        corpus: List[Dict[str, str]],
        batch_size: int = 16,
        **kwargs,
    ) -> np.ndarray:
        texts = []
        for x in corpus:
            text = x["text"]
            texts.append(f"{self.document_prefix} {text}")

        return self._batch_encode_texts(
            texts=texts,
            batch_size=batch_size,
        )

class ReasonIREncoder:
    def __init__(
        self,
        model_name_or_path: str = "model/ReasonIR-8B",
        device: str = "cuda",
        query_instruction: str = "",
        doc_instruction: str = "",
        normalize_embeddings: bool = False,
        trust_remote_code: bool = True,
        torch_dtype: Union[str, torch.dtype] = "auto",
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.query_instruction = query_instruction
        self.doc_instruction = doc_instruction
        self.normalize_embeddings = normalize_embeddings

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self.model = self.model.to(device)
        self.model.eval()

    def _to_numpy(self, emb) -> np.ndarray:
        if isinstance(emb, np.ndarray):
            arr = emb.astype(np.float32)
        elif torch.is_tensor(emb):
            arr = emb.detach().cpu().float().numpy().astype(np.float32)
        else:
            arr = np.asarray(emb, dtype=np.float32)

        if arr.ndim == 1:
            arr = arr[None, :]

        if self.normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-12, None)
            arr = arr / norms

        return arr

    def _encode_texts(
        self,
        texts: List[str],
        instruction: str,
        batch_size: int,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            with torch.no_grad():
                try:
                    # 优先尝试批量编码
                    batch_emb = self.model.encode(batch_texts, instruction=instruction)
                except Exception:
                    # 若远程实现只支持单条字符串，则退回逐条编码
                    batch_emb = [
                        self.model.encode(text, instruction=instruction)
                        for text in batch_texts
                    ]

            batch_emb = self._to_numpy(batch_emb)
            all_embeddings.append(batch_emb)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(all_embeddings, axis=0)

    def encode_queries(
        self,
        queries: List[str],
        batch_size: int = 16,
        **kwargs,
    ) -> np.ndarray:
        return self._encode_texts(
            texts=queries,
            instruction=self.query_instruction,
            batch_size=batch_size,
        )

    def encode_corpus(
        self,
        corpus: List[Dict[str, str]],
        batch_size: int = 16,
        **kwargs,
    ) -> np.ndarray:
        texts = [x["text"] for x in corpus]
        return self._encode_texts(
            texts=texts,
            instruction=self.doc_instruction,
            batch_size=batch_size,
        )

class INFRetrieverEncoder:
    def __init__(
        self,
        model_name_or_path: str = "model/inf-retriever-v1-pro",
        device: str = "cuda",
        query_instruction: str = "Given a web search query, retrieve paragraphs that answer the question",
        max_length: int = 2048,
        normalize_embeddings: bool = True,
        trust_remote_code: bool = True,
        use_fp16: bool = False,
        use_bf16: bool = True,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.query_instruction = query_instruction
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        model_kwargs = dict(kwargs)
        model_kwargs["trust_remote_code"] = trust_remote_code

        if self.use_bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif self.use_fp16:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

        self.model.to(device)
        self.model.eval()

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"

    def _tokenize_texts(self, texts: List[str]):
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        return batch_dict

    def _batch_encode_texts(
        self,
        texts: List[str],
        batch_size: int,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]

            batch_dict = self._tokenize_texts(batch_texts)

            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = self.last_token_pool(
                    outputs.last_hidden_state,
                    batch_dict["attention_mask"],
                )

                if self.normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(
                embeddings.detach().cpu().float().numpy().astype(np.float32)
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(all_embeddings, axis=0)

    def encode_queries(
        self,
        queries: List[str],
        batch_size: int = 16,
        **kwargs,
    ) -> np.ndarray:
        instructed_queries = [
            self.get_detailed_instruct(self.query_instruction, q)
            for q in queries
        ]
        return self._batch_encode_texts(
            texts=instructed_queries,
            batch_size=batch_size,
        )

    def encode_corpus(
        self,
        corpus: List[Dict[str, str]],
        batch_size: int = 16,
        **kwargs,
    ) -> np.ndarray:
        texts = [x["text"] for x in corpus]
        return self._batch_encode_texts(
            texts=texts,
            batch_size=batch_size,
        )

class ReasonEmbedQwen3Encoder:
    def __init__(
        self,
        model_name_or_path: str = "model/reason-embed-qwen3-8b-0928",
        device: str = "cuda",
        query_instruction: str = "Given a web search query, retrieve paragraphs that answer the question",
        max_length: int = 2048,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        use_bf16: bool = False,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.query_instruction = query_instruction
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **kwargs,
        )

        model_kwargs = dict(kwargs)
        if self.use_bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif self.use_fp16:
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )

        self.model.to(device)
        self.model.eval()

    @staticmethod
    def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    @staticmethod
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery: {query}"

    def _tokenize_texts(self, texts: List[str]):
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            pad_to_multiple_of=8,
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        return batch_dict

    def _batch_encode_texts(self, texts: List[str], batch_size: int) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            batch_dict = self._tokenize_texts(batch_texts)

            with torch.no_grad():
                outputs = self.model(**batch_dict)
                embeddings = self.last_token_pool(
                    outputs.last_hidden_state,
                    batch_dict["attention_mask"],
                )
                if self.normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(
                embeddings.detach().cpu().float().numpy().astype(np.float32)
            )

        return np.concatenate(all_embeddings, axis=0)

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> np.ndarray:
        instructed_queries = [
            self.get_detailed_instruct(self.query_instruction, q)
            for q in queries
        ]
        return self._batch_encode_texts(instructed_queries, batch_size)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 16, **kwargs) -> np.ndarray:
        texts = [x["text"] for x in corpus]
        return self._batch_encode_texts(texts, batch_size)

class SentenceTransformerEncoder:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        normalize_embeddings: bool = True,
        model_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.model_kwargs = model_kwargs or {}

        self.model = SentenceTransformer(
            model_name_or_path,
            device=device,
            local_files_only=True,
            trust_remote_code=True,
            model_kwargs=self.model_kwargs,
            **kwargs,
        )

    def _encode_texts(
        self,
        texts: List[str],
        batch_size: int,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=True,
        )
        return embeddings.astype(np.float32)

    def encode_queries(
        self,
        queries: List[str],
        batch_size: int = 32,
        **kwargs,
    ) -> np.ndarray:
        return self._encode_texts(queries, batch_size=batch_size)

    def encode_corpus(
        self,
        corpus: List[Dict[str, str]],
        batch_size: int = 32,
        **kwargs,
    ) -> np.ndarray:
        texts = [x.get("text", "") for x in corpus]
        return self._encode_texts(texts, batch_size=batch_size)

class MatSciBertEncoder:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        max_length: int = 512,
        pooling: str = "mean",   # "mean" or "cls"
        normalize_embeddings: bool = True,
        use_fp16: bool = False,
        **kwargs,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.max_length = max_length
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **kwargs,
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path,
            local_files_only=True,
            trust_remote_code=True,
            **kwargs,
        )

        self.model.to(device)
        self.model.eval()

        if self.use_fp16 and "cuda" in device:
            self.model.half()

    @staticmethod
    def _mean_pooling(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def _pool(self, outputs, attention_mask):
        last_hidden_state = outputs.last_hidden_state

        if self.pooling == "cls":
            return last_hidden_state[:, 0]
        elif self.pooling == "mean":
            return self._mean_pooling(last_hidden_state, attention_mask)
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")

    def _encode_texts(self, texts: List[str], batch_size: int) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = self._pool(outputs, encoded["attention_mask"])

                if self.normalize_embeddings:
                    embeddings = F.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(
                embeddings.detach().cpu().float().numpy().astype(np.float32)
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.concatenate(all_embeddings, axis=0)

    def encode_queries(self, queries: List[str], batch_size: int = 32, **kwargs) -> np.ndarray:
        return self._encode_texts(queries, batch_size)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 32, **kwargs) -> np.ndarray:
        texts = [x.get("text", "") for x in corpus]
        return self._encode_texts(texts, batch_size)



if __name__ == '__main__':
    test = 'reasonEmbed'
    if  test == 'nv_embed':
        encoder = NV_Embed2Encoder(
            model_name_or_path="model/NV-Embed-v2",
            device="cuda",
            query_instruction="Given a question, retrieve passages that answer the question",
            max_length=1024,  
            normalize=True,
            num_workers=0,
        )

        queries = [
            'are judo throws allowed in wrestling?', 
            'how to become a radiology technician in michigan?'
            ]

        corpus = [
            {
                "title": "Electrochemical Measurement",
                "text": "Since you're reading this, you are probably someone from a judo background or someone who is just wondering how judo techniques can be applied under wrestling rules. So without further ado, let's get to the question. Are Judo throws allowed in wrestling? Yes, judo throws are allowed in freestyle and folkstyle wrestling. You only need to be careful to follow the slam rules when executing judo throws. In wrestling, a slam is lifting and returning an opponent to the mat with unnecessary force."
            },
            {
                "title": "",
                "text": "Below are the basic steps to becoming a radiologic technologist in Michigan:Earn a high school diploma. As with most careers in health care, a high school education is the first step to finding entry-level employment. Taking classes in math and science, such as anatomy, biology, chemistry, physiology, and physics, can help prepare students for their college studies and future careers.Earn an associate degree. Entry-level radiologic positions typically require at least an Associate of Applied Science. Before enrolling in one of these degree programs, students should make sure it has been properly accredited by the Joint Review Committee on Education in Radiologic Technology (JRCERT).Get licensed or certified in the state of Michigan."
            }
        ]

        q_emb = encoder.encode_queries(queries, batch_size=8)
        d_emb = encoder.encode_corpus(corpus, batch_size=8)

        scores = q_emb @ d_emb.T
        print(scores)
        # [[87.42693328857422, 0.46283677220344543], [0.965264618396759, 86.03721618652344]]

    if test == 'bge_reasoner':
        encoder = BGEReasonerEncoder(
            model_name_or_path="model/bge-reasoner",
            device="cuda:0",
            query_instruction="Given a Math problem, retrieve relevant theorems that help answer the problem.",
            max_length=1024,
            normalize_embeddings=True,
            use_fp16=True,
        )

        queries = [
            "Imagine you have a magical box that transforms any object you put inside it, where the object is represented by the column vector x = (x_1, x_2). The box's transformation can be represented by the matrix A = [[5, 4], [1, 2]], so when given an object x, the box outputs the new object Ax. On some special objects, this new object is just a constant multiple of the original object, λx = (λx_1, λx_2). Find both possible values of λ where this occurs — note that these are the box's eigenvalues.",
            "Imagine you're comparing three digital images that are extremely simplified down to a grid of 5 pixels each, represented by either black (0) or white (1) pixels. The images are as follows: Image A: 00000, Image B: 10101, and Image C: 01010. By counting the number of pixels that differ between each pair of images, find the smallest number of differing pixels."
        ]

        corpus = [
            {
                "text": "\\begin{definition}[Definition:Eigenvector/Linear Operator]\nLet $K$ be a field.\nLet $V$ be a vector space over $K$. \nLet $A : V \\to V$ be a linear operator.\nLet $\\lambda \\in K$ be an eigenvalue of $A$.\nA non-zero vector $v \\in V$ is an '''eigenvector corresponding to $\\lambda$''' {{iff}}:\n:$v \\in \\map \\ker {A - \\lambda I}$\nwhere: \n:$I : V \\to V$ is the identity mapping on $V$\n:$\\map \\ker {A - \\lambda I}$ denotes the kernel of $A - \\lambda I$.\nThat is, {{iff}}: \n:$A v = \\lambda v$\n\\end{definition}"
            },
            {
                "text": "\\section{Error Correction Capability of Linear Code}\nTags: Linear Codes\n\n\\begin{theorem}\nLet $C$ be a linear code.\nLet $C$ have a minimum distance $d$.\nThen $C$ corrects $e$ transmission errors for all $e$ such that $2 e + 1 \\le d$.\n\\end{theorem}\n\n\\begin{proof}\nLet $C$ be a linear code whose master code is $V$.\nLet $c \\in C$ be a transmitted codeword.\nLet $v$ be the received word from $c$.\nBy definition, $v$ is an element of $V$.\nLet $v$ have a distance $e$ from $c$, where $2 e + 1 \\le d$.\nThus there have been $e$ transmission errors.\n{{AimForCont}} $c_1$ is a codeword of $C$, distinct from $c$, such that $\\map d {v, c_1} \\le e$.\nThen:\n{{begin-eqn}}\n{{eqn | l = \\map d {c, c_1}\n      | o = \\le\n      | r = \\map d {c, v} + \\map d {v, c_1}\n      | c = \n}}\n{{eqn | o = \\le\n      | r = e + e\n      | c = \n}}\n{{eqn | o = <\n      | r = d\n      | c = \n}}\n{{end-eqn}}\nSo $c_1$ has a distance from $c$ less than $d$.\nBut $C$ has a minimum distance $d$.\nThus $c_1$ cannot be a codeword of $C$.\nFrom this contradiction it follows that there is no codeword of $C$ closer to $v$ than $c$.\nHence there is a unique codeword of $C$ which has the smallest distance from $v$.\nHence it can be understood that $C$ has corrected the transmission errors of $v$.\n{{Qed}}\n\\end{proof}\n\n"
            }
        ]

        q_emb = encoder.encode_queries(queries, batch_size=1)
        d_emb = encoder.encode_corpus(corpus, batch_size=1)

        scores = (q_emb @ d_emb.T) * 100
        print(scores.tolist())
        # [[82.25, 53.84375], [57.53125, 65.3125]]

    if test== 'qwen3-8b':
        encoder = Qwen3EmbeddingEncoder(
            model_name_or_path="model/Qwen3-Embedding-8B",
            device="cuda",
            query_instruction='Given a web search query, retrieve relevant passages that answer the query',
            max_length=1024,
            normalize_embeddings=True,
            use_fp16=True,
        )
        queries = [
            'What is the capital of China?',
            'Explain gravity'
        ]
        # No need to add instruction for retrieval documents
        corpus = [
            {
                "text": "The capital of China is Beijing."
            },
            {
                "text": "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
            }
        ]
        
        q_emb = encoder.encode_queries(queries, batch_size=1)
        d_emb = encoder.encode_corpus(corpus, batch_size=1)
        scores = (q_emb @ d_emb.T) * 100
        print(scores.tolist())
        # [[0.7493016123771667, 0.0750647559762001], [0.08795969933271408, 0.6318399906158447]]

    if test == 'diver_retriever':
        encoder = Qwen3EmbeddingEncoder(
            model_name_or_path="model/Diver-Retriever-4B-1020",
            device="cuda",
            query_instruction='Given a web search query, retrieve relevant passages that answer the query',
            max_length=8192,
            normalize_embeddings=True,
            use_fp16=True,
        )
        queries = [
            'What is the capital of China?',
            'Explain gravity'
        ]
        # No need to add instruction for retrieval documents
        corpus = [
            {
                "text": "The capital of China is Beijing."
            },
            {
                "text": "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
            }
        ]
        
        q_emb = encoder.encode_queries(queries, batch_size=1)
        d_emb = encoder.encode_corpus(corpus, batch_size=1)
        scores = (q_emb @ d_emb.T) * 100
        print(scores.tolist())
    
    if test == 'llama-nv-embed':
        encoder = NVEmbedReasoningEncoder(
            model_name_or_path="model/llama-nv-embed-reasoning-3b",
            device="cuda",
        )
        queries = [
            "how much protein should a female eat",
            "summit define",
        ]
        # No need to add instruction for retrieval documents
        corpus = [
            {
                "text": "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
            },
            {
                "text": "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
            }
        ]
        
        q_emb = encoder.encode_queries(queries, batch_size=1)
        d_emb = encoder.encode_corpus(corpus, batch_size=1)
        scores = (q_emb @ d_emb.T) * 100
        print(q_emb.shape)
        print(d_emb.shape)
        print(scores.tolist())
        #[[0.6688634157180786, 0.23073062300682068], [0.24395054578781128, 0.5622682571411133]]

    if test == 'reasonIR':
        encoder = ReasonIREncoder(
            model_name_or_path="model/ReasonIR-8B",
            device="cuda",
        )

        queries = ["The quick brown fox jumps over the lazy dog."]
        corpus = [{"text": "The quick brown fox jumps over the lazy dog."}]

        q_emb = encoder.encode_queries(queries, batch_size=1)
        d_emb = encoder.encode_corpus(corpus, batch_size=1)

        scores = q_emb @ d_emb.T
        print(scores.tolist())

    if test == 'inf-retriever-v1':
        encoder = INFRetrieverEncoder(
            model_name_or_path="model/inf-retriever-v1-pro",
            device="cuda",
        )

        queries = [
            "how much protein should a female eat",
            "summit define",
        ]
        corpus = [
            {
                "text": "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
            },
            {
                "text": "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
            }
        ]
        
        q_emb = encoder.encode_queries(queries, batch_size=1)
        d_emb = encoder.encode_corpus(corpus, batch_size=1)
        scores = (q_emb @ d_emb.T) * 100

        print(scores.tolist())
        # [[91.46114349365234, 76.98332214355469], [70.7035140991211, 87.158203125]]

    if test == 'reasonEmbed':
        encoder = ReasonEmbedQwen3Encoder(
            model_name_or_path="model/reason-embed-qwen3-8b-0928",
            device="cuda",
            query_instruction='Given a Math problem, retrieve relevant theorems that help answer the problem.',
        )

        queries = [
            "Imagine you have a magical box that transforms any object you put inside it, where the object is represented by the column vector x = (x_1, x_2). The box's transformation can be represented by the matrix A = [[5, 4], [1, 2]], so when given an object x, the box outputs the new object Ax. On some special objects, this new object is just a constant multiple of the original object, λx = (λx_1, λx_2). Find both possible values of λ where this occurs — note that these are the box's eigenvalues.",
            "Imagine you're comparing three digital images that are extremely simplified down to a grid of 5 pixels each, represented by either black (0) or white (1) pixels. The images are as follows: Image A: 00000, Image B: 10101, and Image C: 01010. By counting the number of pixels that differ between each pair of images, find the smallest number of differing pixels."
        ]
        corpus = [
            {
                "text": "\\begin{definition}[Definition:Eigenvector/Linear Operator]\nLet $K$ be a field.\nLet $V$ be a vector space over $K$. \nLet $A : V \\to V$ be a linear operator.\nLet $\\lambda \\in K$ be an eigenvalue of $A$.\nA non-zero vector $v \\in V$ is an '''eigenvector corresponding to $\\lambda$''' {{iff}}:\n:$v \\in \\map \\ker {A - \\lambda I}$\nwhere: \n:$I : V \\to V$ is the identity mapping on $V$\n:$\\map \\ker {A - \\lambda I}$ denotes the kernel of $A - \\lambda I$.\nThat is, {{iff}}: \n:$A v = \\lambda v$\n\\end{definition}",
            },
            {
                "text": "\\section{Error Correction Capability of Linear Code}\nTags: Linear Codes\n\n\\begin{theorem}\nLet $C$ be a linear code.\nLet $C$ have a minimum distance $d$.\nThen $C$ corrects $e$ transmission errors for all $e$ such that $2 e + 1 \\le d$.\n\\end{theorem}\n\n\\begin{proof}\nLet $C$ be a linear code whose master code is $V$.\nLet $c \\in C$ be a transmitted codeword.\nLet $v$ be the received word from $c$.\nBy definition, $v$ is an element of $V$.\nLet $v$ have a distance $e$ from $c$, where $2 e + 1 \\le d$.\nThus there have been $e$ transmission errors.\n{{AimForCont}} $c_1$ is a codeword of $C$, distinct from $c$, such that $\\map d {v, c_1} \\le e$.\nThen:\n{{begin-eqn}}\n{{eqn | l = \\map d {c, c_1}\n      | o = \\le\n      | r = \\map d {c, v} + \\map d {v, c_1}\n      | c = \n}}\n{{eqn | o = \\le\n      | r = e + e\n      | c = \n}}\n{{eqn | o = <\n      | r = d\n      | c = \n}}\n{{end-eqn}}\nSo $c_1$ has a distance from $c$ less than $d$.\nBut $C$ has a minimum distance $d$.\nThus $c_1$ cannot be a codeword of $C$.\nFrom this contradiction it follows that there is no codeword of $C$ closer to $v$ than $c$.\nHence there is a unique codeword of $C$ which has the smallest distance from $v$.\nHence it can be understood that $C$ has corrected the transmission errors of $v$.\n{{Qed}}\n\\end{proof}\n\n"
            }
        ]
        
        q_emb = encoder.encode_queries(queries, batch_size=1)
        d_emb = encoder.encode_corpus(corpus, batch_size=1)
        scores = (q_emb @ d_emb.T) * 100

        print(scores.tolist())
   





# CUDA_VISIBLE_DEVICES=5,4 python -m scripts.eval.model_class

