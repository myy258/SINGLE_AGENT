# 本地 Embedding 模块：用 BAAI/bge-small-zh-v1.5 做文本向量化
# Co-authored with CoCo

"""
本版改动：

1. 「一次性 encode 整个知识库会 OOM」
   原来 encode(texts) 把所有 chunk 塞进同一个 batch，知识库稍大就爆显存/内存。
   现在按 config.EMBEDDING_BATCH_SIZE 分批。

2. 「每次启动都重算全部向量」
   原来 DenseRetriever 在构造时同步 encode 全库，没有任何缓存，进程每次启动
   都重跑一遍。现在按"文档内容的 sha256"做缓存键落盘（npz），内容没变就直接
   加载；变了自动重算。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_CACHE_PATH, EMBEDDING_MODEL_PATH

# BGE 系列模型官方推荐：编码查询时加此前缀，编码文档时不加。
_BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def _corpus_fingerprint(texts: list[str]) -> str:
    """按文档内容算指纹，作为向量缓存的键。"""
    h = hashlib.sha256()
    h.update(str(len(texts)).encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    h.update(EMBEDDING_MODEL_PATH.encode("utf-8"))
    return h.hexdigest()


class Embedder:
    def __init__(self):
        # 放在这里而不是模块顶层：transformers/torch 加载很重，
        # 用 bm25 模式的用户不该被迫付这个代价。
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()

        model_path = Path(EMBEDDING_MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Embedding 模型路径不存在：{model_path}\n"
                "请用环境变量 SINGLE_AGENT_EMBEDDING_MODEL_PATH 指定实际路径，"
                "或把 config.RETRIEVAL_MODE 改成 'bm25'（不需要 embedding 模型）。"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)
        self.model.eval()

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        import torch

        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            output = self.model(**encoded)
        embeddings = output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.numpy()

    def encode(self, texts: list[str]) -> np.ndarray:
        """编码文档文本（不加指令前缀）。分批执行，避免一次性占满内存。"""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        batch_size = max(1, EMBEDDING_BATCH_SIZE)
        chunks = [
            self._encode_batch(texts[i : i + batch_size])
            for i in range(0, len(texts), batch_size)
        ]
        return np.vstack(chunks)

    def encode_query(self, query: str) -> np.ndarray:
        """编码查询文本，自动添加 BGE 查询指令前缀。"""
        return self._encode_batch([_BGE_QUERY_INSTRUCTION + query])

    # ── 带缓存的语料编码 ──────────────────────────────────────────────────
    def encode_corpus_cached(self, texts: list[str]) -> np.ndarray:
        """编码整个语料库，命中缓存则直接读盘。"""
        fingerprint = _corpus_fingerprint(texts)
        cache_path = Path(EMBEDDING_CACHE_PATH)

        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    if str(data["fingerprint"]) == fingerprint:
                        # 命中缓存是正常路径，不打印
                        return data["embeddings"]
            except (OSError, KeyError, ValueError) as exc:
                print(f"[RAG] 向量缓存读取失败，将重新编码：{exc}")

        # 编码可能耗时十几秒，这条进度提示保留——否则会像卡死
        print(f"[RAG] 正在编码知识库（{len(texts)} 个 chunk）...")
        embeddings = self.encode(texts)

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path, embeddings=embeddings, fingerprint=np.array(fingerprint)
            )
        except OSError as exc:
            print(f"[RAG] 向量缓存写入失败（不影响本次使用）：{exc}")

        return embeddings
