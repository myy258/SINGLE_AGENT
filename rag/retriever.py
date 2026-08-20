# 本地检索模块：Dense（向量）/ BM25（关键词）/ Hybrid（RRF 融合）三种模式（照搬 AGENT/retriever.py）

import re

import numpy as np

from rag.embedder import Embedder


def _tokenize(text: str) -> list[str]:
    """把文本切成 token 列表：ASCII 单词保持整体，CJK 字符逐个拆开。"""
    tokens = re.findall(r'[A-Za-z0-9]+|[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text)
    return [t.lower() for t in tokens] if tokens else list(text)


class DenseRetriever:
    """基于 BGE embedding + 余弦相似度的密集向量检索。"""

    min_margin: float = 0.03

    def __init__(self, documents: list[str], embedder: Embedder):
        self.documents = documents
        self.embedder = embedder
        self.doc_embeddings = embedder.encode(documents)

    def _all_scores(self, query: str) -> np.ndarray:
        q_emb = self.embedder.encode_query(query)
        return np.dot(self.doc_embeddings, q_emb.T).reshape(-1)

    def retrieve(self, query: str, top_k: int = 2) -> list[str]:
        docs, _ = self.retrieve_with_scores(query, top_k)
        return docs

    def retrieve_with_scores(self, query: str, top_k: int = 2) -> tuple[list[str], list[float]]:
        scores = self._all_scores(query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_idx], [float(scores[i]) for i in top_idx]


class BM25Retriever:
    """基于 BM25Okapi 的稀疏关键词检索。"""

    min_margin: float = 0.5

    def __init__(self, documents: list[str]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "BM25 检索需要 rank-bm25 库，请运行：pip install rank-bm25"
            ) from exc
        self.documents = documents
        tokenized_corpus = [_tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _all_scores(self, query: str) -> np.ndarray:
        return np.array(self.bm25.get_scores(_tokenize(query)))

    def retrieve(self, query: str, top_k: int = 2) -> list[str]:
        docs, _ = self.retrieve_with_scores(query, top_k)
        return docs

    def retrieve_with_scores(self, query: str, top_k: int = 2) -> tuple[list[str], list[float]]:
        scores = self._all_scores(query)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_idx], [float(scores[i]) for i in top_idx]


class HybridRetriever:
    """Dense + BM25 双路 RRF 融合。"""

    _RRF_K: int = 60
    min_margin: float = 0.0

    def __init__(self, documents: list[str], embedder: Embedder):
        self.documents = documents
        self.dense = DenseRetriever(documents, embedder)
        self.bm25 = BM25Retriever(documents)

    def retrieve(self, query: str, top_k: int = 2) -> list[str]:
        docs, _ = self.retrieve_with_scores(query, top_k)
        return docs

    def retrieve_with_scores(self, query: str, top_k: int = 2) -> tuple[list[str], list[float]]:
        n = len(self.documents)

        dense_scores = self.dense._all_scores(query)
        bm25_scores = self.bm25._all_scores(query)

        dense_rank = np.empty(n, dtype=int)
        dense_rank[np.argsort(dense_scores)[::-1]] = np.arange(1, n + 1)

        bm25_rank = np.empty(n, dtype=int)
        bm25_rank[np.argsort(bm25_scores)[::-1]] = np.arange(1, n + 1)

        rrf_scores = (
            1.0 / (self._RRF_K + dense_rank) +
            1.0 / (self._RRF_K + bm25_rank)
        )

        top_idx = np.argsort(rrf_scores)[::-1][:top_k]
        return [self.documents[i] for i in top_idx], [float(rrf_scores[i]) for i in top_idx]


Retriever = DenseRetriever
