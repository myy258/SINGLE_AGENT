# 本地检索模块：Dense（向量）/ BM25（关键词）/ Hybrid（RRF 融合）三种模式
# Co-authored with CoCo

"""
本版改动的核心是**相关性闸门**。

原来 rag_tool.py 靠 `scores[0] - scores[1] > min_margin` 判断"是否足够相关"，
而 HybridRetriever.min_margin = 0.0。原因是 RRF 分数是 1/(60+rank) 量级，
top1 和 top2 的差值大约 1e-4，用 dense 的默认阈值 0.03 会把**所有**结果都拒掉，
所以只能设成 0.0——代价是 hybrid 模式下任何查询都必然返回一条文档，哪怕完全
无关。模型于是拿着噪声当事实回答。

根因：排名差本质上无法表达"相关性"。两条同样无关的文档，排名差也可能很大。

现在每个 Hit 除了排序分数（score）之外，额外带一个 0~1 的绝对相关度
（relevance），闸门用它跟 config.RETRIEVAL_MIN_SCORE 比：
- dense  ：relevance = 余弦相似度，本身就是绝对量
- hybrid ：relevance = 该文档的 dense 余弦相似度（RRF 只用来排序，不用来判定）
- bm25   ：BM25 分数没有天然上界，用单调压缩 s/(s+5) 映射到 0~1，
           所以纯 bm25 模式下阈值语义跟另两种不完全可比——这是已知取舍，
           不是遗漏。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from rag.knowledge_base import Chunk


@dataclass(frozen=True)
class Hit:
    """一条检索结果。

    score     ：该检索器自己的排序分数（不同模式量纲不同，仅用于排序/调试）
    relevance ：0~1 的绝对相关度，用于跟 RETRIEVAL_MIN_SCORE 比较
    """

    chunk: Chunk
    score: float
    relevance: float


def _tokenize(text: str) -> list[str]:
    """把文本切成 token 列表：ASCII 单词保持整体，CJK 字符逐个拆开。"""
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]", text)
    return [t.lower() for t in tokens] if tokens else list(text)


def _bm25_to_relevance(score: float) -> float:
    """把无上界的 BM25 分数单调压缩到 0~1。"""
    return float(score / (score + 5.0)) if score > 0 else 0.0


class DenseRetriever:
    """基于 BGE embedding + 余弦相似度的密集向量检索。"""

    def __init__(self, chunks: list[Chunk], embedder):
        self.chunks = chunks
        self.embedder = embedder
        # 走带缓存的编码：内容没变时不重复算（原来每次启动全量重算）
        self.doc_embeddings = embedder.encode_corpus_cached([c.text for c in chunks])

    def all_scores(self, query: str) -> np.ndarray:
        q_emb = self.embedder.encode_query(query)
        return np.dot(self.doc_embeddings, q_emb.T).reshape(-1)

    def retrieve(self, query: str, top_k: int = 3) -> list[Hit]:
        scores = self.all_scores(query)
        order = np.argsort(scores)[::-1][:top_k]
        return [
            Hit(
                chunk=self.chunks[i],
                score=float(scores[i]),
                # 余弦相似度理论范围 [-1,1]，负相关一律当 0
                relevance=max(0.0, float(scores[i])),
            )
            for i in order
        ]


class BM25Retriever:
    """基于 BM25Okapi 的稀疏关键词检索。"""

    def __init__(self, chunks: list[Chunk]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "BM25 检索需要 rank-bm25 库，请运行：pip install rank-bm25"
            ) from exc
        self.chunks = chunks
        self.bm25 = BM25Okapi([_tokenize(c.text) for c in chunks])

    def all_scores(self, query: str) -> np.ndarray:
        return np.array(self.bm25.get_scores(_tokenize(query)))

    def retrieve(self, query: str, top_k: int = 3) -> list[Hit]:
        scores = self.all_scores(query)
        order = np.argsort(scores)[::-1][:top_k]
        return [
            Hit(
                chunk=self.chunks[i],
                score=float(scores[i]),
                relevance=_bm25_to_relevance(float(scores[i])),
            )
            for i in order
        ]


class HybridRetriever:
    """Dense + BM25 双路 RRF 融合排序，但相关性判定仍用 dense 余弦的绝对值。"""

    _RRF_K: int = 60

    def __init__(self, chunks: list[Chunk], embedder):
        self.chunks = chunks
        self.dense = DenseRetriever(chunks, embedder)
        self.bm25 = BM25Retriever(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[Hit]:
        n = len(self.chunks)
        dense_scores = self.dense.all_scores(query)
        bm25_scores = self.bm25.all_scores(query)

        dense_rank = np.empty(n, dtype=int)
        dense_rank[np.argsort(dense_scores)[::-1]] = np.arange(1, n + 1)

        bm25_rank = np.empty(n, dtype=int)
        bm25_rank[np.argsort(bm25_scores)[::-1]] = np.arange(1, n + 1)

        rrf = 1.0 / (self._RRF_K + dense_rank) + 1.0 / (self._RRF_K + bm25_rank)
        order = np.argsort(rrf)[::-1][:top_k]

        return [
            Hit(
                chunk=self.chunks[i],
                score=float(rrf[i]),
                # 关键：闸门用 dense 余弦这个绝对量，而不是 RRF 的排名差
                relevance=max(0.0, float(dense_scores[i])),
            )
            for i in order
        ]
