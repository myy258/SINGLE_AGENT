# 本地知识库检索工具：dense/bm25/hybrid 三模式可切，懒加载单例
# Co-authored with CoCo

"""
本版改动：

1. 「取了 top2 却只返回 top1」
   原来 retrieve_with_scores(query, top_k=2) 拿两条，然后 `return docs[0]`——
   第二条纯粹浪费一次检索。现在按 config.RETRIEVAL_TOP_K 取并全部返回。

2. 「相关性闸门在 hybrid 模式下等于没有」
   见 rag/retriever.py 顶部的说明。现在用绝对相关度跟
   config.RETRIEVAL_MIN_SCORE 比较，真正能判定"知识库里没有"。

3. 「返回内容不带来源，答案无法溯源」
   现在每条结果都带 `来源: 文件名#片段序号`，system prompt 也要求模型
   引用时把出处写进回答。
"""

from __future__ import annotations

import threading

from langchain_core.tools import tool

from config import RETRIEVAL_MIN_SCORE, RETRIEVAL_MODE, RETRIEVAL_TOP_K

_retriever = None
_retriever_lock = threading.Lock()

_NOT_FOUND = (
    "本地知识库里没有足够相关的内容（检索到的最高相关度低于阈值）。"
    "如果这个问题需要外部信息，可以改用 web_search；"
    "如果应该在知识库里，说明 texts/ 目录下缺少对应资料。"
)


def _build_retriever():
    from rag.knowledge_base import load_all_documents
    from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever

    chunks = load_all_documents()
    if not chunks:
        raise RuntimeError("知识库为空，请在 texts/ 目录下放入 .txt 文件后重启。")

    mode = (RETRIEVAL_MODE or "hybrid").strip().lower()

    if mode == "bm25":
        return BM25Retriever(chunks)

    if mode == "dense":
        from rag.embedder import Embedder

        return DenseRetriever(chunks, Embedder())

    # hybrid（默认）：缺 rank-bm25 时退回 dense，缺 embedding 模型时退回 bm25，
    # 每次降级都明确说明，不静默改变行为。
    from rag.embedder import Embedder

    try:
        return HybridRetriever(chunks, Embedder())
    except ImportError as exc:
        print(f"[RAG] hybrid 模式缺少依赖（{exc}），降级为 dense 模式。")
        return DenseRetriever(chunks, Embedder())
    except FileNotFoundError as exc:
        print(f"[RAG] 找不到 embedding 模型（{exc}），降级为 bm25 模式。")
        return BM25Retriever(chunks)


def _get_retriever():
    """按 config.RETRIEVAL_MODE 首次构建检索器，之后复用同一实例。"""
    global _retriever
    if _retriever is not None:
        return _retriever
    with _retriever_lock:
        if _retriever is None:
            _retriever = _build_retriever()
    return _retriever


@tool
def search_local_knowledge_base(query: str) -> str:
    """在本地知识库中检索相关内容。当用户问的是具体的人物、事实、术语、
    公司规章、内部资料等你自己不确定的信息时，应该优先调用这个工具查一下。
    返回结果带来源标注，请在回答里保留出处。

    Args:
        query: 用中文简洁描述要检索的问题、人名或主题。
    """
    try:
        retriever = _get_retriever()
    except (RuntimeError, ImportError, FileNotFoundError) as exc:
        return f"本地知识库不可用：{exc}"

    try:
        hits = retriever.retrieve(query, top_k=max(1, RETRIEVAL_TOP_K))
    except Exception as exc:
        return f"本地知识库检索失败：{exc}"

    relevant = [h for h in hits if h.relevance >= RETRIEVAL_MIN_SCORE]
    if not relevant:
        best = max((h.relevance for h in hits), default=0.0)
        return f"{_NOT_FOUND}（最高相关度 {best:.3f} < 阈值 {RETRIEVAL_MIN_SCORE}）"

    blocks = [
        f"[{i}] 来源: {h.chunk.citation}　相关度: {h.relevance:.3f}\n{h.chunk.text}"
        for i, h in enumerate(relevant, 1)
    ]
    return "\n\n".join(blocks)


def get_rag_tools() -> list:
    """返回 RAG 工具列表，供 local_tools.py 的 create_tools() 调用。"""
    return [search_local_knowledge_base]
