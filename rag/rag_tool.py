# 本地知识库检索工具：dense/bm25/hybrid 三模式可切，懒加载单例（照搬 AGENT/rag_tool.py 逻辑）

from langchain_core.tools import tool

_retriever = None  # 懒加载单例


def _get_retriever():
    """按 config.RETRIEVAL_MODE 首次构建检索器，之后复用同一实例。"""
    global _retriever
    if _retriever is not None:
        return _retriever

    from config import RETRIEVAL_MODE
    from rag.knowledge_base import documents
    from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever

    if not documents:
        raise RuntimeError("知识库为空，请在 texts/ 目录下放入 .txt 文件后重启。")

    mode = RETRIEVAL_MODE.strip().lower()

    if mode == "bm25":
        _retriever = BM25Retriever(documents)

    elif mode == "dense":
        from rag.embedder import Embedder
        _retriever = DenseRetriever(documents, Embedder())

    else:
        from rag.embedder import Embedder
        try:
            _retriever = HybridRetriever(documents, Embedder())
        except ImportError:
            _retriever = DenseRetriever(documents, Embedder())

    return _retriever


@tool
def search_local_knowledge_base(query: str) -> str:
    """在本地知识库中检索相关内容。当用户问的是具体的人物、事实、术语、
    公司规章、内部资料等你自己不确定的信息时，应该优先调用这个工具查一下。

    Args:
        query: 用中文简洁描述要检索的问题、人名或主题。
    """
    try:
        print("[RAG] 正在检索本地知识库...")
        retriever = _get_retriever()
        docs, scores = retriever.retrieve_with_scores(query, top_k=2)

        if not docs:
            return "本地知识库未找到足够相关的内容，建议改用其他方式获取信息。"

        min_margin = getattr(retriever, "min_margin", 0.03)
        margin = scores[0] - scores[1] if len(scores) > 1 else 1.0

        if margin < min_margin:
            return "本地知识库未找到足够相关的内容，建议改用其他方式获取信息。"

        return docs[0]

    except Exception as e:
        return f"本地知识库检索失败：{e}"


def get_rag_tools() -> list:
    """返回 RAG 工具列表，供 tools.py 的 create_tools() 调用。"""
    return [search_local_knowledge_base]
