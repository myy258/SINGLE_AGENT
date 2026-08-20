# 知识库文档加载：从 config.KNOWLEDGE_BASE_TEXTS_DIR 加载 .txt 并按段落切 chunk（照搬 AGENT/knowledge_base.py）

import re
from pathlib import Path

from config import KNOWLEDGE_BASE_TEXTS_DIR

TEXTS_DIR = Path(KNOWLEDGE_BASE_TEXTS_DIR)

_MAX_CHUNK_SIZE = 200
_OVERLAP = 40
_MIN_CHUNK_SIZE = 10


def _split_into_chunks(text: str) -> list[str]:
    """按空行粗切段落，超长段落再按句末标点细切并保留 OVERLAP 字符重叠。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks = []
    for para in paragraphs:
        if len(para) <= _MAX_CHUNK_SIZE:
            if len(para) >= _MIN_CHUNK_SIZE:
                chunks.append(para)
        else:
            sentences = re.split(r"(?<=[。！？\n])", para)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) <= _MAX_CHUNK_SIZE:
                    current += sent
                else:
                    if len(current.strip()) >= _MIN_CHUNK_SIZE:
                        chunks.append(current.strip())
                    tail = current[-_OVERLAP:] if len(current) > _OVERLAP else current
                    current = tail + sent
            if len(current.strip()) >= _MIN_CHUNK_SIZE:
                chunks.append(current.strip())

    return chunks


def load_texts() -> list[str]:
    """加载 TEXTS_DIR 下所有 .txt，按段落切成独立 chunk。"""
    docs = []
    if not TEXTS_DIR.exists():
        return docs
    for f in sorted(TEXTS_DIR.glob("*.txt")):
        content = f.read_text(encoding="utf-8").strip()
        if not content:
            continue
        docs.extend(_split_into_chunks(content))
    return docs


def load_all_documents() -> list[str]:
    """加载知识库全部文档。"""
    return load_texts()


documents = load_all_documents()
