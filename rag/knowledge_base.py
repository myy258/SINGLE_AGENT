# 知识库文档加载：从 config.KNOWLEDGE_BASE_TEXTS_DIR 加载 .txt 并按段落切 chunk
# Co-authored with CoCo

"""
本版改动：

1. 「import 期就做磁盘 IO」
   原来模块底部有一行 `documents = load_all_documents()`，只要 import 这个模块
   就会去读盘。测试和只用 bm25 的场景都被迫付这个代价。现在改成懒加载函数。

2. 「切出来的片段不知道来自哪个文件」
   原来 chunk 是纯字符串，检索结果无法溯源，模型没法在回答里标注出处。
   现在每个 chunk 带上来源文件名和序号。

3. 「chunk 太小」
   原来固定 200 字符，中文一个完整段落经常被切断，检索到的片段缺上下文。
   现在走 config.CHUNK_MAX_CHARS（默认 500）+ CHUNK_OVERLAP_CHARS（默认 80）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from config import (
    CHUNK_MAX_CHARS,
    CHUNK_MIN_CHARS,
    CHUNK_OVERLAP_CHARS,
    KNOWLEDGE_BASE_TEXTS_DIR,
)


@dataclass(frozen=True)
class Chunk:
    """知识库里的一个片段。source 用于在回答里标注出处。"""

    text: str
    source: str
    index: int

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.index}"


def _texts_dir() -> Path:
    return Path(KNOWLEDGE_BASE_TEXTS_DIR)


def _split_into_chunks(text: str) -> list[str]:
    """按空行粗切段落，超长段落再按句末标点细切并保留 OVERLAP 字符重叠。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= CHUNK_MAX_CHARS:
            if len(para) >= CHUNK_MIN_CHARS:
                chunks.append(para)
            continue

        sentences = re.split(r"(?<=[。！？\n])", para)
        current = ""
        for sent in sentences:
            if len(current) + len(sent) <= CHUNK_MAX_CHARS:
                current += sent
                continue
            if len(current.strip()) >= CHUNK_MIN_CHARS:
                chunks.append(current.strip())
            tail = current[-CHUNK_OVERLAP_CHARS:] if len(current) > CHUNK_OVERLAP_CHARS else current
            current = tail + sent
        if len(current.strip()) >= CHUNK_MIN_CHARS:
            chunks.append(current.strip())

    return chunks


def load_all_documents() -> list[Chunk]:
    """加载 TEXTS_DIR 下所有 .txt，按段落切成带来源信息的 chunk。"""
    texts_dir = _texts_dir()
    docs: list[Chunk] = []
    if not texts_dir.exists():
        print(f"[RAG] 警告：知识库目录不存在：{texts_dir}")
        return docs

    files = sorted(texts_dir.glob("*.txt"))
    if not files:
        print(f"[RAG] 警告：{texts_dir} 下没有 .txt 文件，知识库为空。")
        return docs

    for f in files:
        try:
            content = f.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as exc:
            print(f"[RAG] 警告：读取 {f.name} 失败，已跳过：{exc}")
            continue
        if not content:
            continue
        for i, piece in enumerate(_split_into_chunks(content)):
            docs.append(Chunk(text=piece, source=f.name, index=i))

    # 加载成功不打印（启动期保持安静）；目录缺失/为空/读失败已在上面告警。
    return docs
