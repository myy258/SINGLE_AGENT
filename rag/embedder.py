# 本地 Embedding 模块：用 BAAI/bge-small-zh-v1.5 做文本向量化（照搬 AGENT/embedder.py）

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging as hf_logging

from config import EMBEDDING_MODEL_PATH

# 关掉 transformers 加载模型时自带的进度条 / LOAD REPORT 日志
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()

# BGE 系列模型官方推荐：编码查询时加此前缀，编码文档时不加。
_BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class Embedder:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)
        self.model.eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        """编码文档文本（不加指令前缀）。用于构建向量索引。"""
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            output = self.model(**encoded)
        embeddings = output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.numpy()

    def encode_query(self, query: str) -> np.ndarray:
        """编码查询文本，自动添加 BGE 查询指令前缀。"""
        return self.encode([_BGE_QUERY_INSTRUCTION + query])
