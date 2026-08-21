# 多 Agent 系统的全局配置：LLM 后端选择、Snowflake/DashScope/Ollama 参数、RAG 开关、路径白名单
"""
只改这一个文件即可切换 LLM 后端 / 调整路径 / 打开关闭 RAG。

设计原则：
  - DEFAULT_WORK_DIR：用户没说目录时，文件默认放这里（精确）
  - ALLOWED_DIRS    ：MCP 沙箱白名单，覆盖所有允许访问的位置（可以宽）
"""

import os
from pathlib import Path


# ── LLM 后端选择 ─────────────────────────────────────────────────────────
# "ollama"     : 本地 Ollama 服务
# "dashscope"  : 阿里云 Qwen API
# "snowflake"  : Snowflake Cortex REST API（云端 LLM，OpenAI 兼容协议）
LLM_BACKEND: str = "snowflake"

# Ollama 模型名（LLM_BACKEND="ollama" 时生效）
OLLAMA_MODEL: str = "qwen3.6-27b"
OLLAMA_BASE_URL: str = ""

# DashScope 模型名（LLM_BACKEND="dashscope" 时生效）
DASHSCOPE_MODEL: str = "qwen3.5-27b"
DASHSCOPE_BASE_URL: str = ""

# DashScope API Key
DASHSCOPE_API_KEY: str = ""

# ── Snowflake Cortex 配置（LLM_BACKEND="snowflake" 时生效）─────────────────
# 模型名可选：claude-sonnet-4-5、llama3.1-70b、mistral-large2 等，
# 具体以账号所在区域支持的模型列表为准。
SNOWFLAKE_MODEL: str = "claude-sonnet-4-5"
# 账号地址，形如 https://<account-identifier>.snowflakecomputing.com
SNOWFLAKE_ACCOUNT_URL: str = ""
# Programmatic Access Token（PAT），在 Snowsight 用户设置里生成。
# 建议用环境变量 SNOWFLAKE_PAT 传入，避免把密钥写进代码。
SNOWFLAKE_PAT: str = ""


# ── 用户没指定路径时的默认工作目录 ─────────────────────────────────────
# 比如用户说"创建 hello.txt"（没说目录），就放到这里。
DEFAULT_WORK_DIR: str = str(
    Path(os.path.expanduser("~")) / "Desktop" / "qwen3_agent_output"
)


# ── 沙箱白名单（MCP filesystem 最大允许访问范围）─────────────────────────
ALLOWED_DIRS: list[str] = [
    DEFAULT_WORK_DIR,
    str(Path(os.path.expanduser("~"))),
]


# ── RAG 开关 ──────────────────────────────────────────────────────────────
# 关掉后 rag_tool 不会被加载，其它工具/专员不受影响。
ENABLE_RAG: bool = True

# 本地 Embedding 模型路径（bge-small-zh-v1.5）
EMBEDDING_MODEL_PATH: str = "F:/Max/llm/bge-small-zh-v1.5"

# 知识库文档目录：默认指向 MULTI_AGENT/texts/，保持系统自包含
KNOWLEDGE_BASE_TEXTS_DIR: str = str(Path(__file__).resolve().parent / "texts")

# 检索模式：
#   "dense"  — 仅用向量相似度
#   "bm25"   — 仅用 BM25 关键词匹配（无需 embedding 模型）
#   "hybrid" — Dense + BM25 双路 RRF 融合（推荐）
RETRIEVAL_MODE: str = "hybrid"


# ── 多 Agent 系统运行参数 ─────────────────────────────────────────────────
# Supervisor 最多路由多少轮（防死循环，跟业务无关）
SUPERVISOR_MAX_STEPS: int = 10
# 每个 worker 内部 ReAct 循环最多多少步
WORKER_MAX_STEPS: int = 12
# 历史 worker 产出注入 supervisor 前截断的字符数（迁就上下文 32k~200k 的最小公约数）
WORKER_OUTPUT_TRUNCATE: int = 4000


def get_allowed_dirs() -> list[str]:
    """返回允许目录列表，并保证默认工作目录存在。"""
    Path(DEFAULT_WORK_DIR).mkdir(parents=True, exist_ok=True)
    return ALLOWED_DIRS


def get_default_work_dir() -> str:
    return DEFAULT_WORK_DIR


def format_dirs_for_prompt() -> str:
    """把目录列表格式化成 prompt 里的 markdown 列表。"""
    return "\n".join(f"  - {d}" for d in ALLOWED_DIRS)
