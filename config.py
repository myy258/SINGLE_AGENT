# 单 Agent 系统的全局配置：LLM 后端选择、Snowflake/DashScope/Ollama 参数、RAG 开关、路径白名单
# Co-authored with CoCo
"""
只改这一个文件即可切换 LLM 后端 / 调整路径 / 打开关闭 RAG。

设计原则：
  - DEFAULT_WORK_DIR：用户没说目录时，文件默认放这里（精确）
  - ALLOWED_DIRS    ：沙箱白名单，覆盖所有允许访问的位置（可以宽）
                      注意：这个白名单现在对 **所有** 落盘类工具生效
                      （write_file / python_exec / run_python_script /
                      git_command / MCP filesystem），不再只约束 MCP。

所有敏感值和机器相关路径都支持用环境变量覆盖，优先级：环境变量 > 本文件默认值。
"""

import os
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    """读环境变量，去掉首尾空白；没设置就返回默认值。"""
    return (os.environ.get(name) or "").strip() or default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


# ── LLM 后端选择 ─────────────────────────────────────────────────────────
# "ollama"     : 本地 Ollama 服务
# "dashscope"  : 阿里云 Qwen API
# "snowflake"  : Snowflake Cortex REST API（云端 LLM，OpenAI 兼容协议）
LLM_BACKEND: str = _env("LLM_BACKEND", "")

# Ollama 模型名（LLM_BACKEND="ollama" 时生效）
# 必须是 `ollama list` 里真实存在的 tag；写错的话启动时会报错并列出本机实际有哪些模型。
OLLAMA_MODEL: str = _env("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL: str = _env("OLLAMA_BASE_URL", "")
# Ollama 的上下文窗口。注意这个值直接决定 agent 的输入预算（见 CONTEXT_LIMITS）。
OLLAMA_NUM_CTX: int = _env_int("OLLAMA_NUM_CTX", 8192)
# Ollama 桌面应用路径（仅 Windows 下用于自动拉起服务）。
# 留空则自动在 PATH 和常见安装位置里找，找不到就提示手动启动。
OLLAMA_APP_PATH: str = _env("OLLAMA_APP_PATH", "")

# DashScope 模型名（LLM_BACKEND="dashscope" 时生效）
# 这里填的必须是该 base_url 下真实可用的模型/部署名。
DASHSCOPE_MODEL: str = _env("DASHSCOPE_MODEL", "qwen3.8-27b")
DASHSCOPE_BASE_URL: str = _env(
    "",
    "",
)

# DashScope API Key
DASHSCOPE_API_KEY: str = ""

# ── Snowflake Cortex 配置（LLM_BACKEND="snowflake" 时生效）─────────────────
# 模型名可选：claude-sonnet-4-5
# 具体以账号所在区域支持的模型列表为准。
SNOWFLAKE_MODEL: str = _env("SNOWFLAKE_MODEL", "claude-sonnet-4-5")
# 账号地址，形如 https://<account-identifier>.snowflakecomputing.com
SNOWFLAKE_ACCOUNT_URL: str = _env(
    "SNOWFLAKE_ACCOUNT_URL", ""
)
# Programmatic Access Token（PAT），在 Snowsight 用户设置里生成。
# 建议用环境变量 SNOWFLAKE_PAT 传入，避免把密钥写进代码。
SNOWFLAKE_PAT: str = ""


# ── LLM 通用调用参数 ─────────────────────────────────────────────────────
LLM_TEMPERATURE: float = 0.2
LLM_MAX_TOKENS: int = _env_int("LLM_MAX_TOKENS", 8192)
LLM_TIMEOUT: int = _env_int("LLM_TIMEOUT", 120)
LLM_MAX_RETRIES: int = _env_int("LLM_MAX_RETRIES", 3)

# 启动时是否做一次真实调用做健康检查。
# 每次启动都 invoke("ping") 是一次真实计费调用，默认关掉；
# 需要排查连通性时设 SINGLE_AGENT_STARTUP_HEALTHCHECK=1 打开。
STARTUP_HEALTHCHECK: bool = _env_bool("SINGLE_AGENT_STARTUP_HEALTHCHECK", False)


# ── 上下文预算（按后端区分）───────────────────────────────────────────────
# 之前只有一个硬编码的 600_000 字符阈值，既写死了 Claude 的 200K context，
# 又用 "4 字符≈1 token" 估算——对中文严重失准（中文约 1~1.5 字符 = 1 token），
# 结果用 Ollama（8K context）时要超限 70 倍才会告警。
#
# 现在改成：按后端取真实 context 窗口，再乘 CONTEXT_INPUT_RATIO 作为输入预算，
# 并且优先用模型自带 tokenizer 精确计数（见 agent.py），估算只作为兜底。
CONTEXT_LIMITS: dict[str, int] = {
    "ollama": OLLAMA_NUM_CTX,
    "dashscope": _env_int("DASHSCOPE_CONTEXT", 32_768),
    "snowflake": _env_int("SNOWFLAKE_CONTEXT", 200_000),
}
# 输入最多占 context 的比例，其余留给 system prompt、工具定义和模型输出
CONTEXT_INPUT_RATIO: float = 0.6
# 兜底估算用的"每 token 平均字符数"。中文取偏保守的 1.5，宁可早报警不要晚报警。
CHARS_PER_TOKEN_FALLBACK: float = 1.5


# ── 用户没指定路径时的默认工作目录 ─────────────────────────────────────
# 比如用户说"创建 hello.txt"（没说目录），就放到这里。
# 不再硬假设 ~/Desktop 存在（Linux/macOS 上常常没有，或者名字不叫 Desktop）。
def _default_work_dir() -> str:
    override = _env("SINGLE_AGENT_WORK_DIR")
    if override:
        return override
    home = Path(os.path.expanduser("~"))
    desktop = home / "Desktop"
    base = desktop if desktop.is_dir() else home
    # 目录名保持历史值不变，避免用户已有的产出文件被"孤立"在旧目录里
    return str(base / "qwen3_agent_output")


DEFAULT_WORK_DIR: str = _default_work_dir()


# ── 沙箱白名单（所有落盘类工具的最大允许访问范围）─────────────────────────
# 这里列出的目录对以下工具统一生效：
#   write_file / python_exec / run_python_script / git_command /
#   MCP filesystem 全部工具
# 用 SINGLE_AGENT_ALLOWED_DIRS 环境变量可以覆盖（os.pathsep 分隔）。
def _default_allowed_dirs() -> list[str]:
    override = _env("SINGLE_AGENT_ALLOWED_DIRS")
    if override:
        return [p for p in override.split(os.pathsep) if p.strip()]
    return [DEFAULT_WORK_DIR, str(Path(os.path.expanduser("~")))]


ALLOWED_DIRS: list[str] = _default_allowed_dirs()


# ── 代码执行限制 ─────────────────────────────────────────────────────────
# python_exec / run_python_script 子进程的资源上限。
# 0 表示不限制（Windows 上 resource 模块不可用，会自动跳过内存限制）。
EXEC_TIMEOUT_DEFAULT: int = _env_int("SINGLE_AGENT_EXEC_TIMEOUT", 60)
EXEC_MAX_MEMORY_MB: int = _env_int("SINGLE_AGENT_EXEC_MAX_MEMORY_MB", 2048)
EXEC_MAX_OUTPUT_CHARS: int = 20_000


# ── 文件读取截断阈值 ─────────────────────────────────────────────────────
# 避免一次性把超大文件塞进对话上下文顶爆 context。
MAX_READ_CHARS: int = _env_int("SINGLE_AGENT_MAX_READ_CHARS", 40_000)


# ── 界面显示语言 ───────────────────────────────────────────────────────────
# 注意：这个只控制命令行界面的文案（提示符、启动横幅、错误提示等），
# 跟 AI 的回答语言无关——AI 回答语言是自动跟随用户每一条消息的输入语言判断的
# （中文提问用中文答，英文提问用英文答），不受此项或 lang 命令影响。
# "zh" : 界面显示中文（默认）
# "en" : 界面显示 English
# 运行时可在命令行输入 "lang en" / "lang zh" 随时切换，这里只是启动默认值。
DEFAULT_LANGUAGE: str = _env("SINGLE_AGENT_LANGUAGE", "en")


# ── 会话历史与持久化 ─────────────────────────────────────────────────────
# 历史现在由 LangGraph 的 SqliteSaver checkpointer 管理（完整保留工具调用
# 上下文，进程重启也不丢），不再手工重放 message list。
# 空字符串表示只用内存（MemorySaver），进程退出即丢。
CHECKPOINT_DB_PATH: str = _env(
    "SINGLE_AGENT_CHECKPOINT_DB", str(Path(__file__).resolve().parent / "core" / "checkpoints.sqlite")
)
# 单轮对话最多保留多少条历史消息进入模型（按 token 预算裁剪后的硬上限兜底）
HISTORY_MAX_MESSAGES: int = _env_int("SINGLE_AGENT_HISTORY_MAX_MESSAGES", 60)


# ── RAG 开关 ──────────────────────────────────────────────────────────────
# 关掉后 rag_tool 不会被加载，其它工具不受影响。
ENABLE_RAG: bool = _env_bool("SINGLE_AGENT_ENABLE_RAG", True)

# 本地 Embedding 模型路径（bge-small-zh-v1.5）
# 机器相关，务必用环境变量 SINGLE_AGENT_EMBEDDING_MODEL_PATH 覆盖。
EMBEDDING_MODEL_PATH: str = _env(
    "SINGLE_AGENT_EMBEDDING_MODEL_PATH", "F:/Max/llm/bge-small-zh-v1.5"
)
# embedding 分批大小：一次性 encode 整个知识库会 OOM
EMBEDDING_BATCH_SIZE: int = _env_int("SINGLE_AGENT_EMBEDDING_BATCH_SIZE", 32)
# 向量缓存文件：知识库内容不变时跳过重新编码（按内容 hash 判定）
EMBEDDING_CACHE_PATH: str = _env(
    "SINGLE_AGENT_EMBEDDING_CACHE",
    str(Path(__file__).resolve().parent / "rag" / "cache" / "embeddings.npz"),
)

# 知识库文档目录：指向本项目自带的 texts/，保持系统自包含
KNOWLEDGE_BASE_TEXTS_DIR: str = _env(
    "SINGLE_AGENT_TEXTS_DIR", str(Path(__file__).resolve().parent / "texts")
)

# 检索模式：
#   "dense"  — 仅用向量相似度
#   "bm25"   — 仅用 BM25 关键词匹配（无需 embedding 模型）
#   "hybrid" — Dense + BM25 双路 RRF 融合（推荐）
RETRIEVAL_MODE: str = _env("SINGLE_AGENT_RETRIEVAL_MODE", "hybrid")
# 返回给模型的检索结果条数
RETRIEVAL_TOP_K: int = _env_int("SINGLE_AGENT_RETRIEVAL_TOP_K", 3)
# 相关性闸门：dense 余弦相似度的绝对下限。低于这个值视为"知识库里没有"。
# 之前用 top1/top2 的分差做闸门，在 hybrid（RRF）模式下分差恒为 1e-4 量级，
# 只能把阈值设成 0.0，结果任何查询都必然返回一条文档（哪怕完全无关）。
RETRIEVAL_MIN_SCORE: float = float(_env("SINGLE_AGENT_RETRIEVAL_MIN_SCORE", "0.35"))
# 知识库切块参数。原来固定 200 字符太小，一个完整段落经常被切断，
# 检索到的片段常常缺上下文；500 字符 + 80 重叠更适合中文段落。
CHUNK_MAX_CHARS: int = _env_int("SINGLE_AGENT_CHUNK_MAX_CHARS", 500)
CHUNK_OVERLAP_CHARS: int = _env_int("SINGLE_AGENT_CHUNK_OVERLAP_CHARS", 80)
CHUNK_MIN_CHARS: int = _env_int("SINGLE_AGENT_CHUNK_MIN_CHARS", 20)


# ── 联网搜索后端 ─────────────────────────────────────────────────────────
# "baidu"  : 抓百度 HTML（无需 API Key，但依赖页面结构、易被反爬拦、不返回 URL）
# "tavily" : Tavily Search API（需要 TAVILY_API_KEY，返回结构化结果含 URL）
# "serper" : Serper.dev Google API（需要 SERPER_API_KEY）
SEARCH_BACKEND: str = _env("SINGLE_AGENT_SEARCH_BACKEND", "baidu")
SEARCH_TIMEOUT: int = _env_int("SINGLE_AGENT_SEARCH_TIMEOUT", 10)
TAVILY_API_KEY: str = _env("TAVILY_API_KEY")
SERPER_API_KEY: str = _env("SERPER_API_KEY")


# ── ReAct 循环运行参数 ───────────────────────────────────────────────────
# 单轮对话里 agent 最多走多少步（一步 = 一次模型输出或一次工具返回）
AGENT_MAX_STEPS: int = _env_int("SINGLE_AGENT_MAX_STEPS", 20)


# ── 工具函数 ─────────────────────────────────────────────────────────────
def get_allowed_dirs() -> list[str]:
    """返回允许目录列表，并保证默认工作目录存在。"""
    Path(DEFAULT_WORK_DIR).mkdir(parents=True, exist_ok=True)
    return ALLOWED_DIRS


def get_default_work_dir() -> str:
    """返回默认工作目录，并保证它存在。"""
    Path(DEFAULT_WORK_DIR).mkdir(parents=True, exist_ok=True)
    return DEFAULT_WORK_DIR


def format_dirs_for_prompt() -> str:
    """把目录列表格式化成 prompt 里的 markdown 列表。"""
    return "\n".join(f"  - {d}" for d in ALLOWED_DIRS)


def get_context_limit() -> int:
    """当前后端的 context 窗口大小（token）。未知后端给一个保守值。"""
    return CONTEXT_LIMITS.get(LLM_BACKEND, 8192)


def get_input_token_budget() -> int:
    """单轮输入允许占用的最大 token 数。"""
    return max(1024, int(get_context_limit() * CONTEXT_INPUT_RATIO))
