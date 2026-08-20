# 本 Agent 系统的全局配置：LLM 后端选择、Snowflake/DashScope/Ollama 参数、RAG 开关、路径等。


"""
只需这一个文件，就能切换 LLM 后端 / 调整路径 / 打开关闭 RAG。

核心原则：
  - DEFAULT_WORK_DIR：用户没说目录时，文件默认放哪儿（确保存在）
  - ALLOWED_DIRS    ：MCP 沙箱工具（文件读写）允许访问的位置，可以多个
"""

import os
from pathlib import Path


# 选择 LLM 后端（三选一）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# "ollama"     : 本地 Ollama 服务
# "dashscope"  : 阿里云 Qwen API
# "snowflake"  : Snowflake Cortex REST API（云端 LLM，OpenAI 兼容协议）
LLM_BACKEND: str = "snowflake"

# Ollama 模型配置（LLM_BACKEND="ollama" 时有效）
OLLAMA_MODEL: str = "qwen3.6-27b"
OLLAMA_BASE_URL: str = "http://localhost:11434"

# DashScope 模型配置（LLM_BACKEND="dashscope" 时有效）
DASHSCOPE_MODEL: str = "qwen3.5-27b"
DASHSCOPE_BASE_URL: str = "https://ws-xdlfwulc72sarf83.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

# DashScope API Key（请替换为你自己的密钥）
DASHSCOPE_API_KEY: str = "your-dashscope-api-key-here"

# 配置 Snowflake Cortex 参数（LLM_BACKEND="snowflake" 时有效）━━━━━━━━━━━━━━━━
# 模型名称（可选：claude-sonnet-4-5、llama3.1-70b、mistral-large2 等）
# 具体以你账号所在区域支持的模型列表为准。
SNOWFLAKE_MODEL: str = "claude-sonnet-4-5"
# 账号地址（格式：https://<account-identifier>.snowflakecomputing.com）
SNOWFLAKE_ACCOUNT_URL: str = "https://your-account.snowflakecomputing.com"
# Programmatic Access Token（PAT）：在 Snowsight 用户设置中生成。
# 建议从环境变量 SNOWFLAKE_PAT 读入，或者自己写在这里。
SNOWFLAKE_PAT: str = "your-snowflake-pat-token-here"


# 配置 「用户没指定路径时」默认工作目录 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 即：用户说"生成 hello.txt"，没说目录，就放到这里
DEFAULT_WORK_DIR: str = str(
    Path(os.path.expanduser("~")) / "Desktop" / "qwen3_agent_output"
)

# 配置 MCP 文件工具（read_file/write_file/list_directory 等）允许访问的目录 ━━━━━━━
# 可以多个，用列表表示。Agent 只能在这些目录及其子目录下读写文件。
ALLOWED_DIRS: list[str] = [
    DEFAULT_WORK_DIR,
    str(Path(os.path.expanduser("~"))),  # 用户主目录
]

# 配置 RAG（本地知识库检索）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 是否启用 RAG 功能
ENABLE_RAG: bool = True
# 知识库文本文件所在目录（相对于本项目根目录）
RAG_TEXT_DIR: str = "texts"
# 向量数据库存储路径（相对于本项目根目录）
RAG_VECTOR_DB_PATH: str = "vector_db"
# 检索时返回的最相关文档数量
RAG_TOP_K: int = 3
