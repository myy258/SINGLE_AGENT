# LLM 工厂：按 config.LLM_BACKEND 切换 Ollama / DashScope / Snowflake Cortex
# Co-authored with CoCo

import os
import time
import urllib.request
from pathlib import Path
from urllib.error import URLError

from config import (
    LLM_BACKEND,
    OLLAMA_MODEL, OLLAMA_BASE_URL,
    DASHSCOPE_MODEL, DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY,
    SNOWFLAKE_MODEL, SNOWFLAKE_ACCOUNT_URL, SNOWFLAKE_PAT,
)


_OLLAMA_APP_CANDIDATES = [
    Path("C:/Users/M172504/AppData/Local/Programs/Ollama/ollama app.exe"),
]


def _ensure_ollama_running(timeout: int = 30) -> None:
    """确保本地 Ollama 服务可用，没跑就启动桌面应用。"""
    def alive() -> bool:
        try:
            urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
            return True
        except (URLError, ConnectionError, OSError):
            return False

    if alive():
        return

    app_path = next((p for p in _OLLAMA_APP_CANDIDATES if p.exists()), None)
    if app_path is None:
        raise RuntimeError(
            "找不到 Ollama 桌面应用。请确认已安装 Ollama，或手动打开它。"
        )

    print(f"[Ollama] 启动桌面应用：{app_path.name}")
    os.startfile(str(app_path))   # noqa: use only on Windows

    for _ in range(timeout):
        if alive():
            print("[Ollama] 启动成功")
            return
        time.sleep(1)

    raise RuntimeError(f"Ollama 启动超时（{timeout}s），请手动检查任务栏 🦙 图标。")


def build_llm():
    """返回一个 LangChain ChatModel（支持 bind_tools），上层不区分后端。"""
    # 三家最大公约数：只用 temperature（+ Ollama 独有的 num_ctx）
    if LLM_BACKEND == "ollama":
        _ensure_ollama_running()
        from langchain_ollama import ChatOllama
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
            num_ctx=8192,
        )
        print(f"[LLM] 已连接本地 Ollama：{OLLAMA_MODEL}（num_ctx=8192）")
        return llm

    if LLM_BACKEND == "dashscope":
        from langchain_openai import ChatOpenAI
        api_key = os.environ.get("DASHSCOPE_API_KEY") or DASHSCOPE_API_KEY
        if not api_key:
            raise RuntimeError(
                "未找到 DashScope API Key。\n"
                "方式 1：在 config.py 里设置 DASHSCOPE_API_KEY = 'sk-...'\n"
                "方式 2：设置环境变量 DASHSCOPE_API_KEY=sk-..."
            )
        llm = ChatOpenAI(
            model=DASHSCOPE_MODEL,
            base_url=DASHSCOPE_BASE_URL,
            api_key=api_key,
            temperature=0.2,
            max_tokens=8192,
            max_retries=3,
            timeout=120,
        )
        try:
            llm.invoke("ping")
            print(f"[LLM] 已连接 DashScope API：{DASHSCOPE_MODEL}")
        except Exception as e:
            raise RuntimeError(
                f"DashScope API 调用失败：{e}\n"
                "请检查 API Key / 网络 / 模型名。"
            )
        return llm

    if LLM_BACKEND == "snowflake":
        from langchain_openai import ChatOpenAI
        account_url = os.environ.get("SNOWFLAKE_ACCOUNT_URL") or SNOWFLAKE_ACCOUNT_URL
        api_key = os.environ.get("SNOWFLAKE_PAT") or SNOWFLAKE_PAT
        if not account_url:
            raise RuntimeError("未找到 Snowflake 账号地址（SNOWFLAKE_ACCOUNT_URL）。")
        if not api_key:
            raise RuntimeError("未找到 Snowflake Programmatic Access Token（SNOWFLAKE_PAT）。")
        common_kwargs = dict(
            model=SNOWFLAKE_MODEL,
            base_url=f"{account_url.rstrip('/')}/api/v2/cortex/v1",
            api_key=api_key,
            temperature=0.2,
            max_tokens=16000,
            max_retries=3,
            timeout=120,
        )
        # 该模型经由 Bedrock 调用，对"一轮里并行发起多个工具调用"的配对校验很严格
        # （每个 toolUse 都必须配上 toolResult，少一个就 400），
        # 先尝试关掉并行工具调用降低这类错误概率；如果 Cortex 不支持这个参数，
        # ping 会失败，自动退回不带这个参数的版本。
        try:
            llm = ChatOpenAI(**common_kwargs, model_kwargs={"parallel_tool_calls": False})
            llm.invoke("ping")
            print(f"[LLM] 已连接 Snowflake Cortex REST API：{SNOWFLAKE_MODEL}")
            return llm
        except Exception:
            print("[LLM] parallel_tool_calls=False 不被支持，回退为默认并行工具调用。")

        try:
            llm = ChatOpenAI(**common_kwargs)
            llm.invoke("ping")
            print(f"[LLM] 已连接 Snowflake Cortex REST API：{SNOWFLAKE_MODEL}")
        except Exception as e:
            raise RuntimeError(
                f"Snowflake Cortex API 调用失败：{e}\n"
                "请检查账号地址、PAT、CORTEX_USER 权限、模型名。"
            )
        return llm

    raise ValueError(f"未知的 LLM_BACKEND: {LLM_BACKEND}")
