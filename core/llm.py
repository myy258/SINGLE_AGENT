# LLM 工厂：按 config.LLM_BACKEND 切换 Ollama / DashScope / Snowflake Cortex
# Co-authored with CoCo

"""
本版相对最初版本的改动：

1. 「只能在 Windows 上跑，而且带着别人的用户名」
   原来硬编码了 C:/Users/M172504/AppData/Local/Programs/Ollama/ollama app.exe，
   并用 os.startfile（仅 Windows）拉起服务。换台机器或换 OS 全部失效。
   现在：先看 config.OLLAMA_APP_PATH（可用环境变量给），再用 shutil.which 在
   PATH 里找 ollama，最后才试各平台的常见安装位置；启动方式按平台分流。

2. 「每次启动都花一次钱做健康检查」
   原来无条件 llm.invoke("ping")，snowflake 分支最多 ping 两次——每次启动都是
   真实计费调用。现在默认关闭（config.STARTUP_HEALTHCHECK），需要排查连通性时
   用环境变量打开。Ollama 分支改成查 /api/tags（免费且顺带校验模型名）。

3. 「模型名写错只在第一次真实对话时才炸」
   Ollama 分支现在会拿 /api/tags 的结果核对模型名，写错就直接报错并列出本机
   实际有哪些模型，而不是等到用户提第一个问题时才失败。

4. 调用参数（temperature / max_tokens / timeout / retries）从这里挪进 config，
   不再散落在函数体里。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.error import URLError

from config import (
    DASHSCOPE_API_KEY,
    DASHSCOPE_BASE_URL,
    DASHSCOPE_MODEL,
    LLM_BACKEND,
    LLM_MAX_RETRIES,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    OLLAMA_APP_PATH,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    SNOWFLAKE_ACCOUNT_URL,
    SNOWFLAKE_MODEL,
    SNOWFLAKE_PAT,
    STARTUP_HEALTHCHECK,
)

# 各平台上 Ollama 的常见安装位置（找不到就提示用户手动启动，不再猜某个人的用户名）
_OLLAMA_APP_CANDIDATES: tuple[Path, ...] = (
    # Windows
    Path(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama app.exe")),
    Path(os.path.expandvars(r"%PROGRAMFILES%\Ollama\ollama app.exe")),
    # macOS
    Path("/Applications/Ollama.app/Contents/MacOS/Ollama"),
    # Linux
    Path("/usr/local/bin/ollama"),
    Path("/usr/bin/ollama"),
)


def _ollama_tags() -> dict | None:
    """查 /api/tags。服务没起来返回 None。这是一个免费调用，可放心用作健康检查。"""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=3) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, ConnectionError, OSError, json.JSONDecodeError):
        return None


def _find_ollama_executable() -> Path | None:
    if OLLAMA_APP_PATH:
        p = Path(OLLAMA_APP_PATH).expanduser()
        if p.exists():
            return p
        print(f"[Ollama] 警告：config.OLLAMA_APP_PATH 指向的文件不存在：{p}")

    found = shutil.which("ollama")
    if found:
        return Path(found)

    return next((p for p in _OLLAMA_APP_CANDIDATES if p.exists()), None)


def _launch_ollama(app_path: Path) -> None:
    """按平台启动 Ollama。"""
    print(f"[Ollama] 尝试启动：{app_path}")
    if sys.platform == "win32":
        os.startfile(str(app_path))  # type: ignore[attr-defined]  # noqa: S606 - 仅 Windows
        return
    if sys.platform == "darwin" and app_path.suffix != "":
        subprocess.Popen(["open", "-a", "Ollama"], start_new_session=True)
        return
    # Linux / macOS 命令行版：起一个后台 serve
    subprocess.Popen(
        [str(app_path), "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _ensure_ollama_running(timeout: int = 30) -> dict:
    """确保本地 Ollama 服务可用，没跑就尝试启动。返回 /api/tags 的内容。"""
    tags = _ollama_tags()
    if tags is not None:
        return tags

    app_path = _find_ollama_executable()
    if app_path is None:
        raise RuntimeError(
            "Ollama 服务没有运行，也没能在本机找到 ollama 可执行文件。\n"
            "请先手动启动 Ollama，或用环境变量 OLLAMA_APP_PATH 指定它的路径。\n"
            f"（当前查询的服务地址：{OLLAMA_BASE_URL}）"
        )

    try:
        _launch_ollama(app_path)
    except OSError as exc:
        raise RuntimeError(f"启动 Ollama 失败：{exc}，请手动启动后重试。") from exc

    for _ in range(timeout):
        tags = _ollama_tags()
        if tags is not None:
            print("[Ollama] 启动成功")
            return tags
        time.sleep(1)

    raise RuntimeError(f"Ollama 启动超时（{timeout}s），请手动检查 Ollama 是否正常运行。")


def _assert_ollama_model_available(tags: dict) -> None:
    """核对模型名。写错就立刻报错并列出本机实际可用的模型。"""
    available = [m.get("name", "") for m in tags.get("models", []) if isinstance(m, dict)]
    if not available:
        print("[Ollama] 警告：/api/tags 没返回任何模型，跳过模型名校验。")
        return
    # ollama 的 tag 形如 "qwen2.5:7b"；允许用户省略 ":latest"
    normalized = {name.removesuffix(":latest") for name in available}
    if OLLAMA_MODEL in normalized or OLLAMA_MODEL in available:
        return
    raise RuntimeError(
        f"Ollama 里没有模型「{OLLAMA_MODEL}」。\n"
        f"本机实际可用的模型：{', '.join(sorted(available)) or '（无）'}\n"
        f"请改 config.OLLAMA_MODEL（或设环境变量 OLLAMA_MODEL），"
        f"或先执行：ollama pull {OLLAMA_MODEL}"
    )


def _healthcheck(llm, label: str) -> None:
    """可选的真实调用健康检查。默认不做——每次启动 ping 一次是要计费的。

    成功路径不打印任何东西（启动期保持安静）；只有失败才抛错。
    """
    if not STARTUP_HEALTHCHECK:
        return
    try:
        llm.invoke("ping")
    except Exception as exc:
        raise RuntimeError(
            f"{label} 调用失败：{exc}\n请检查 API Key / 网络 / 模型名 / 权限。"
        ) from exc


def _build_ollama():
    tags = _ensure_ollama_running()
    _assert_ollama_model_available(tags)

    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=LLM_TEMPERATURE,
        num_ctx=OLLAMA_NUM_CTX,
    )
    return llm


def _build_dashscope():
    from langchain_openai import ChatOpenAI

    api_key = os.environ.get("DASHSCOPE_API_KEY") or DASHSCOPE_API_KEY
    if not api_key:
        raise RuntimeError(
            "未找到 DashScope API Key。\n"
            "推荐：设置环境变量 DASHSCOPE_API_KEY=sk-...\n"
            "（也可以写在 config.py 里，但不要把带密钥的 config.py 提交进 git）"
        )
    llm = ChatOpenAI(
        model=DASHSCOPE_MODEL,
        base_url=DASHSCOPE_BASE_URL,
        api_key=api_key,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        max_retries=LLM_MAX_RETRIES,
        timeout=LLM_TIMEOUT,
    )
    _healthcheck(llm, f"DashScope API：{DASHSCOPE_MODEL}")
    return llm


def _build_snowflake():
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
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        max_retries=LLM_MAX_RETRIES,
        timeout=LLM_TIMEOUT,
    )

    # 该模型经由 Bedrock 调用，对"一轮里并行发起多个工具调用"的配对校验很严格
    # （每个 toolUse 都必须配上 toolResult，少一个就 400）。关掉并行工具调用可以
    # 降低这类错误的概率。
    #
    # 注意：原来这里用"构造 + invoke('ping') 是否成功"来判断 Cortex 支不支持这个
    # 参数，等于把一次计费调用当成特性探测，而且 ping 失败的真实原因可能只是网络。
    # 现在不做探测：直接带上这个参数（OpenAI 兼容端点对未知参数通常忽略），
    # 真出问题时由 agent.py 的断点续跑重试兜底。
    llm = ChatOpenAI(**common_kwargs, model_kwargs={"parallel_tool_calls": False})
    _healthcheck(llm, f"Snowflake Cortex REST API：{SNOWFLAKE_MODEL}")
    return llm


_BUILDERS = {
    "ollama": _build_ollama,
    "dashscope": _build_dashscope,
    "snowflake": _build_snowflake,
}


def build_llm():
    """返回一个 LangChain ChatModel（支持 bind_tools），上层不区分后端。"""
    builder = _BUILDERS.get(LLM_BACKEND)
    if builder is None:
        raise ValueError(
            f"未知的 LLM_BACKEND: {LLM_BACKEND}。可选：{', '.join(sorted(_BUILDERS))}"
        )
    return builder()
