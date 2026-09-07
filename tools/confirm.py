# 写入/编辑/删除类操作前的人工审核闸门
# Co-authored with CoCo

"""
所有会修改或删除文件的工具（write_file / edit_file / move_file /
create_directory / python_exec / run_python_script / git_command）都要在真正
执行前调用 confirm_action，得到用户明确许可才继续。

这是整个系统**最后也是最重要**的一道安全防线：静态代码检查和路径白名单都只是
纵深防御，真正兜底的是"用户看到摘要后点了允许"。

三处相对最初版本的改动：

1. 可插拔 provider（替代原来的 config.CONFIRM_MODE 分支）
   Web 版原来在这个文件里 `from web import event_bus`，导致 confirm.py 和
   config.py 在 CLI/Web 两棵树里长期分叉。现在改成 set_provider() 注入：
   CLI 用默认的终端实现，Web 层启动时注入自己的实现。两棵树的这个文件可以
   保持字节一致。

2. 异步安全
   MCP 工具的确认是在 async 协程内部同步调用的，阻塞式 input() 会把整个事件
   循环卡死（Web 版正因如此必须把 agent 丢到独立线程里跑）。现在额外提供
   aconfirm_action()，把阻塞调用挪到线程池，async 调用点改 await 它即可。

3. 会话内"始终允许"
   一个多步任务原来要按十几次 y。确认疲劳下用户会无脑 y，防护形同虚设。
   现在支持 a（本次会话全部允许）和 t（本次会话该工具全部允许），
   让用户有意识地一次性授权，而不是被迫机械点击。
"""

from __future__ import annotations

import asyncio
import sys
import threading
from typing import Callable

# provider 签名：(摘要, 工具名) -> 是否允许
ConfirmProvider = Callable[[str, str], bool]

_lock = threading.Lock()
_provider: ConfirmProvider | None = None

# 会话内的批量授权状态
_allow_all: bool = False
_allowed_tools: set[str] = set()


# ── provider 注入 ─────────────────────────────────────────────────────────
def set_provider(provider: ConfirmProvider | None) -> None:
    """注入确认实现（Web 层用）。传 None 恢复成终端实现。"""
    global _provider
    with _lock:
        _provider = provider


def reset_session_grants() -> None:
    """清空"本次会话始终允许"的授权。开启新对话时应该调用。"""
    global _allow_all
    with _lock:
        _allow_all = False
        _allowed_tools.clear()


def _check_session_grant(tool_name: str) -> bool:
    with _lock:
        if _allow_all:
            return True
        return bool(tool_name) and tool_name in _allowed_tools


def _grant_all() -> None:
    global _allow_all
    with _lock:
        _allow_all = True


def _grant_tool(tool_name: str) -> None:
    with _lock:
        if tool_name:
            _allowed_tools.add(tool_name)


# ── 默认实现：终端交互 ────────────────────────────────────────────────────
def _confirm_via_cli(summary: str, tool_name: str) -> bool:
    print("\n" + "=" * 60)
    print("[审核] 即将执行以下操作，需要你确认：")
    print(summary)
    print("=" * 60)

    options = "(y=允许一次 / n=拒绝"
    if tool_name:
        options += f" / t=本次会话都允许 {tool_name}"
    options += " / a=本次会话全部允许)"

    while True:
        try:
            choice = input(f"是否允许执行？{options}: ").strip().lower()
        except EOFError:
            # 非交互环境（stdin 被重定向/关闭）：拒绝，而不是抛异常炸掉整轮对话
            print("\n[审核] 检测到非交互式输入，无法确认，按拒绝处理。")
            return False
        except KeyboardInterrupt:
            print("\n[审核] 已中断，按拒绝处理。")
            return False

        if choice in ("y", "yes"):
            return True
        if choice in ("n", "no"):
            return False
        if choice == "t" and tool_name:
            _grant_tool(tool_name)
            print(f"[审核] 已授权：本次会话内 {tool_name} 的调用不再逐次询问。")
            return True
        if choice == "a":
            print(
                "[审核] ⚠️ 已授权：本次会话内所有写入/执行操作都不再询问。\n"
                "        这会关闭最后一道安全闸门，请确认你清楚接下来要做什么。\n"
                "        输入 new 开启新对话会自动恢复逐次确认。"
            )
            _grant_all()
            return True

        print(f"请输入 y 或 n{'、t' if tool_name else ''}、a。")


# ── 对外接口 ─────────────────────────────────────────────────────────────
def confirm_action(summary: str, tool_name: str = "") -> bool:
    """请求人工确认（同步）。返回是否允许执行。

    Args:
        summary: 给用户看的操作摘要，必须包含足以判断风险的信息。
        tool_name: 工具名。给了才能使用"本次会话都允许该工具"这个选项。
    """
    if _check_session_grant(tool_name):
        return True

    with _lock:
        provider = _provider
    return (provider or _confirm_via_cli)(summary, tool_name)


async def aconfirm_action(summary: str, tool_name: str = "") -> bool:
    """请求人工确认（异步安全版）。

    provider 是阻塞式的同步函数，直接在协程里调用会卡死整个事件循环，
    进而导致 Web 版的 /confirm 回调永远进不来（死锁）。这里挪到线程池执行。
    """
    if _check_session_grant(tool_name):
        return True
    return await asyncio.to_thread(confirm_action, summary, tool_name)


def is_interactive() -> bool:
    """当前是否处于可交互终端。非交互时所有确认都会被拒绝，启动时可据此提前告警。"""
    with _lock:
        if _provider is not None:
            return True  # Web 等自定义 provider 自行负责交互
    return bool(getattr(sys.stdin, "isatty", lambda: False)())
