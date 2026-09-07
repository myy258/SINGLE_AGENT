# 单 Agent 系统的交互入口：加载 LLM、本地工具、MCP 工具，进入命令行对话循环
# Co-authored with CoCo

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# 让 `python /任意路径/SINGLE_AGENT/main.py` 也能跑。
# 原来 `from config import ...` 依赖当前工作目录恰好是 SINGLE_AGENT/，
# 从别处启动会直接 ImportError。
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except AttributeError:
        pass

from agent import SingleAgent  # noqa: E402
from config import DEFAULT_LANGUAGE, LLM_BACKEND  # noqa: E402
from core.llm import build_llm  # noqa: E402
from skills.skill_loader import get_skill_tools  # noqa: E402
from tools.confirm import is_interactive  # noqa: E402
from tools.git_tool import get_git_tools  # noqa: E402
from tools.local_tools import create_tools  # noqa: E402
from tools.mcp_setup import get_mcp_tools, shutdown_mcp  # noqa: E402
from tools.rollback import get_rollback_tools  # noqa: E402

_UI_TEXT = {
    "zh": {
        "ready": "单 Agent 系统已就绪（后端：{backend}，界面语言：{lang}）",
        "arch": "  架构：Single ReAct Agent + 全部工具",
        "tool_count": "  可用工具总数：{n}",
        "note_auto": "  AI 回答语言自动跟随你的输入（中文提问用中文答，英文提问用英文答）",
        "note_confirm": "  写入/执行类操作会逐次询问；确认时可选 a=本次会话全部允许",
        "hint_new": "  输入 new  → 开启新对话（清空历史）",
        "hint_lang": "  输入 lang en / lang zh → 切换界面显示语言（不影响 AI 回答语言）",
        "hint_exit": "  输入 exit → 退出",
        "warn_noninteractive": (
            "  ⚠️ 当前不是交互式终端，所有需要确认的操作都会被自动拒绝。"
        ),
        "prompt": "\n用户: ",
        "bye": "再见！",
        "lang_switched": "界面已切换为中文显示。",
        "lang_unchanged": "界面当前已经是中文显示。",
        "lang_usage": "用法：lang en（界面切英文） / lang zh（界面切中文）",
        "answer_prefix": "\nAI: ",
        "thinking": "\nAI 正在处理...",
        "error": (
            "\nAI: 抱歉，这次请求处理时出现异常，请重新提问一次，"
            "不会影响之前的对话记录。完整技术细节已记录在会话日志里，"
            "日志路径：{log_path}"
        ),
    },
    "en": {
        "ready": "Single Agent system ready (backend: {backend}, UI language: {lang})",
        "arch": "  Architecture: Single ReAct Agent + all tools",
        "tool_count": "  Total tools available: {n}",
        "note_auto": (
            "  AI response language auto-follows your input "
            "(Chinese in -> Chinese out, English in -> English out)"
        ),
        "note_confirm": (
            "  Write/exec actions ask for confirmation; press a to allow all for this session"
        ),
        "hint_new": "  Type new  -> start a new conversation (clears history)",
        "hint_lang": (
            "  Type lang en / lang zh -> switch UI display language "
            "(does not affect AI response language)"
        ),
        "hint_exit": "  Type exit -> quit",
        "warn_noninteractive": (
            "  ⚠️ Not an interactive terminal; every action needing confirmation "
            "will be auto-denied."
        ),
        "prompt": "\nYou: ",
        "bye": "Goodbye!",
        "lang_switched": "UI switched to English display.",
        "lang_unchanged": "UI is already displayed in English.",
        "lang_usage": "Usage: lang en (switch UI to English) / lang zh (switch UI to Chinese)",
        "answer_prefix": "\nAI: ",
        "thinking": "\nAI is working...",
        "error": (
            "\nAI: Sorry, this request hit an error — please try asking again, "
            "this won't affect earlier conversation history. Full technical details "
            "were logged to the session log, path: {log_path}"
        ),
    },
}


def _t(ui_lang: str, key: str, **kwargs) -> str:
    text = _UI_TEXT.get(ui_lang, _UI_TEXT["zh"])[key]
    return text.format(**kwargs) if text and kwargs else text


def _print_banner(lang: str, tool_count: int) -> None:
    print("=" * 60)
    print(_t(lang, "ready", backend=LLM_BACKEND, lang=lang))
    print(_t(lang, "arch"))
    print(_t(lang, "tool_count", n=tool_count))
    print(_t(lang, "note_auto"))
    print(_t(lang, "note_confirm"))
    print(_t(lang, "hint_new"))
    print(_t(lang, "hint_lang"))
    print(_t(lang, "hint_exit"))
    if not is_interactive():
        print(_t(lang, "warn_noninteractive"))
    print("=" * 60)


async def _ask(agent: SingleAgent, user_input: str, lang: str) -> None:
    """提问并流式打印回答。

    原来用 ainvoke，长任务期间用户面对空白终端，不知道是在跑还是卡死了。
    现在边生成边打印；如果后端不支持流式（一个 token 都没回调），
    再把最终答案整体打印出来，行为不会退化。
    """
    printed_any = False
    prefix = _t(lang, "answer_prefix")

    def on_token(piece: str) -> None:
        nonlocal printed_any
        if not printed_any:
            print(prefix, end="", flush=True)
            printed_any = True
        print(piece, end="", flush=True)

    answer = await agent.arun(user_input, on_token=on_token)

    if printed_any:
        print()  # 收尾换行
    else:
        print(f"{prefix}{answer}")


async def main() -> None:
    llm = build_llm()
    local_tools = create_tools()
    mcp_tools = await get_mcp_tools()
    all_tools = (
        local_tools + get_skill_tools() + get_rollback_tools()
        + get_git_tools() + mcp_tools
    )

    lang = DEFAULT_LANGUAGE if DEFAULT_LANGUAGE in ("zh", "en") else "zh"
    agent = SingleAgent(llm=llm, tools=all_tools)

    _print_banner(lang, len(all_tools))

    try:
        while True:
            try:
                user_input = input(_t(lang, "prompt")).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{_t(lang, 'bye')}")
                break

            if not user_input:
                continue

            lowered = user_input.lower()
            if lowered == "exit":
                print(_t(lang, "bye"))
                break
            if lowered == "new":
                agent.new_conversation()
                continue
            if lowered in ("lang en", "lang zh"):
                new_lang = "en" if lowered == "lang en" else "zh"
                if new_lang == lang:
                    print(_t(lang, "lang_unchanged"))
                else:
                    lang = new_lang
                    print(_t(lang, "lang_switched"))
                continue
            if lowered == "lang":
                print(_t(lang, "lang_usage"))
                continue

            try:
                await _ask(agent, user_input, lang)
            except Exception:
                print(_t(lang, "error", log_path=agent.log.path))
    finally:
        # 原来这些清理只在部分退出分支里做，Ctrl-C 走 except 分支时
        # MCP 的 npx 子进程会留着不回收。
        agent.close()
        await shutdown_mcp()


if __name__ == "__main__":
    asyncio.run(main())
else:
    async def run_in_jupyter() -> None:
        await main()
