import sys
import asyncio

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except AttributeError:
        pass

from core.llm import build_llm
from tools.local_tools import create_tools
from agent import SingleAgent
from tools.mcp_setup import get_mcp_tools
from skills.skill_loader import get_skill_tools
from tools.rollback import get_rollback_tools
from tools.git_tool import get_git_tools
from config import LLM_BACKEND, DEFAULT_LANGUAGE


_UI_TEXT = {
    "zh": {
        "ready": "单 Agent 系统已就绪（后端：{backend}，界面语言：{lang}）",
        "arch": "  架构：Single ReAct Agent + 全部工具",
        "tool_count": "  可用工具总数：{n}",
        "note_auto": "  AI 回答语言自动跟随你的输入（中文提问用中文答，英文提问用英文答）",
        "hint_new": "  输入 new  → 开启新对话（清空历史）",
        "hint_lang": "  输入 lang en / lang zh → 切换界面显示语言（不影响 AI 回答语言）",
        "hint_exit": "  输入 exit → 退出",
        "prompt": "\n用户: ",
        "bye": "再见！",
        "lang_switched": "界面已切换为中文显示。",
        "lang_unchanged": "界面当前已经是中文显示。",
        "lang_usage": "用法：lang en（界面切英文） / lang zh（界面切中文）",
        "answer_prefix": "\nAI: ",
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
        "note_auto": "  AI response language auto-follows your input (Chinese in -> Chinese out, English in -> English out)",
        "hint_new": "  Type new  -> start a new conversation (clears history)",
        "hint_lang": "  Type lang en / lang zh -> switch UI display language (does not affect AI response language)",
        "hint_exit": "  Type exit -> quit",
        "prompt": "\nYou: ",
        "bye": "Goodbye!",
        "lang_switched": "UI switched to English display.",
        "lang_unchanged": "UI is already displayed in English.",
        "lang_usage": "Usage: lang en (switch UI to English) / lang zh (switch UI to Chinese)",
        "answer_prefix": "\nAI: ",
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


async def main():
    llm = build_llm()
    local_tools = create_tools()
    mcp_tools = await get_mcp_tools()
    all_tools = (
        local_tools + get_skill_tools() + get_rollback_tools()
        + get_git_tools() + mcp_tools
    )

    lang = DEFAULT_LANGUAGE if DEFAULT_LANGUAGE in ("zh", "en") else "zh"
    agent = SingleAgent(llm=llm, tools=all_tools)

    print("=" * 60)
    print(_t(lang, "ready", backend=LLM_BACKEND, lang=lang))
    print(_t(lang, "arch"))
    print(_t(lang, "tool_count", n=len(all_tools)))
    print(_t(lang, "note_auto"))
    print(_t(lang, "hint_new"))
    print(_t(lang, "hint_lang"))
    print(_t(lang, "hint_exit"))
    print("=" * 60)

    while True:
        try:
            user_input = input(_t(lang, "prompt")).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_t(lang, 'bye')}")
            agent.close()
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered == "exit":
            print(_t(lang, "bye"))
            agent.close()
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
            answer = await agent.arun(user_input)
            print(f"{_t(lang, 'answer_prefix')}{answer}")
        except Exception:
            print(_t(lang, "error", log_path=agent.log.path))


if __name__ == "__main__":
    asyncio.run(main())
else:
    async def run_in_jupyter():
        await main()
