# 单 Agent 系统的交互入口：加载 LLM、本地工具、MCP 工具，进入命令行对话循环

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
from config import LLM_BACKEND


async def main():
    llm = build_llm()
    local_tools = create_tools()
    mcp_tools = await get_mcp_tools()
    all_tools = (
        local_tools + get_skill_tools() + get_rollback_tools()
        + get_git_tools() + mcp_tools
    )

    agent = SingleAgent(llm=llm, tools=all_tools)

    print("=" * 60)
    print(f"单 Agent 系统已就绪（后端：{LLM_BACKEND}）")
    print("  架构：Single ReAct Agent + 全部工具")
    print(f"  可用工具总数：{len(all_tools)}")
    print("  输入 new  → 开启新对话（清空历史）")
    print("  输入 exit → 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n用户: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            agent.close()
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("再见！")
            agent.close()
            break
        if user_input.lower() == "new":
            agent.new_conversation()
            continue

        try:
            answer = await agent.arun(user_input)
            print(f"\nAI: {answer}")
        except Exception:
            print(
                "\nAI: 抱歉，这次请求处理时出现异常，请重新提问一次，"
                "不会影响之前的对话记录。完整技术细节已记录在会话日志里，"
                f"日志路径：{agent.log.path}"
            )


if __name__ == "__main__":
    asyncio.run(main())
else:
    async def run_in_jupyter():
        await main()
