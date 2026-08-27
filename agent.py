"""
架构（极简）：
    用户输入 ─► SingleAgent.arun ─► create_react_agent (ReAct 循环)
                     │                     │
                     │            所有工具（write_file / python_exec /
                     │            run_python_script / calculator /
                     │            current_time / baidu_search /
                     │            search_local_knowledge_base /
                     │            MCP filesystem 工具集）
                     │
                     └─► 会话记忆（instance 属性）+ 日志

关键设计：
- 单个 LLM，无 supervisor / worker 分工
- 直接 create_react_agent 挂全部工具
- 会话历史用 list[BaseMessage] 保存，每轮追加
- 硬上限：ReAct 步数 ≤ SINGLE_AGENT_MAX_STEPS
"""

from langchain_core.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from langgraph.prebuilt import create_react_agent

from config import (
    LLM_BACKEND, SNOWFLAKE_MODEL, OLLAMA_MODEL, DASHSCOPE_MODEL,
    get_default_work_dir, format_dirs_for_prompt,
)
from core.logger import SessionLogger
from skills.skill_loader import format_skill_index_for_prompt


SINGLE_AGENT_MAX_STEPS: int = 20

# 留给 Cortex claude-sonnet-4-5（200K token context）的安全余量：
# 按 4 字符≈1 token 估算，预留 system prompt / 工具定义 / 输出空间后的警戒线
SAFE_INPUT_CHAR_LIMIT: int = 600_000

# 模型经由 Bedrock 调用，偶发会在一轮里并行发起多个工具调用导致
# "toolUse 块没有配对上 toolResult" 的 400 报错——这是概率性的生成异常，
# 同样的输入重新生成一次通常就不会再触发，所以做应用层重试而不是直接报错。
_TOOL_PAIR_ERROR_MARKERS: tuple[str, str] = ("toolUse", "toolResult")
_TOOL_PAIR_ERROR_MAX_ATTEMPTS: int = 3


def _estimate_char_len(messages: list[BaseMessage]) -> int:
    total = 0
    for m in messages:
        content = m.content
        if isinstance(content, list):
            content = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b) for b in content
            )
        total += len(str(content))
    return total


def _get_model_identity() -> str:
    if LLM_BACKEND == "snowflake":
        return f"{SNOWFLAKE_MODEL}（Snowflake Cortex）"
    if LLM_BACKEND == "ollama":
        return f"{OLLAMA_MODEL}（本地 Ollama）"
    if LLM_BACKEND == "dashscope":
        return f"{DASHSCOPE_MODEL}（DashScope API）"
    return "未知模型"


def _build_system_prompt(model_identity: str) -> str:
    return f"""\
你是基于 {model_identity} 的智能助手。你能独立完成多种任务：
写代码、跑分析、写作、翻译、常识问答、联网搜索、本地知识库检索、文本文件读写等。

【工具使用准则】
1. 涉及数据分析、数值计算、跑代码：
   - 首选 python_exec 直接在内部运行代码。用 print(...) 把结果打印出来。
   - 报错就根据 stderr 修正后重跑，不要原样重复。
   - **只有用户明确要求"生成 py 脚本 / 保存成 .py 文件"时**才用
     write_file + run_python_script。
2. 简单算数用 calculator；查时间用 current_time。
3. 涉及具体人物、公司规章、内部资料等本地信息 →
   先用 search_local_knowledge_base 检索本地知识库；本地找不到再用 baidu_search。
4. 文本文件读写：优先用 read_file / write_file / list_directory 等。
5. 用户要求"撤销/回滚/恢复上一个版本"某次写入/编辑/移动/建目录操作时：
   先用 list_backups 看看有哪些可回滚的记录，找到对应 id 后调用 rollback(id)。
   注意 python_exec / run_python_script 跑的代码不在可回滚范围内。
6. **git_command 不受 MCP 文件访问白名单（read_file/write_file/list_directory 那套
   ALLOWED_DIRS 限制）的约束，可以操作系统上任意路径**。涉及 git/GitHub 相关任务
   （提交、推送、打 tag、发布等）时，第一步永远直接用 git_command，不要先用
   list_directory/read_file 去试探目录是否存在——那样做即使那个目录不在 MCP 白名单
   里也完全不影响 git_command 正常工作。如果 list_directory/read_file 对某个路径报
   "不在允许访问范围"，那只是这些工具自己的限制，绝不代表 git_command 也用不了，
   也绝不代表整个任务做不到，不要因此下结论说"无法访问"或让用户手动执行命令。

【可用技能】
遇到以下场景时，先调用 load_skill(名字) 拿到完整步骤指引，再照着执行，
不要凭自己猜测流程：
{format_skill_index_for_prompt()}

【文件路径规则】
- 相对路径 / 纯文件名 → 落到默认工作目录（用户看得到）
- 绝对路径 → 用户明确指定的位置，保留不动

【禁止】
- 不要凭猜测编造"运行成功 / 输出是..."；一切结论以真实工具返回为准。
- 不要重复调用已经明确返回"未找到"的工具。

【允许的目录】
{format_dirs_for_prompt()}

【默认工作目录】
{get_default_work_dir()}

【回答语言（自动跟随）】
根据用户最近这一条消息使用的语言来回答：
- 用户用中文提问 → 用中文回答
- 用户用英文提问 → 用英文回答
- 语言不明确、中英混用，或消息只是代码/数字等无法判断语言时 → 默认用中文回答
每一轮独立判断，不要被上一轮的语言"锁定"；同一轮回答内不要中英文混杂。
"""


class SingleAgent:
    """单 agent 版：一个 ReAct 循环包干所有任务。"""

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool]):
        self.llm = llm
        self.tools = tools
        self.model_identity = _get_model_identity()
        self.system_prompt = _build_system_prompt(self.model_identity)

        print(f"[SingleAgent] 装载工具 {len(tools)} 个：{[t.name for t in tools]}")

        self._history: list[BaseMessage] = []
        self.log = SessionLogger()
        print(f"[SingleAgent] 会话日志：{self.log.path}")

        self.agent = create_react_agent(llm, tools=tools, prompt=self.system_prompt)

    # ── 公开接口 ─────────────────────────────────────────────────────────
    async def arun(self, user_input: str) -> str:
        return await self._run(user_input)

    def run(self, user_input: str) -> str:
        import asyncio
        return asyncio.run(self._run(user_input))

    async def _run(self, user_input: str) -> str:
        self.log.start_turn(user_input)

        messages: list[BaseMessage] = list(self._history) + [HumanMessage(content=user_input)]

        char_len = _estimate_char_len(messages)
        if char_len > SAFE_INPUT_CHAR_LIMIT:
            self.log.event(
                "Warning",
                f"本轮输入约 {char_len} 字符（≈{char_len // 4} tokens），"
                f"接近/超过模型 200K context 上限，可能触发 500 错误。建议分段处理大文件。",
            )
            print(f"[SingleAgent] 警告：本轮输入约 {char_len} 字符，接近上下文上限，可能报错。")
        else:
            self.log.event("Info", f"本轮输入约 {char_len} 字符（≈{char_len // 4} tokens）")

        result = None
        for attempt in range(1, _TOOL_PAIR_ERROR_MAX_ATTEMPTS + 1):
            try:
                result = await self.agent.ainvoke(
                    {"messages": messages},
                    config={"recursion_limit": SINGLE_AGENT_MAX_STEPS * 2 + 4},
                )
                break
            except Exception as e:
                msg = str(e)
                is_tool_pair_error = all(marker in msg for marker in _TOOL_PAIR_ERROR_MARKERS)
                if is_tool_pair_error and attempt < _TOOL_PAIR_ERROR_MAX_ATTEMPTS:
                    self.log.event(
                        "Agent",
                        f"第 {attempt} 次尝试触发工具调用配对异常，自动重试：{e}",
                    )
                    continue
                self.log.event("Agent", f"执行异常：{e}")
                raise

        new_msgs = result.get("messages", [])
        answer = self._extract_final_answer(new_msgs)
        self._log_trace(new_msgs, base_len=len(messages))

        self._history.append(HumanMessage(content=user_input))
        self._history.append(AIMessage(content=answer))
        self._history = self._history[-20:]

        self.log.end_turn(answer)
        return answer

    def new_conversation(self) -> None:
        self._history = []
        try:
            self.log.close()
        except Exception:
            pass
        self.log = SessionLogger()
        print(f"[SingleAgent] 已开启新会话，日志：{self.log.path}")

    def close(self) -> None:
        try:
            self.log.close()
        except Exception:
            pass

    # ── 辅助 ─────────────────────────────────────────────────────────────
    @staticmethod
    def _extract_final_answer(messages: list[BaseMessage]) -> str:
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                content = m.content
                if isinstance(content, list):
                    content = "".join(
                        b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                    )
                text = str(content).strip()
                if text:
                    return text
        return "（未生成有效答复）"

    def _log_trace(self, messages: list[BaseMessage], base_len: int) -> None:
        step = 0
        for m in messages[base_len:]:
            if isinstance(m, AIMessage):
                step += 1
                content = m.content
                if isinstance(content, list):
                    content = "".join(
                        b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                    )
                text = str(content).strip()
                if text:
                    self.log.agent_step(step, text if len(text) < 800 else text[:800] + "...(截断)")
                for tc in getattr(m, "tool_calls", []) or []:
                    self.log.event(
                        f"TOOL_CALL:{tc.get('name', '?')}",
                        f"args={tc.get('args', {})}",
                    )
            elif isinstance(m, ToolMessage):
                result_text = str(m.content)
                self.log.tool_call(
                    tool_name=getattr(m, "name", "?"),
                    args={},
                    result=result_text,
                )
