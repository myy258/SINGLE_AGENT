# 单 Agent 系统：直接用 create_react_agent 挂载所有工具，无 supervisor 无 worker
# Co-authored with CoCo
"""
架构（极简）：
    用户输入 ─► SingleAgent.arun ─► create_react_agent (ReAct 循环)
                     │                     │
                     │            所有工具（write_file / python_exec /
                     │            run_python_script / calculator /
                     │            current_time / web_search /
                     │            search_local_knowledge_base /
                     │            MCP filesystem 工具集）
                     │
                     └─► 会话记忆（LangGraph checkpointer）+ 日志

关键设计：
- 单个 LLM，无 supervisor / worker 分工
- 直接 create_react_agent 挂全部工具
- 会话历史交给 LangGraph 的 checkpointer（thread_id 隔离），**完整保留工具调用
  上下文**，不再只存最终答案文本
- 每次模型调用前用 pre_model_hook 按真实 token 预算裁剪历史

本版修正的四个问题：

1. 「历史只存最终文本，工具上下文跨轮全丢」
   原来 _history 只 append(HumanMessage, AIMessage(answer))，中间所有
   AIMessage(tool_calls) / ToolMessage 都被丢掉。后果是"刚才那个文件再改一处"
   这类接续请求，模型看不到上一轮读到的内容和写入的路径，只能重新摸索或者猜。
   而且 _history[-20:] 是按**条数**截断，一条 5 万字的消息和一条 5 字的消息同权，
   跟按字符估算的上下文守卫是两套互相矛盾的口径。
   现在统一交给 checkpointer + 按 token 裁剪。

2. 「上下文守卫阈值对中文严重失准，且不随后端变化」
   原来 SAFE_INPUT_CHAR_LIMIT = 600_000（按 4 字符≈1 token 算成 150K），
   但中文约 1~1.5 字符 = 1 token，60 万中文字符实际是 40~60 万 token，
   是 200K 上限的 2~3 倍——守卫在真正超限时根本不会报警。而且这个常量硬绑
   Claude 的 200K，用 Ollama（num_ctx=8192）时要超限 70 倍才吭一声。
   现在按 config.CONTEXT_LIMITS 取当前后端的真实窗口，优先用模型自带
   tokenizer 精确计数，并且**主动裁剪**而不只是打印警告。

3. 「工具配对错误重试会重复执行副作用」
   原来捕获到 toolUse/toolResult 配对异常后，用同一份 messages 重跑整个
   ainvoke。但上一次尝试里已经真实执行过的工具（写文件、git push、git tag）
   不会回退，重试会从头再走一遍 ReAct 循环，可能重复 push、重复写文件。
   现在重试改成 ainvoke(None, config)：LangGraph 从最后一个 checkpoint
   **断点续跑**，已完成的工具调用不会重放。

4. 「_extract_final_answer 可能把中间思考当成答案」
   原来从后往前找第一个有文本的 AIMessage。如果最后一条 AIMessage 是纯
   tool_call（content 为空，很常见）而循环因步数上限中断，它会往前捞到某条
   中间推理输出当成最终答案返回，用户看到半成品且没有任何"未完成"提示。
   现在只认"没有 tool_calls 的 AIMessage"，捞不到就明确告知本轮未产出结论。
"""

from __future__ import annotations

import time
import warnings
import uuid
from typing import Any, Awaitable, Callable, Iterable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from config import (
    AGENT_MAX_STEPS,
    CHARS_PER_TOKEN_FALLBACK,
    DASHSCOPE_MODEL,
    HISTORY_MAX_MESSAGES,
    LLM_BACKEND,
    OLLAMA_MODEL,
    SNOWFLAKE_MODEL,
    format_dirs_for_prompt,
    get_context_limit,
    get_default_work_dir,
    get_input_token_budget,
)
from core import events
from core.checkpointer import build_checkpointer
from core.logger import SessionLogger
from skills.skill_loader import format_skill_index_for_prompt
from tools.confirm import reset_session_grants

# 模型经由 Bedrock 调用时，偶发会在一轮里并行发起多个工具调用导致
# "toolUse 块没有配对上 toolResult" 的 400 报错——这是概率性的生成异常，
# 断点续跑一次通常就不会再触发，所以做应用层重试而不是直接报错。
_TOOL_PAIR_ERROR_MARKERS: tuple[str, str] = ("toolUse", "toolResult")
_TOOL_PAIR_ERROR_MAX_ATTEMPTS: int = 3

TokenCallback = Callable[[str], None] | Callable[[str], Awaitable[None]] | None


# ── token 计数 ────────────────────────────────────────────────────────────
def _make_token_counter(llm: BaseChatModel) -> Callable[[Iterable[BaseMessage]], int]:
    """优先用模型自带的 tokenizer；不可用或不可信时退回按字符估算。

    两个坑都得躲开：
    - 原来按 4 字符≈1 token 估算，中文（约 1~1.5 字符/token）会严重**低估**，
      守卫在真正超限时不报警。config.CHARS_PER_TOKEN_FALLBACK 默认 1.5。
    - langchain 对未知模型名会静默退回 GPT-2 tokenizer，而 GPT-2 的 BPE 对中文
      是逐字节切的，一个汉字常常算 2~3 个 token，会严重**高估**，导致历史被
      过度裁剪、白白浪费上下文。所以这里先探测一次：一旦发现走的是 fallback
      分支，就改用我们自己的字符估算。
    """

    def _fallback(messages: list[BaseMessage]) -> int:
        total = 0
        for m in messages:
            content = m.content
            if isinstance(content, list):
                content = "".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                )
            total += len(str(content))
        return int(total / CHARS_PER_TOKEN_FALLBACK) + 1

    # None = 还没探测过
    native_usable: bool | None = None

    def _probe() -> bool:
        probe_msgs = [HumanMessage(content="探测")]
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                llm.get_num_tokens_from_messages(probe_msgs)
            if any("fallback" in str(w.message).lower() for w in caught):
                print(
                    "[SingleAgent] 模型没有提供准确的 tokenizer（langchain 退回 GPT-2），"
                    "改用按字符估算来算 token 预算。"
                )
                return False
            return True
        except Exception:
            return False

    def _counter(messages: Iterable[BaseMessage]) -> int:
        nonlocal native_usable
        msgs = list(messages)
        if native_usable is None:
            native_usable = _probe()
        if not native_usable:
            return _fallback(msgs)
        try:
            return llm.get_num_tokens_from_messages(msgs)
        except Exception:
            native_usable = False
            return _fallback(msgs)

    return _counter


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, list):
        content = "".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )
    return str(content).strip()


# LangGraph 触到 recursion_limit 时会自己塞一条占位 AIMessage。它不是模型的
# 输出，不该被当成"中断前的部分结果"展示给用户。
_FRAMEWORK_PLACEHOLDERS: tuple[str, ...] = (
    "Sorry, need more steps to process this request",
)


def _is_framework_placeholder(text: str) -> bool:
    return any(marker in text for marker in _FRAMEWORK_PLACEHOLDERS)


# ── system prompt ────────────────────────────────────────────────────────
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
   - python_exec 只跑分析/计算：不能执行系统命令，不能用 subprocess /
     ctypes / socket / importlib，不能访问 __class__ / __globals__ 之类的
     内部属性，也不能出现【允许的目录】之外的绝对路径。要读写文件请用
     read_file / write_file 工具。
2. 简单算数用 calculator；查时间用 current_time。
3. 涉及具体人物、公司规章、内部资料等本地信息 →
   先用 search_local_knowledge_base 检索本地知识库；本地找不到再用 web_search。
   引用 web_search 的结果时，把它返回的来源 URL 一并写进回答，方便用户核查。
4. 文本文件读写：优先用 read_file / write_file / list_directory 等。
5. 用户要求"撤销/回滚/恢复上一个版本"某次写入/编辑/移动/建目录操作时：
   先用 list_backups 看看有哪些可回滚的记录，找到对应 id 后调用 rollback(id)。
   注意 python_exec / run_python_script 跑的代码不在可回滚范围内。
6. git 相关任务（提交、推送、打 tag、发布等）用 git_command。它的 cwd 跟
   read_file / write_file 受同一套【允许的目录】约束，所以用户给的仓库路径
   如果不在允许范围内，git_command 也一样用不了——这种情况直接把实际情况
   告诉用户（并说明可以把该目录加进 config.ALLOWED_DIRS），不要反复重试。

【可用技能】
遇到以下场景时，先调用 load_skill(名字) 拿到完整步骤指引，再照着执行，
不要凭自己猜测流程：
{format_skill_index_for_prompt()}

【文件路径规则】
- 相对路径 / 纯文件名 → 落到默认工作目录（用户看得到）
- 绝对路径 → 用户明确指定的位置，但必须在【允许的目录】范围内

【禁止】
- 不要凭猜测编造"运行成功 / 输出是..."；一切结论以真实工具返回为准。
- 不要重复调用已经明确返回"未找到"的工具。
- 工具报"不在允许访问范围"时，那是真实的权限边界，不要换着花样绕，
  直接把情况和解决办法（加白名单）告诉用户。

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

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        session_id: str = "default",
    ):
        self.llm = llm
        self.tools = tools
        self.session_id = session_id
        self.model_identity = _get_model_identity()
        self.system_prompt = _build_system_prompt(self.model_identity)

        # 启动期保持安静：工具清单/上下文预算/日志路径都属于"一切正常"的播报。
        # 只有降级和警告才打印（见下面 _build_graph 里的 warning 分支）。
        self._count_tokens = _make_token_counter(llm)
        self._token_budget = get_input_token_budget()

        self.log = SessionLogger()

        self._checkpointer = build_checkpointer()
        self._thread_id = uuid.uuid4().hex
        self.agent = self._build_graph()

    # ── 构图 ─────────────────────────────────────────────────────────────
    def _build_graph(self):
        """尽量把 checkpointer / pre_model_hook 交给 create_react_agent。

        老版本 langgraph 没有这两个参数，逐级降级而不是直接崩——降级时会明确
        打印失去了哪项能力，不做静默降级。
        """
        base_kwargs: dict[str, Any] = dict(
            tools=self.tools, prompt=self.system_prompt
        )
        attempts = [
            (
                dict(checkpointer=self._checkpointer, pre_model_hook=self._trim_hook),
                None,
            ),
            (
                dict(checkpointer=self._checkpointer),
                "当前 langgraph 不支持 pre_model_hook，改为在每轮开始时裁剪历史。",
            ),
            (
                {},
                "当前 langgraph 不支持 checkpointer/pre_model_hook，历史仅在进程内保留。",
            ),
        ]
        for extra, warning in attempts:
            try:
                graph = create_react_agent(self.llm, **base_kwargs, **extra)
            except TypeError:
                continue
            self._has_checkpointer = "checkpointer" in extra
            self._has_trim_hook = "pre_model_hook" in extra
            if warning:
                print(f"[SingleAgent] 警告：{warning}")
            return graph
        raise RuntimeError("create_react_agent 调用失败：langgraph 版本过旧，请升级。")

    def _trim_hook(self, state: dict) -> dict:
        """每次调模型前按 token 预算裁剪历史。

        返回 llm_input_messages 而不是 messages：只影响"这次喂给模型的内容"，
        不改动 checkpointer 里保存的完整历史，后续回溯/审计仍然完整。
        """
        from langchain_core.messages.utils import trim_messages

        messages: list[BaseMessage] = state.get("messages", [])
        if not messages:
            return {}
        try:
            trimmed = trim_messages(
                messages,
                strategy="last",
                max_tokens=self._token_budget,
                token_counter=self._count_tokens,
                start_on="human",
                end_on=("human", "tool"),
                include_system=True,
                allow_partial=False,
            )
        except Exception as exc:
            # 裁剪失败不能让整轮对话挂掉，退回按条数硬截断
            self.log.event("Warning", f"按 token 裁剪历史失败，退回条数截断：{exc}")
            trimmed = messages[-HISTORY_MAX_MESSAGES:]

        if len(trimmed) < len(messages):
            self.log.event(
                "Trim",
                f"历史裁剪：{len(messages)} 条 → {len(trimmed)} 条"
                f"（预算 {self._token_budget} tokens）",
            )
        return {"llm_input_messages": trimmed}

    # ── 公开接口 ─────────────────────────────────────────────────────────
    async def arun(self, user_input: str, on_token: TokenCallback = None) -> str:
        return await self._run(user_input, on_token)

    def run(self, user_input: str) -> str:
        import asyncio

        return asyncio.run(self._run(user_input, None))

    def new_conversation(self) -> None:
        """开启新对话：换 thread_id（历史清零）、重开日志、收回批量授权。"""
        self._thread_id = uuid.uuid4().hex
        try:
            self.log.close()
        except Exception:
            pass
        self.log = SessionLogger()
        # 上一轮授权的"本次会话全部允许"不应延续到新对话
        reset_session_grants()
        print(f"[SingleAgent] 已开启新会话，日志：{self.log.path}")

    def close(self) -> None:
        try:
            self.log.close()
        except Exception:
            pass

    # ── 主流程 ───────────────────────────────────────────────────────────
    def _config(self) -> dict:
        return {
            "configurable": {"thread_id": self._thread_id},
            # 一"步"在 LangGraph 里是一个节点执行；一次工具往返 = 2 步，
            # 再留几步给首尾节点。
            "recursion_limit": AGENT_MAX_STEPS * 2 + 4,
        }

    async def _run(self, user_input: str, on_token: TokenCallback) -> str:
        events.set_current_session_id(self.session_id)
        self.log.start_turn(user_input)
        events.push(self.session_id, "user_message", {"text": user_input})

        started = time.monotonic()
        config = self._config()
        base_len = len(self._state_messages(config))

        approx_tokens = self._count_tokens([HumanMessage(content=user_input)])
        if approx_tokens > self._token_budget:
            msg = (
                f"这一条输入本身就约 {approx_tokens} tokens，超过了当前后端"
                f"（{LLM_BACKEND}，窗口 {get_context_limit()}）的输入预算 "
                f"{self._token_budget} tokens。请把内容分段后分多次提交。"
            )
            self.log.event("Warning", msg)
            events.push(self.session_id, "error", {"text": msg})
            return msg

        try:
            result = await self._invoke_with_resume(user_input, config, on_token)
        except GraphRecursionError:
            return self._handle_step_limit(config, started, base_len)
        except Exception as exc:
            self.log.event("Agent", f"执行异常：{exc}")
            events.push(self.session_id, "error", {"text": str(exc)})
            raise

        new_msgs = result.get("messages", []) if isinstance(result, dict) else []
        if not new_msgs:
            new_msgs = self._state_messages(config)

        answer = self._extract_final_answer(new_msgs[base_len:] or new_msgs)
        steps = self._log_trace(new_msgs, base_len=base_len)

        self.log.turn_stats(
            duration_s=time.monotonic() - started,
            input_tokens=self._count_tokens(new_msgs),
            output_chars=len(answer),
            steps=steps,
        )
        self.log.end_turn(answer)
        events.push(self.session_id, "final_answer", {"text": answer})
        events.push(self.session_id, "turn_end", {})
        return answer

    async def _invoke_with_resume(
        self, user_input: str, config: dict, on_token: TokenCallback
    ) -> dict:
        """首次带输入调用；遇到工具配对异常时**断点续跑**而不是重放整轮。

        这是关键区别：原来重试传的是同一份完整 messages，等于让 ReAct 循环
        从头再走一遍，已经真实执行过的工具（写文件、git push）会被重复执行。
        改成 ainvoke(None, config) 后，LangGraph 从最后一个 checkpoint 继续，
        已完成的节点不会重跑。
        """
        payload: dict | None = {"messages": [HumanMessage(content=user_input)]}

        for attempt in range(1, _TOOL_PAIR_ERROR_MAX_ATTEMPTS + 1):
            try:
                if on_token is not None:
                    return await self._astream(payload, config, on_token)
                return await self.agent.ainvoke(payload, config=config)
            except GraphRecursionError:
                raise
            except Exception as exc:
                msg = str(exc)
                is_pair_error = all(m in msg for m in _TOOL_PAIR_ERROR_MARKERS)
                resumable = self._has_checkpointer
                if is_pair_error and resumable and attempt < _TOOL_PAIR_ERROR_MAX_ATTEMPTS:
                    self.log.event(
                        "Agent",
                        f"第 {attempt} 次尝试触发工具调用配对异常，从断点续跑重试：{exc}",
                    )
                    # 续跑：不再传输入，已执行的工具结果保留在 checkpoint 里
                    payload = None
                    continue
                if is_pair_error and not resumable:
                    self.log.event(
                        "Agent",
                        "触发工具调用配对异常，但当前没有 checkpointer，"
                        "无法安全续跑（重放会重复执行已完成的工具），直接上报。",
                    )
                raise
        raise RuntimeError("重试次数用尽")  # pragma: no cover - 循环内必定 return/raise

    async def _astream(self, payload: dict | None, config: dict, on_token: TokenCallback) -> dict:
        """流式执行：边生成边回调 token，避免长任务期间用户面对空白终端。"""
        import inspect

        async for chunk, _meta in self.agent.astream(
            payload, config=config, stream_mode="messages"
        ):
            text = getattr(chunk, "content", None)
            if not text or not isinstance(chunk, AIMessage):
                continue
            piece = text if isinstance(text, str) else _message_text(chunk)
            if not piece:
                continue
            events.push(self.session_id, "token", {"text": piece})
            out = on_token(piece)
            if inspect.isawaitable(out):
                await out

        # 流式模式下拿不到聚合结果，从 checkpoint 里读回完整状态
        return {"messages": self._state_messages(config)}

    # ── 辅助 ─────────────────────────────────────────────────────────────
    def _state_messages(self, config: dict) -> list[BaseMessage]:
        """从 checkpointer 读取当前 thread 的完整消息历史。"""
        if not getattr(self, "_has_checkpointer", False):
            return []
        try:
            state = self.agent.get_state(config)
        except Exception:
            return []
        values = getattr(state, "values", None) or {}
        return list(values.get("messages", []))

    def _handle_step_limit(self, config: dict, started: float, base_len: int) -> str:
        """步数触顶：明确告诉用户是"步数用完"，而不是笼统的"出现异常"。

        原来 GraphRecursionError 没有单独处理，用户只会看到 main.py 那句
        "请重新提问"，完全不知道是步数上限导致的，也拿不到已完成的中间结果。
        """
        msgs = self._state_messages(config)
        partial = self._extract_final_answer(msgs[base_len:] or msgs, allow_empty=True)
        steps = self._log_trace(msgs, base_len=base_len)
        self.log.event(
            "StepLimit",
            f"单轮步数达到上限（AGENT_MAX_STEPS={AGENT_MAX_STEPS}），本轮中断。",
        )
        self.log.turn_stats(duration_s=time.monotonic() - started, steps=steps)

        answer = (
            f"本轮达到步数上限（{AGENT_MAX_STEPS} 步）仍未完成，已中断"
            "（可以在 config.AGENT_MAX_STEPS 里调大，或把任务拆小一点再说）。"
        )
        if partial:
            answer += f"\n\n中断前已经得到的部分结果：\n{partial}"
        events.push(self.session_id, "error", {"text": answer})
        self.log.end_turn(answer)
        return answer

    @staticmethod
    def _extract_final_answer(messages: list[BaseMessage], allow_empty: bool = False) -> str:
        """取本轮最终答复。

        规则：**只认最后一条 AIMessage**，且它必须没有 tool_calls。

        为什么不能"从后往前找第一条有文本的 AIMessage"（原实现，以及第一版修复）：
        带 tool_calls 的 AIMessage 是中间步骤（模型在说"我要调工具"）。如果循环
        因步数上限中断在一个 tool_call 上，往前扫就会捞到某条中间推理输出，把
        半成品思考当成结论交给用户，而且完全没有"未完成"的提示。
        最后一条是 tool_call 就意味着这一轮**没有**结论，这是事实，要如实说。
        """
        ai_messages = [m for m in messages if isinstance(m, AIMessage)]

        if ai_messages and not getattr(ai_messages[-1], "tool_calls", None):
            text = _message_text(ai_messages[-1])
            if text:
                return text

        if allow_empty:
            # 步数触顶时展示"中断前的部分结果"，这里才允许往前找
            for m in reversed(ai_messages):
                text = _message_text(m)
                if text and not _is_framework_placeholder(text):
                    return text
            return ""

        return (
            "本轮没有产出最终结论（模型最后一步仍在调用工具就结束了）。"
            "可以再说一次或把任务拆得更小一点。"
        )

    def _log_trace(self, messages: list[BaseMessage], base_len: int) -> int:
        """把本轮新增的消息写进日志并推事件，返回 agent 步数。"""
        step = 0
        for m in messages[base_len:]:
            if isinstance(m, AIMessage):
                step += 1
                text = _message_text(m)
                if text:
                    self.log.agent_step(step, text)
                    events.push(
                        self.session_id, "agent_step", {"step": step, "text": text}
                    )
                for tc in getattr(m, "tool_calls", []) or []:
                    tool_name = tc.get("name", "?")
                    tool_args = tc.get("args", {})
                    self.log.tool_call(tool_name, tool_args)
                    events.push(
                        self.session_id, "tool_call", {"name": tool_name, "args": tool_args}
                    )
            elif isinstance(m, ToolMessage):
                tool_name = getattr(m, "name", "?")
                result_text = str(m.content)
                self.log.tool_result(tool_name, result_text)
                events.push(
                    self.session_id,
                    "tool_result",
                    {
                        "name": tool_name,
                        "result": result_text
                        if len(result_text) < 2000
                        else result_text[:2000] + "...(截断)",
                    },
                )
        return step
