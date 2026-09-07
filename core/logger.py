# 会话日志系统（单 agent 版）：每对话一个 .txt
# Co-authored with CoCo

"""
写入策略：
- 一个 SessionLogger 实例对应一个 .txt 文件；每次 new_conversation 重开新文件。
- 只写文件，不 print 到控制台。
- 用统一的 [AGENT] / [TOOL] 事件标签，不区分 supervisor/worker。

本版相对最初版本的改动：

1. 去掉死参数：tool_call(args=...) 的调用方永远传 {}，纯噪声。
2. 补上可观测性：原来只记工具名和参数，不记结果、不记耗时、不记 token 数，
   出问题基本只能靠猜。现在记录每轮耗时、输入/输出 token 估算、工具结果长度
   和截断后的预览。
3. 加脱敏：工具参数和结果都可能带上凭证（例如 git remote add origin
   https://user:token@github.com/...）。落盘前统一过一遍 _redact。
4. 用 contextmanager 保证文件句柄一定关闭。
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"

# 落盘前需要打码的模式：URL 里的 basic auth、常见 token 前缀、长 JWT
_REDACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(https?://)([^/\s:@]+):([^/\s@]+)@"), r"\1\2:***@"),
    (re.compile(r"\b(sk-[A-Za-z0-9._-]{8})[A-Za-z0-9._-]+"), r"\1***"),
    (re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{6})[A-Za-z0-9]+"), r"\1***"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]+"), "<JWT已打码>"),
    (
        re.compile(
            r"(?i)\b((?:api[_-]?key|token|secret|password|passwd|pat)\s*[=:]\s*)"
            r"['\"]?([^\s'\",;]{6,})"
        ),
        r"\1***",
    ),
)

_PREVIEW_LIMIT = 400


def _redact(text: str) -> str:
    for pattern, repl in _REDACT_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def _preview(text: str, limit: int = _PREVIEW_LIMIT) -> str:
    text = _redact(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(截断，共 {len(text)} 字符)"


class SessionLogger:
    """一个会话对应一个日志文件。线程内串行使用，不加锁。"""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or uuid.uuid4().hex[:8]
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = LOG_DIR / f"session_{ts}_{self.session_id}.txt"
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        self._turn = 0
        self._closed = False
        self._write_header()

    # ── 内部 ─────────────────────────────────────────────────────────────
    def _write_header(self) -> None:
        self._write(
            f"{'=' * 70}\n"
            f"会话日志 session_id={self.session_id}\n"
            f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'=' * 70}\n\n"
        )

    def _stamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _write(self, text: str) -> None:
        if self._closed:
            return
        try:
            self._fh.write(text)
        except (OSError, ValueError):
            pass

    # ── 对外接口 ─────────────────────────────────────────────────────────
    def start_turn(self, user_input: str) -> None:
        self._turn += 1
        self._write(
            f"\n----- Turn {self._turn} [{self._stamp()}] -----\n"
            f"[USER] {_redact(user_input)}\n"
        )

    def agent_step(self, step: int, thought: str) -> None:
        """记录 agent 的一次内部推理输出。"""
        self._write(f"[AGENT step={step}] {_preview(thought, 800)}\n")

    def tool_call(self, tool_name: str, args: dict | None = None) -> None:
        """记录一次工具调用的入参。"""
        arg_text = _preview(str(args)) if args else "（无）"
        self._write(f"[TOOL_CALL:{tool_name}] args={arg_text}\n")

    def tool_result(self, tool_name: str, result: str) -> None:
        """记录一次工具返回。记长度 + 截断预览，便于排查而不至于把日志写爆。"""
        self._write(
            f"[TOOL_RESULT:{tool_name}] len={len(result)} {_preview(result)}\n"
        )

    def event(self, tag: str, msg: str) -> None:
        """记录任意事件（异常、降级、上限触发等）。"""
        self._write(f"[{tag}] {_redact(msg)}\n")

    def turn_stats(
        self,
        duration_s: float,
        input_tokens: int | None = None,
        output_chars: int | None = None,
        steps: int | None = None,
    ) -> None:
        """记录本轮的耗时/规模指标。"""
        parts = [f"耗时={duration_s:.1f}s"]
        if input_tokens is not None:
            parts.append(f"输入≈{input_tokens}tokens")
        if output_chars is not None:
            parts.append(f"输出={output_chars}字符")
        if steps is not None:
            parts.append(f"步数={steps}")
        self._write(f"[STATS] {'　'.join(parts)}\n")

    def end_turn(self, answer: str) -> None:
        """记录本轮结束。"""
        self._write(f"[TURN_END] {self._stamp()} 答复长度={len(answer)}\n")

    def close(self) -> None:
        if self._closed:
            return
        self._write(
            f"\n{'=' * 70}\n会话结束 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        self._closed = True
        try:
            self._fh.close()
        except (OSError, ValueError):
            pass

    def __enter__(self) -> "SessionLogger":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
