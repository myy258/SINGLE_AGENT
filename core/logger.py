# 会话日志系统（单 agent 版）：每对话一个 .txt，不区分 supervisor/worker
# Co-authored with CoCo

"""
写入策略：
- 一个 SessionLogger 实例对应一个 .txt 文件；每次 new_conversation 重开新文件。
- 只写文件，不 print 到控制台。
- 单 agent 场景下不再区分 supervisor/worker，改用统一的 [AGENT] / [TOOL] 事件标签。
"""

import uuid
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class SessionLogger:
    """一个会话对应一个日志文件。线程内串行使用，不加锁。"""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or uuid.uuid4().hex[:8]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = LOG_DIR / f"session_{ts}_{self.session_id}.txt"
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)
        self._turn = 0
        self._write_header()

    def _write_header(self):
        self._fh.write(
            f"{'=' * 70}\n"
            f"会话日志 session_id={self.session_id}\n"
            f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'=' * 70}\n\n"
        )

    def _stamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _write(self, text: str):
        try:
            self._fh.write(text)
        except Exception:
            pass

    def start_turn(self, user_input: str):
        self._turn += 1
        self._write(
            f"\n----- Turn {self._turn} [{self._stamp()}] -----\n"
            f"[USER] {user_input}\n"
        )

    def agent_step(self, step: int, thought: str):
        """记录 agent 的一次内部推理输出。"""
        self._write(f"[AGENT step={step}] {thought}\n")

    def tool_call(self, tool_name: str, args: dict, result: str):
        """记录一次工具调用（只留工具名+参数，不落地 result 内容）。"""
        arg_preview = str(args)
        if len(arg_preview) > 400:
            arg_preview = arg_preview[:400] + "...(截断)"
        self._write(f"[TOOL:{tool_name}] args={arg_preview}\n")

    def event(self, tag: str, msg: str):
        """记录任意事件（异常、降级、上限触发等）。"""
        self._write(f"[{tag}] {msg}\n")

    def end_turn(self, answer: str):
        """记录本轮结束（只留时间戳标记，不落地最终答案内容）。"""
        self._write(f"[TURN_END] {self._stamp()}\n")

    def close(self):
        try:
            self._write(f"\n{'=' * 70}\n会话结束 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            self._fh.close()
        except Exception:
            pass
