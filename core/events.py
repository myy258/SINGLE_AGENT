# 事件汇 + 当前会话 id：让 agent/工具层无条件推事件，CLI 下是 no-op，Web 层注入实现
# Co-authored with CoCo

"""
为什么需要这一层：

SINGLE_AGENT（CLI）和 SINGLE_AGENT_WEB 之前是整树复制，Web 版为了把 agent 内部
事件推给前端，直接在 agent.py 里 `from web import event_bus` 并散落插了 8 处
event_bus.push(...)，导致 agent.py 在两棵树里长期分叉，每次修 bug 都要改两遍。

改法：agent.py 只依赖这里的 push()，默认实现是 no-op（CLI 下零开销）；
Web 层在启动时调 set_sink() 注入真正的实现。于是 agent.py / config.py /
tools/confirm.py 在两棵树里可以保持字节一致。

线程安全：sink 的替换和读取都在锁内完成；push() 本身把异常吞掉——事件推送是
旁路可观测性，永远不应该让主流程失败。
"""

from __future__ import annotations

import threading
from typing import Any, Protocol


class EventSink(Protocol):
    """事件接收端协议。Web 层实现它并通过 set_sink 注入。"""

    def push(self, session_id: str, event_type: str, data: dict[str, Any]) -> None:
        ...


class _NullSink:
    """默认实现：什么都不做。"""

    def push(self, session_id: str, event_type: str, data: dict[str, Any]) -> None:
        return None


_lock = threading.Lock()
_sink: EventSink = _NullSink()
_current_session_id: str = "default"


def set_sink(sink: EventSink | None) -> None:
    """注入事件接收端；传 None 恢复成 no-op。"""
    global _sink
    with _lock:
        _sink = sink if sink is not None else _NullSink()


def push(session_id: str, event_type: str, data: dict[str, Any] | None = None) -> None:
    """推送一个事件。任何异常都被吞掉，不影响主流程。"""
    with _lock:
        sink = _sink
    try:
        sink.push(session_id, event_type, data or {})
    except Exception:
        pass


# ── 当前会话 id ───────────────────────────────────────────────────────────
# 深层工具调用（例如 tools/confirm.py）没法拿到 agent 实例，需要一个地方查
# "现在是哪个会话在跑"。沿用项目既有的"单会话串行"假设，用带锁的全局变量。
# 未来若要支持多会话真并发，把这里换成 contextvars 即可，调用点不用改。

def set_current_session_id(session_id: str) -> None:
    global _current_session_id
    with _lock:
        _current_session_id = session_id


def get_current_session_id() -> str:
    with _lock:
        return _current_session_id
