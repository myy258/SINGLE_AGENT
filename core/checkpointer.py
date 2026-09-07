# 会话历史持久化：LangGraph checkpointer 工厂
# Co-authored with CoCo

"""
为什么用 checkpointer 而不是自己维护 message list：

原来 agent.py 自己保存 `_history: list[BaseMessage]`，每轮把
`list(self._history) + [HumanMessage(...)]` 整体重新喂给 graph，并且只把
最终答案文本存回历史。三个后果：
- 中间的 AIMessage(tool_calls) / ToolMessage 全部丢失，跨轮接续任务时模型
  看不到上一轮读到什么、写到哪；
- 进程退出历史即消失；
- 出错重试只能整轮重放，已执行的工具会重复执行副作用。

create_react_agent 原生支持 checkpointer，这三件事都是它本来就解决好的。
默认落到 sqlite 文件（进程重启还在），config.CHECKPOINT_DB_PATH 设为空串
则退回纯内存。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from config import CHECKPOINT_DB_PATH


def build_checkpointer():
    """返回一个 LangGraph checkpointer。sqlite 不可用时降级到内存并明确告警。"""
    from langgraph.checkpoint.memory import MemorySaver

    path = (CHECKPOINT_DB_PATH or "").strip()
    if not path:
        print("[Checkpoint] CHECKPOINT_DB_PATH 为空，使用内存存储（进程退出后历史丢失）。")
        return MemorySaver()

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError:
        print(
            "[Checkpoint] 警告：未安装 langgraph-checkpoint-sqlite，"
            "历史只保存在内存里（进程退出即丢失）。"
            "装上它可以让对话历史跨重启保留：pip install langgraph-checkpoint-sqlite"
        )
        return MemorySaver()

    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False：Web 版会在独立线程里跑 agent
        conn = sqlite3.connect(path, check_same_thread=False)
        saver = SqliteSaver(conn)
        saver.setup()
        # 成功路径保持安静；只有降级/失败才打印
        return saver
    except (sqlite3.Error, OSError) as exc:
        print(f"[Checkpoint] 警告：sqlite 初始化失败（{exc}），退回内存存储。")
        return MemorySaver()
