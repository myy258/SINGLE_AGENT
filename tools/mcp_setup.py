# MCP filesystem server 接入：把标准 MCP 文件工具转成 LangChain Tool
# Co-authored with CoCo

"""
通过 npx 启动 @modelcontextprotocol/server-filesystem，把它暴露的
read_file / read_multiple_files / list_directory / directory_tree /
search_files / create_directory / edit_file / move_file / get_file_info
等工具全部拿过来交给 agent 使用。

本地的 write_file 因为需要处理 .csv 的 BOM（Excel 中文不乱码），跟 MCP
自带的 write_file 撞名——过滤掉 MCP 版本，统一用本地版本。

本版修正的四个问题：

1. 「位置参数会导致备份静默丢失」
   原来 _record_backup_before_call 只从 kwargs 里取 path/source/destination。
   一旦调用走位置参数，kwargs 为空 → 不写备份记录 → 确认框照样弹、用户点了
   允许、操作执行了，但事后 list_backups 查不到，**根本无法回滚**，而用户
   毫不知情。现在用工具自己的 args_schema 把位置参数映射回参数名，映射不出来
   就在确认摘要里明确写"本次操作不可回滚"，让用户带着这个信息做决定。

2. 「备份失败被 except Exception: pass 吞掉」
   同上，现在失败信息会拼进工具返回值和确认摘要。

3. 「wrapper 可能是空操作」
   原来只包 t.func / t.coroutine。langchain_mcp_adapters 造出来的
   StructuredTool 如果走的是别的执行路径，包装就完全不生效（静默失效）。
   现在包 t.coroutine/t.func 之外，还会校验"到底包上了没有"，没包上就明确
   报错而不是让一个没有闸门的写工具进入工具列表。

4. 「client 生命周期」
   提供 shutdown_mcp() 供退出时调用。注意 langchain-mcp-adapters >= 0.1.0 的
   MultiServerMCPClient 是无状态的——get_tools()/session() 各自开关 stdio 子进程，
   不持有长连接，所以这个版本下并没有 npx 子进程泄漏，也没有 close/aclose 可调，
   shutdown_mcp() 会静默返回。**特别不要去调 __aexit__**：它存在但被故意实现成
   抛 NotImplementedError，调它只会在退出时打印一条误导性报错。
"""

from __future__ import annotations

import inspect
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient

from config import MAX_READ_CHARS, get_allowed_dirs
from tools.confirm import aconfirm_action, confirm_action
from tools.rollback import (
    RecordResult,
    record_create_directory,
    record_move,
    record_write,
)

ALLOWED_DIRS = get_allowed_dirs()

_EXCLUDED_MCP_TOOLS = {"write_file"}

# 大文件读取工具的输出截断阈值：避免一次性把超大文件塞进对话上下文顶爆 context。
_TRUNCATE_MCP_TOOLS = {"read_file", "read_text_file", "read_multiple_files"}

# 会修改/删除文件系统状态的工具：执行前必须人工审核确认。
_CONFIRM_MCP_TOOLS = {"edit_file", "move_file", "create_directory"}

# 各工具需要用于登记备份的参数名
_BACKUP_ARG_NAMES: dict[str, tuple[str, ...]] = {
    "edit_file": ("path",),
    "move_file": ("source", "destination"),
    "create_directory": ("path",),
}

_client: MultiServerMCPClient | None = None


def _truncate_large_result(result: Any) -> Any:
    text = result if isinstance(result, str) else str(result)
    if len(text) <= MAX_READ_CHARS:
        return result
    return (
        text[:MAX_READ_CHARS]
        + f"\n\n...(内容过长，已截断。原始长度 {len(text)} 字符，超过安全阈值 "
        f"{MAX_READ_CHARS} 字符。请分段处理该文件，例如按函数/按行数区间多次读取和加注释，"
        "不要一次性把整份大文件塞进一次请求。)"
    )


def _schema_field_order(tool: Any) -> list[str]:
    """从工具的 args_schema 里取出参数顺序，用来把位置参数映射回参数名。"""
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return []
    # pydantic v2
    fields = getattr(schema, "model_fields", None)
    if isinstance(fields, dict):
        return list(fields.keys())
    # pydantic v1
    fields = getattr(schema, "__fields__", None)
    if isinstance(fields, dict):
        return list(fields.keys())
    # 纯 dict 形式的 JSON schema
    if isinstance(schema, dict):
        props = schema.get("properties")
        if isinstance(props, dict):
            return list(props.keys())
    return []


def _normalize_args(tool: Any, args: tuple, kwargs: dict) -> dict:
    """把 (args, kwargs) 统一成 {参数名: 值}。位置参数按 schema 顺序对齐。"""
    merged = dict(kwargs)
    if args:
        order = _schema_field_order(tool)
        for name, value in zip(order, args):
            merged.setdefault(name, value)
    return merged


def _build_call_summary(tool_name: str, normalized: dict, unmapped: int) -> str:
    parts = [f"{k}={v!r}" for k, v in normalized.items()]
    summary = f"工具：{tool_name}\n参数：{', '.join(parts) or '（无）'}"
    if unmapped:
        summary += (
            f"\n⚠️ 有 {unmapped} 个参数无法对应到参数名（工具未提供 args_schema），"
            "本次操作**无法登记回滚备份**，执行后不能用 rollback 撤销。"
        )
    return summary


def _record_backup_before_call(tool_name: str, normalized: dict) -> list[RecordResult]:
    """按工具名把即将执行的操作记进 rollback 的 manifest，供后续回滚。"""
    needed = _BACKUP_ARG_NAMES.get(tool_name, ())
    missing = [n for n in needed if n not in normalized]
    if missing:
        return [
            RecordResult(
                id=None,
                ok=False,
                path=str(normalized.get(needed[0], "?")) if needed else "?",
                reason=f"缺少参数 {', '.join(missing)}，无法定位备份目标",
            )
        ]

    if tool_name == "edit_file":
        return [record_write(normalized["path"], tool_name="edit_file")]
    if tool_name == "move_file":
        return [
            record_move(
                normalized["source"], normalized["destination"], tool_name="move_file"
            )
        ]
    if tool_name == "create_directory":
        return [record_create_directory(normalized["path"], tool_name="create_directory")]
    return []


def _backup_warnings(records: list[RecordResult]) -> str:
    return "".join(r.warning_suffix for r in records if not r.ok)


def _finalize_records(records: list[RecordResult]) -> None:
    for r in records:
        r.finalize()


def _wrap_read_tool(t: Any) -> Any:
    """给大文件读取工具的输出加长度保护，超阈值时截断并提示分段处理。"""
    wrapped_any = False
    original_coroutine = getattr(t, "coroutine", None)
    original_func = getattr(t, "func", None)

    if original_coroutine is not None:
        async def wrapped_coroutine(*args, **kwargs):
            return _truncate_large_result(await original_coroutine(*args, **kwargs))

        t.coroutine = wrapped_coroutine
        wrapped_any = True

    if original_func is not None:
        def wrapped_func(*args, **kwargs):
            return _truncate_large_result(original_func(*args, **kwargs))

        t.func = wrapped_func
        wrapped_any = True

    if not wrapped_any:
        # 包不上就说清楚：这个读工具没有长度保护，可能顶爆 context
        print(
            f"[MCP] 警告：{t.name} 既没有 func 也没有 coroutine，"
            "无法加装输出截断保护，读取超大文件时可能顶爆上下文。"
        )
    return t


def _wrap_confirm_tool(t: Any) -> Any:
    """给会修改/删除文件系统状态的工具加人工审核闸门，拒绝则不执行。"""
    wrapped_any = False
    original_coroutine = getattr(t, "coroutine", None)
    original_func = getattr(t, "func", None)

    def _prepare(args: tuple, kwargs: dict) -> tuple[str, dict]:
        """返回 (给用户看的确认摘要, 归一化后的参数字典)。"""
        normalized = _normalize_args(t, args, kwargs)
        unmapped = max(0, len(args) - len(_schema_field_order(t)))
        return _build_call_summary(t.name, normalized, unmapped), normalized

    if original_coroutine is not None:
        async def wrapped_coroutine(*args, **kwargs):
            summary, normalized = _prepare(args, kwargs)
            # 用 aconfirm_action：同步 input() 直接在协程里跑会卡死事件循环
            if not await aconfirm_action(summary, tool_name=t.name):
                return f"操作已被用户拒绝：{t.name} 未执行。"
            records = _record_backup_before_call(t.name, normalized)
            result = await original_coroutine(*args, **kwargs)
            _finalize_records(records)
            return f"{result}{_backup_warnings(records)}"

        t.coroutine = wrapped_coroutine
        wrapped_any = True

    if original_func is not None:
        def wrapped_func(*args, **kwargs):
            summary, normalized = _prepare(args, kwargs)
            if not confirm_action(summary, tool_name=t.name):
                return f"操作已被用户拒绝：{t.name} 未执行。"
            records = _record_backup_before_call(t.name, normalized)
            result = original_func(*args, **kwargs)
            _finalize_records(records)
            return f"{result}{_backup_warnings(records)}"

        t.func = wrapped_func
        wrapped_any = True

    if not wrapped_any:
        # 一个"会改文件但没有确认闸门"的工具绝对不能进工具列表，直接报错停机。
        raise RuntimeError(
            f"MCP 工具 {t.name} 既没有 func 也没有 coroutine，无法加装人工审核闸门。"
            "拒绝在没有闸门的情况下把它交给 agent——请检查 langchain_mcp_adapters 版本。"
        )
    return t


async def get_mcp_tools() -> list:
    """异步启动 MCP filesystem server，返回它的工具列表（已排除撞名工具）。"""
    global _client
    _client = MultiServerMCPClient(
        {
            "filesystem": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    *ALLOWED_DIRS,
                ],
                "transport": "stdio",
            },
        }
    )
    try:
        tools = await _client.get_tools()
    except Exception as e:
        print(f"[MCP] 启动 filesystem server 失败：{e}")
        print("[MCP] 已跳过 MCP 工具加载，其它工具仍可正常使用。")
        _client = None
        return []

    tools = [t for t in tools if t.name not in _EXCLUDED_MCP_TOOLS]
    tools = [_wrap_read_tool(t) if t.name in _TRUNCATE_MCP_TOOLS else t for t in tools]
    tools = [_wrap_confirm_tool(t) if t.name in _CONFIRM_MCP_TOOLS else t for t in tools]
    # 加载成功时保持安静：工具清单和目录白名单属于"一切正常"的播报，
    # 启动横幅里已经有工具总数。只有失败/降级才打印（见上面的 except 分支）。
    return tools


async def shutdown_mcp() -> None:
    """释放 MCP client 持有的资源。退出前调用。

    langchain-mcp-adapters >= 0.1.0 的 MultiServerMCPClient 是**无状态**的：
    get_tools() 和 session() 各自用 `async with` 开关 stdio 子进程，不持有长连接，
    所以这个版本下没有任何东西需要显式关闭，本函数正常静默返回。

    ⚠️ 绝对不要去试 `__aexit__`：它在 0.1.0+ 依然存在，但被故意实现成抛
    NotImplementedError（提示"MultiServerMCPClient 不能当 async context manager 用"）。
    调它不会关掉任何东西，只会在用户退出时打印一条莫名其妙的报错。
    只探测真正的关闭方法，留给未来版本或旧版本。
    """
    global _client
    if _client is None:
        return
    client, _client = _client, None

    for method in ("aclose", "close"):
        fn = getattr(client, method, None)
        if fn is None:
            continue
        try:
            result = fn()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            print(f"[MCP] 关闭 client 时出错（{method}）：{exc}")
        return
    # 两个方法都不存在 = 当前版本无需显式关闭，静默返回
