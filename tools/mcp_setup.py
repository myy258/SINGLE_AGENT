# MCP filesystem server 接入：把标准 MCP 文件工具转成 LangChain Tool

"""
通过 npx 启动 @modelcontextprotocol/server-filesystem，把它暴露的
read_file / read_multiple_files / list_directory / directory_tree /
search_files / create_directory / edit_file / move_file / get_file_info
等工具全部拿过来，交给各 worker 使用。

本地的 write_file 因为需要处理 .csv 的 BOM（Excel 中文不乱码），跟 MCP
自带的 write_file 撞名——过滤掉 MCP 版本，统一用本地版本。
"""

from langchain_mcp_adapters.client import MultiServerMCPClient
from config import get_allowed_dirs
from tools.confirm import confirm_action
from tools.rollback import record_write, record_move, record_create_directory

ALLOWED_DIRS = get_allowed_dirs()

_EXCLUDED_MCP_TOOLS = {"write_file"}

# 大文件读取工具的输出截断阈值：避免一次性把超大文件塞进对话上下文，
# 顶到 claude-sonnet-4-5 的 200K token context 上限触发 500 错误。
_TRUNCATE_MCP_TOOLS = {"read_file", "read_text_file", "read_multiple_files"}
MAX_READ_CHARS: int = 40_000

# 会修改/删除文件系统状态的工具：执行前必须人工审核确认。
_CONFIRM_MCP_TOOLS = {"edit_file", "move_file", "create_directory"}


def _truncate_large_result(result):
    text = result if isinstance(result, str) else str(result)
    if len(text) <= MAX_READ_CHARS:
        return result
    return (
        text[:MAX_READ_CHARS]
        + f"\n\n...(内容过长，已截断。原始长度 {len(text)} 字符，超过安全阈值 "
        f"{MAX_READ_CHARS} 字符。请分段处理该文件，例如按函数/按行数区间多次读取和加注释，"
        "不要一次性把整份大文件塞进一次请求。)"
    )


def _wrap_read_tool(t):
    """给大文件读取工具的输出加长度保护，超阈值时截断并提示分段处理。"""
    original_func = getattr(t, "func", None)
    original_coroutine = getattr(t, "coroutine", None)

    if original_coroutine is not None:
        async def wrapped_coroutine(*args, **kwargs):
            result = await original_coroutine(*args, **kwargs)
            return _truncate_large_result(result)
        t.coroutine = wrapped_coroutine

    if original_func is not None:
        def wrapped_func(*args, **kwargs):
            result = original_func(*args, **kwargs)
            return _truncate_large_result(result)
        t.func = wrapped_func

    return t


def _build_call_summary(tool_name: str, args: tuple, kwargs: dict) -> str:
    parts = [repr(a) for a in args] + [f"{k}={v!r}" for k, v in kwargs.items()]
    return f"工具：{tool_name}\n参数：{', '.join(parts)}"


def _record_backup_before_call(tool_name: str, kwargs: dict) -> None:
    """按工具名把即将执行的操作记进 rollback 的 manifest，供后续回滚。"""
    try:
        if tool_name == "edit_file" and "path" in kwargs:
            record_write(kwargs["path"], tool_name="edit_file")
        elif tool_name == "move_file" and "source" in kwargs and "destination" in kwargs:
            record_move(kwargs["source"], kwargs["destination"], tool_name="move_file")
        elif tool_name == "create_directory" and "path" in kwargs:
            record_create_directory(kwargs["path"], tool_name="create_directory")
    except Exception:
        pass  # 备份失败不应阻断正常操作


def _wrap_confirm_tool(t):
    """给会修改/删除文件系统状态的工具加人工审核闸门，拒绝则不执行。"""
    original_func = getattr(t, "func", None)
    original_coroutine = getattr(t, "coroutine", None)

    if original_coroutine is not None:
        async def wrapped_coroutine(*args, **kwargs):
            summary = _build_call_summary(t.name, args, kwargs)
            if not confirm_action(summary):
                return f"操作已被用户拒绝：{t.name} 未执行。"
            _record_backup_before_call(t.name, kwargs)
            return await original_coroutine(*args, **kwargs)
        t.coroutine = wrapped_coroutine

    if original_func is not None:
        def wrapped_func(*args, **kwargs):
            summary = _build_call_summary(t.name, args, kwargs)
            if not confirm_action(summary):
                return f"操作已被用户拒绝：{t.name} 未执行。"
            _record_backup_before_call(t.name, kwargs)
            return original_func(*args, **kwargs)
        t.func = wrapped_func

    return t


async def get_mcp_tools() -> list:
    """异步启动 MCP filesystem server，返回它的工具列表（已排除撞名工具）。"""
    client = MultiServerMCPClient(
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
        tools = await client.get_tools()
    except Exception as e:
        print(f"[MCP] 启动 filesystem server 失败：{e}")
        print("[MCP] 已跳过 MCP 工具加载，其它工具仍可正常使用。")
        return []
    tools = [t for t in tools if t.name not in _EXCLUDED_MCP_TOOLS]
    tools = [_wrap_read_tool(t) if t.name in _TRUNCATE_MCP_TOOLS else t for t in tools]
    tools = [_wrap_confirm_tool(t) if t.name in _CONFIRM_MCP_TOOLS else t for t in tools]
    print(f"[MCP] 从 filesystem server 加载了 {len(tools)} 个工具：")
    for t in tools:
        print(f"  - {t.name}")
    print(f"[MCP] 允许操作的目录：{ALLOWED_DIRS}")
    return tools
