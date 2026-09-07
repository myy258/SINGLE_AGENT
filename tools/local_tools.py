# 本地工具集：write_file / calculator / current_time / python_exec / run_python_script / web_search + 可插拔 RAG
# Co-authored with CoCo

"""
安全模型（相对最初版本的变化）：

1. 路径：所有落盘/读取都过 core.paths，统一受 config.ALLOWED_DIRS 约束。
   之前绝对路径完全不校验，写哪都行，等于白名单只对 MCP 生效。
2. 代码执行：静态检查从"字符串子串黑名单"换成 core.code_guard 的 AST 检查
   （见该模块的说明，原来的名单既拦不住拼接绕过，又会误伤字符串里的字面量）。
3. 子进程：限定 cwd 在工作目录、注入资源上限（内存/CPU/文件大小）、
   洗掉环境变量里的凭证，输出做长度截断。
4. 数学表达式：calculator 从裸 eval 换成 core.safe_eval 的 AST 白名单求值。

⚠️ 依然要清楚：这些是纵深防御，不是真隔离。python_exec 跑的是完整 CPython
子进程，有决心的攻击者仍可能绕过静态检查。真正的隔离需要容器/seccomp。
每次执行前的人工确认（tools.confirm）是最后一道也是最重要的一道闸门。
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from config import (
    ALLOWED_DIRS,
    ENABLE_RAG,
    EXEC_MAX_MEMORY_MB,
    EXEC_MAX_OUTPUT_CHARS,
    EXEC_TIMEOUT_DEFAULT,
    get_default_work_dir,
)
from core import code_guard
from core.paths import PathNotAllowed, resolve_user_path
from core.safe_eval import UnsafeExpression, safe_eval
from tools.confirm import confirm_action
from tools.rollback import record_write


def _work_dir() -> Path:
    """默认工作目录。做成函数而不是 import 期的模块级常量：
    原来 local_tools 自己硬编码了一份 ~/Desktop/qwen3_agent_output，跟
    config.DEFAULT_WORK_DIR 重复定义——改 config 不会改 write_file 的落盘位置。
    而且模块级 mkdir 意味着"import 一下就在用户桌面建目录"，测试时很讨厌。
    """
    return Path(get_default_work_dir())


def _truncate_output(text: str) -> str:
    if len(text) <= EXEC_MAX_OUTPUT_CHARS:
        return text
    return (
        text[:EXEC_MAX_OUTPUT_CHARS]
        + f"\n...(输出过长已截断，原始长度 {len(text)} 字符，上限 {EXEC_MAX_OUTPUT_CHARS})"
    )


# ── 工具 1：写文件（本地版，处理 CSV BOM）─────────────────────────────────
@tool
def write_file(filename: str, content: str) -> str:
    """把文字内容写入本地文件。默认生成 .txt；文件名以 .csv 结尾会按 CSV 格式
    保存（带 BOM 的 UTF-8，Excel 中文不乱码）。当用户要求生成表格/Excel 数据时
    用 .csv 后缀，不要用 .xlsx。

    Args:
        filename: 目标文件名，例如 'hello.txt' 或 'data.csv'。
            相对路径 / 纯文件名 → 落到默认工作目录；绝对路径 → 原样使用，
            但必须落在允许访问的目录范围内。
        content: 要写入的文字内容；CSV 每行换行、每列逗号。
    """
    try:
        filepath = resolve_user_path(filename, _work_dir(), ALLOWED_DIRS)
    except PathNotAllowed as exc:
        return f"写入被拒绝：{exc}"

    # 后缀补全：只对"用户没给扩展名"的情况生效，不改动用户明确给出的后缀
    if not filepath.suffix:
        filepath = filepath.with_suffix(".txt")
    encoding = "utf-8-sig" if filepath.suffix.lower() == ".csv" else "utf-8"

    preview = content if len(content) < 500 else content[:500] + "...(截断)"
    summary = f"工具：write_file\n目标文件：{filepath}\n内容预览：\n{preview}"
    if not confirm_action(summary, tool_name="write_file"):
        return f"操作已被用户拒绝：未写入文件 {filepath}。"

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        backup_note = record_write(str(filepath), tool_name="write_file")
        filepath.write_text(content, encoding=encoding)
    except OSError as exc:
        return f"写入失败：{exc}"

    # 记录写入后的内容指纹：回滚时用它判断"文件之后有没有被别人改过"
    backup_note.finalize()
    return f"已写入文件：{filepath}{backup_note.warning_suffix}"


# ── 工具 2：计算器 ───────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。当用户问算术、求和、乘除、百分比等数值问题时使用。

    Args:
        expression: 一个纯数学表达式字符串，例如 '12*(3+4)'、'(100-15)/5'、'2**10'。
            支持 abs/round/min/max/pow/sqrt/log/exp/sin/cos/tan 和常量 pi、e。
            不要包含中文或单位。
    """
    try:
        return f"计算结果：{safe_eval(expression)}"
    except UnsafeExpression as exc:
        return f"计算失败：{exc}"
    except (ArithmeticError, ValueError, TypeError) as exc:
        return f"计算失败：{exc}"


# ── 工具 3：当前时间 ─────────────────────────────────────────────────────
@tool
def current_time() -> str:
    """返回当前的本地日期和时间。当用户问 '现在几点'、'今天几号'、
    '当前时间' 时使用。无需任何参数。
    """
    return f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


# ── 子进程执行的公共设施 ─────────────────────────────────────────────────
# 从子进程环境里摘掉的变量名片段：防止被执行的代码把凭证 print 出来回灌进对话
_SECRET_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "PAT", "CREDENTIAL")


def _clean_env() -> dict[str, str]:
    """复制一份环境变量，去掉看起来像凭证的项。"""
    return {
        k: v
        for k, v in os.environ.items()
        if not any(marker in k.upper() for marker in _SECRET_ENV_MARKERS)
    }


def _resource_limiter():
    """返回给 subprocess 的 preexec_fn，设置子进程资源上限。

    Windows 上没有 resource 模块，返回 None（跳过限制）——这是已知的平台差异，
    不是遗漏；Windows 下只剩 timeout 和静态检查两道防线。
    """
    try:
        import resource
    except ImportError:
        return None

    mem_bytes = EXEC_MAX_MEMORY_MB * 1024 * 1024 if EXEC_MAX_MEMORY_MB > 0 else 0

    def _apply() -> None:  # pragma: no cover - 只在子进程里执行
        if mem_bytes:
            resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
        # 单个文件最大 256MB，防止把磁盘写满
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 * 1024,) * 2)
        # 禁止创建 core dump
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return _apply


def _run_subprocess(argv: list[str], timeout: int) -> subprocess.CompletedProcess:
    work_dir = _work_dir()
    work_dir.mkdir(parents=True, exist_ok=True)
    kwargs: dict = dict(
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(work_dir),
        encoding="utf-8",
        errors="replace",
        env=_clean_env(),
    )
    limiter = _resource_limiter()
    if limiter is not None:
        kwargs["preexec_fn"] = limiter
    return subprocess.run(argv, **kwargs)


def _format_proc_result(label: str, proc: subprocess.CompletedProcess, empty_hint: str) -> str:
    output = _truncate_output((proc.stdout or "").strip())
    error = _truncate_output((proc.stderr or "").strip())
    if proc.returncode != 0:
        return f"{label}执行出错（退出码 {proc.returncode}）：\n{error or output or '（无输出）'}"
    return f"{label}执行成功。\n标准输出：\n{output or empty_hint}"


# ── 工具 4：Python 代码执行 ──────────────────────────────────────────────
@tool
def python_exec(code: str, timeout: int = 0) -> str:
    """直接在内部运行一段 Python 代码，返回它 print 出来的内容。
    **这是分析/计算/临时验证的首选**：无需先写脚本文件，代码由 agent 在内部
    子进程中即时执行，只把 stdout / stderr 返回。适合数据分析、快速验证、
    数值计算、字符串处理等一切"我只关心结果"的场景。

    当用户明确说"生成一个 py 脚本" / "保存成 .py 文件"时，才改用
    write_file + run_python_script。

    安全限制：不允许执行系统命令、不允许 subprocess/ctypes/socket/importlib
    这类模块、不允许访问 __class__/__globals__ 等内部属性、不允许出现允许目录
    之外的绝对路径。需要读写文件请改用 read_file / write_file 工具。

    Args:
        code: 要运行的 Python 代码字符串。**必须用 print(...) 把结果打印出来**，
            否则返回值为空。可以 import numpy、pandas 等分析库。
        timeout: 最长允许运行的秒数；传 0 或省略则用配置里的默认值。
    """
    try:
        code_guard.scan(code, allowed_dirs=ALLOWED_DIRS)
    except code_guard.UnsafeCode as exc:
        return str(exc)

    effective_timeout = timeout if timeout and timeout > 0 else EXEC_TIMEOUT_DEFAULT

    code_preview = code if len(code) < 1000 else code[:1000] + "...(截断)"
    summary = (
        f"工具：python_exec\n工作目录：{_work_dir()}\n"
        f"超时：{effective_timeout}s　内存上限：{EXEC_MAX_MEMORY_MB}MB\n"
        f"即将执行代码：\n{code_preview}"
    )
    if not confirm_action(summary, tool_name="python_exec"):
        return "操作已被用户拒绝：未执行代码。"

    try:
        proc = _run_subprocess([sys.executable, "-E", "-s", "-c", code], effective_timeout)
    except subprocess.TimeoutExpired:
        return f"运行超时：代码超过 {effective_timeout} 秒未结束，已强制终止。"
    except OSError as exc:
        return f"运行失败：{exc}"

    return _format_proc_result(
        "代码", proc, "（无输出——记得用 print(...) 打印你要的结果）"
    )


@tool
def run_python_script(filename: str, timeout: int = 0) -> str:
    """运行一个已经存在的本地 Python 脚本文件，返回它的标准输出和报错。
    使用前必须先用 write_file 把代码保存成 .py 文件。不要凭空编造"运行成功"，
    必须用真实运行结果说话。安全限制与 python_exec 相同。

    Args:
        filename: 要运行的脚本文件名，例如 'analyze.py'（不带 .py 会自动加）。
            相对路径默认从默认工作目录读取；绝对路径必须在允许访问范围内。
        timeout: 最长允许运行的秒数；传 0 或省略则用配置里的默认值。
    """
    try:
        filepath = resolve_user_path(filename, _work_dir(), ALLOWED_DIRS)
    except PathNotAllowed as exc:
        return f"运行被拒绝：{exc}"

    if not filepath.suffix:
        filepath = filepath.with_suffix(".py")

    if not filepath.exists():
        return f"运行失败：文件不存在 {filepath}，请先用 write_file 把代码写入这个文件。"

    try:
        code = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"运行失败：读取 {filepath} 出错：{exc}"

    try:
        code_guard.scan(code, allowed_dirs=ALLOWED_DIRS)
    except code_guard.UnsafeCode as exc:
        return f"脚本 {filepath.name} 未通过安全检查。\n{exc}"

    effective_timeout = timeout if timeout and timeout > 0 else EXEC_TIMEOUT_DEFAULT

    code_preview = code if len(code) < 1000 else code[:1000] + "...(截断)"
    summary = (
        f"工具：run_python_script\n脚本文件：{filepath}\n"
        f"超时：{effective_timeout}s　内存上限：{EXEC_MAX_MEMORY_MB}MB\n"
        f"代码内容：\n{code_preview}"
    )
    if not confirm_action(summary, tool_name="run_python_script"):
        return f"操作已被用户拒绝：未运行脚本 {filepath.name}。"

    try:
        proc = _run_subprocess([sys.executable, "-E", "-s", str(filepath)], effective_timeout)
    except subprocess.TimeoutExpired:
        return f"运行超时：脚本 {filepath.name} 超过 {effective_timeout} 秒未结束，已强制终止。"
    except OSError as exc:
        return f"运行失败：{exc}"

    return _format_proc_result(
        f"脚本 {filepath.name} ", proc, "（无输出，脚本正常结束）"
    )


# ── 工具 5：联网搜索 ─────────────────────────────────────────────────────
@tool
def web_search(query: str, top_k: int = 5) -> str:
    """在互联网上搜索实时信息。当用户询问新闻、时事、天气、人物、
    最新事件等需要联网才能回答的问题时使用。返回结果**带来源 URL**，
    引用外部信息时请把 URL 一并给出，便于用户核查。

    Args:
        query: 搜索关键词，简洁描述要搜索的内容。
        top_k: 返回的搜索结果条数，默认 5 条。
    """
    from tools.web_search import SearchUnavailable, format_results, run_search

    try:
        results = run_search(query, top_k)
    except SearchUnavailable as exc:
        # 明确区分"搜不了"和"搜了但没有"，避免模型把工具故障当成事实结论
        return f"联网搜索这次没能完成：{exc}"

    if not results:
        return "搜索正常完成，但没有返回任何结果条目，请换个关键词再试。"
    return format_results(results)


# ── 工厂：返回所有本地工具（含 RAG 插件）──────────────────────────────
def create_tools() -> list:
    """返回本地定义的全部工具。RAG 工具可插拔，缺依赖时自动跳过。"""
    base_tools = [
        write_file,
        calculator,
        current_time,
        python_exec,
        run_python_script,
        web_search,
    ]

    if ENABLE_RAG:
        try:
            from rag.rag_tool import get_rag_tools

            base_tools += get_rag_tools()
        except ImportError as exc:
            print(f"[提示] RAG 工具加载失败，已跳过（其它工具不受影响）：{exc}")

    return base_tools
