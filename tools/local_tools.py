# 本地工具集：write_file / calculator / current_time / python_exec / run_python_script / baidu_search + 可插拔 RAG

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

from config import ENABLE_RAG
from tools.confirm import confirm_action
from tools.rollback import record_write

OUTPUT_DIR = Path(os.path.expanduser("~")) / "Desktop" / "qwen3_agent_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 工具 1：写文件（本地版，处理 CSV BOM）─────────────────────────────────
@tool
def write_file(filename: str, content: str) -> str:
    """把文字内容写入本地文件。默认生成 .txt；文件名以 .csv 结尾会按 CSV 格式
    保存（带 BOM 的 UTF-8，Excel 中文不乱码）。当用户要求生成表格/Excel 数据时
    用 .csv 后缀，不要用 .xlsx。

    Args:
        filename: 目标文件名，例如 'hello.txt' 或 'data.csv'。
            相对路径 / 纯文件名 → 落到默认工作目录；绝对路径 → 原样使用。
        content: 要写入的文字内容；CSV 每行换行、每列逗号。
    """
    safe_name = Path(filename).name
    if safe_name.lower().endswith(".csv"):
        encoding = "utf-8-sig"
    else:
        if not safe_name.endswith(".txt") and "." not in safe_name:
            safe_name += ".txt"
        encoding = "utf-8"
    raw = Path(filename)
    if raw.is_absolute():
        filepath = raw
    else:
        filepath = OUTPUT_DIR / safe_name

    preview = content if len(content) < 500 else content[:500] + "...(截断)"
    summary = f"工具：write_file\n目标文件：{filepath}\n内容预览：\n{preview}"
    if not confirm_action(summary):
        return f"操作已被用户拒绝：未写入文件 {filepath}。"

    if raw.is_absolute():
        filepath.parent.mkdir(parents=True, exist_ok=True)
    record_write(str(filepath), tool_name="write_file")
    filepath.write_text(content, encoding=encoding)
    return f"已写入文件：{filepath}"


# ── 工具 2：计算器 ───────────────────────────────────────────────────────
_SAFE_NAMES = {
    "abs": abs, "min": min, "max": max, "round": round,
    "pow": pow, "sum": sum, "len": len,
}


@tool
def calculator(expression: str) -> str:
    """计算数学表达式。当用户问算术、求和、乘除、百分比等数值问题时使用。

    Args:
        expression: 一个纯数学表达式字符串，例如 '12*(3+4)'、'(100-15)/5'、'2**10'。
            不要包含中文或单位。
    """
    try:
        expr = expression.strip().strip("\"'`")
        result = eval(expr, {"__builtins__": {}}, _SAFE_NAMES)
        return f"计算结果：{result}"
    except Exception as e:
        return f"计算失败：{e}"


# ── 工具 3：当前时间 ─────────────────────────────────────────────────────
@tool
def current_time() -> str:
    """返回当前的本地日期和时间。当用户问 '现在几点'、'今天几号'、
    '当前时间' 时使用。无需任何参数。
    """
    return f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


# ── 工具 4：Python 代码执行 ──────────────────────────────────────────────
_DANGEROUS_CODE_PATTERNS = (
    "os.system(", "os.popen(", "os.remove(", "os.unlink(", "os.rmdir(",
    "shutil.rmtree(", "subprocess.", "eval(", "exec(", "__import__(",
    "ctypes", "socket.",
)


def _scan_dangerous_code(code: str) -> str | None:
    for pattern in _DANGEROUS_CODE_PATTERNS:
        if pattern in code:
            return pattern
    return None


@tool
def python_exec(code: str, timeout: int = 60) -> str:
    """直接在内部运行一段 Python 代码，返回它 print 出来的内容。
    **这是分析/计算/临时验证的首选**：无需先写脚本文件，代码由 agent 在内部
    子进程中即时执行，只把 stdout / stderr 返回。适合数据分析、快速验证、
    数值计算、字符串处理等一切"我只关心结果"的场景。

    当用户明确说"生成一个 py 脚本" / "保存成 .py 文件"时，才改用
    write_file + run_python_script。

    基础安全限制：不允许 os.system / subprocess / eval / exec / ctypes /
    socket 等高风险调用。

    Args:
        code: 要运行的 Python 代码字符串。**必须用 print(...) 把结果打印出来**，
            否则返回值为空。可以 import numpy、pandas 等标准库。
        timeout: 最长允许运行的秒数，默认 60。
    """
    hit = _scan_dangerous_code(code)
    if hit:
        return (
            f"运行被拒绝：代码里包含高风险调用模式 `{hit}`。请改写后重试。"
        )

    code_preview = code if len(code) < 1000 else code[:1000] + "...(截断)"
    summary = f"工具：python_exec\n即将执行代码：\n{code_preview}"
    if not confirm_action(summary):
        return "操作已被用户拒绝：未执行代码。"

    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(OUTPUT_DIR), encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"运行超时：代码超过 {timeout} 秒未结束，已强制终止。"
    except Exception as e:
        return f"运行失败：{e}"

    output = proc.stdout.strip()
    error = proc.stderr.strip()
    if proc.returncode != 0:
        return f"代码执行出错（退出码 {proc.returncode}）：\n{error or output or '（无输出）'}"
    return f"代码执行成功。\n标准输出：\n{output or '（无输出——记得用 print(...) 打印你要的结果）'}"


@tool
def run_python_script(filename: str, timeout: int = 30) -> str:
    """运行一个已经存在的本地 Python 脚本文件，返回它的标准输出和报错。
    使用前必须先用 write_file 把代码保存成 .py 文件。不要凭空编造"运行成功"，
    必须用真实运行结果说话。基础安全限制：不允许 os.system / subprocess /
    eval / exec / ctypes / socket 等高风险调用。

    Args:
        filename: 要运行的脚本文件名，例如 'analyze.py'（不带 .py 会自动加）。
            相对路径默认从默认工作目录读取；绝对路径原样使用。
        timeout: 最长允许运行的秒数，默认 30。
    """
    raw = Path(filename)
    if raw.is_absolute():
        filepath = raw
    else:
        safe_name = raw.name
        if not safe_name.endswith(".py"):
            safe_name += ".py"
        filepath = OUTPUT_DIR / safe_name

    if not filepath.exists():
        return f"运行失败：文件不存在 {filepath}，请先用 write_file 把代码写入这个文件。"

    code = filepath.read_text(encoding="utf-8", errors="replace")
    hit = _scan_dangerous_code(code)
    if hit:
        return (
            f"运行被拒绝：脚本 {filepath.name} 里包含高风险代码模式 `{hit}`。"
            "请去掉这类写法后再重新写入文件运行。"
        )

    code_preview = code if len(code) < 1000 else code[:1000] + "...(截断)"
    summary = f"工具：run_python_script\n脚本文件：{filepath}\n代码内容：\n{code_preview}"
    if not confirm_action(summary):
        return f"操作已被用户拒绝：未运行脚本 {filepath.name}。"

    try:
        proc = subprocess.run(
            [sys.executable, str(filepath)],
            capture_output=True, text=True, timeout=timeout, cwd=str(OUTPUT_DIR),
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        return f"运行超时：脚本 {filepath.name} 超过 {timeout} 秒未结束，已强制终止。"
    except Exception as e:
        return f"运行失败：{e}"

    output = proc.stdout.strip()
    error = proc.stderr.strip()
    if proc.returncode != 0:
        return f"脚本 {filepath.name} 执行出错（退出码 {proc.returncode}）：\n{error or output or '（无输出）'}"
    return f"脚本 {filepath.name} 执行成功。\n标准输出：\n{output or '（无输出，脚本正常结束）'}"


# ── 工具 5：百度搜索 ─────────────────────────────────────────────────────
_BAIDU_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}


@tool
def baidu_search(query: str, top_k: int = 5) -> str:
    """在百度上搜索实时信息。当用户询问新闻、时事、天气、人物、
    最新事件等需要联网才能回答的问题时使用。

    Args:
        query: 搜索关键词，用中文简洁描述要搜索的内容。
        top_k: 返回的搜索结果条数，默认 5 条。
    """
    try:
        url = "https://www.baidu.com/s"
        params = {"wd": query, "rn": top_k}
        resp = requests.get(url, params=params, headers=_BAIDU_HEADERS, timeout=8)
        resp.encoding = "utf-8"
        soup = BeautifulSoup(resp.text, "html.parser")

        results = []
        for item in soup.select("div.result, div.result-op"):
            title_tag = item.select_one("h3 a")
            abstract_tag = item.select_one("div.c-abstract, span.content-right_8Zs40")
            if not title_tag:
                continue
            title = title_tag.get_text(strip=True)
            abstract = abstract_tag.get_text(strip=True) if abstract_tag else ""
            results.append(f"【{title}】\n{abstract}")
            if len(results) >= top_k:
                break

        if not results:
            return "百度搜索未返回有效结果，请换个关键词再试。"
        return "\n\n".join(results)

    except requests.exceptions.ConnectionError:
        return "无法联网搜索：连接百度失败，可能是当前网络环境限制了访问，暂时无法使用联网搜索，建议换用其他信息来源或稍后再试。"
    except requests.exceptions.Timeout:
        return "搜索请求超时，请检查网络连接后重试。"
    except Exception:
        return "联网搜索暂时不可用，可能是网络限制或百度页面结构变化导致，建议换个方式获取信息。"


# ── 工厂：返回所有本地工具（含 RAG 插件）──────────────────────────────
def create_tools() -> list:
    """返回本地定义的全部工具。RAG 工具可插拔，缺依赖时自动跳过。"""
    base_tools = [write_file, calculator, current_time, python_exec, run_python_script, baidu_search]

    if ENABLE_RAG:
        try:
            from rag.rag_tool import get_rag_tools
            base_tools += get_rag_tools()
        except ImportError:
            print("[提示] 未找到 rag_tool.py，跳过 RAG 工具加载。")

    return base_tools
