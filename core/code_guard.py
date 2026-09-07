# 代码执行前的 AST 静态检查：替代原来的字符串子串黑名单
# Co-authored with CoCo

"""
原来的做法是子串匹配：

    _DANGEROUS_CODE_PATTERNS = ("os.system(", "subprocess.", "eval(", ...)
    if pattern in code: 拒绝

两类问题：

假阴性（拦不住）——随手就能绕过：
    getattr(__builtins__, 'sys' + 'tem')
    importlib.import_module('os').system(...)     # importlib 没在名单里
    open('/etc/passwd').read()                   # open 根本没拦
    __import__('os').popen(...)                   # 拼接/间接调用一律漏过
    (1).__class__.__mro__[1].__subclasses__()    # 经典逃逸链

假阳性（误拦）——只要注释或字符串里出现这些字面量就被拒：
    print("不要用 subprocess.run")   # 明明是纯文本，也被拦

改成 AST 遍历后，判定的是"代码真正表达的语义"而不是文本，上面两类问题都消失。

⚠️ 重要认知：这仍然只是纵深防御的一层，**不是真沙箱**。子进程跑的是完整
CPython，理论上总有绕过空间。真正的隔离要靠 config.ALLOWED_DIRS 之外的
容器/虚拟机，或者 seccomp。这一层的目标是"挡住模型无意间写出的危险代码"，
而不是"防住有针对性的攻击者"。
"""

from __future__ import annotations

import ast

# 完全禁止导入的模块（含其子模块）
_DENIED_MODULES: frozenset[str] = frozenset({
    "subprocess",
    "ctypes",
    "socket",
    "multiprocessing",
    "importlib",
    "pty",
    "fcntl",
    "signal",
    "winreg",
    "webbrowser",
    "http",
    "ftplib",
    "telnetlib",
    "smtplib",
    "xmlrpc",
    "pickle",
    "marshal",
    "shelve",
})

# 允许导入但禁止调用的具体属性（模块名 → 属性集合）
_DENIED_ATTRS: dict[str, frozenset[str]] = {
    "os": frozenset({
        "system", "popen", "remove", "unlink", "rmdir", "removedirs",
        "execv", "execve", "execl", "execlp", "execvp", "spawnv", "spawnl",
        "fork", "forkpty", "kill", "killpg", "chmod", "chown", "chroot",
        "setuid", "setgid", "abort", "truncate", "rename", "renames", "replace",
    }),
    "shutil": frozenset({"rmtree", "move", "chown"}),
    "sys": frozenset({"exit", "settrace", "setprofile", "setrecursionlimit"}),
}

# 禁止调用的内建函数
_DENIED_BUILTINS: frozenset[str] = frozenset({
    "eval", "exec", "compile", "__import__",
    "globals", "locals", "vars",
    "getattr", "setattr", "delattr",
    "input", "breakpoint", "help", "exit", "quit",
})

# 禁止访问的 dunder 属性/名字：这是绝大多数逃逸链的入口
_DENIED_DUNDERS: frozenset[str] = frozenset({
    "__class__", "__base__", "__bases__", "__mro__", "__subclasses__",
    "__globals__", "__builtins__", "__import__", "__code__", "__closure__",
    "__dict__", "__getattribute__", "__reduce__", "__reduce_ex__",
    "__init_subclass__", "__subclasshook__", "__loader__", "__spec__",
})


class UnsafeCode(ValueError):
    """代码里含有被禁止的操作。message 直接面向模型，说明拒绝原因和改法。"""


def _root_module(dotted: str) -> str:
    return dotted.split(".", 1)[0]


def _check_import(node: ast.Import | ast.ImportFrom, problems: list[str]) -> None:
    if isinstance(node, ast.Import):
        names = [a.name for a in node.names]
    else:
        # `from . import x` 的 module 是 None
        names = [node.module] if node.module else []

    for name in names:
        if _root_module(name) in _DENIED_MODULES:
            problems.append(f"第 {node.lineno} 行：禁止导入模块 `{name}`")


def _check_attribute(node: ast.Attribute, problems: list[str]) -> None:
    if node.attr in _DENIED_DUNDERS:
        problems.append(f"第 {node.lineno} 行：禁止访问内部属性 `{node.attr}`")
        return
    # 只判定 `模块名.属性` 这种最直接的形式；间接引用交给 dunder/getattr 拦截兜底
    if isinstance(node.value, ast.Name):
        denied = _DENIED_ATTRS.get(node.value.id)
        if denied and node.attr in denied:
            problems.append(
                f"第 {node.lineno} 行：禁止调用 `{node.value.id}.{node.attr}`"
            )


def _looks_absolute(text: str) -> bool:
    """跨平台判断一个字符串字面量是否长得像绝对路径。"""
    if not text or len(text) > 4096:
        return False
    if text.startswith(("/", "\\\\")):
        return True
    # Windows 盘符：C:\... 或 C:/...
    return len(text) >= 3 and text[1] == ":" and text[2] in ("\\", "/")


def _check_path_literal(
    node: ast.Constant,
    allowed_dirs: list[str] | tuple[str, ...],
    problems: list[str],
) -> None:
    if not isinstance(node.value, str) or not _looks_absolute(node.value):
        return
    # 延迟导入，避免 core.paths 与本模块产生循环依赖
    from core.paths import is_within

    if not is_within(node.value, allowed_dirs):
        problems.append(
            f"第 {node.lineno} 行：代码里出现了白名单外的绝对路径 `{node.value}`"
        )


def scan(code: str, allowed_dirs: list[str] | tuple[str, ...] | None = None) -> None:
    """静态检查一段 Python 代码。有问题就抛 UnsafeCode，干净则正常返回。

    Args:
        code: 待检查的源码。
        allowed_dirs: 若提供，代码里出现的绝对路径字面量必须落在这些目录内，
            否则拒绝。这是为了堵住 python_exec 绕过 ALLOWED_DIRS 白名单最
            现实的一条路（直接写死一个绝对路径去读写）。非字面量路径（拼接、
            变量）静态查不出来，只能靠人工确认那一关兜底。
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeCode(
            f"代码语法错误，第 {exc.lineno} 行：{exc.msg}。请修正语法后重试。"
        ) from exc

    problems: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            _check_import(node, problems)
        elif isinstance(node, ast.Attribute):
            _check_attribute(node, problems)
        elif isinstance(node, ast.Name):
            if node.id in _DENIED_DUNDERS:
                problems.append(f"第 {node.lineno} 行：禁止引用 `{node.id}`")
            elif node.id in _DENIED_BUILTINS and isinstance(node.ctx, ast.Load):
                problems.append(f"第 {node.lineno} 行：禁止使用 `{node.id}`")
        elif allowed_dirs and isinstance(node, ast.Constant):
            _check_path_literal(node, allowed_dirs, problems)

    if problems:
        # 去重但保持顺序，避免同一问题在 walk 里重复出现
        seen: set[str] = set()
        unique = [p for p in problems if not (p in seen or seen.add(p))]
        raise UnsafeCode(
            "运行被拒绝，代码里含有不允许的操作：\n"
            + "\n".join(f"  - {p}" for p in unique)
            + "\n\n说明：这里只跑分析/计算类代码。需要读写文件请用 read_file / "
            "write_file 工具，需要执行系统命令请把命令告诉用户由他手动执行。"
        )
