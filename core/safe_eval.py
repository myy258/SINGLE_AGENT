# 数学表达式安全求值：AST 白名单，替代原来的裸 eval()
# Co-authored with CoCo

"""
原来 tools/local_tools.py 里的 calculator 用的是：

    eval(expr, {"__builtins__": {}}, _SAFE_NAMES)

两个问题：
1. `{"__builtins__": {}}` 挡不住经典逃逸链，形如
   `(1).__class__.__mro__[1].__subclasses__()` 能一路摸到任意类，
   进而拿到 os/subprocess。
2. 就算不逃逸，`2**999999999` 也能直接把进程 CPU / 内存打满。

这里改成 AST 白名单：只允许数字字面量、四则运算/幂/取模/整除、一元正负号、
括号，以及少数几个纯数值函数。任何其它节点类型（属性访问、下标、名字引用、
函数定义、推导式……）一律拒绝，从根上没有 eval 那种"先执行再看"的问题。

同时对幂运算做规模预检查，避免合法语法造成的资源耗尽。
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable

# 允许的二元运算
_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# 允许的一元运算
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# 允许调用的纯数值函数（都不接受可迭代对象，避免藉此构造复杂表达式）
_FUNCS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}

_CONSTS: dict[str, float] = {"pi": math.pi, "e": math.e}

# 规模上限：防止合法语法把 CPU/内存打满
_MAX_NODES: int = 200
_MAX_POW_EXPONENT: float = 1024.0
_MAX_ABS_RESULT: float = 1e308


class UnsafeExpression(ValueError):
    """表达式包含不允许的语法结构，或规模超出安全上限。"""


def _check_pow(base: Any, exponent: Any) -> None:
    try:
        if abs(exponent) > _MAX_POW_EXPONENT:
            raise UnsafeExpression(
                f"幂运算指数过大（|{exponent}| > {_MAX_POW_EXPONENT:.0f}），已拒绝以避免资源耗尽。"
            )
        if abs(base) > 1 and abs(exponent) * math.log10(abs(base)) > 308:
            raise UnsafeExpression("幂运算结果超出可表示范围，已拒绝。")
    except TypeError as exc:
        raise UnsafeExpression(f"幂运算的操作数类型不支持：{exc}") from exc


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise UnsafeExpression(f"只允许数字字面量，收到：{node.value!r}")
        return node.value

    if isinstance(node, ast.BinOp):
        op_func = _BIN_OPS.get(type(node.op))
        if op_func is None:
            raise UnsafeExpression(f"不支持的运算符：{type(node.op).__name__}")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            _check_pow(left, right)
        return op_func(left, right)

    if isinstance(node, ast.UnaryOp):
        op_func = _UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise UnsafeExpression(f"不支持的一元运算符：{type(node.op).__name__}")
        return op_func(_eval_node(node.operand))

    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise UnsafeExpression(
            f"不允许引用名字「{node.id}」。只支持数字、运算符和这些常量：{', '.join(_CONSTS)}"
        )

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise UnsafeExpression("只允许直接调用白名单函数，不允许属性访问式调用。")
        func = _FUNCS.get(node.func.id)
        if func is None:
            raise UnsafeExpression(
                f"不允许调用「{node.func.id}」。可用函数：{', '.join(sorted(_FUNCS))}"
            )
        if node.keywords:
            raise UnsafeExpression("函数调用不支持关键字参数。")
        args = [_eval_node(a) for a in node.args]
        if node.func.id == "pow" and len(args) >= 2:
            _check_pow(args[0], args[1])
        return func(*args)

    raise UnsafeExpression(f"不允许的语法结构：{type(node).__name__}")


def safe_eval(expression: str) -> Any:
    """求值一个纯数学表达式。任何越界语法都抛 UnsafeExpression。"""
    expr = expression.strip().strip("\"'`")
    if not expr:
        raise UnsafeExpression("表达式为空。")
    if len(expr) > 500:
        raise UnsafeExpression(f"表达式过长（{len(expr)} 字符 > 500）。")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"表达式语法错误：{exc.msg}") from exc

    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > _MAX_NODES:
        raise UnsafeExpression(f"表达式过于复杂（{node_count} 个语法节点 > {_MAX_NODES}）。")

    result = _eval_node(tree)

    if isinstance(result, float) and (math.isinf(result) or math.isnan(result)):
        raise UnsafeExpression(f"计算结果不是有限数值：{result}")
    if isinstance(result, (int, float)) and abs(result) > _MAX_ABS_RESULT:
        raise UnsafeExpression("计算结果超出可表示范围。")
    return result
