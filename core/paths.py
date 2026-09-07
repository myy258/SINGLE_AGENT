# 路径解析与白名单收口：所有落盘/读取类工具共用同一套路径判定
# Co-authored with CoCo

"""
之前的问题：ALLOWED_DIRS 白名单只约束 MCP 的 read_file/list_directory 那一套，
而本地的 write_file / python_exec / run_python_script / git_command 完全不校验
路径——绝对路径想写哪写哪，等于整个白名单模型在这几个工具面前形同虚设。

这里把"路径怎么解析"和"路径允许不允许"抽成纯 stdlib 的函数，便于单测，
并让所有工具走同一套判定，避免各写一份互相不一致。

判定规则：
- 相对路径 / 纯文件名  → 落到默认工作目录（并剥掉路径分量，防 ../../ 逃逸）
- 绝对路径            → 保留，但必须落在 allowed_dirs 的某一棵子树内
- 一律先 resolve()     → 同时消解 ..、. 和符号链接，防止用软链接绕过白名单
"""

from __future__ import annotations

from pathlib import Path


class PathNotAllowed(PermissionError):
    """目标路径不在允许访问的目录白名单内。"""


def _resolve(path: Path) -> Path:
    """解析成绝对真实路径。文件不存在也不报错（strict=False）。"""
    return Path(path).expanduser().resolve(strict=False)


def is_within(path: str | Path, allowed_dirs: list[str] | tuple[str, ...]) -> bool:
    """path 是否落在 allowed_dirs 任一目录（含其子目录）内。"""
    target = _resolve(Path(path))
    for d in allowed_dirs:
        try:
            root = _resolve(Path(d))
        except Exception:
            continue
        if target == root or target.is_relative_to(root):
            return True
    return False


def ensure_within(path: str | Path, allowed_dirs: list[str] | tuple[str, ...]) -> Path:
    """校验并返回解析后的路径；不在白名单内则抛 PathNotAllowed。"""
    target = _resolve(Path(path))
    if not is_within(target, allowed_dirs):
        allowed = "\n".join(f"  - {d}" for d in allowed_dirs)
        raise PathNotAllowed(
            f"路径不在允许访问的范围内：{target}\n当前允许的目录：\n{allowed}\n"
            "如果确实需要访问这个位置，请把它加进 config.ALLOWED_DIRS 后重启。"
        )
    return target


def resolve_user_path(
    filename: str,
    default_dir: str | Path,
    allowed_dirs: list[str] | tuple[str, ...],
) -> Path:
    """把用户/模型给的文件名解析成最终落盘路径，并做白名单校验。

    相对路径只取文件名部分（`Path(x).name`），这样 "../../etc/passwd" 会被压成
    "passwd" 落到默认工作目录，而不是逃出去。绝对路径保留原样但必须过白名单。
    """
    raw = Path(filename).expanduser()
    if raw.is_absolute():
        return ensure_within(raw, allowed_dirs)

    name = raw.name
    if not name:
        raise PathNotAllowed(f"非法的文件名：{filename!r}")
    return ensure_within(Path(default_dir) / name, allowed_dirs)
