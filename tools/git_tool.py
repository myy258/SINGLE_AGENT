# Git 版本管理工具：受限子命令白名单 + 冲突/历史安全检查 + 变更类操作前人工审核
# Co-authored with CoCo

"""
只支持有限的 git 子命令，防止模型拼出任意危险命令；
add/commit/push/pull/tag/checkout/init/clone 这类会改变仓库状态或远程内容的操作，
执行前都要经过 confirm_action() 人工确认，跟其它写入类工具的防护思路一致。

不内置任何具体仓库地址——远程仓库 URL 由用户在对话里提供，agent 通过
'git remote add origin <url>' 配置，不要在代码里硬编码任何人的仓库。

额外的安全检查（防止合并冲突把 <<<<<<< HEAD 这类标记误提交进代码）：
1. add/commit 前检查是否存在未解决的合并冲突（git status --porcelain 里的
   UU/AA/DD/AU/UA/UD/DU 状态码），存在就直接拒绝执行，不管确认框选了什么。
2. pull/merge 前自动打一个 "presync_<时间戳>" 标签作为安全快照，方便出问题时
   用 git checkout/reset 一键恢复。
3. pull/merge 前检查工作区是否干净（无未提交的本地改动），不干净就拒绝执行，
   避免本地改动和远程改动混在一起更容易冲突。
4. 拒绝在 pull/merge 里传 --allow-unrelated-histories，这个参数专门用来强行合并
   两段完全不相关的历史，几乎必然产生大量冲突，不允许模型自己决定用这个。
5. 上面 1/3 两条检查如果自身执行失败，不会当成"没问题"静默放行，而是直接拒绝
   执行并提示用户手动核实。
6. push --force / reset --hard 这类会不可逆丢弃内容的操作，人工审核的确认摘要里
   会追加醒目警告。

本版修正的三个问题：

1. 「带空格的 commit message 会被拆坏」
   原来用 git_args.strip().split()。模型按 docstring 示例传
       commit -m "fix: 修复xxx"
   split 后变成 ['commit', '-m', '"fix:', '修复xxx"']，commit message 变成
   `"fix:`（带一个字面双引号），后半截被当成额外的 pathspec 参数。
   现在改用 shlex.split()，按 shell 规则正确处理引号。

2. 「git status 的 returncode 从不检查 → 安全检查静默失效」
   原来只把"抛异常"算作检查失败。但 git status 在非仓库目录里是正常返回、
   returncode=128、stdout 为空——于是冲突检查看到空 stdout 判定"无冲突"放行
   add/commit，干净性检查同样判定"工作区干净"放行 pull/merge。
   模块说明第 5 条承诺的"不静默放行"，恰好在最常见的失败模式下失效了。
   现在显式检查 returncode。

3. 「cwd 完全不校验 → 沙箱逃逸」
   git_command 之前可以在系统上任意路径执行，clone 能往任意位置落盘，
   checkout/reset 能销毁任意仓库的改动，而 system prompt 还专门强调这一点。
   现在 cwd 同样受 config.ALLOWED_DIRS 约束，跟其它工具口径一致。
"""

from __future__ import annotations

import shlex
import subprocess
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from config import ALLOWED_DIRS
from core.paths import PathNotAllowed, ensure_within
from tools.confirm import confirm_action

_ALLOWED_SUBCOMMANDS = {
    "status", "diff", "log", "branch", "remote",
    "add", "commit", "push", "pull", "tag", "checkout", "init", "clone", "reset", "merge",
}

# 会改变仓库状态/远程内容的子命令：执行前必须人工审核确认
_MUTATING_SUBCOMMANDS = {
    "add", "commit", "push", "pull", "tag", "checkout", "init", "clone", "reset", "merge",
}

# 会把远程内容合并进本地工作区的子命令：需要额外的冲突/历史安全检查
_MERGE_LIKE_SUBCOMMANDS = {"pull", "merge"}

# 未解决合并冲突在 `git status --porcelain` 里的状态码
_UNMERGED_STATUS_CODES = {"UU", "AA", "DD", "AU", "UA", "UD", "DU"}

_DANGEROUS_FLAGS = {"--allow-unrelated-histories"}

# push 时会强行覆盖远程历史的高危参数：不拦截执行，但确认框里必须醒目提示
_FORCE_PUSH_FLAGS = {"--force", "-f", "--force-with-lease"}

# reset 时会直接丢弃工作区改动的参数
_HARD_RESET_FLAGS = {"--hard"}


class GitCheckFailed(RuntimeError):
    """安全检查本身没能完成（比如 git status 执行失败）。绝不能当成"检查通过"。"""


def _run_git(parts: list[str], cwd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git"] + parts, capture_output=True, text=True, timeout=timeout,
        cwd=cwd, encoding="utf-8", errors="replace",
    )


def _git_status_porcelain(cwd: str) -> str:
    """跑 git status --porcelain 并返回 stdout。任何失败都抛 GitCheckFailed。

    关键点：returncode != 0 也必须算失败。非 git 仓库时 git status 会正常返回、
    returncode=128、stdout 为空——如果只看异常，这里会被误判成"工作区干净、
    没有冲突"，把两道安全检查同时架空。
    """
    try:
        proc = _run_git(["status", "--porcelain"], cwd)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitCheckFailed(f"git status 执行异常：{exc}") from exc

    if proc.returncode != 0:
        detail = ((proc.stderr or "") + (proc.stdout or "")).strip() or "（无输出）"
        raise GitCheckFailed(
            f"git status 返回非零退出码 {proc.returncode}：{detail}"
        )
    return proc.stdout or ""


def _get_unmerged_paths(cwd: str) -> list[str]:
    """返回未解决合并冲突的文件路径列表。检查失败时抛 GitCheckFailed。"""
    unmerged = []
    for line in _git_status_porcelain(cwd).splitlines():
        if len(line) >= 2 and line[:2] in _UNMERGED_STATUS_CODES:
            unmerged.append(line[3:].strip())
    return unmerged


def _is_working_tree_clean(cwd: str) -> bool:
    """工作区是否干净。检查失败时抛 GitCheckFailed。"""
    return _git_status_porcelain(cwd).strip() == ""


def _make_safety_tag(cwd: str) -> str | None:
    """pull/merge 前打一个安全快照 tag，返回 tag 名（打失败返回 None，不阻断主流程）。"""
    tag_name = f"presync_{datetime.now():%Y%m%d_%H%M%S}"
    try:
        proc = _run_git(["tag", tag_name], cwd)
        return tag_name if proc.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _build_confirm_summary(git_args: str, cwd: str, subcmd: str, parts: list[str],
                           safety_tag: str | None) -> str:
    summary = f"工具：git_command\n目录：{cwd}\n即将执行：git {git_args}"
    if subcmd in _MERGE_LIKE_SUBCOMMANDS:
        summary += (
            "\n⚠️ 此操作可能引入冲突标记（<<<<<<< / ======= / >>>>>>>），"
            "如果冲突未妥善处理会污染文件内容。"
        )
        if safety_tag:
            summary += f"\n已自动打安全快照 tag：{safety_tag}（出问题可用它恢复）。"
    if subcmd == "push" and any(flag in _FORCE_PUSH_FLAGS for flag in parts):
        summary += (
            "\n🚨 检测到强制推送参数（--force/-f/--force-with-lease）：这会"
            "覆盖远程分支的历史记录，可能导致他人的提交丢失，且无法撤销，"
            "请务必确认这确实是你想要的操作。"
        )
    if subcmd == "reset" and any(flag in _HARD_RESET_FLAGS for flag in parts):
        summary += (
            "\n🚨 检测到 reset --hard：这会**直接丢弃工作区所有未提交的改动**，"
            "被丢弃的内容不在 rollback 的可恢复范围内，无法找回。"
        )
    if subcmd == "checkout":
        summary += (
            "\n⚠️ checkout 在指定文件路径时会用版本库内容覆盖本地改动，"
            "被覆盖的内容无法通过 rollback 找回。"
        )
    return summary


@tool
def git_command(git_args: str, cwd: str = ".") -> str:
    """在指定目录下执行 git 命令，用于版本管理和发布（提交代码、推送、打 tag 等）。
    只支持部分子命令：status / diff / log / branch / remote / add / commit / push /
    pull / tag / checkout / init / clone / reset / merge。不支持的子命令会被拒绝。

    cwd 必须落在允许访问的目录范围内（跟 read_file/write_file 等工具同一套白名单）。

    Args:
        git_args: git 子命令及参数（不要带 "git" 前缀），例如：
            "status"、'commit -m "fix: 修复xxx"'、"remote add origin <仓库URL>"、
            "push -u origin main"、"tag v1.0.0"、"push origin v1.0.0"。
            带空格的参数请用引号括起来，会按 shell 规则正确解析。
            不要传 --allow-unrelated-histories，这个会被拒绝。
        cwd: 执行命令的目录，默认当前目录；用户应先告知项目所在路径。
    """
    try:
        parts = shlex.split(git_args.strip())
    except ValueError as exc:
        return f"git 参数解析失败（引号没有闭合？）：{exc}"

    if not parts:
        return "git_command 参数为空，请提供要执行的 git 子命令。"

    subcmd = parts[0]
    if subcmd not in _ALLOWED_SUBCOMMANDS:
        return (
            f"不支持的 git 子命令「{subcmd}」。仅支持：{', '.join(sorted(_ALLOWED_SUBCOMMANDS))}"
        )

    if any(flag in _DANGEROUS_FLAGS for flag in parts):
        return (
            "操作被拒绝：不允许使用 --allow-unrelated-histories 强行合并两段不相关的历史，"
            "这几乎必然产生大量冲突。如果确实需要合并不相关的仓库，请手动在终端操作。"
        )

    # ── cwd 白名单校验 ──────────────────────────────────────────────────
    try:
        cwd_path = ensure_within(Path(cwd), ALLOWED_DIRS)
    except PathNotAllowed as exc:
        return f"操作被拒绝：{exc}"
    if not cwd_path.is_dir():
        return f"操作被拒绝：目录不存在或不是目录：{cwd_path}"
    cwd_str = str(cwd_path)

    # ── add/commit 前：硬性检查是否存在未解决的合并冲突 ──────────────────
    if subcmd in ("add", "commit"):
        try:
            unmerged = _get_unmerged_paths(cwd_str)
        except GitCheckFailed as exc:
            return (
                f"操作被拒绝：未能确认当前是否存在未解决的合并冲突（{exc}），"
                "为安全起见不会自动 add/commit。请手动执行 git status 核实工作区状态后再重试。"
            )
        if unmerged:
            return (
                "操作被拒绝：检测到以下文件存在未解决的合并冲突（git status 里状态是 "
                "UU/AA/DD 等），不能直接 add/commit：\n"
                + "\n".join(f"  - {p}" for p in unmerged)
                + "\n请先用 read_file 查看这些文件内容，去掉 <<<<<<< / ======= / >>>>>>> "
                "标记并合并出正确内容，用 write_file 写回干净版本后再重试。"
            )

    # ── pull/merge 前：工作区必须干净 + 自动打安全快照 ──────────────────
    safety_tag = None
    if subcmd in _MERGE_LIKE_SUBCOMMANDS:
        try:
            is_clean = _is_working_tree_clean(cwd_str)
        except GitCheckFailed as exc:
            return (
                f"操作被拒绝：未能确认工作区是否干净（{exc}），"
                "为安全起见不会自动 pull/merge。请手动执行 git status 核实后再重试。"
            )
        if not is_clean:
            return (
                "操作被拒绝：工作区有未提交的本地改动，不能直接 pull/merge。"
                "请先 commit 这些改动，或明确告知要放弃它们后再重试。"
            )
        safety_tag = _make_safety_tag(cwd_str)

    if subcmd in _MUTATING_SUBCOMMANDS:
        summary = _build_confirm_summary(git_args, cwd_str, subcmd, parts, safety_tag)
        if not confirm_action(summary, tool_name="git_command"):
            return f"操作已被用户拒绝：未执行 git {git_args}。"

    try:
        proc = _run_git(parts, cwd_str, timeout=60)
    except FileNotFoundError:
        return "未检测到 git，请先在本机安装 git 并确保命令行可以直接运行 git。"
    except subprocess.TimeoutExpired:
        return f"git {git_args} 执行超时（60秒），push/pull/clone 需要联网，请检查网络后重试。"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"执行 git 命令失败：{exc}"

    output = (proc.stdout or "") + (proc.stderr or "")

    if subcmd in _MERGE_LIKE_SUBCOMMANDS and "CONFLICT" in output:
        note = (
            "\n\n⚠️ 检测到合并冲突（CONFLICT）。禁止直接 add/commit！"
            "请先用 read_file 查看冲突文件内容，去掉 <<<<<<< / ======= / >>>>>>> 标记，"
            "确认内容正确后用 write_file 写回，再重新 add/commit。"
        )
        if safety_tag:
            note += f"\n如果想放弃这次合并，可以执行 git_command('reset --hard {safety_tag}')。"
        return f"git {git_args} 执行结果：\n{output.strip()}{note}"

    if proc.returncode != 0:
        return f"git {git_args} 执行失败（returncode={proc.returncode}）：\n{output.strip()}"
    return output.strip() or f"git {git_args} 执行成功（无输出）。"


def get_git_tools() -> list:
    """返回 git 相关工具列表，供 main.py 挂载。"""
    return [git_command]
