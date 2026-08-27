# Git 版本管理工具：受限子命令白名单 + 冲突/历史安全检查 + 变更类操作前人工审核
# Co-authored with CoCo

"""
只支持有限的 git 子命令，防止模型拼出任意危险命令；
add/commit/push/pull/tag/checkout/init/clone 这类会改变仓库状态或远程内容的操作，
执行前都要经过 confirm_action() 人工确认，跟 tools.py 里其它写入类工具的防护思路一致。

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
5. 上面 1/3 两条检查如果自身执行失败（比如 git status 报错），不会当成"没问题"
   静默放行，而是直接拒绝执行并提示用户手动核实。
6. push 命令带 --force/-f/--force-with-lease 时，人工审核的确认摘要里会追加醒目
   警告，提醒这会覆盖远程历史且不可撤销。
"""

import subprocess
from datetime import datetime

from langchain_core.tools import tool

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


def _run_git(parts: list[str], cwd: str, timeout: int = 30):
    return subprocess.run(
        ["git"] + parts, capture_output=True, text=True, timeout=timeout,
        cwd=cwd, encoding="utf-8", errors="replace",
    )


def _get_unmerged_paths(cwd: str) -> tuple[list[str] | None, bool]:
    """返回 (未解决合并冲突的文件路径列表, 检查是否失败)。
    检查本身失败时返回 (None, True)，调用方必须把"无法确认"如实告知用户，
    不能当成"没有冲突"直接放行。
    """
    try:
        proc = _run_git(["status", "--porcelain"], cwd)
    except Exception:
        return None, True
    unmerged = []
    for line in proc.stdout.splitlines():
        if len(line) >= 2 and line[:2] in _UNMERGED_STATUS_CODES:
            unmerged.append(line[3:].strip())
    return unmerged, False


def _is_working_tree_clean(cwd: str) -> tuple[bool | None, bool]:
    """返回 (是否干净, 检查是否失败)。检查本身失败时返回 (None, True)。"""
    try:
        proc = _run_git(["status", "--porcelain"], cwd)
    except Exception:
        return None, True
    return proc.stdout.strip() == "", False


def _make_safety_tag(cwd: str) -> str | None:
    """pull/merge 前打一个安全快照 tag，返回 tag 名（打失败返回 None，不阻断主流程）。"""
    tag_name = f"presync_{datetime.now():%Y%m%d_%H%M%S}"
    try:
        proc = _run_git(["tag", tag_name], cwd)
        return tag_name if proc.returncode == 0 else None
    except Exception:
        return None


@tool
def git_command(git_args: str, cwd: str = ".") -> str:
    """在指定目录下执行 git 命令，用于版本管理和发布（提交代码、推送、打 tag 等）。
    只支持部分子命令：status / diff / log / branch / remote / add / commit / push /
    pull / tag / checkout / init / clone / reset / merge。不支持的子命令会被拒绝。

    Args:
        git_args: git 子命令及参数（不要带 "git" 前缀），例如：
            "status"、'commit -m "fix: 修复xxx"'、"remote add origin <仓库URL>"、
            "push -u origin main"、"tag v1.0.0"、"push origin v1.0.0"。
            不要传 --allow-unrelated-histories，这个会被拒绝。
        cwd: 执行命令的目录，默认当前目录；用户应先告知项目所在路径。
    """
    parts = git_args.strip().split()
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

    # ── add/commit 前：硬性检查是否存在未解决的合并冲突 ──────────────────
    if subcmd in ("add", "commit"):
        unmerged, check_failed = _get_unmerged_paths(cwd)
        if check_failed:
            return (
                "操作被拒绝：未能确认当前是否存在未解决的合并冲突（git status 执行异常），"
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
        is_clean, check_failed = _is_working_tree_clean(cwd)
        if check_failed:
            return (
                "操作被拒绝：未能确认工作区是否干净（git status 执行异常），"
                "为安全起见不会自动 pull/merge。请手动执行 git status 核实后再重试。"
            )
        if not is_clean:
            return (
                "操作被拒绝：工作区有未提交的本地改动，不能直接 pull/merge。"
                "请先 commit 这些改动，或明确告知要放弃它们后再重试。"
            )
        safety_tag = _make_safety_tag(cwd)

    if subcmd in _MUTATING_SUBCOMMANDS:
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
        if not confirm_action(summary):
            return f"操作已被用户拒绝：未执行 git {git_args}。"

    try:
        proc = _run_git(parts, cwd, timeout=60)
    except FileNotFoundError:
        return "未检测到 git，请先在本机安装 git 并确保命令行可以直接运行 git。"
    except subprocess.TimeoutExpired:
        return f"git {git_args} 执行超时（60秒），push/pull/clone 需要联网，请检查网络后重试。"
    except Exception as e:
        return f"执行 git 命令失败：{e}"

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
