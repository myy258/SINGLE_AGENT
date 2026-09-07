# 操作回滚：写入/编辑/移动/建目录前自动备份旧状态，可按 id 回滚
# Co-authored with CoCo

"""
原理：
- write_file / edit_file 执行前，把"旧文件内容"备份成一个 .bak 文件；
- move_file 执行前，只记"从哪移到哪"（回滚 = 反向移动一次）；
- create_directory 执行前，只记"这个目录是新建的"（回滚 = 目录为空才删除）。
- 所有记录写进 backups/manifest.json，每条记录一个唯一 id。
- rollback(id) 按记录反向执行一次，本身也会走 confirm_action 确认，
  并且回滚动作也会生成一条新记录（可以对回滚再回滚）。
- 不覆盖 python_exec / run_python_script：那两个工具跑任意代码，没法预先
  知道会改动哪些文件，无法安全备份/回滚。

保留策略（避免 backups/ 无限增长）：
- 同一个路径最多保留最近 5 条备份记录；
- manifest 总条目数最多保留 200 条；
- 超出的从最旧开始淘汰，连带删除对应的 .bak 文件。

本版修正的两个问题：

1. 「回滚会静默抹掉他人改动」
   原来回滚 write 时直接 shutil.copy2(backup, path)。如果该文件在备份之后
   被别的流程改过（人工编辑、python_exec 里的代码、另一个工具），这些改动
   会被无声覆盖。现在写入完成后会记录一份 after_sha256，回滚时先比对：
   对不上就说明文件被改过，默认拒绝并要求用户再确认一次。

2. 「备份失败无人知晓」
   原来 record_* 的异常在调用方被 `except Exception: pass` 吞掉，结果是
   确认框照弹、操作照做，但事后 list_backups 里查不到、根本回滚不了，
   用户毫不知情。现在 record_* 返回 RecordResult，调用方把
   result.warning_suffix 拼进工具返回值，让"这次不可回滚"显式暴露出来。

并发：manifest 是"读-改-写"整个 JSON，多线程/多请求同时写会互相覆盖
（Web 版尤其明显）。现在用 进程内 threading.Lock + 跨进程的原子锁文件
双重保护。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from tools.confirm import confirm_action

_BACKUP_DIR = Path(__file__).resolve().parent / "backups"
_MANIFEST_PATH = _BACKUP_DIR / "manifest.json"
_LOCK_PATH = _BACKUP_DIR / "manifest.lock"

_MAX_PER_PATH: int = 5
_MAX_TOTAL: int = 200

# 锁文件等待参数
_LOCK_TIMEOUT: float = 5.0
_LOCK_POLL: float = 0.02

_thread_lock = threading.Lock()


# ── 备份记录结果 ─────────────────────────────────────────────────────────
@dataclass
class RecordResult:
    """一次备份登记的结果。ok=False 表示这次操作事后无法回滚。"""

    id: str | None
    ok: bool
    path: str
    reason: str = ""

    @property
    def warning_suffix(self) -> str:
        """拼在工具返回值末尾的提示。备份成功时是空串。"""
        if self.ok:
            return ""
        return (
            f"\n⚠️ 注意：本次操作的回滚备份没有成功（{self.reason}），"
            "因此这次改动**无法用 rollback 撤销**。"
        )

    def finalize(self) -> None:
        """操作真正完成后调用：记录改动后的内容指纹，供回滚时校验是否被二次修改。"""
        if not self.ok or self.id is None:
            return
        digest = _sha256_of(Path(self.path))
        if digest is None:
            return
        try:
            with _manifest_lock():
                entries = _load_manifest()
                for e in entries:
                    if e.get("id") == self.id:
                        e["after_sha256"] = digest
                        break
                _save_manifest(entries)
        except Exception:
            # 指纹只是额外的安全校验，写不进去不影响回滚本身能用
            pass


_FAILED_SENTINEL_REASON = "备份写入异常"


# ── 底层：锁、指纹、manifest 读写 ──────────────────────────────────────────
@contextmanager
def _manifest_lock():
    """进程内 + 跨进程的 manifest 互斥。拿不到锁就抛 TimeoutError。"""
    with _thread_lock:
        _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + _LOCK_TIMEOUT
        fd = None
        while True:
            try:
                fd = os.open(str(_LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.monotonic() > deadline:
                    # 锁文件可能是上次崩溃留下的陈旧文件，超时后强行接管
                    try:
                        _LOCK_PATH.unlink()
                        continue
                    except OSError as exc:
                        raise TimeoutError(f"获取 manifest 锁超时：{exc}") from exc
                time.sleep(_LOCK_POLL)
        try:
            yield
        finally:
            try:
                os.close(fd)
            finally:
                try:
                    _LOCK_PATH.unlink()
                except OSError:
                    pass


def _sha256_of(path: Path) -> str | None:
    """计算文件内容的 sha256。文件不存在或读不了返回 None。"""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(65536), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def _load_manifest() -> list[dict]:
    if not _MANIFEST_PATH.exists():
        return []
    try:
        data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _save_manifest(entries: list[dict]) -> None:
    """原子写：先写临时文件再 replace，避免中途崩溃留下半截 JSON。"""
    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _MANIFEST_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, _MANIFEST_PATH)


def _delete_backup_file(entry: dict) -> None:
    backup_file = entry.get("backup_file")
    if backup_file:
        p = _BACKUP_DIR / backup_file
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass


def _prune(entries: list[dict], new_path: str | None) -> list[dict]:
    """按路径保留最近 N 条 + 总条目数上限，淘汰最旧的记录（连带删除 .bak 文件）。"""
    if new_path is not None:
        same_path = [e for e in entries if e.get("path") == new_path]
        while len(same_path) > _MAX_PER_PATH:
            oldest = same_path.pop(0)
            _delete_backup_file(oldest)
            # 按 id 删除，而不是按字典相等——两条内容完全相同的记录会误删错的那条
            entries = [e for e in entries if e.get("id") != oldest.get("id")]
    while len(entries) > _MAX_TOTAL:
        oldest = entries.pop(0)
        _delete_backup_file(oldest)
    return entries


def _append_entry(entry: dict) -> None:
    with _manifest_lock():
        entries = _load_manifest()
        entries.append(entry)
        entries = _prune(entries, entry.get("path"))
        _save_manifest(entries)


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


# ── 登记接口（供各写入类工具在执行前调用）────────────────────────────────
def record_write(path: str, tool_name: str) -> RecordResult:
    """写入/编辑前调用：备份旧内容（如果文件存在）。返回 RecordResult。"""
    p = Path(path)
    try:
        existed_before = p.exists()
        backup_file = None
        before_sha = None
        if existed_before:
            before_sha = _sha256_of(p)
            backup_file = f"{_new_id()}.bak"
            _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, _BACKUP_DIR / backup_file)

        entry_id = _new_id()
        _append_entry({
            "id": entry_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tool": tool_name,
            "op": "write",
            "path": str(p),
            "existed_before": existed_before,
            "backup_file": backup_file,
            "before_sha256": before_sha,
        })
        return RecordResult(id=entry_id, ok=True, path=str(p))
    except (OSError, TimeoutError) as exc:
        return RecordResult(id=None, ok=False, path=str(p), reason=f"{_FAILED_SENTINEL_REASON}：{exc}")


def record_move(source: str, destination: str, tool_name: str = "move_file") -> RecordResult:
    """移动/重命名前调用：只记来源/目标路径。"""
    try:
        entry_id = _new_id()
        _append_entry({
            "id": entry_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tool": tool_name,
            "op": "move",
            "source": str(source),
            "destination": str(destination),
            "path": str(destination),
        })
        return RecordResult(id=entry_id, ok=True, path=str(destination))
    except (OSError, TimeoutError) as exc:
        return RecordResult(
            id=None, ok=False, path=str(destination), reason=f"{_FAILED_SENTINEL_REASON}：{exc}"
        )


def record_create_directory(path: str, tool_name: str = "create_directory") -> RecordResult:
    """建目录前调用：只记这个目录是新建的。"""
    try:
        entry_id = _new_id()
        _append_entry({
            "id": entry_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tool": tool_name,
            "op": "create_directory",
            "path": str(path),
        })
        return RecordResult(id=entry_id, ok=True, path=str(path))
    except (OSError, TimeoutError) as exc:
        return RecordResult(
            id=None, ok=False, path=str(path), reason=f"{_FAILED_SENTINEL_REASON}：{exc}"
        )


# ── 查询与回滚 ───────────────────────────────────────────────────────────
def _find_entry(entries: list[dict], backup_id: str) -> dict | None:
    for e in entries:
        if e.get("id") == backup_id:
            return e
    return None


def _describe_entry(e: dict) -> str:
    op = e.get("op")
    if op == "write":
        target = f"{e['path']}（原来{'存在' if e.get('existed_before') else '不存在'}）"
    elif op == "move":
        target = f"{e.get('source')} → {e.get('destination')}"
    else:
        target = e.get("path", "?")
    return f"[{e.get('id')}] {e.get('timestamp')}  {e.get('tool')}({op})  {target}"


@tool
def list_backups(limit: int = 10) -> str:
    """列出最近的可回滚操作记录（write_file/edit_file/move_file/create_directory）。
    回滚前先调用这个工具看看有哪些 id 可选。

    Args:
        limit: 最多列出多少条，默认 10（从最新往前数）。
    """
    entries = _load_manifest()
    if not entries:
        return "目前没有任何可回滚的操作记录。"
    recent = entries[-max(1, limit):][::-1]
    return "\n".join(_describe_entry(e) for e in recent)


def _rollback_write(entry: dict) -> str:
    path = Path(entry["path"])

    if not entry.get("existed_before"):
        if path.exists():
            record_write(str(path), tool_name="rollback")
            try:
                path.unlink()
            except OSError as exc:
                return f"回滚失败：删除 {path} 出错：{exc}"
        return f"已回滚：删除 {path}（该文件是这次操作新建的）。"

    backup_name = entry.get("backup_file")
    if not backup_name:
        return f"回滚失败：这条记录没有关联备份文件，无法恢复 {path}。"
    backup_path = _BACKUP_DIR / backup_name
    if not backup_path.exists():
        return f"回滚失败：备份文件已丢失（{backup_name}）。"

    # 关键校验：文件在这次操作之后是否又被改过？
    after_sha = entry.get("after_sha256")
    current_sha = _sha256_of(path)
    if after_sha and current_sha and current_sha != after_sha:
        extra = (
            f"⚠️ {path} 在这次操作之后又被修改过（内容指纹不一致）。\n"
            "直接回滚会把那之后的改动一起抹掉，且无法恢复。\n"
            "确认仍要回滚吗？"
        )
        if not confirm_action(extra, tool_name="rollback_overwrite"):
            return (
                f"回滚已中止：{path} 在该操作之后被改动过，为避免丢失这些改动未执行回滚。"
            )

    record_write(str(path), tool_name="rollback")
    try:
        shutil.copy2(backup_path, path)
    except OSError as exc:
        return f"回滚失败：恢复 {path} 出错：{exc}"
    return f"已回滚：恢复 {path} 到 {entry.get('timestamp')} 之前的内容。"


@tool
def rollback(backup_id: str) -> str:
    """按 id 回滚一次写入/编辑/移动/建目录操作，恢复到该操作之前的状态。
    执行前会弹出人工审核确认。只覆盖 write_file/edit_file/move_file/
    create_directory 四类工具，不覆盖 python_exec/run_python_script 的副作用。

    如果目标文件在那次操作之后又被改动过，会额外提示并要求二次确认，
    避免静默抹掉别人的改动。

    Args:
        backup_id: 要回滚的记录 id，从 list_backups 的结果里获取。
    """
    entries = _load_manifest()
    entry = _find_entry(entries, backup_id)
    if entry is None:
        return f"未找到 id 为「{backup_id}」的回滚记录，先用 list_backups 查一下有哪些。"

    summary = f"工具：rollback\n即将回滚这条记录：\n{_describe_entry(entry)}"
    if not confirm_action(summary, tool_name="rollback"):
        return "回滚已被用户拒绝，未执行。"

    op = entry.get("op")
    try:
        if op == "write":
            return _rollback_write(entry)

        if op == "move":
            source, destination = Path(entry["source"]), Path(entry["destination"])
            if not destination.exists():
                return f"回滚失败：{destination} 不存在，可能已被后续操作改动。"
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(source))
            record_move(str(destination), str(source), tool_name="rollback")
            return f"已回滚：{destination} 移回 {source}。"

        if op == "create_directory":
            path = Path(entry["path"])
            if not path.exists():
                return f"「{path}」已经不存在，无需回滚。"
            if any(path.iterdir()):
                return f"回滚失败：目录 {path} 非空（里面有后续新增的内容），未删除。"
            path.rmdir()
            return f"已回滚：删除新建的空目录 {path}。"

        return f"未知的操作类型：{op}"

    except OSError as exc:
        return f"回滚失败：{exc}"


def get_rollback_tools() -> list:
    """返回回滚相关工具列表，供 main.py 挂载。"""
    return [list_backups, rollback]
