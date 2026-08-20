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
"""

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

from tools.confirm import confirm_action

_BACKUP_DIR = Path(__file__).resolve().parent / "backups"
_MANIFEST_PATH = _BACKUP_DIR / "manifest.json"

_MAX_PER_PATH: int = 5
_MAX_TOTAL: int = 200


def _load_manifest() -> list[dict]:
    if not _MANIFEST_PATH.exists():
        return []
    try:
        return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_manifest(entries: list[dict]) -> None:
    _BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    _MANIFEST_PATH.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _delete_backup_file(entry: dict) -> None:
    backup_file = entry.get("backup_file")
    if backup_file:
        p = _BACKUP_DIR / backup_file
        if p.exists():
            p.unlink()


def _prune(entries: list[dict], new_path: str | None) -> list[dict]:
    """按路径保留最近 N 条 + 总条目数上限，淘汰最旧的记录（连带删除 .bak 文件）。"""
    if new_path is not None:
        same_path = [e for e in entries if e.get("path") == new_path]
        while len(same_path) > _MAX_PER_PATH:
            oldest = same_path.pop(0)
            _delete_backup_file(oldest)
            entries.remove(oldest)
    while len(entries) > _MAX_TOTAL:
        oldest = entries.pop(0)
        _delete_backup_file(oldest)
    return entries


def _append_entry(entry: dict) -> None:
    entries = _load_manifest()
    entries.append(entry)
    entries = _prune(entries, entry.get("path"))
    _save_manifest(entries)


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def record_write(path: str, tool_name: str) -> str:
    """写入/编辑前调用：备份旧内容（如果文件存在），返回本次记录的 id。"""
    p = Path(path)
    existed_before = p.exists()
    backup_file = None
    if existed_before:
        backup_id = _new_id()
        backup_file = f"{backup_id}.bak"
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
    })
    return entry_id


def record_move(source: str, destination: str, tool_name: str = "move_file") -> str:
    """移动/重命名前调用：只记来源/目标路径。"""
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
    return entry_id


def record_create_directory(path: str, tool_name: str = "create_directory") -> str:
    """建目录前调用：只记这个目录是新建的。"""
    entry_id = _new_id()
    _append_entry({
        "id": entry_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tool": tool_name,
        "op": "create_directory",
        "path": str(path),
    })
    return entry_id


def _find_entry(entries: list[dict], backup_id: str) -> dict | None:
    for e in entries:
        if e["id"] == backup_id:
            return e
    return None


def _describe_entry(e: dict) -> str:
    if e["op"] == "write":
        target = f"{e['path']}（原来{'存在' if e['existed_before'] else '不存在'}）"
    elif e["op"] == "move":
        target = f"{e['source']} → {e['destination']}"
    else:
        target = e["path"]
    return f"[{e['id']}] {e['timestamp']}  {e['tool']}({e['op']})  {target}"


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
    recent = entries[-limit:][::-1]
    return "\n".join(_describe_entry(e) for e in recent)


@tool
def rollback(backup_id: str) -> str:
    """按 id 回滚一次写入/编辑/移动/建目录操作，恢复到该操作之前的状态。
    执行前会弹出人工审核确认。只覆盖 write_file/edit_file/move_file/
    create_directory 四类工具，不覆盖 python_exec/run_python_script 的副作用。

    Args:
        backup_id: 要回滚的记录 id，从 list_backups 的结果里获取。
    """
    entries = _load_manifest()
    entry = _find_entry(entries, backup_id)
    if entry is None:
        return f"未找到 id 为「{backup_id}」的回滚记录，先用 list_backups 查一下有哪些。"

    summary = f"工具：rollback\n即将回滚这条记录：\n{_describe_entry(entry)}"
    if not confirm_action(summary):
        return "回滚已被用户拒绝，未执行。"

    op = entry["op"]
    try:
        if op == "write":
            path = Path(entry["path"])
            if entry["existed_before"]:
                backup_path = _BACKUP_DIR / entry["backup_file"]
                if not backup_path.exists():
                    return f"回滚失败：备份文件已丢失（{entry['backup_file']}）。"
                # 回滚前先把"回滚之前的当前内容"也备份一次，回滚本身可再撤销
                record_write(str(path), tool_name="rollback")
                shutil.copy2(backup_path, path)
                return f"已回滚：恢复 {path} 到 {entry['timestamp']} 之前的内容。"
            else:
                if path.exists():
                    record_write(str(path), tool_name="rollback")
                    path.unlink()
                return f"已回滚：删除 {path}（该文件是这次操作新建的）。"

        elif op == "move":
            source, destination = Path(entry["source"]), Path(entry["destination"])
            if not destination.exists():
                return f"回滚失败：{destination} 不存在，可能已被后续操作改动。"
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(source))
            record_move(str(destination), str(source), tool_name="rollback")
            return f"已回滚：{destination} 移回 {source}。"

        elif op == "create_directory":
            path = Path(entry["path"])
            if not path.exists():
                return f"「{path}」已经不存在，无需回滚。"
            if any(path.iterdir()):
                return f"回滚失败：目录 {path} 非空（里面有后续新增的内容），未删除。"
            path.rmdir()
            return f"已回滚：删除新建的空目录 {path}。"

        else:
            return f"未知的操作类型：{op}"

    except Exception as e:
        return f"回滚失败：{e}"


def get_rollback_tools() -> list:
    """返回回滚相关工具列表，供 main.py 挂载。"""
    return [list_backups, rollback]
