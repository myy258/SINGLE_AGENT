# 技能加载器：扫描 skills/*.md，按需把技能正文注入对话上下文
# Co-authored with CoCo

"""
每个 skill 是 skills/ 目录下一个 .md 文件，格式：

    ---
    name: 技能名
    description: 一句话简介（什么时候用）
    ---
    （正文：详细步骤指引）

启动时只把 name+description 拼成"技能目录"塞进 system prompt；
正文很长的话不常驻上下文，模型判断任务匹配某个技能时才调用 load_skill
拿到完整正文，避免 prompt 随技能数量线性膨胀。
"""

import re
from pathlib import Path

from langchain_core.tools import tool

# 本文件就在 skills/ 目录内，所以技能目录是 parent 本身。
# （曾经写成 parent / "skills" → 指向不存在的 skills/skills/，导致 _SKILLS 恒为空、
#   所有 load_skill 调用永远返回"未找到"，而且是完全静默的失败。）
_SKILLS_DIR = Path(__file__).resolve().parent

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_skill_file(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"[skills] 警告：读取 {path.name} 失败，已跳过：{exc}")
        return None
    m = _FRONTMATTER_RE.match(text)
    if not m:
        print(
            f"[skills] 警告：{path.name} 缺少合法的 frontmatter（--- name/description ---），已跳过。"
        )
        return None
    header, body = m.group(1), m.group(2).strip()
    meta = {}
    for line in header.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    name = meta.get("name")
    description = meta.get("description", "")
    if not name:
        print(f"[skills] 警告：{path.name} 的 frontmatter 里没有 name 字段，已跳过。")
        return None
    return {"name": name, "description": description, "body": body}


def _load_all_skills() -> dict[str, dict]:
    skills: dict[str, dict] = {}
    if not _SKILLS_DIR.exists():
        print(f"[skills] 警告：技能目录不存在：{_SKILLS_DIR}")
        return skills

    md_files = sorted(_SKILLS_DIR.glob("*.md"))
    for f in md_files:
        parsed = _parse_skill_file(f)
        if parsed:
            if parsed["name"] in skills:
                print(f"[skills] 警告：技能名「{parsed['name']}」重复，{f.name} 覆盖了之前的定义。")
            skills[parsed["name"]] = parsed

    # 静默失败是这个模块历史上最大的坑：目录里明明有 .md 却一个都没加载成功时，
    # 必须吵出来，而不是让 system prompt 悄悄渲染成"（当前没有可用技能）"。
    if md_files and not skills:
        print(
            f"[skills] 严重警告：{_SKILLS_DIR} 下有 {len(md_files)} 个 .md 文件，"
            "但没有一个解析成功——技能系统当前完全不可用，请检查上面的逐条警告。"
        )
    elif not md_files:
        print(f"[skills] 提示：{_SKILLS_DIR} 下没有 .md 技能文件。")

    return skills


_SKILLS: dict[str, dict] | None = None


def _get_skills() -> dict[str, dict]:
    """懒加载 + 缓存。避免 import 期就做磁盘 IO 和打印，也方便测试里 reload。"""
    global _SKILLS
    if _SKILLS is None:
        _SKILLS = _load_all_skills()
    return _SKILLS


def reload_skills() -> dict[str, dict]:
    """强制重新扫描技能目录（改完 .md 不用重启进程）。"""
    global _SKILLS
    _SKILLS = _load_all_skills()
    return _SKILLS


def format_skill_index_for_prompt() -> str:
    """生成"技能目录"文本（name + 一句话简介），供拼进 system prompt。"""
    skills = _get_skills()
    if not skills:
        return "（当前没有可用技能）"
    return "\n".join(f"  - {name}：{info['description']}" for name, info in skills.items())


@tool
def load_skill(skill_name: str) -> str:
    """加载某个技能的详细步骤指引。当当前任务匹配【可用技能】目录里的某一项时，
    先调用这个工具拿到完整步骤，再照着执行，而不是凭自己猜测流程。

    Args:
        skill_name: 技能名称，必须跟【可用技能】目录里列出的名字完全一致。
    """
    skills = _get_skills()
    skill = skills.get(skill_name)
    if skill is None:
        available = "、".join(skills) or "（无）"
        return f"未找到名为「{skill_name}」的技能。当前可用技能：{available}"
    return skill["body"]


def get_skill_tools() -> list:
    """返回技能相关工具列表，供 main.py 挂载。"""
    return [load_skill]
