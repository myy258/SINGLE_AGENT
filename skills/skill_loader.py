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

_SKILLS_DIR = Path(__file__).resolve().parent / "skills"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)


def _parse_skill_file(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
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
        return None
    return {"name": name, "description": description, "body": body}


def _load_all_skills() -> dict[str, dict]:
    skills = {}
    if not _SKILLS_DIR.exists():
        return skills
    for f in sorted(_SKILLS_DIR.glob("*.md")):
        parsed = _parse_skill_file(f)
        if parsed:
            skills[parsed["name"]] = parsed
    return skills


_SKILLS: dict[str, dict] = _load_all_skills()


def format_skill_index_for_prompt() -> str:
    """生成"技能目录"文本（name + 一句话简介），供拼进 system prompt。"""
    if not _SKILLS:
        return "（当前没有可用技能）"
    return "\n".join(f"  - {name}：{info['description']}" for name, info in _SKILLS.items())


@tool
def load_skill(skill_name: str) -> str:
    """加载某个技能的详细步骤指引。当当前任务匹配【可用技能】目录里的某一项时，
    先调用这个工具拿到完整步骤，再照着执行，而不是凭自己猜测流程。

    Args:
        skill_name: 技能名称，必须跟【可用技能】目录里列出的名字完全一致。
    """
    skill = _SKILLS.get(skill_name)
    if skill is None:
        available = "、".join(_SKILLS) or "（无）"
        return f"未找到名为「{skill_name}」的技能。当前可用技能：{available}"
    return skill["body"]


def get_skill_tools() -> list:
    """返回技能相关工具列表，供 main.py 挂载。"""
    return [load_skill]
