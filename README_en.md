# Local Agent System Overview

<p align="center">
  <b>English | <a href="README.md">Chinese</a></b>
</p>

## 1. Purpose

Deploy a local agent that can: write and run code (mainly Python), run analysis, write, translate, answer general-knowledge questions, search the web, retrieve from a local knowledge base, and read/write files.

## 2. Architecture

- Entry point `main.py` assembles: `core/llm.py` (LLM) + `tools/local_tools.py` (local tools) +
  `skills/skill_loader.py` (skills) + `tools/mcp_setup.py` (MCP filesystem tools) +
  `tools/rollback.py` / `tools/git_tool.py` (rollback / Git)
- All tools are attached to `SingleAgent` in `agent.py` (`create_react_agent`). The system prompt =
  tool-usage guidelines + skill catalog; conversation history keeps the most recent 20 messages; ReAct loop ≤ 20 steps
- Tools fall into two categories:
  - **Read-only / side-effect-free**: executed directly (e.g. calculator, read_file, search_local_knowledge_base)
  - **Write / execute type**: first goes through human confirmation in `confirm.py`; once confirmed, `rollback.py`
    automatically backs up before executing (write_file, python_exec, edit_file, move_file, Git mutating subcommands, etc.)
- Each run's execution trace (excluding content) is written to the session log produced by `core/logger.py`

**No supervisor, no workers** — just one LLM + one prompt + a set of tools.

## 3. File Manifest

```
SINGLE_AGENT/
├── main.py               # Command-line interactive entry point
├── config.py             # LLM backend selection + RAG toggle + path allowlist
├── agent.py              # SingleAgent class (create_react_agent)
│
├── core/                 # Underlying engine the agent runs on
│   ├── llm.py             # Factory for the three LLM backends
│   ├── logger.py          # Session logging
│   ├── events.py          # Event sink abstraction (no-op in CLI, implemented by the Web layer)
│   ├── checkpointer.py    # Conversation history persistence (LangGraph SqliteSaver)
│   ├── paths.py           # Path resolution + single source of truth for the ALLOWED_DIRS allowlist
│   ├── code_guard.py      # AST static checks for python_exec
│   └── safe_eval.py       # AST-allowlist evaluation for calculator
│
├── tools/                # Tool implementations attached to the agent
│   ├── local_tools.py     # Local tools
│   ├── mcp_setup.py       # MCP filesystem server integration + truncation/confirmation/backup wrapping
│   ├── git_tool.py        # Git version management
│   ├── rollback.py        # Operation rollback
│   └── confirm.py         # Terminal blocking human review
│
├── rag/                  # Local knowledge base retrieval
│   ├── embedder.py         # BGE-small-zh Chinese text vectorization
│   ├── retriever.py        # Dense / BM25 / Hybrid retriever
│   ├── knowledge_base.py   # Loads .txt files under texts/ and chunks them by paragraph
│   └── rag_tool.py         # search_local_knowledge_base tool
│
├── skills/               # Skill loader kept together with skill content
│   ├── skill_loader.py     # Scans skills/*.md, exposes the load_skill tool
│   ├── 代码加注释.md
│   ├── 翻译.md
│   └── Git版本管理与发布.md
│
├── texts/                # Local knowledge base documents

```

> `core/`, `tools/`, `rag/`, and `skills/` are all ordinary Python packages (each with its own `__init__.py`).
> The project is always started with `python main.py` (from within the `SINGLE_AGENT/` directory), so `sys.path[0]`
> will be `SINGLE_AGENT/` itself. This means submodules can use absolute imports such as
> `from tools.confirm import confirm_action` to reference each other, and modules kept at the root
> (`config.py` / `agent.py`) can be imported directly from submodules via `from config import ...`,
> with no need for relative imports.

## 4. Tool Set

- **Local tools**: write_file, python_exec, run_python_script, calculator,
  current_time, baidu_search, search_local_knowledge_base
- **Skill tool**: `load_skill(name)` loads the full guidance for a skill on demand
- **Rollback tools**: `list_backups` / `rollback(id)`
- **Git tool**: `git_command`, with an allowlist of subcommands; mutating ones require review; includes built-in
  merge-conflict checks, automatically tags a safety point before pull/merge, and rejects `--allow-unrelated-histories`
- **MCP filesystem tools** (launched via npx): read/list/search/get_file_info etc. are read-only, plus
  create_directory/edit_file/move_file

## 5. Agent Skills

- `skills/*.md`, with frontmatter containing `name` + `description`; the catalog stays resident in the prompt at
  startup, and the body is injected on demand via `load_skill`
- Adding a new skill only requires adding a `.md` file — no code changes needed; `skill_loader.py` scans automatically

## 6. Self-Built RAG Local Knowledge Base

`texts/*.txt` files are chunked by paragraph, with three retrieval modes (default: hybrid):
- **Dense**: strong at semantic/synonym matching, depends on the embedding model, less sensitive to proper nouns
- **BM25**: accurate and fast at keyword/proper-noun matching, does not understand semantics
- **Hybrid**: fuses both via RRF, offers the broadest coverage and is the default recommendation; use BM25 alone
  when the knowledge base is small and queries are keyword-heavy

## 7. LLM Backends

You can import a local model or use an external model via API. Due to hardware constraints, locally you can only
run models under 27B — for example qwen2.5-7b runs smoothly, while qwen3.6-27b becomes noticeably slower.

## 8. Hard Limits

ReAct ≤ 20 steps (recursion limit 44); a single MCP read is auto-truncated beyond 40,000 characters; input is
flagged at ≈600,000 characters; rollback records are capped at ≤5 per path / ≤200 total.

## 9. Running

```bash
cd SINGLE_AGENT && python main.py
```
Type `new` to start a new session, `exit` to quit.

## 12. Hardware Configuration

- Intel(R) Xeon(R) Gold 5115 CPU @ 2.40GHz (2.39 GHz)
- RAM 64.0 GB
- NVIDIA Quadro P4000

## 13. Notes

For learning and discussion purposes only. See the [`LICENSE`](LICENSE) file for details.
