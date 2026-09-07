# 本地 Agent 系统介绍

<p align="center">
  <b><a href="README_en.md">English</a> | 中文</b>
</p>

## 1. 定位

部署本地agent：写和跑代码（主python）、跑分析、写作、翻译、常识问答、联网搜索、本地知识库检索、文件读写。

## 2. 架构

- `main.py` 启动，组装：`core/llm.py`（LLM）+ `tools/local_tools.py`（本地工具）+
  `skills/skill_loader.py`（技能）+ `tools/mcp_setup.py`（MCP 文件系统工具）+
  `tools/rollback.py` / `tools/git_tool.py`（回滚/Git）
- 全部工具挂给 `agent.py` 里的 `SingleAgent`（`create_react_agent`），system prompt =
  工具使用准则 + 技能目录；会话历史留最近 20 条，ReAct ≤ 20 步
- 工具分两类：
  - **只读/无副作用**：直接执行（如 calculator、read_file、search_local_knowledge_base）
  - **写入/执行类**：先经 `confirm.py` 人工确认，确认后 `rollback.py` 自动备份再执行
    （write_file、python_exec、edit_file、move_file、git 变更类子命令等）
- 每轮运行轨迹（不含内容）写入 `core/logger.py` 生成的会话日志

**无 supervisor，无 workers**——一个 LLM + 一份 prompt + 一堆工具。

## 3. 文件清单

```
SINGLE_AGENT/
├── main.py               # 命令行交互入口
├── config.py             # LLM 后端选择 + RAG 开关 + 路径白名单
├── agent.py              # SingleAgent 类（create_react_agent）
│
├── core/                 # agent 运行的底层引擎
│   ├── llm.py             # 三后端 LLM 工厂
│   ├── logger.py          # 会话日志
│   ├── events.py          # 事件汇抽象（CLI 下 no-op，Web 层注入实现）
│   ├── checkpointer.py    # 会话历史持久化（LangGraph SqliteSaver）
│   ├── paths.py           # 路径解析 + ALLOWED_DIRS 白名单收口
│   ├── code_guard.py      # python_exec 的 AST 静态检查
│   └── safe_eval.py       # calculator 的 AST 白名单求值
│
├── tools/                # 挂载给 agent 的工具实现
│   ├── local_tools.py     # 本地工具
│   ├── mcp_setup.py       # MCP filesystem server 接入 + 截断/审核/备份包装
│   ├── git_tool.py        # Git 版本管理
│   ├── rollback.py        # 操作回滚
│   └── confirm.py         # 终端阻塞式人工审核
│
├── rag/                  # 本地知识库检索
│   ├── embedder.py         # BGE-small-zh 中文向量化
│   ├── retriever.py        # Dense / BM25 / Hybrid 检索器
│   ├── knowledge_base.py   # 加载 texts/ 下 .txt 并段落切 chunk
│   └── rag_tool.py         # search_local_knowledge_base 工具
│
├── skills/               # 技能加载器 + 技能内容放在一起
│   ├── skill_loader.py     # 扫描 skills/*.md，暴露 load_skill 工具
│   ├── 代码加注释.md
│   ├── 翻译.md
│   └── Git版本管理与发布.md
│
├── texts/                # 本地知识库文档

```

> `core/`、`tools/`、`rag/`、`skills/` 都是普通 Python package（各自有 `__init__.py`）。
> 项目始终以 `python main.py`（在 `SINGLE_AGENT/` 目录下）启动，`sys.path[0]` 会是
> `SINGLE_AGENT/` 本身，所以子包内部可以用 `from tools.confirm import confirm_action`
> 这种绝对导入互相引用，`config.py`/`agent.py` 留在根目录的模块也能被子包直接
> `from config import ...` 引用，不需要相对导入。

## 4. Tool集

- **本地工具**：write_file、python_exec、run_python_script、calculator、
  current_time、baidu_search、search_local_knowledge_base
- **技能工具**：`load_skill(name)` 按需加载某技能完整指引
- **回滚工具**：`list_backups` / `rollback(id)`
- **Git 工具**：`git_command`，子命令白名单，变更类需审核；自带合并冲突检查、
  pull/merge 前自动打安全 tag、拒绝 `--allow-unrelated-histories`
- **MCP 文件系统工具**（npx 拉起）：read/list/search/get_file_info 等只读，加上
  create_directory/edit_file/move_file

## 5. Agent Skills

- `skills/*.md`，frontmatter 含 `name` + `description`；启动时目录常驻 prompt，
  正文按需 `load_skill` 注入
- 新增技能只需加一个 `.md`，无需改代码，`skill_loader.py` 自动扫描

## 6. 自搭建RAG 本地知识库

`texts/*.txt` 段落切 chunk，三种检索模式（默认 hybrid）：
- **Dense**：语义相近/同义词匹配强，依赖 embedding 模型，对专有名词不敏感
- **BM25**：关键词/专有名词匹配准、快，不理解语义
- **Hybrid**：RRF 融合两者，覆盖最广，默认推荐；知识库小且偏关键词查询时可单用 BM25

## 7. LLM 后端

可导入本地模型或者通过API使用外部模型。受硬件配置限制，本地只能使用27b以下的模型，如qwen2.5-7b能流畅跑起来，qwen3.6-27b效率会变慢。

## 8. 硬上限

ReAct ≤20 步（递归上限 44）；MCP 单次读取 ≤4万字符自动截断；输入≈60万字符预警；
回滚记录同路径≤5条/总量≤200条。

## 9. 运行

```bash
cd SINGLE_AGENT && python main.py
```
输入 `new` 开新会话，`exit` 退出。

## 12. 硬件配置

- Intel(R) Xeon(R) Gold 5115 CPU @ 2.40GHz (2.39 GHz)
- RAM 64.0 GB
- NVIDIA Quadro P4000

## 13. 备注

供交流学习使用，点击查看 [`LICENSE`](LICENSE) 文件

