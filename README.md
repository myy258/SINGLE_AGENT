# 本地 Agent 系统介绍

## 1. 定位

部署本地agent：写代码、跑分析、写作、翻译、常识问答、联网搜索、本地知识库检索、文件读写。

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
├── main.py / agent.py / config.py     # 入口 / Agent 类 / 后端与路径配置
├── core/        llm.py（LLM 工厂） logger.py（会话日志）
├── tools/       local_tools.py mcp_setup.py git_tool.py rollback.py confirm.py
├── rag/         embedder.py retriever.py knowledge_base.py rag_tool.py
├── skills/      skill_loader.py + 各技能 .md（代码加注释/翻译/Git版本管理与发布）
├── texts/       本地知识库文档
└── backups/ logs/

```

> 始终在 `SINGLE_AGENT/` 目录下 `python main.py` 启动，模块间用绝对导入
> （如 `from tools.confirm import confirm_action`）。改代码要同步整个文件夹。

## 4. 工具集

- **本地工具**：write_file、python_exec（首选）、run_python_script、calculator、
  current_time、baidu_search、search_local_knowledge_base
- **技能工具**：`load_skill(name)` 按需加载某技能完整指引
- **回滚工具**：`list_backups` / `rollback(id)`（见 §6）
- **Git 工具**：`git_command`，子命令白名单，变更类需审核；自带合并冲突检查、
  pull/merge 前自动打安全 tag、拒绝 `--allow-unrelated-histories`
- **MCP 文件系统工具**（npx 拉起）：read/list/search/get_file_info 等只读，加上
  create_directory/edit_file/move_file（需审核+自动备份）

## 5. 技能包

- `skills/*.md`，frontmatter 含 `name` + `description`；启动时目录常驻 prompt，
  正文按需 `load_skill` 注入
- 新增技能只需加一个 `.md`，无需改代码，`skill_loader.py` 自动扫描

## 6. 写入类操作的安全机制

- **人工审核**（`confirm.py`）：写入/编辑/删除类工具执行前打印摘要，阻塞等待 y/n，
  拒绝则不执行
- **自动回滚**（`rollback.py`）：审核通过后先备份旧状态（写/编辑存旧内容，移动记录
  原路径，建目录记"新建"），存入 `backups/manifest.json` + `<id>.bak`；
  `rollback(id)` 可反向恢复。保留策略：同路径≤5条，总量≤200条，超出淘汰最旧
- 覆盖范围不含 `python_exec`/`run_python_script`（任意代码不可控）；Git 有独立安全网

## 7. RAG 本地知识库

`texts/*.txt` 段落切 chunk，三种检索模式（默认 hybrid）：
- **Dense**：语义相近/同义词匹配强，依赖 embedding 模型，对专有名词不敏感
- **BM25**：关键词/专有名词匹配准、快，不理解语义
- **Hybrid**：RRF 融合两者，覆盖最广，默认推荐；知识库小且偏关键词查询时可单用 BM25

链路：`embedder.py` → `retriever.py` → `knowledge_base.py` → `rag_tool.py`
（暴露为 `search_local_knowledge_base`）。控制台只打印一行 `[RAG] 正在检索...`。

## 8. LLM 后端

可导入本地模型或者通过API使用外部模型。

## 9. 会话记忆与日志

历史留最近 20 条（10 轮），超 60 万字符打印上下文警告。日志每对话一个 `.txt`，
只记 `[USER]`/`[AGENT step=N]`/`[TOOL:xxx]`（仅参数）/`[TURN_END]`，不落地结果内容。

## 10. 硬上限

ReAct ≤20 步（递归上限 44）；MCP 单次读取 ≤4万字符自动截断；输入≈60万字符预警；
回滚记录同路径≤5条/总量≤200条。

## 11. 运行

```bash
cd SINGLE_AGENT && python main.py
```
输入 `new` 开新会话，`exit` 退出。

## 12. 硬件配置

- Intel(R) Xeon(R) Gold 5115 CPU @ 2.40GHz (2.39 GHz)
- RAM 64.0 GB
- NVIDIA Quadro P4000

## 13. 备注

供交流学习使用，点击查看 LICENSE 文件

