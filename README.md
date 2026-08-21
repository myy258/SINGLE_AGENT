# SINGLE_AGENT 单 Agent 系统设计文档

---

## 1. 定位

- **目标**：交互式中文助手。用户提问 → agent 自主选择工具 → 返回答案。
- **能力**：写代码、跑分析、写作、翻译、常识问答、联网搜索、本地知识库检索、文本文件读写。

## 2. 架构

```
用户输入 ─► SingleAgent.arun ─► create_react_agent (ReAct 循环)
                                        │
                                        ├── write_file / python_exec / run_python_script
                                        ├── calculator / current_time / baidu_search
                                        ├── search_local_knowledge_base
                                        ├── load_skill（技能加载，见 §5）
                                        ├── list_backups / rollback（操作回滚，见 §7）
                                        ├── git_command（Git 版本管理，见 §4）
                                        └── MCP filesystem 工具集
                                              (read_file / edit_file / list_directory / ...)
```

## 3. 文件清单

```
SINGLE_AGENT/
├── main.py               # 命令行交互入口
├── config.py             # LLM 后端选择 + RAG 开关 + 路径白名单
├── agent.py              # SingleAgent 类（create_react_agent）
│
├── core/                 # agent 运行的底层引擎
│   ├── llm.py             # 三后端 LLM 工厂（max_tokens / 重试 / 超时）
│   └── logger.py          # 会话日志（每对话一个 .txt，只记运行轨迹）
│
├── tools/                # 挂载给 agent 的工具实现
│   ├── local_tools.py     # 本地工具（write_file / python_exec / calculator / ...）
│   ├── mcp_setup.py       # MCP filesystem server 接入 + 截断/审核/备份包装
│   ├── git_tool.py        # Git 版本管理（受限子命令白名单 + 冲突/历史安全检查）
│   ├── rollback.py        # 操作回滚（写入/编辑/移动/建目录前自动备份，可按 id 回滚）
│   └── confirm.py         # 终端阻塞式人工审核（写入/编辑/删除前确认）
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
├── backups/              # tools/rollback.py 的备份文件 + manifest.json（运行后自动生成）
├── logs/                 # core/logger.py 的会话日志（运行后自动生成）
├── DESIGN.md             # 本文档
└── TUTORIAL.md           # 从零搭建教程（简化扁平结构，教学用）
```

> `core/`、`tools/`、`rag/`、`skills/` 都是普通 Python package（各自有 `__init__.py`）。
> 项目始终以 `python main.py`（在 `SINGLE_AGENT/` 目录下）启动，`sys.path[0]` 会是
> `SINGLE_AGENT/` 本身，所以子包内部可以用 `from tools.confirm import confirm_action`
> 这种绝对导入互相引用，`config.py`/`agent.py` 留在根目录的模块也能被子包直接
> `from config import ...` 引用，不需要相对导入。

## 4. 工具集

**本地工具**（`tools/local_tools.py`，写入/执行类都有人工审核确认，见 §6）：
- `write_file(filename, content)` — 相对路径落到默认目录；绝对路径原样使用
- `python_exec(code, timeout=60)` — 首选！直接跑代码，print 出结果
- `run_python_script(filename, timeout=30)` — 用户明说要脚本时才用
- `calculator(expression)` — 数学表达式
- `current_time()` — 当前时间
- `baidu_search(query, top_k=5)` — 联网搜索
- `search_local_knowledge_base(query)` — 本地知识库检索（RAG）

**技能工具**（`skills/skill_loader.py`）：
- `load_skill(skill_name)` — 按名字取某个 skill 的完整步骤指引，见 §5

**回滚工具**（`tools/rollback.py`）：
- `list_backups(limit=10)` — 列出最近的可回滚操作记录
- `rollback(backup_id)` — 按 id 回滚一次操作，见 §7

**Git 工具**（`tools/git_tool.py`）：
- `git_command(git_args, cwd=".")` — 执行受限的 git 子命令（status/diff/log/branch/remote/
  add/commit/push/pull/tag/checkout/init/clone/reset），变更类子命令执行前人工审核确认。
  不内置任何仓库地址，远程 URL 由用户在对话里提供。配套技能见
  `skills/Git版本管理与发布.md`。
  - **合并冲突安全网**：`add`/`commit` 前会检查 `git status --porcelain`，
    只要发现未解决的合并冲突（`UU`/`AA`/`DD` 等状态码）就直接拒绝执行，防止
    带着 `<<<<<<< HEAD` 标记的文件被误提交（已用真实冲突场景验证过这个检查逻辑）。
  - **pull/merge 安全网**：执行前检查工作区是否干净（不干净直接拒绝），并自动打一个
    `presync_<时间戳>` 的 git tag 作为安全快照，出问题可以 `reset --hard` 一键恢复；
    确认弹窗会额外提示"可能引入冲突标记"；结果里出现 `CONFLICT` 会附加处理指引。
  - **拒绝危险合并**：不允许在 `git_args` 里传 `--allow-unrelated-histories`
    （强行合并两段不相关的历史，几乎必然产生大量冲突）。

**MCP filesystem 工具**（`tools/mcp_setup.py` 通过 npx 拉起）：
- read_file / read_multiple_files / read_text_file（超 4 万字符自动截断并提示分段处理）
- list_directory / list_directory_with_sizes / directory_tree
- search_files / get_file_info / list_allowed_directories
- create_directory / edit_file / move_file（执行前人工审核确认+自动备份，见 §6、§7）

## 5. 技能包（Skills）

- 每个 skill 是 `skills/` 目录下一个 `.md` 文件（跟 `skill_loader.py` 放在同一个包里），
  格式：
  ```
  ---
  name: 技能名
  description: 一句话简介（什么时候用）
  ---
  （正文：详细步骤指引）
  ```
- 启动时只把 `name` + `description` 拼成"技能目录"塞进 system prompt（§10），正文很长
  不常驻上下文；模型判断任务匹配某个技能时，调用 `load_skill(名字)` 才把完整正文注入
  当前对话，用完即走，避免技能数量增多时把 prompt 撑爆。
- 新增技能只需要在 `skills/` 下加一个新的 `.md` 文件，不用改 `agent.py` / `main.py`。
- 现有示范：
  - `skills/代码加注释.md`（原来硬编码在 system prompt 里的"读文件→加注释→写回"流程，
    已迁移成独立技能）。
  - `skills/翻译.md`（要求地道、无语病，不做逐字直译；一次性输出【口语/日常】
    【正式书面】【商务邮件】【社交媒体】四个场合版本供用户自选，不追问场合）。
  - `skills/Git版本管理与发布.md`（提交/推送/打 tag 发布流程，不硬编码仓库地址，
    远程 URL 由用户提供；配套 `git_command` 工具）。

### 5.1 如何创建一个新技能（教程）

1. **先想清楚触发条件**：这个技能解决什么问题？用户会怎么问？跟已有技能的
   `description` 有没有重叠——重叠太多模型容易选错。
2. **在 `skills/` 目录下新建一个 `.md` 文件**，文件名随意（建议用技能名，方便管理），
   比如 `skills/数据分析.md`。
3. **写头部（frontmatter）**，格式固定，两个字段必填：
   ```
   ---
   name: 数据分析
   description: 对一批数据做统计分析、画图、找异常值；用户要求"分析一下这份
     数据/看看这些数字有什么规律"时使用
   ---
   ```
   - `name` 是 `load_skill(name)` 调用时要传的参数，跟文件名可以不一致，但**同一个
     技能只应该有一个 `name`**，不要跟其它技能重名。
   - `description` 是模型判断"要不要用这个技能"的唯一依据，尽量：
     - 说清楚"做什么" + "什么时候用"；
     - 带上用户大概率会用到的关键词（比如"分析/统计/画图"）；
     - 跟其它技能的描述保持区分度，不要含糊到跟别的技能都能对上。
   - **注意**：`description` 必须写在一行内，不要跨行折行——`skill_loader.py` 用简单
     逐行解析 frontmatter，折行会导致后半句内容被丢弃。
4. **写正文**——正文只有匹配到才会被加载，可以写得详细，建议包含：
   - **步骤**：按顺序列出该怎么做，每步该用哪个已有工具（`python_exec`、
     `read_file`、`write_file` 等），参考 `skills/代码加注释.md` 的写法。
   - **输出格式要求**（如果有）：比如翻译技能要求"一次性给 4 个场合标签"，
     参考 `skills/翻译.md`。
   - **禁止/注意事项**：比如"不要凭空编造数据""遇到 XX 情况要先问用户"。
5. **保存即生效，不需要改代码**：`skills/skill_loader.py` 启动时会自动扫描
   `skills/*.md`，新技能的 `name`+`description` 会自动出现在 system prompt 的
   【可用技能】目录里。
6. **测试**：用一句大概会命中这个技能的话去问 agent，观察它是否先调用了
   `load_skill('你的技能名')` 再执行。如果没触发或触发错了，回头把 `description`
   写得更贴近"用户真实的问法"，而不是急着改代码逻辑。

## 6. 人工审核确认（写入/编辑/删除类操作）

- `tools/confirm.py` 提供 `confirm_action(summary)`：打印操作摘要，阻塞 `input()` 等待
  `y`/`n`，拒绝则该次操作直接返回"已被用户拒绝"，不执行。
- 覆盖范围：
  - 本地：`write_file`、`python_exec`、`run_python_script`
  - MCP：`edit_file`、`move_file`、`create_directory`
  - Git：`add`/`commit`/`push`/`pull`/`tag`/`checkout`/`init`/`clone`/`reset`
- 只读类工具（`read_file`、`list_directory`、`git status` 等）不需要审核。

## 6. 操作回滚（Rollback）

- **原理**：写入/编辑/移动/建目录类操作在通过 `confirm_action` 确认之后、真正执行
  之前，先把"改之前的状态"记一条日志（`tools/rollback.py`）：
  - `write_file` / `edit_file`（覆盖或编辑内容）→ 备份**旧文件内容**（原来不存在
    就记"这是新建的"，回滚 = 删除）。
  - `move_file`（移动/重命名）→ 只记"从哪移到哪"，回滚 = 反向移动一次。
  - `create_directory`（新建目录）→ 只记"这个目录是新建的"，回滚 = 目录仍为空
    才删除，非空则不动并提示。
- 所有记录写进 `backups/manifest.json`，每条记录一个唯一 id；内容备份存成
  `backups/<id>.bak`（`backups/` 在 `SINGLE_AGENT/` 根目录下，跟随代码目录）。
- **`list_backups(limit=10)`**：列出最近的可回滚记录（id、工具名、路径、时间）。
- **`rollback(backup_id)`**：按 id 反向执行一次，本身也要走 `confirm_action` 确认，
  并且回滚动作也会生成一条新记录——相当于一个"撤销栈"，理论上可以对回滚再回滚。
- **保留策略（避免 `backups/` 无限增长）**：
  - 同一个路径最多保留最近 5 条记录；
  - `manifest.json` 总条目数最多保留 200 条；
  - 超出部分从最旧的开始淘汰，连带删除对应的 `.bak` 文件。
- **覆盖边界**：`python_exec` / `run_python_script` 跑的是任意代码，没法预先知道
  会改动哪些文件，**不在回滚覆盖范围内**。Git 相关操作有自己独立的安全网（见 §4
  `git_command` 的合并冲突/安全快照机制），不复用这套 `rollback.py`。

## 7. RAG（本地知识库检索）

- 数据源：`texts/*.txt` 段落级切 chunk
- 三种检索模式（`config.RETRIEVAL_MODE`）：`dense` / `bm25` / `hybrid`（推荐）
- `hybrid` = Dense + BM25 双路 RRF 融合
- 相关代码都在 `rag/` 包下：`rag/embedder.py`（向量化）→ `rag/retriever.py`
  （检索器）→ `rag/knowledge_base.py`（加载切 chunk）→ `rag/rag_tool.py`
  （组装成 `search_local_knowledge_base` 工具）。

### 9.1 三种检索模式的优缺点

| 模式 | 优点 | 缺点 |
|---|---|---|
| **Dense（向量语义检索）** | 能理解语义相近但字面不同的表达（比如"请假流程"能匹配到"休假申请"）；对同义词、口语化提问、跨表达方式的匹配能力强 | 依赖 embedding 模型质量，本地模型（BGE-small-zh）语义理解能力有限；对精确关键词/专有名词（人名、编号、型号）反而不够敏感，容易"理解过度"匹配到不相关但语义相近的内容；计算量比 BM25 大（要跑模型推理） |
| **BM25（关键词检索）** | 对精确关键词、专有名词、编号、人名等**字面匹配**非常准；不需要 embedding 模型，速度快、资源占用低、结果可解释（能看出命中了哪些词） | 完全不理解语义，同义词/换个说法就搜不到（比如问"离职流程"搜不到写着"员工退出"的内容）；对中文分词质量敏感，分词切得不好直接影响效果 |
| **Hybrid（Dense + BM25 的 RRF 融合）** | 两者互补：关键词匹配和语义匹配的结果都能捞到，覆盖面最广，是目前项目里默认推荐的模式；对"既有专有名词又要理解语义"的复合查询效果最好 | 两套检索都要跑，延迟和资源消耗是三者中最高的；融合排序（RRF）不是每次都直觉正确，有时会把两边都排名靠后但恰好都出现的无关结果拉到前面；调试起来比单一模式更复杂（出问题要分别看两路结果才能定位） |

`config.RETRIEVAL_MODE` 默认是 `"hybrid"`，属于"稳妥优先"的选择——牺牲一点性能换取
召回率。如果知识库很小（就是 `texts/*.txt` 那三份）、查询大多是关键词式的（比如查
具体制度条款名），BM25 单用可能已经够、还更快。

- 控制台只保留一行简洁提醒：`[RAG] 正在检索本地知识库...`；加载 Embedding 模型、
  构建索引、检索模式公告、`top_score/margin` 调试信息均不打印（`transformers` 库自带
  的进度条/LOAD REPORT 也已通过 `hf_logging.set_verbosity_error()` 关闭）。

## 8. 硬上限

- ReAct 步数 ≤ 20（防死循环）
- 图递归上限 = 44
- MCP 读文件单次返回 ≤ 4 万字符（超出自动截断并提示分段处理）
- 单轮输入 ≈ 60 万字符时打印上下文警告（见 §11）
- 回滚记录：同路径 ≤ 5 条、总条目 ≤ 200 条（见 §7）

## 9. 运行

```bash
cd SINGLE_AGENT
python main.py
```

- 输入 `new` 开启新会话（清空历史）
- 输入 `exit` 退出

> 项目目录已按职责拆成 `core/`（引擎）、`tools/`（工具）、`rag/`（知识库检索）、
> `skills/`（技能）几个子包，详见 §3 文件清单和 §2.1 详细结构图。改代码时记得
> 同步整个 `SINGLE_AGENT/` 文件夹到本机，不要只挑单个文件——子包之间的相对导入
> 关系一起变了，漏同步会导致 `ImportError`。
