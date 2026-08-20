# 从零搭建一个 ReAct 单 Agent —— 代码实操教程

> 以 `SINGLE_AGENT` 这套真实项目为基础，带你一步步敲代码搭出同款 Agent。
> 每一步都给出可以直接抄的代码片段，跟着做完能跑通一个完整的中文智能助手。

---

## 第 0 步：环境准备

### 安装依赖

```bash
pip install langchain-core langgraph langchain-openai langchain-ollama \
            langchain-mcp-adapters requests beautifulsoup4 \
            transformers torch rank-bm25 numpy
```

- `langchain-core` / `langgraph`：Agent 循环的核心框架
- `langchain-openai`：用 OpenAI 兼容协议连接 Snowflake Cortex / DashScope
- `langchain-mcp-adapters`：接入 MCP（Model Context Protocol）标准工具
- `transformers` / `torch`：本地跑 Embedding 模型（RAG 用）
- `rank-bm25`：BM25 关键词检索

### 项目骨架

先建好这些空文件，后面一步步填内容：

```
my_agent/
├── config.py
├── llm.py
├── tools.py
├── agent.py
├── main.py
└── logger.py
```

---

## 第 1 步：配置层 `config.py`

把所有"会调整的参数"集中放一个文件，不要散落在各处：

```python
# config.py
LLM_BACKEND = "snowflake"   # "ollama" / "dashscope" / "snowflake" 三选一

SNOWFLAKE_MODEL = "claude-sonnet-4-5"
SNOWFLAKE_ACCOUNT_URL = "https://<account>.snowflakecomputing.com"
SNOWFLAKE_PAT = "..."   # 建议用环境变量传入，不要硬编码进代码

OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"
```

**设计习惯**：不管以后加多少后端、多少功能开关，都往这个文件里加变量，`llm.py`/`tools.py`
只 `import` 用，不自己硬编码参数——切后端、调参数只改这一个文件。

---

## 第 2 步：LLM 工厂 `llm.py`

用 `ChatOpenAI` 的 OpenAI 兼容协议连接 Snowflake Cortex：

```python
# llm.py
from langchain_openai import ChatOpenAI
from config import SNOWFLAKE_MODEL, SNOWFLAKE_ACCOUNT_URL, SNOWFLAKE_PAT

def build_llm():
    llm = ChatOpenAI(
        model=SNOWFLAKE_MODEL,
        base_url=f"{SNOWFLAKE_ACCOUNT_URL.rstrip('/')}/api/v2/cortex/v1",
        api_key=SNOWFLAKE_PAT,
        temperature=0.2,
        max_tokens=16000,     # 单次回复输出上限，别设太小，多工具调用容易被截断
        max_retries=3,        # 遇到瞬时 500/超时自动重试
        timeout=120,
    )
    # 建好就探活一下，坏了立刻报错，不要等用户问问题才发现连不上
    llm.invoke("ping")
    print(f"[LLM] 已连接：{SNOWFLAKE_MODEL}")
    return llm
```

**为什么要 `invoke("ping")` 探活**：LLM 客户端对象创建成功不代表真的能连上（账号/网络/权限都可能有
问题）。启动阶段就试一次真实调用，坏了直接崩在启动时,比运行到一半才发现要好排查得多。

---

## 第 3 步：第一个工具 `tools.py`

从最简单的 `calculator` 开始，理解 `@tool` 的写法：

```python
# tools.py
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """计算一个数学表达式，比如 "3 + 5 * 2"。只支持 +-*/() 和数字。

    Args:
        expression: 要计算的数学表达式字符串。
    """
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"计算失败：{e}"
```

**关键点**：docstring 不是给人看的注释，是**给模型看的说明书**——模型完全靠这段文字判断
"这个工具是干什么的、什么时候该调用它、参数怎么填"。写得越清楚具体，模型选得越准。

再加一个会产生副作用（写文件）的工具，体会一下"执行代码/写文件类工具"要多一层小心：

```python
from pathlib import Path

OUTPUT_DIR = Path.home() / "agent_output"
OUTPUT_DIR.mkdir(exist_ok=True)

@tool
def write_file(filename: str, content: str) -> str:
    """把文字内容写入本地文件。

    Args:
        filename: 文件名，比如 'hello.txt'。
        content: 要写入的文字内容。
    """
    filepath = OUTPUT_DIR / filename
    filepath.write_text(content, encoding="utf-8")
    return f"已写入：{filepath}"
```

（后面第 8 步会给这类工具加人工审核确认——这里先写最简版，跑通再加防护。）

组装成工具列表：

```python
def create_tools() -> list:
    return [calculator, write_file]
```

---

## 第 4 步：把工具挂到 Agent 上

### `agent.py`——最小可跑通版本

```python
# agent.py
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

class SingleAgent:
    def __init__(self, llm, tools):
        self.agent = create_react_agent(
            llm, tools=tools,
            prompt="你是一个中文智能助手，善用挂载的工具完成任务。",
        )

    async def arun(self, user_input: str) -> str:
        result = await self.agent.ainvoke(
            {"messages": [HumanMessage(content=user_input)]}
        )
        # 取最后一条 AI 消息的文本作为答案
        for m in reversed(result["messages"]):
            if m.type == "ai" and m.content:
                return m.content
        return "（未生成有效答复）"
```

### `main.py`——命令行入口

```python
# main.py
import asyncio
from llm import build_llm
from tools import create_tools
from agent import SingleAgent

async def main():
    llm = build_llm()
    agent = SingleAgent(llm=llm, tools=create_tools())
    print("Agent 已就绪，输入 exit 退出")
    while True:
        user_input = input("\n用户: ").strip()
        if user_input.lower() == "exit":
            break
        answer = await agent.arun(user_input)
        print(f"AI: {answer}")

if __name__ == "__main__":
    asyncio.run(main())
```

**跑一下**：`python main.py`，输入"帮我算一下 3+5*2"，能得到 `13` 就说明最小链路通了——
LLM 工厂、工具定义、ReAct 循环、CLI 输入输出，四块都接上了。先跑通再加功能，不要一次性把
所有模块都写完才第一次运行。

---

## 第 5 步：加 RAG 检索能力

### `embedder.py`——文本向量化

```python
# embedder.py
import torch
from transformers import AutoTokenizer, AutoModel

class Embedder:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()

    def encode(self, texts: list[str]):
        encoded = self.tokenizer(texts, padding=True, truncation=True,
                                  max_length=512, return_tensors="pt")
        with torch.no_grad():
            output = self.model(**encoded)
        emb = output.last_hidden_state[:, 0, :]
        return torch.nn.functional.normalize(emb, p=2, dim=1).numpy()
```

### `retriever.py`——检索器（先只写 Dense 版）

```python
# retriever.py
import numpy as np

class DenseRetriever:
    def __init__(self, documents: list[str], embedder: Embedder):
        self.documents = documents
        self.embedder = embedder
        self.doc_embeddings = embedder.encode(documents)

    def retrieve(self, query: str, top_k: int = 2) -> list[str]:
        q_emb = self.embedder.encode([query])
        scores = np.dot(self.doc_embeddings, q_emb.T).reshape(-1)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_idx]
```

### `rag_tool.py`——组装成一个工具（懒加载单例）

```python
# rag_tool.py
from langchain_core.tools import tool

_retriever = None  # 懒加载单例：不用就不加载模型，省启动时间

def _get_retriever():
    global _retriever
    if _retriever is None:
        from embedder import Embedder
        from retriever import DenseRetriever
        documents = ["公司规定每周一开例会。", "请假需提前一天申请。"]  # 示例数据
        _retriever = DenseRetriever(documents, Embedder("BAAI/bge-small-zh-v1.5"))
    return _retriever

@tool
def search_local_knowledge_base(query: str) -> str:
    """在本地知识库中检索相关内容。涉及公司规章、内部资料等信息时用这个。

    Args:
        query: 要检索的问题或关键词。
    """
    docs = _get_retriever().retrieve(query, top_k=1)
    return docs[0] if docs else "本地知识库未找到相关内容。"
```

**为什么要懒加载**：Embedding 模型加载很慢（几秒到几十秒），如果用户这次对话根本用不到 RAG，
没必要在启动时就白白等这个加载时间——第一次真正调用工具时才加载，之后复用同一个实例。

把这个工具加进 `create_tools()` 返回的列表即可接入。

---

## 第 6 步：接入外部工具集（MCP）

MCP（Model Context Protocol）是一套标准协议，很多现成的工具服务器（比如文件系统操作）
不用自己写，直接接进来就能用：

```python
# mcp_setup.py
from langchain_mcp_adapters.client import MultiServerMCPClient

async def get_mcp_tools() -> list:
    client = MultiServerMCPClient({
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user/agent_output"],
            "transport": "stdio",
        },
    })
    return await client.get_tools()   # 自动拿到 read_file / edit_file / move_file 等一整套工具
```

`main.py` 里改成：

```python
mcp_tools = await get_mcp_tools()
all_tools = create_tools() + mcp_tools
agent = SingleAgent(llm=llm, tools=all_tools)
```

**包装模式（进阶）**：拿到的现成工具不一定完全符合你的需求（比如读超大文件会撑爆 context），
可以给它们的 `.func`/`.coroutine` 属性做一层包装，在真正执行前后插入自己的逻辑：

```python
def _wrap_truncate(t, max_chars=40000):
    original = t.coroutine
    async def wrapped(*args, **kwargs):
        result = await original(*args, **kwargs)
        text = str(result)
        if len(text) > max_chars:
            return text[:max_chars] + "\n\n...(内容过长已截断)"
        return result
    t.coroutine = wrapped
    return t

mcp_tools = [_wrap_truncate(t) if t.name == "read_file" else t for t in mcp_tools]
```

这个"包一层再塞回去"的模式后面第 8 步的人工审核、备份逻辑也会用到，是很通用的技巧。

---

## 第 7 步：技能系统（Skills）

当"工具"不够表达一整套多步骤流程时（比如"翻译成 4 种场合"这种有明确规范的任务），
用**技能包**——一段按需加载的详细指令，而不是把所有细节写进 system prompt。

### 技能文件格式

新建 `skills/翻译.md`：

```markdown
---
name: 翻译
description: 中外文互译，要求地道无语病；用户说"翻译/译成"时使用
---

不要逐字直译，按目标语言自然表达习惯重组句子；习语要意译成对等表达。
一次性给出【口语】【正式书面】【邮件】三个版本，不要追问场合。
```

### `skill_loader.py`——扫描 + 按需加载

```python
# skill_loader.py
import re
from pathlib import Path
from langchain_core.tools import tool

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n(.*)$", re.DOTALL)

def _load_skills(skills_dir: Path) -> dict:
    skills = {}
    for f in skills_dir.glob("*.md"):
        m = _FRONTMATTER_RE.match(f.read_text(encoding="utf-8"))
        if not m:
            continue
        meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
        name = meta["name"].strip()
        skills[name] = {"description": meta["description"].strip(), "body": m.group(2).strip()}
    return skills

_SKILLS = _load_skills(Path(__file__).parent / "skills")

def format_skill_index_for_prompt() -> str:
    return "\n".join(f"  - {name}：{info['description']}" for name, info in _SKILLS.items())

@tool
def load_skill(skill_name: str) -> str:
    """加载某个技能的详细步骤指引，任务匹配【可用技能】目录时先调用这个。

    Args:
        skill_name: 技能名称，必须跟目录里的名字完全一致。
    """
    skill = _SKILLS.get(skill_name)
    return skill["body"] if skill else f"未找到技能「{skill_name}」。"
```

`agent.py` 的 system prompt 里加一段：

```python
from skill_loader import format_skill_index_for_prompt

prompt = f"""你是中文智能助手。

【可用技能】
遇到以下场景先调用 load_skill(名字) 拿步骤再执行：
{format_skill_index_for_prompt()}
"""
```

**以后加新能力**：只要在 `skills/` 下新增一个 `.md` 文件，不用改任何代码，`_load_skills`
启动时会自动扫描到。

---

## 第 8 步：安全闸门 + 回滚

### `confirm.py`——写入前人工确认

```python
# confirm.py
def confirm_action(summary: str) -> bool:
    print(f"\n[审核] 即将执行：\n{summary}")
    while True:
        choice = input("是否允许？(y/n): ").strip().lower()
        if choice in ("y", "n"):
            return choice == "y"
```

在 `write_file` 里接入：

```python
@tool
def write_file(filename: str, content: str) -> str:
    filepath = OUTPUT_DIR / filename
    if not confirm_action(f"写入 {filepath}\n内容预览：{content[:200]}"):
        return "操作已被用户拒绝。"
    filepath.write_text(content, encoding="utf-8")
    return f"已写入：{filepath}"
```

### `rollback.py`——写入前顺手备份，出问题能恢复

```python
# rollback.py
import json, shutil, uuid
from datetime import datetime
from pathlib import Path

_BACKUP_DIR = Path(__file__).parent / "backups"
_MANIFEST = _BACKUP_DIR / "manifest.json"

def record_write(path: str) -> str:
    p = Path(path)
    entries = json.loads(_MANIFEST.read_text()) if _MANIFEST.exists() else []
    backup_file = None
    if p.exists():
        backup_file = f"{uuid.uuid4().hex[:8]}.bak"
        _BACKUP_DIR.mkdir(exist_ok=True)
        shutil.copy2(p, _BACKUP_DIR / backup_file)
    entry_id = uuid.uuid4().hex[:8]
    entries.append({"id": entry_id, "path": str(p), "backup_file": backup_file,
                     "existed_before": backup_file is not None,
                     "timestamp": datetime.now().isoformat()})
    _MANIFEST.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
    return entry_id
```

**原理只有一句话**：改之前先把"改之前的状态"存一份，回滚就是把这份状态原样放回去。
真实项目里还要加保留策略（避免备份文件无限增长）、加 `rollback(id)` 工具本身，逻辑是
上面这个最小版本的自然扩展。

---

## 第 9 步：日志 + 完整 `main.py`

```python
# logger.py
from pathlib import Path
from datetime import datetime

class SessionLogger:
    def __init__(self):
        log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        self.path = log_dir / f"session_{datetime.now():%Y%m%d_%H%M%S}.txt"
        self._fh = open(self.path, "a", encoding="utf-8")

    def event(self, tag: str, msg: str):
        self._fh.write(f"[{tag}] {msg}\n")
```

**只记运行轨迹，别记结果内容**：日志里记"调用了哪个工具、什么时候、传了什么参数"就够排查
问题了，不需要把工具返回的具体内容/模型最终答案也存下来——既省空间，也避免敏感信息落地。

最终 `main.py` 把所有工具来源拼起来：

```python
async def main():
    llm = build_llm()
    all_tools = (
        create_tools()
        + [load_skill]
        + await get_mcp_tools()
    )
    agent = SingleAgent(llm=llm, tools=all_tools)
    while True:
        user_input = input("\n用户: ").strip()
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "new":
            agent.new_conversation()   # 需要在 SingleAgent 里实现清空历史
            continue
        try:
            print(f"AI: {await agent.arun(user_input)}")
        except Exception:
            print("AI: 抱歉，这次请求处理出现异常，请重新提问一次。")
```

---

## 第 10 步：调试技巧

1. **模型选错工具/技能** → 不是改代码逻辑，先去改 docstring / skill 的 `description`，
   写得越贴近用户真实的问法，命中率越高。
2. **看不出问题出在哪一步** → 查日志文件，按时间线看 `[TOOL:xxx]` 记录，确认到底调用
   没调用、调用了几次。
3. **怀疑是 context 塞爆** → 打印一下本轮消息的总字符数，跟模型的 context window 上限
   做个粗略比例估算（4 字符 ≈ 1 token），超过大半就要考虑截断/分段。
4. **怀疑是并发/异步 bug** → 先退回到同步单步调试，确认工具本身逻辑没问题，再排查是不是
   多个工具并行调用时互相干扰。

---

## 第 11 步：落地到实际生产的痛点与难点总结

搭一个能跑的 demo 不难，真正难的是下面这几类问题：

### 1. Context / Token 限制是硬约束，不是可选项
模型的 context window（比如 200K token）看着很大，但只要 agent 会调用"读文件"之类的
工具，一次读一个大文件就可能占掉大半——必须在架构层面就规划好截断/分段策略，不能等
线上报错了再补。而且报错形式往往不是"温和地告诉你超限了"，可能直接是网关层的 500/400，
排查成本很高。

### 2. 云端模型底层网关的隐藏校验规则
不同云服务商在 API 兼容层背后走的是不同的底层协议（比如 Bedrock 的 `toolUse`/`toolResult`
严格配对校验），这些规则往往不会写在你直接调用的那层文档里，只有在"模型并行发起多个工具
调用又被截断"这种边界情况下才会暴露出来。这类问题很难在开发阶段发现，只能靠线上真实使用
反馈，加固手段（限制并行工具调用、加大输出上限）也只能降低概率，不能 100% 杜绝。

### 3. 外部依赖（网络/第三方 API）的容错设计
公司网络环境限制、第三方接口不稳定是常态，工具层必须对"调用失败"这件事做充分预期，
区分"网络问题""接口变更""参数错误"等不同失败原因，给出对应的提示，而不是让一个异常
直接把整个对话打断。

### 4. 任意代码执行/文件写入的安全边界
一旦给了 agent "写文件""跑代码"的能力，就等于给了它修改/破坏本地环境的能力。
黑名单式的危险代码扫描永远是"防不住聪明人"的，真正兜底的是人工审核确认 + 操作可回滚——
两道防线都要有，不能只靠模型"自律"。

### 5. 错误处理的"开发者视角"陷阱
写代码的人天然倾向于把原始异常信息暴露出来方便自己调试，但普通用户看到一段 JSON 格式的
API 报错只会觉得"坏了、不能用"。技术细节应该下沉到日志，用户看到的永远应该是"发生了什么 +
下一步该怎么做"的人话。

### 6. Skill 与 Tool 的边界依赖模型自主判断，没有强保证
不管描述写得多精确，"该用哪个工具/技能"本质上是模型的一次概率性推理，不是确定性路由。
技能/工具数量越多，越需要靠测试驱动去迭代描述文字，而不是假设第一版写的说明就一定管用。

### 7. 日志设计的两难
记录太详细（把每次工具调用的完整输出、模型的最终回答都存下来）既占空间又有敏感信息泄露
风险；记录太简略又会在出问题时啥都查不到。合理的折中通常是"只记运行轨迹（谁在什么时候
调用了什么），不记具体内容"，需要有意识地做这个取舍，而不是随手全存或全不存。

### 8. 多后端兼容性差异
即使都包装成"OpenAI 兼容协议"，不同后端对 `max_tokens`、`parallel_tool_calls` 等参数
的支持程度并不一致，有些参数换个后端就直接报错或被静默忽略。切换/新增后端时必须做好
探测和优雅降级（不支持就退回默认配置），不能假设所有后端行为一致。
