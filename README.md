# AGENT

### 目录结构
```
qwen2.5_langgraph/
├── main.py            # 程序入口：组装 LLM + 工具 + Agent，跑交互式 CLI 循环
├── config.py           # 全局配置：LLM 后端、白名单目录、RAG 开关等（唯一改动入口）
├── llm.py              # LLM 工厂：按 config 选择 Ollama 本地模型 or 阿里云 DashScope
├── tools.py             # 本地工具定义：write_file / calculator / current_time / baidu_search
├── tool_selector.py     # 关键词匹配的动态工具选择器，减少每轮传给 LLM 的工具数量
├── mcp_setup.py         # 连接 MCP filesystem server，把其工具转换成 LangChain Tool
├── agent.py             # 核心：基于 LangGraph 的状态图 Agent（本项目的心脏）
├── rag_tool.py          # 把本地知识库检索包装成一个可插拔的 LangChain 工具
├── embedder.py          # 文本向量化（BGE 中文小模型）
├── retriever.py         # 向量检索（余弦相似度 Top-K）
├── knowledge_base.py    # 加载 texts/ 下的 .txt 文档，切分成 chunk
├── texts/               # RAG 知识库原始文档（.txt）
└── MULTI_AGENT_GUIDE.md # 如何把单 Agent 升级成多 Agent 协作的操作指南
```

### 依赖方向
```
main.py
 ├── llm.py          （构建 ChatModel）
 ├── tools.py         ├── rag_tool.py → embedder.py / retriever.py / knowledge_base.py
 ├── mcp_setup.py      → config.py（读白名单目录）
 └── agent.py
      ├── tool_selector.py
      └── config.py（读白名单 / 默认目录，拼 system prompt
```
