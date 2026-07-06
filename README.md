# SmartClean-AI

基于 LangGraph + Streamlit + Qwen 的扫地机器人智能客服系统。支持 RAG 知识库问答、多智能体路由、对话压缩、用户画像分析。

## 功能

- **多轮对话** — LangGraph ReAct Agent + MemorySaver 持久化会话
- **模型路由** — 简单问题（问候/天气）走 `qwen-turbo`，复杂问题走 `qwen3-max`
- **RAG 知识库** — ChromaDB 检索 + 余弦相似度重排（召回 5 取 Top 3）
- **报告生成** — 独立 Agent，输入关键词自动生成结构化报告
- **对话压缩** — 每 10 轮自动摘要历史，避免 Token 超限
- **用户画像** — 5+ 会话后自动分析用户特征，侧边栏展示
- **数据本地化** — 所有数据存本地文件，无需数据库

## 架构

```
SmartClean-AI/
├── app.py                  # Streamlit 前端入口
├── agent/
│   ├── react_agent.py      # 主对话 Agent（LangGraph）
│   ├── report_agent.py     # 报告生成 Agent
│   ├── model_router.py     # 模型路由规则
│   ├── trace_manager.py    # 对话追踪 & 持久化
│   ├── user_profile.py     # 用户画像
│   └── tools/              # Agent 工具集
│       ├── agent_tools.py  # 主 Agent 工具（RAG/天气/查询）
│       └── report_tools.py # 报告 Agent 工具
├── model/
│   └── factory.py           # LLM & Embedding 工厂
├── rag/
│   ├── rag_service.py       # RAG 检索 + 重排
│   └── vector_store.py      # ChromaDB 向量库
├── utils/                   # 工具模块
├── config/                  # YAML 配置文件
├── prompts/                 # 提示词模板
├── chroma_db/               # 向量数据库文件
├── traces/                  # 对话历史 JSON
├── profiles/                # 用户画像
└── logs/                    # 日志文件
```

## 快速开始

```bash
# 1. 克隆
git clone https://github.com/<your>/SmartClean-AI.git
cd SmartClean-AI

# 2. 安装依赖（推荐 conda）
conda create -n langchain-env python=3.12
conda activate langchain-env
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env 填入 DASHSCOPE_API_KEY

# 4. 运行
streamlit run app.py
```

## 配置

| 文件 | 说明 |
|---|---|
| `config/rag.yml` | 模型名称、chunk 参数 |
| `config/chroma.yml` | 检索数量 `k` |
| `config/agent.yml` | Agent 行为参数 |
| `config/prompts.yml` | 提示词路径 |

## 技术栈

- **框架**: LangGraph v0.4+ / LangChain v0.4+
- **前端**: Streamlit 1.58
- **模型**: Qwen3-Max / Qwen-Turbo (DashScope)
- **向量库**: ChromaDB
- **Embedding**: text-embedding-v4 (DashScope)
