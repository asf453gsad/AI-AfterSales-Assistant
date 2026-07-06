# ============================================================
# ReactAgent — 基于 LangGraph 的主智能体
# 功能：ReAct 思考-行动循环，支持双模型路由、上下文保持、压缩
# ============================================================
import re
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model_expensive, chat_model_cheap
from utils.prompt_loader import load_system_prompts
from agent.tools.agent_tools import (
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report,
)
from agent.model_router import SIMPLE_PATTERNS
from agent.trace_manager import TraceManager
from utils.logger_handler import logger

# 主Agent可调用的7个工具
ALL_TOOLS = [
    rag_summarize,
    get_weather,
    get_user_location,
    get_user_id,
    get_current_month,
    fetch_external_data,
    fill_context_for_report,
]


class ReactAgent:
    # ============================================================
    # 初始化：加载提示词，创建便宜/昂贵两个agent实例
    # ============================================================
    def __init__(self):
        self.system_prompt = load_system_prompts()
        self.trace_manager = TraceManager()
        self._init_agents()

    def _init_agents(self):
        self.agents = {}
        for model_type, model in [("cheap", chat_model_cheap), ("expensive", chat_model_expensive)]:
            self.agents[model_type] = create_react_agent(
                model=model,
                tools=ALL_TOOLS,
                prompt=self.system_prompt,
                checkpointer=MemorySaver(),
            )

    # ============================================================
    # 模型路由：匹配SIMPLE_PATTERNS → 便宜模型，否则 → qwen3-max
    # ============================================================
    def _is_simple_query(self, query: str) -> bool:
        for pattern in SIMPLE_PATTERNS:
            if re.search(pattern, query):
                return True
        return False

    def _get_app(self, query: str):
        if self._is_simple_query(query):
            return self.agents["cheap"]
        return self.agents["expensive"]

    # ============================================================
    # 流式执行：检查摘要 → 构建输入 → stream_mode="values" 逐块输出
    # 使用 checkpointer + thread_id 保持跨轮对话上下文
    # ============================================================
    def execute_stream(self, query: str, thread_id: str):
        app = self._get_app(query)
        config = {"configurable": {"thread_id": thread_id}}

        summary = self.trace_manager.get_summary(thread_id)
        if summary:
            input_data = {
                "messages": [
                    SystemMessage(content=f"之前的对话摘要：{summary}\n请基于此摘要和当前问题回答。"),
                    ("user", query),
                ]
            }
        else:
            input_data = {"messages": [("user", query)]}

        for event in app.stream(
            input_data,
            config,
            stream_mode="values"
        ):
            if "messages" in event:
                messages = event["messages"]
                last_msg = messages[-1]
                if hasattr(last_msg, "content") and last_msg.content and getattr(last_msg, "type", "") == "ai":
                    yield last_msg.content + "\n"

    # ============================================================
    # 会话压缩：每10轮用户消息触发一次，用便宜模型总结对话历史
    # ============================================================
    def check_and_compress(self, thread_id: str):
        user_count = self.trace_manager.count_user_messages(thread_id)
        if user_count > 0 and user_count % 10 == 0:
            logger.info(f"[compress] 对话 {thread_id} 超过10轮，开始压缩")
            msgs = self.trace_manager.get_messages(thread_id)
            text = "\n".join([f"{m['role']}: {m['content']}" for m in msgs])
            template = PromptTemplate.from_template("请总结以下对话的核心内容：\n\n{messages}")
            chain = template | chat_model_cheap
            result = chain.invoke({"messages": text})
            self.trace_manager.save_summary(thread_id, result.content)
            logger.info(f"[compress] 对话 {thread_id} 压缩完成")
