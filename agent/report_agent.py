# ============================================================
# ReportAgent — 报告生成专用智能体
# 功能：独立的 agent，使用报告提示词，专注于用户使用报告生成
# ============================================================
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from model.factory import chat_model_expensive
from utils.prompt_loader import load_report_prompts
from agent.tools.report_tools import (
    get_user_id,
    get_current_month,
    fetch_external_data,
    rag_summarize,
)

REPORT_TOOLS = [
    get_user_id,
    get_current_month,
    fetch_external_data,
    rag_summarize,
]


class ReportAgent:
    # ============================================================
    # 初始化：加载报告专用提示词，创建独立agent（使用昂贵模型）
    # thread_id 带 report_ 前缀与主agent隔离
    # ============================================================
    def __init__(self):
        self.system_prompt = load_report_prompts()
        self.app = create_react_agent(
            model=chat_model_expensive,
            tools=REPORT_TOOLS,
            prompt=self.system_prompt,
            checkpointer=MemorySaver(),
        )

    def execute_stream(self, query: str, thread_id: str):
        config = {"configurable": {"thread_id": f"report_{thread_id}"}}
        input_data = {"messages": [("user", query)]}

        for event in self.app.stream(
            input_data,
            config,
            stream_mode="values"
        ):
            if "messages" in event:
                messages = event["messages"]
                last_msg = messages[-1]
                if hasattr(last_msg, "content") and last_msg.content and getattr(last_msg, "type", "") == "ai":
                    yield last_msg.content + "\n"
