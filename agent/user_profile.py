# ============================================================
# 用户画像 — 分析对话历史生成用户特征画像
# 当 traces/ 中对话数量 ≥5 时自动触发一次
# ============================================================
import json
import os
from utils.path_tool import get_abs_path
from agent.trace_manager import TraceManager
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model_cheap


class UserProfile:

    def __init__(self):
        self.profile_dir = get_abs_path("profiles")
        os.makedirs(self.profile_dir, exist_ok=True)
        self.trace_manager = TraceManager()

    # ============================================================
    # 检查并生成：仅当有5+对话且尚未生成画像时触发
    # ============================================================
    def check_and_generate(self):
        threads = self.trace_manager.list_threads()
        if len(threads) >= 5:
            profile_path = os.path.join(self.profile_dir, "profile.json")
            if not os.path.exists(profile_path):
                self._generate_profile()

    # ============================================================
    # 内部：用便宜模型分析所有对话文本生成画像
    # ============================================================
    def _generate_profile(self):
        conv_text = self.trace_manager.get_conversation_text()
        if not conv_text.strip():
            return

        template = PromptTemplate.from_template(
            "分析以下用户与智能客服的对话记录，生成用户画像，包括：用户关心的问题类型、使用偏好、常见需求、用户特征等。\n\n对话记录：\n{conversations}\n\n用户画像："
        )
        chain = template | chat_model_cheap
        result = chain.invoke({"conversations": conv_text})

        profile_path = os.path.join(self.profile_dir, "profile.json")
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump({"profile": result.content}, f, ensure_ascii=False, indent=2)

    # ============================================================
    # 读取已保存的画像（供前端侧边栏展示）
    # ============================================================
    def get_profile(self):
        profile_path = os.path.join(self.profile_dir, "profile.json")
        if os.path.exists(profile_path):
            with open(profile_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
