# ============================================================
# 模型路由器 — 根据问题类型动态选择便宜/昂贵模型
# 简单问题（天气、问候等）→ qwen-turbo，否则 → qwen3-max
# ============================================================
import re
from model.factory import chat_model_cheap, chat_model_expensive

# 匹配简单问题的正则模式列表
SIMPLE_PATTERNS = [
    r"天气", r"温度", r"湿度", r"下雨", r"晴天",
    r"地点", r"位置", r"城市",
    r"你好", r"您好", r"hi", r"hello",
    r"再见", r"拜拜", r"谢谢", r"感谢",
    r"你是谁", r"你能做什么",
]


def route_model(query: str, purpose: str = "qa"):
    if purpose == "compression":
        return chat_model_cheap
    if purpose == "qa":
        for pattern in SIMPLE_PATTERNS:
            if re.search(pattern, query):
                return chat_model_cheap
    return chat_model_expensive
