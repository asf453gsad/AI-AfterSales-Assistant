# ============================================================
# 模型工厂 — 创建 LLM 和 Embedding 模型实例
# 三个全局单例：昂贵模型(qwen3-max)、便宜模型(qwen-turbo)、嵌入模型
# ============================================================
import os
from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_handler import rag_conf

load_dotenv()

chat_model_expensive = ChatTongyi(model=rag_conf["chat_model_name"])
chat_model_cheap = ChatTongyi(model=rag_conf.get("cheap_model_name", "qwen-turbo"))
embed_model = DashScopeEmbeddings(model=rag_conf["embedding_model_name"])
