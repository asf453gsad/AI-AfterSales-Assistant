# ============================================================
# RAG 总结服务 — 检索 + 重排序 + LLM 生成
# 流程：query → 向量检索5个 → embedding重排序保留Top3 → LLM总结
# ============================================================
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model_expensive, embed_model
import numpy as np


# ============================================================
# 重排序：用 embedding 余弦相似度对检索结果重新打分，保留Top K
# ============================================================
def rerank_docs(query: str, docs: list[Document], k: int = 3) -> list[Document]:
    if not docs:
        return []
    query_embedding = embed_model.embed_query(query)
    doc_texts = [doc.page_content for doc in docs]
    doc_embeddings = embed_model.embed_documents(doc_texts)

    similarities = []
    for doc_emb in doc_embeddings:
        query_norm = np.linalg.norm(query_embedding)
        doc_norm = np.linalg.norm(doc_emb)
        if query_norm == 0 or doc_norm == 0:
            sim = 0
        else:
            sim = float(np.dot(query_embedding, doc_emb) / (query_norm * doc_norm))
        similarities.append(sim)

    sorted_pairs = sorted(zip(docs, similarities), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_pairs[:k]]


class RagSummarizeService(object):

    def __init__(self):
        self.vector_store = VectorStoreService()
        self.retriever = self.vector_store.get_retriever()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model_expensive
        self.chain = self._init_chain()

    # ============================================================
    # LCEL 链：prompt_template → model → 字符串解析
    # ============================================================
    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    # ============================================================
    # 检索文档（k=5，由 chroma.yml 配置）
    # ============================================================
    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    # ============================================================
    # 完整 RAG 流程：检索 → 重排序 → 组装context → LLM总结
    # ============================================================
    def rag_summarize(self, query: str) -> str:
        context_docs = self.retriever_docs(query)
        reranked_docs = rerank_docs(query, context_docs, k=3)

        context = ""
        counter = 0
        for doc in reranked_docs:
            counter += 1
            context += (
                f"【参考资料{counter}】: "
                f"参考资料：{doc.page_content} | "
                f"参考元数据：{doc.metadata}\n"
            )

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )
