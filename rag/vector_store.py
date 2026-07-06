# ============================================================
# ChromaDB 向量库服务 — 文档存储、检索、增量更新
# 支持 TXT/PDF 加载、MD5 去重、文本分块
# ============================================================
from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
import os


class VectorStoreService:

    def __init__(self):
        # 初始化 ChromaDB 连接
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=chroma_conf["persist_directory"],
        )

        # 文本分块器（递归分割，优先按段落/句子切分）
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    # ============================================================
    # 获取检索器（k=5，由chroma.yml配置）
    # ============================================================
    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    # ============================================================
    # 加载知识库文档：扫描 data/ 目录 → MD5去重 → 分块 → 存入向量库
    # ============================================================
    def load_document(self):
        # 内部：检查MD5是否已存在（去重）
        def check_md5_hex(md5_for_check: str):
            md5_file = get_abs_path(chroma_conf["md5_hex_store"])
            if not os.path.exists(md5_file):
                open(md5_file, "w", encoding="utf-8").close()
                return False
            with open(md5_file, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True
                return False

        # 内部：保存已处理文件的MD5
        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        # 内部：按文件类型选择加载器
        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        # 扫描 data/ 目录获取所有支持类型的文件
        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        # 逐文件处理：去重检查 → 加载 → 分块 → 存入向量库
        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)
                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                self.vector_store.add_documents(split_document)
                save_md5_hex(md5_hex)
                logger.info(f"[加载知识库]{path} 内容加载成功")

            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue
