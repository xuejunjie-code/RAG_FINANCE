import os
import torch
import logging
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# 使用日志记录代替 print
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentProcessor:
    """
    一个负责处理文档（加载、分割、嵌入）并更新向量数据库的类。
    """
    def __init__(self, pdf_path = "dataset", txt_path = "dataset", vector_store_path = "vec_store", model_name="BAAI/bge-large-zh-v1.5"):
        # 1. 配置通过 __init__ 传入，更灵活
        self.pdf_path = pdf_path
        self.txt_path = txt_path
        self.vector_store_path = vector_store_path
        self.processed_log_path = os.path.join(vector_store_path, "processed_files.log")
        
        # 2. 将组件初始化也放在 __init__ 中
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vector_store = Chroma(
            persist_directory=self.vector_store_path,
            embedding_function=self.embedding_model
        )
        logging.info("DocumentProcessor 初始化完成。")

    def _get_processed_files(self):
        """读取已处理文件日志"""
        if not os.path.exists(self.processed_log_path):
            return set()
        with open(self.processed_log_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)

    def _log_processed_file(self, filename):
        """将新处理的文件名写入日志"""
        with open(self.processed_log_path, 'a', encoding='utf-8') as f:
            f.write(filename + '\n')

    def _load_new_documents(self):
        """只加载新文件，实现增量更新"""
        processed_files = self._get_processed_files()#获取已处理的文件
        new_docs = []
        files_to_process = []

        # 收集所有PDF和TXT文件
        all_files = [os.path.join(self.pdf_path, f) for f in os.listdir(self.pdf_path) if f.endswith('.pdf')]
        all_files.extend([os.path.join(self.txt_path, f) for f in os.listdir(self.txt_path) if f.endswith('.txt')])
        
        for file_path in all_files:
            if os.path.basename(file_path) not in processed_files:
                files_to_process.append(file_path)

        if not files_to_process:
            logging.info("没有发现新文件需要处理。")
            return [], []

        logging.info(f"发现 {len(files_to_process)} 个新文件，开始加载...")
        for file_path in files_to_process:
            try:
                if file_path.endswith('.pdf'):
                    loader = PyMuPDFLoader(file_path)
                else: # .txt
                    loader = TextLoader(file_path, encoding='utf-8')
                new_docs.extend(loader.load())
                logging.info(f"成功加载: {os.path.basename(file_path)}")
            except Exception as e:
                logging.error(f"加载文件 {os.path.basename(file_path)} 失败: {e}")
        
        return new_docs, [os.path.basename(f) for f in files_to_process]

    def update_vector_store(self):
        """
        主方法：执行完整的文档处理和入库流程
        """
        # 步骤 1: 加载新文档
        new_docs, processed_filenames = self._load_new_documents()
        
        if not new_docs:
            return False # 没有新内容，直接返回

        # 步骤 2: 分割文档
        logging.info("开始分割新文档...")
        split_docs = self.splitter.split_documents(new_docs)
        logging.info(f"新文档被分割成 {len(split_docs)} 个片段。")
        
        # 步骤 3: 嵌入并存储到 ChromaDB
        logging.info("开始将新文档片段添加至向量数据库...")
        self.vector_store.add_documents(documents=split_docs)
        # self.vector_store.persist() # 新版ChromaDB通常是自动持久化的
        
        # 步骤 4: 更新处理日志
        for filename in processed_filenames:
            self._log_processed_file(filename)
            
        logging.info("向量数据库更新成功！")
    