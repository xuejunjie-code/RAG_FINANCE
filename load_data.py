import os
import torch
import logging
from typing import List
from paddleocr import PPStructure
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import DistanceStrategy # 导入距离策略

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FAISSDocumentProcessor:
    """
    使用本地PaddleOCR/PP-Structure进行文档深度解析，
    并为每个文档创建一个独立的、使用FAISS (IndexFlatIP) 存储的向量数据库。
    """
    def __init__(self, data_path="dataset", dbs_root_path="faiss_dbs", model_name="BAAI/bge-large-zh-v1.5"):
        self.data_path = data_path
        self.dbs_root_path = dbs_root_path # 存储所有FAISS数据库的根目录
        self.processed_log_path = os.path.join(dbs_root_path, "processed_files.log")
        
        # 确保数据库根目录存在
        os.makedirs(self.dbs_root_path, exist_ok=True)
        
        # 初始化Embedding模型 (保持不变)
        logging.info(f"正在加载Embedding模型: {model_name}")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}, # 归一化对于IP距离至关重要
        )
        
        # 初始化备用的文本分割器
        self.recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        
        # 初始化PP-Structure模型
        logging.info("正在初始化PP-Structure模型，首次运行可能需要下载模型文件...")
        self.structure_engine = PPStructure(show_log=False, layout=True)
        logging.info("FAISSDocumentProcessor 初始化完成。")

    def _get_processed_files(self) -> set:
        if not os.path.exists(self.processed_log_path): return set()
        with open(self.processed_log_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f)

    def _log_processed_file(self, filename: str):
        with open(self.processed_log_path, 'a', encoding='utf-8') as f:
            f.write(filename + '\n')

    def _parse_with_ppstructure(self, file_path: str) -> List[Document]:
        """
        核心函数：使用本地的PP-Structure模型解析单个文档文件。
        (此函数与之前版本基本相同)
        """
        logging.info(f"[PP-Structure] 正在解析文件: {os.path.basename(file_path)}")
        result = self.structure_engine(file_path)
        
        doc_chunks = []
        for item in result:
            item_type = item.get('type')
            
            if item_type == 'table':
                table_html = item.get('res', {}).get('html', '')
                if table_html:
                    table_caption = f"文件 {os.path.basename(file_path)} 中的表格"
                    table_content = f"### 表格：{table_caption}\n\n{table_html}"
                    table_doc = Document(
                        page_content=table_content,
                        metadata={"source": os.path.basename(file_path), "type": "table"}
                    )
                    doc_chunks.append(table_doc)
            else:
                text_content = ""
                text_lines = item.get('res', [])
                for line_info in text_lines:
                    text_content += line_info.get('text', '') + '\n'
                
                if text_content.strip():
                    text_doc = Document(
                        page_content=text_content,
                        metadata={"source": os.path.basename(file_path), "type": item_type}
                    )
                    doc_chunks.append(text_doc)
                    
        return doc_chunks

    def create_or_update_faiss_stores(self):
        """
        主方法：扫描新文件，并为每个新文件创建独立的FAISS数据库。
        """
        processed_files = self._get_processed_files()
        all_files_in_dir = [f for f in os.listdir(self.data_path) if f.endswith(('.pdf', '.png', '.jpg'))]

        new_files_count = 0
        for filename in all_files_in_dir:
            if filename in processed_files:
                continue

            new_files_count += 1
            file_path = os.path.join(self.data_path, filename)
            logging.info(f"发现新文件: {filename}，开始处理...")

            try:
                # 1. 使用PP-Structure解析文档，得到初步的块
                docs_from_file = self._parse_with_ppstructure(file_path)
                
                # 2. 对解析出的文本块进行二次切分
                final_chunks = []
                for doc in docs_from_file:
                    if doc.metadata.get("type") == "table":
                        final_chunks.append(doc) # 表格块保持原样
                    else:
                        split_text_docs = self.recursive_splitter.split_documents([doc])
                        final_chunks.extend(split_text_docs)
                
                if not final_chunks:
                    logging.warning(f"文件 {filename} 未解析出任何有效内容，跳过建库。")
                    self._log_processed_file(filename) # 记录下来，避免重复处理
                    continue

                # 3. 创建FAISS数据库
                logging.info(f"正在为 {filename} 创建FAISS索引...")
                # 使用 from_documents 工厂方法，并指定距离策略为内积
                # 对于归一化嵌入，内积等效于余弦相似度，FAISS会使用 IndexFlatIP
                faiss_db = FAISS.from_documents(
                    documents=final_chunks,
                    embedding=self.embedding_model,
                    distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
                )
                
                # 4. 定义并保存FAISS数据库到独立目录
                db_name = os.path.splitext(filename)[0] # 使用文件名（不含扩展名）作为库名
                db_path = os.path.join(self.dbs_root_path, db_name)
                faiss_db.save_local(db_path)
                
                logging.info(f"成功为文件 {filename} 创建了独立的FAISS数据库，路径: {db_path}")
                
                # 5. 记录已处理的文件
                self._log_processed_file(filename)

            except Exception as e:
                logging.error(f"处理文件 {filename} 时发生严重错误: {e}", exc_info=True)

        if new_files_count == 0:
            logging.info("没有发现新文件需要处理。")
        else:
            logging.info(f"处理完成！共为 {new_files_count} 个新文件创建了向量数据库。")
