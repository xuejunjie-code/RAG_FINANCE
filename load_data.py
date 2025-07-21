import os
import torch
import logging
from typing import Tuple, List
from paddleocr import PPStructure # 导入PP-Structure
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentProcessor:
    """
    使用本地部署的PaddleOCR/PP-Structure进行文档深度解析，
    实现高质量的文表分离和结构化分块，并支持增量更新。
    """
    def __init__(self, data_path="dataset", vector_store_path="vec_store", model_name="BAAI/bge-large-zh-v1.5"):
        self.data_path = data_path
        self.vector_store_path = vector_store_path
        self.processed_log_path = os.path.join(vector_store_path, "processed_files.log")
        
        # 初始化Embedding模型
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        # 初始化向量数据库
        self.vector_store = Chroma(
            persist_directory=self.vector_store_path,
            embedding_function=self.embedding_model
        )
        # 初始化备用的文本分割器
        self.recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        
        # 初始化PP-Structure模型
        logging.info("正在初始化PP-Structure模型，首次运行可能需要下载模型文件...")
        # layout=True 开启版面分析功能
        self.structure_engine = PPStructure(show_log=False, layout=True)
        logging.info("DocumentProcessor 初始化完成 (PaddleOCR/PP-Structure 本地模式)。")

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
        直接返回一个包含文本块和表格块的Document列表。
        """
        logging.info(f"[PP-Structure] 正在解析文件: {os.path.basename(file_path)}")
        result = self.structure_engine(file_path)
        
        doc_chunks = []
        for item in result:
            item_type = item.get('type')
            
            if item_type == 'table':
                table_html = item.get('res', {}).get('html', '')
                if table_html:
                    # 对于表格，直接创建一个独立的Document
                    table_caption = f"文件 {os.path.basename(file_path)} 中的表格"
                    table_content = f"### 表格：{table_caption}\n\n{table_html}"
                    table_doc = Document(
                        page_content=table_content,
                        metadata={"source": os.path.basename(file_path), "type": "table"}
                    )
                    doc_chunks.append(table_doc)
            else: # 'text', 'title', 'list', 'figure' 等
                # 对于文本，也将其创建为Document，后续再进行可能的合并与切分
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

    def _load_and_process_new_documents(self) -> Tuple[List[Document], List[str]]:
        """
        使用PP-Structure加载和处理新文件。
        """
        processed_files = self._get_processed_files()
        new_files_to_process_paths = []
        files_to_process_names = []

        all_files_in_dir = [f for f in os.listdir(self.data_path) if f.endswith(('.pdf', '.png', '.jpg'))]
        
        for filename in all_files_in_dir:
            if filename not in processed_files:
                new_files_to_process_paths.append(os.path.join(self.data_path, filename))
                files_to_process_names.append(filename)

        if not new_files_to_process_paths:
            logging.info("没有发现新文件需要处理。")
            return [], []

        logging.info(f"发现 {len(new_files_to_process_paths)} 个新文件，开始使用PP-Structure进行本地解析...")
        
        all_new_chunks = []
        for file_path in new_files_to_process_paths:
            try:
                # 解析文档，直接得到分离好的文本块和表格块
                docs_from_file = self._parse_with_ppstructure(file_path)
                
                # 对解析出的文本块进行二次切分，以防单个文本区域过长
                # 表格块则保持原样
                temp_chunks = []
                for doc in docs_from_file:
                    if doc.metadata.get("type") == "table":
                        temp_chunks.append(doc)
                    else:
                        # 使用递归字符分割器对文本块进行切分
                        split_text_docs = self.recursive_splitter.split_documents([doc])
                        temp_chunks.extend(split_text_docs)
                
                all_new_chunks.extend(temp_chunks)
                logging.info(f"成功解析并切分: {os.path.basename(file_path)}")
            except Exception as e:
                logging.error(f"使用PP-Structure解析文件 {os.path.basename(file_path)} 失败: {e}")
        
        return all_new_chunks, files_to_process_names

    def update_vector_store(self):
        """主方法：执行完整的文档处理和入库流程"""
        new_chunks, processed_filenames = self._load_and_process_new_documents()
        
        if not new_chunks:
            return

        logging.info(f"开始将 {len(new_chunks)} 个新片段添加至向量数据库...")
        self.vector_store.add_documents(documents=new_chunks)
        
        for filename in processed_filenames:
            self._log_processed_file(filename)
            
        logging.info(f"向量数据库更新成功！新增了来自 {len(processed_filenames)} 个文件的 {len(new_chunks)} 个片段。")