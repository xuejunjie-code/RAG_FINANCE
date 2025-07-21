import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from load_data import DocumentProcessor

# 加载 .env 文件中的环境变量
load_dotenv()

class RAG:
    """
    一个分级问答系统：
    1. 优先从本地知识库检索。
    2. 如果本地无相关信息，则利用大模型的通用知识回答。
    """
    def __init__(self):
        """
        初始化RAG系统，设置文档处理器、向量存储和语言模型。
        """
        print("--- Initializing RAG System ---")
        
        # 1. 初始化文档加载和向量存储
        self.loader = DocumentProcessor()
        self.loader.update_vector_store()
        self.vector_store = self.loader.vector_store

        # 2. 检查并设置API密钥
        if "GOOGLE_API_KEY" not in os.environ:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

        # 3. 初始化大语言模型 (LLM)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        
        print("--- RAG System Initialized Successfully ---")

    def rag(self, query: str):
        """
        执行分级问答流程：
        1. 从本地向量库检索文档，如果找到足够相关的，则基于本地知识回答。
        2. 否则，直接调用LLM的通用知识库进行回答。
        """
        print(f"\n--- [User Query]: {query} ---")
        
        # 步骤 1: 从本地向量库检索文档及其分数
        print("--- Retrieving from local vector store... ---")
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=4)
        
        print(f"--- [Debug] Raw search results (doc, score): {[(doc.metadata.get('source', 'N/A'), round(score, 4)) for doc, score in docs_with_scores]} ---")

        # 步骤 2: 手动过滤，只保留足够相关的文档
        score_threshold = 0.6  
        docs = [doc for doc, score in docs_with_scores if score < score_threshold]

        # 步骤 3: 根据检索结果选择回答策略
        if docs:
            # 策略一: 本地知识库中有足够相关的答案
            print(f"--- Found {len(docs)} relevant documents locally. Answering based on context. ---")
            context = "\n\n".join(doc.page_content for doc in docs)
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "你是一个专业的问答助手。请根据下面提供的上下文信息来精确地回答问题。确保你的回答完全基于所提供的上下文，不要添加外部知识。"),
                ("human", "上下文:\n{context}\n\n问题:\n{question}")
            ])
            messages = prompt_template.format_messages(context=context, question=query)
        else:
            # 策略二: 本地知识库没有相关答案，直接利用LLM的通用知识回答
            print(f"--- No relevant documents found locally. Asking the LLM directly. ---")
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "你是一个知识渊博的通用问答助手，请直接回答用户的问题。"),
                ("human", "{question}")
            ])
            messages = prompt_template.format_messages(question=query)
        
        # 步骤 4: 调用LLM生成最终回答
        print("--- Sending request to LLM... ---")
        response = self.llm.invoke(messages)

        return response.content