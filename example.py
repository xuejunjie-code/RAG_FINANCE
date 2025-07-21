#用于实现RAG的整个流程——入门向

from dotenv import load_dotenv
import os
from langchain.document_loaders import DirectoryLoader,PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
import torch
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import PromptTemplate

#step 1: 数据的加载
folder_path = "D:/RAG/RAG_Project/finance_bot/data"

pdf_loader = DirectoryLoader(
    directory_path=folder_path,
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
)

raw_doc = pdf_loader.load()

#step 2: 数据分块
data_spliter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

docs = data_spliter.split_documents(raw_doc)

#step 3: 数据向量化
# 闭源模型
# openai_embedding = OpenAIEmbeddings(model="text-embedding-3-small")
# 开源模型（BGE）

bge_embedding = HuggingFaceEmbeddings(
    model="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    )

#step 4: 数据存储
# 默认索引方式为HNSW
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=bge_embedding,
    persist_directory="D:/RAG/RAG_Project/finance_bot/vector_store",
)

vector_store.persist()

#step 5: 数据检索
retriver = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,  # 返回最相似的5个文档
        # "fetch_k": 10,  # 从中检索10个文档进行相似度计算
        'score_threshold': 0.5,  # 设置相似度阈值
    }
)

#step 6: 生成
# 使用OpenAI模型进行生成
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

#构建提示词
template = """
你是一个问答助手。请根据下面提供的上下文来回答问题。
如果上下文中没有答案，就说你不知道。

上下文: 
{context}

问题: 
{question}

回答:
"""
prompt = ChatPromptTemplate.from_template(template)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# e. 构建RAG链
rag_chain = (
    {"context": retriver | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# f. 调用链并提问
user_question = "LCEL有什么核心优势？"
print(f"用户问题: {user_question}")
response = rag_chain.invoke(user_question)
print(f"RAG系统回答: {response}")

# 测试一个知识库中没有的问题
print("\n--- 测试一个知识库中不存在的问题 ---")
user_question_2 = "什么是Python的GIL？"
print(f"用户问题: {user_question_2}")
response_2 = rag_chain.invoke(user_question_2)
print(f"RAG系统回答: {response_2}")









