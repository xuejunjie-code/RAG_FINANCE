from rag import RAG
from sqlite_read import SQLReader
from agent import create_main_agent

# rag = RAG()
sql = SQLReader()
agent = create_main_agent()
# response = rag.rag("武汉力源信息技术股份有限公司的每股面值是多少?")
# response = rag.rag("RAG是什么?")
sql = SQLReader()
response = sql.query_sql("RAG是什么?")

print(response)