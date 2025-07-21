from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_core.prompts import ChatPromptTemplate

class SQLReader:
    def __init__(self, db_path = "dataset//data.db"):
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    def sql_query(self, query):
        schema = self.db.get_table_info()
        sql_generation_template = """请根据以下数据库模式生成SQL查询语句：
            数据库模式:{schema}    

            请生成SQL查询语句：{question}  
            """


        sql_generation_prompt = ChatPromptTemplate.from_template(sql_generation_template)

        # 格式化prompt，填入schema和问题
        messages = sql_generation_prompt.format_messages(schema=schema, question=query)

        # 调用LLM
        ai_message = self.llm.invoke(messages)
        sql_query = ai_message.content.strip() # .strip()用于去除可能存在的前后空格或换行符
        sql_query = sql_query.replace('```sql', '').replace('```', '')  # 去除Markdown格式

        print(f"3. LLM生成的SQL查询语句: {sql_query}")

        # --- 步骤 3: 执行生成的SQL语句 ---
        try:
            sql_result = self.db.run(sql_query)
            print(f"4. 执行SQL后的原始结果: {sql_result}")
        except Exception as e:
            print(f"Error executing SQL: {e}")
            return f"执行SQL时出错: {e}"

        # --- 步骤 4: 构建并调用LLM以生成最终的自然语言回答 ---
        final_response_template = """
        根据下面的表结构、问题、SQL查询和SQL查询结果，写一个通顺的自然语言回答。
        表结构: {schema}
        问题: {question}
        SQL查询: {query}
        SQL查询结果: {response}
        自然语言回答:
        """
        final_response_prompt = ChatPromptTemplate.from_template(final_response_template)
        
        # 格式化prompt
        messages = final_response_prompt.format_messages(
            schema=schema,
            question=query,
            query=sql_query,
            response=sql_result
        )
        
        # 再次调用LLM
        final_ai_message = self.llm.invoke(messages)
        final_answer = final_ai_message.content
        
        print(f"5. LLM生成的最终自然语言回答: {final_answer}")
        
        return final_answer





































































