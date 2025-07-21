from langchain.agents import tool
from rag import RAG          # 导入你的 RAG 类
from sqlite_read import SQLReader # 导入你的 SQLiteDB 类

# --- 步骤1: 实例化你的核心系统 ---
# 在全局或一个工厂函数中创建实例，以确保它们只被初始化一次。
print("--- Initializing systems for Agent Tools ---")
rag_system = RAG()
db_system = SQLReader()
print("--- Systems initialized ---")


# --- 步骤2: 定义工具，并配上清晰的描述 ---

@tool
def query_knowledge_base(query: str) -> str:
    """
    用于回答关于通用概念、定义、解释或总结性的问题。
    当问题不涉及公司内部的具体数据（如员工、薪水）时，应优先使用此工具。
    例如: '什么是RAG？', '介绍一下LangChain', '唐家三少写过哪些书？'
    """
    print(f"--- [Agent] Routing to RAG Tool with query: {query} ---")
    return rag_system.rag(query)

@tool
def query_company_database(query: str) -> str:
    """
    用于查询公司数据库中的精确、结构化数据，例如员工、部门或薪水信息。
    当问题包含'谁'、'多少'、'哪个部门'、'平均工资'等关键词，并且与公司员工数据相关时，必须使用此工具。
    例如: '销售部有多少名员工？', '谁的工资最高？', '查询Alice的信息'
    """
    print(f"--- [Agent] Routing to SQLite Tool with query: {query} ---")
    return db_system.sql_query(query)


# 接上文...
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

def create_main_agent():
    """
    创建并组装主 Agent Executor。
    """
    # 1. 将所有工具放入一个列表
    tools = [query_knowledge_base, query_company_database]
    
    # 2. 定义 Agent 的思考和行动的 Prompt 模板
    #    这个 ReAct 模板是 LangChain 的标准实践，效果很好。
    #    它指导 LLM 如何进行思考、选择工具、执行并观察结果。
    prompt_template = ChatPromptTemplate.from_template("""
    尽你所能回答以下问题。你可以使用以下工具：

    {tools}

    请使用以下格式：

    Question: 你必须回答的输入问题
    Thought: 你应该总是思考该做什么
    Action: 你要采取的行动，应该是 [{tool_names}] 中的一个
    Action Input: 对行动的输入
    Observation: 行动的结果
    ... (这个“Thought/Action/Action Input/Observation”可以重复N次)
    Thought: 我现在知道最终答案了
    Final Answer: 对原始输入问题的最终回答

    开始！

    Question: {input}
    Thought:{agent_scratchpad}
    """)

    # 3. 初始化 LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # 4. 创建 Agent
    #    create_react_agent 会将 LLM、工具和 Prompt 绑定在一起，
    #    创建一个知道如何遵循 ReAct 格式进行思考的 Agent。
    agent = create_react_agent(llm, tools, prompt_template)

    # 5. 创建 Agent Executor
    #    AgentExecutor 负责实际执行 Agent 的决策循环。
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # 强烈建议设为 True，可以看到 Agent 的完整思考过程
        handle_parsing_errors=True # 处理LLM输出格式错误的情况
    )

    print("--- Agent Executor Created Successfully ---")
    return agent_executor