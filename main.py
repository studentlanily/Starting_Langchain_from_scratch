from anyio.lowlevel import checkpoint
from fontTools.misc.cython import returns
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from numpy.lib.recfunctions import structured_to_unstructured
from pyexpat.errors import messages
from langchain.tools import tool,ToolRuntime
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

#定义系统提示
SYSTEM_PROMPT = """你是一位擅长用双关语表达的专家天气预报员。
你可以使用两个工具：
- get_weather_for_location：用于获取特定地点的天气
- get_user_location：用于获取用户的位置
如果用户询问天气，请确保你知道具体位置。如果从问题中可以判断他们指的是自己所在的位置，请使用 get_user_location 工具来查找他们的位置。"""

#创建工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    return f"{city}总是阳光明媚！"

#定义上下文模式
@dataclass
class Context:
    """自定义运行时上下文"""
    user_id:str

@tool
def get_user_location(runtime:ToolRuntime[Context]) -> str:
    """根据用户ID获取用户信息"""
    user_id=runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


#配置模型
llm = ChatOpenAI(
    model= "qwen2.5:32b",
    base_url="http://localhost:11434/v1",
    api_key="default",
    temperature= 0,
)

local_llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
    base_url="http://127.0.0.1:11434",
)

#定义响应格式
@dataclass
class ResponseFormat:
    """代理的响应方式"""
    #带双关语的回应（始终必需）
    punny_response:str
    #天气的任何有趣信息（如果有）
    weather_conditions:str | None = None

#添加记忆
checkpointer = InMemorySaver()

#创建并运行代理
agent = create_agent(
    model=local_llm,
    system_prompt=SYSTEM_PROMPT,
    tools=[
        get_weather,get_user_location
    ],
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)

#'thread_id'是给定对话的唯一标识符
config = {"configurable":{"thread_id":"1"}}

response = agent.invoke(
    {"messages":[
        {
            "role":"user",
            "content":"广州的天气怎么样？",
         },
    ]},
    config=config,#可选，用于记忆
    context=Context(user_id="1")
)

#输出
#1.尝试取结构化响应（模型需支持 tool calling）
resp = response.get("structured_response")
if resp:
    print("=== 结构化响应 ===")
    print(resp.punny_response)
    print(resp.weather_conditions)
else:
    print("=== 模型未返回结构化响应，使用普通消息 ===")
#2.从messages取最终回复（始终可用）
print("=== 最终回复 ===")
print(response["messages"][-1].content)

#从model层面约束输出
from langgraph.graph import StateGraph,MessagesState,START,END
from langgraph.prebuilt import ToolNode

#1.绑定 tools 到模型
gllm = ChatOllama(model="qwen2.5:32b",temperature=0,base_url="http://127.0.0.1:11434")
llm_with_tools = gllm.bind_tools([get_weather,get_user_location])

#2.定义节点
def call_model(state:MessagesState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages":[response]}

def should_continue(state:MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

#3.构建图
graph = StateGraph(MessagesState)
graph.add_node("agent",call_model)
graph.add_node("tools",ToolNode([get_weather,get_user_location]))
graph.add_edge(START,"agent")
graph.add_conditional_edges("agent",should_continue,{"tools": "tools", END: END})
graph.add_edge("tools","agent")

agent = graph.compile(checkpointer=checkpointer)

#4.调用
response = agent.invoke(
    {"messages":[{"role":"human","content":"广州天气怎么样？"}]},
    config = config
)
print("===结构化输出===")
try:
    print(response)
except:
    returns("输出失败")

# ResponseFormat(
#     punny_response="佛罗里达今天依然是'阳光灿烂'的一天！阳光正在播放'rey-dio'热门歌曲！我得说，这是进行'solar-bration'的完美天气！如果你希望下雨，恐怕这个想法已经'被冲走'了——预报仍然'清晰地'灿烂！",
#     weather_conditions="佛罗里达总是阳光明媚！"
# )

# messages = [
#     ("system","You are a helpful assistant."),
#     ("human","你好，请介绍一下自己。"),
# ]
# response = llm.invoke(messages)
# print(response.content)
#
# for chunk in llm.stream("讲个笑话。"):
#     print(chunk.type+chunk.content,end="\n")
#
# print("==================================\n")
#
# #基本代理
# agent = create_agent(
#     model=llm,
#     tools=[
#         get_weather
#     ],
#     system_prompt="你是一个乐于助人的助手",
# )
#
#
# agent.invoke(
#     {
#         "messages":[{
#             "role":"user",
#             "content":"旧金山的天气怎么样"
#         }]
#     }
# )
