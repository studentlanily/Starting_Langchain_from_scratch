from aiohttp.web_middlewares import middleware
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_ollama import ChatOllama
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, wrap_tool_call
from langchain.tools import tool


#静态模型：静态模型在创建智能体时配置一次，并在整个执行过程中保持不变。这是最常见且直接的方法。
#例如
# agent = create_agent(
#     "qwen2.5:30b",
#     tools=tools
# )

#动态模型——使用@wrap_model_call装饰器创建中间件以修改请求中的模型：
basic_model =ChatOllama(model="qwen2.5:7b",temperature=0,base_url="http://localhost:11434")
advanced_model = ChatOllama(model="qwen2.5:32b",temperature=0,base_url="http://localhost:11434")

@tool
def search(query:str)->str:
    """搜索信息。"""
    return f"结果：{query}"
@tool
def get_weather(location:str)->str:
    """获取位置的天气信息"""
    return f"{location}的天气：晴朗，72°F"


@wrap_model_call
def dynamic_model_selection(request:ModelRequest,handler) ->ModelResponse:
    """根据对话复杂性选择模型"""
    # @wrap_model_call— 装饰器，标记这是一个模型调用中间件
    # request — 包含当前调用的所有信息（模型、消息、工具等）
    # handler — 调用链中的下一个处理函数（调用它才会真正执行LLM）
    # 返回ModelResponse — LLM的响应

    message_count = len(request.state["messages"])
    #获取对话的信息数量

    if message_count > 10:
        # 对较长的对话使用高级模型
        model = advanced_model
    else:
        model = basic_model

    request.override(model=model)
    return handler(request)

@wrap_tool_call
def handle_tool_errors(request,handler):
    """使用自定义消息处理工具执行错误"""
    try:
        return handler(request)
    except Exception as e:
        #向模型返回自定义错误信息
        return ToolMessage(
            content=f"工具错误：请检查您的输入并重试。({str(e)})",
            tool_call_id=request.tool_call["id"]
            #tool_call_id 是工具调用的唯一标识符，用于把工具返回结果和对应的工具调用请求配对。
        )

# 智能体遵循 ReAct（“推理 + 行动”）模式，在简短的推理步骤与针对性的工具调用之间交替，并将结果观察反馈到后续决策中，直到能够提供最终答案。
@wrap_model_call
def log_model_call(request:ModelRequest,handler) -> ModelResponse:
    print(f"\n====AI消息====")
    print(f"消息数：{len(request.state['messages'])}")
    result = handler(request)

    ai_msg = result.result[-1]
    if ai_msg.tool_calls:
        for tc in ai_msg.tool_calls:
            print(f"Action:{tc['name']}({tc['args']})")
    else:
        print(f"Answer:{ai_msg.content}")
    return result

@wrap_tool_call
def log_tool_call(request,handler):
    result = handler(request)
    print(f"Observation:{result.content}")
    return result


# 您可以通过提供提示来塑造智能体处理任务的方式。system_prompt 参数可以作为字符串提供
SYSTEM_PROMPT = """你是一位擅长用双关语表达的专家天气预报员。
你可以使用两个工具：
- get_weather_for_location：用于获取特定地点的天气
- get_user_location：用于获取用户的位置
如果用户询问天气，请确保你知道具体位置。如果从问题中可以判断他们指的是自己所在的位置，请使用 get_user_location 工具来查找他们的位置。"""

agent = create_agent(
    model=basic_model,
    tools=[get_weather,search],
    system_prompt=SYSTEM_PROMPT,
    middleware=[dynamic_model_selection,handle_tool_errors,log_model_call,log_tool_call],# ← 注册中间件
    #config=config,
    #config={"configurable": {"thread_id": "1"}}

)
#请求 → dynamic_model_selection → handle_tool_errors → LLM → handle_tool_errors → dynamic_model_selection → 响应

response=agent.invoke({"messages": [{"role": "human", "content": "广州天气怎么样？"}]})
print(response)