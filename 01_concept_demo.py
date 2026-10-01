"""
阶段 1: RAG 概念演示
====================
目标：理解"为什么要用 RAG"——对比纯 LLM 回答 vs 提供上下文后的回答

流程：
  1. 读取本地文档（公司内部政策）
  2. 直接问 LLM 一个它不可能知道的问题（纯 LLM 模式）
  3. 把相关文档内容作为上下文提供给 LLM（RAG 模式雏形）
  4. 对比两种回答
"""

# 配置 LLM（与你已有的 main.py 一致）
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen2.5:7b",
    base_url="http://localhost:11434/v1",
    api_key="default",
    temperature=0.3,
)

# 1. 读取本地文档
with open("data/company_policy.md", "r", encoding="utf-8") as f:
    document = f.read()

print("=" * 60)
print("[文档] 已加载文档（共 {} 字）".format(len(document)))
print("=" * 60)

# 2. 纯 LLM 模式：直接问，不给任何上下文
question = "星辰科技公司的年假政策是什么？入职4年的员工有多少天年假？"

print("\n[模式一] 纯 LLM 回答（无上下文）")
print("-" * 40)
response_no_context = llm.invoke([("human", question)])
print("回答:", response_no_context.content)

# 3. RAG 模式雏形：先找到相关段落，再让 LLM 回答
# 这里手动模拟"检索"步骤，后面阶段会自动化
print("\n\n[模式二] RAG 模式（提供相关上下文）")
print("-" * 40)

# 手动检索相关段落（模拟 RAG 的检索步骤）
context = """
{relevant_docs}

请基于以上信息回答问题。如果信息不足以回答，请如实说明。
"""
context = context.format(relevant_docs=document)

rag_prompt = f"以下是参考资料：\n\n{context}\n\n问题是：{question}"

response_with_context = llm.invoke([("human", rag_prompt)])
print("回答:", response_with_context.content)

# 4. 总结对比
print("\n\n" + "=" * 60)
print("[总结] 阶段 1 总结")
print("=" * 60)
print("""
  纯 LLM：模型只靠训练时学到的知识回答。
          对于公司内部文档、实时信息、私有数据，它无法准确回答。
          
  RAG 思路：先检索相关文档 → 把文档作为上下文提供给 LLM → LLM 基于事实回答。
  
  下一阶段：我们将用代码自动化"检索"这一步！
""")
