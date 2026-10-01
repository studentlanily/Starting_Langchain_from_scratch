"""
阶段 7: 进阶 RAG 技巧
=====================
目标：优化 RAG 效果

涵盖三个进阶技巧：
  1. 对话式 RAG - 带历史记忆的多轮对话
  2. 查询改写  - 让检索更准确
  3. 重排序    - 对检索结果二次排序
"""

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)

llm = ChatOpenAI(
    model="qwen2.5:7b",
    base_url="http://localhost:11434/v1",
    api_key="default",
    temperature=0.3,
)

# 构建向量库
all_chunks = []
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=40,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)
for filepath in ["data/long_document.txt", "data/company_policy.md"]:
    docs = TextLoader(filepath, encoding="utf-8").load()
    all_chunks.extend(splitter.split_documents(docs))

vector_store = FAISS.from_documents(all_chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

def rag_answer(question, context):
    prompt = """你是一个基于知识库回答问题的助手。
请根据以下参考资料回答用户的问题。如果参考资料不足以回答问题，请如实说明。

参考资料：
{context}

问题：{question}

请用中文回答：""".format(context=context, question=question)
    return llm.invoke([("human", prompt)]).content

print("=" * 60)
print("阶段 7: 进阶 RAG 技巧")
print("=" * 60)

# ========== 1. 对话式 RAG ==========
print("\n[1] 对话式 RAG（带历史记忆）")
print("-" * 40)

history = []
questions_round = [
    "星辰科技有哪些产品？",
    "它们的技术优势是什么？",
    "总结一下我刚才问了什么？",
]

for q in questions_round:
    print("\n用户:", q)

    # 检索
    docs = retriever.invoke(q)
    context = "\n\n".join([doc.page_content for doc in docs])

    # 带历史的 prompt
    history_text = ""
    if history:
        history_text = "以下是之前的对话历史：\n" + "\n".join(history) + "\n\n"

    prompt = """你是一个基于知识库回答问题的助手。
{history_text}
参考资料：
{context}

问题：{question}

请用中文回答：""".format(history_text=history_text, context=context, question=q)

    answer = llm.invoke([("human", prompt)]).content
    print("助手:", answer[:120])

    # 记录历史
    history.append("用户: " + q)
    history.append("助手: " + answer[:50])

# ========== 2. 查询改写 ==========
print("\n\n[2] 查询改写（让检索更准确）")
print("-" * 40)

# 原始问题可能不适合检索
original_query = "他们今年赚了多少钱"
print("原始问题:", original_query)

# 用 LLM 改写为更适合检索的形式
rewrite_prompt = """请将以下问题改写成适合文档检索的形式。
要求：保留核心语义，使用更正式、更完整的关键词。

原始问题：{query}

改写后：""".format(query=original_query)

rewritten = llm.invoke([("human", rewrite_prompt)]).content
print("改写后:", rewritten)

# 对比检索效果
print("\n改写前检索结果:")
for doc in retriever.invoke(original_query):
    print("  -", doc.page_content[:60])

print("\n改写后检索结果:")
for doc in retriever.invoke(rewritten):
    print("  -", doc.page_content[:60])

# ========== 3. 重排序（简易版） ==========
print("\n\n[3] 重排序（Reranker - 用 LLM 对结果排序）")
print("-" * 40)

question = "星辰科技有哪些产品？"
initial_docs = retriever.invoke(question)
print("初始检索到 {} 个文档".format(len(initial_docs)))

# 用 LLM 判断每个文档的相关性并排序
rerank_prompt = """问题：{question}

给定以下文档片段，请判断每个文档与问题的相关程度。
按相关性从高到低排序（只输出排序后的文档编号，用逗号分隔）。

{document_list}

输出格式：编号1, 编号2, 编号3""".format(
    question=question,
    document_list="\n".join([
        "[编号{}] {}".format(i+1, doc.page_content[:100])
        for i, doc in enumerate(initial_docs)
    ])
)

rerank_result = llm.invoke([("human", rerank_prompt)]).content
print("LLM 排序结果:", rerank_result)

# ========== 总结 ==========
print("\n" + "=" * 60)
print("阶段 7 总结 - 你已经学完 RAG 的核心知识！")
print("=" * 60)
print("""
  7 个阶段的学习路线：
    1. 概念 - 为什么需要 RAG
    2. 加载 - 从多种来源读取文档
    3. 分块 - 把文档切成小块
    4. 嵌入 - 文本转向量
    5. 检索 - 找最相关的段落
    6. 生成 - 检索 + LLM 回答问题
    7. 进阶 - 对话、查询改写、重排序
  
  继续深入学习的方向：
    - RAPTOR: 多层摘要检索
    - Agentic RAG: 让 LLM 自己决定检索策略
    - Graph RAG: 基于知识图谱的检索
    - 多模态 RAG: 图片、表格、音频
""")
