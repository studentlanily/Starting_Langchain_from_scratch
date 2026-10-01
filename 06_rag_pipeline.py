"""
阶段 6: 完整 RAG 流水线 (Complete RAG Pipeline)
================================================
目标：将检索 + 生成组合成完整的 RAG 系统

流程：
  用户提问
    │
    ▼
  [1] 检索 - 从向量库中找到相关文档
    │
    ▼
  [2] 增强 - 把文档拼到 prompt 中
    │
    ▼
  [3] 生成 - LLM 基于文档回答问题
"""

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# ========== 初始化组件 ==========
print("=" * 60)
print("阶段 6: 完整 RAG 流水线")
print("=" * 60)

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

# 先重建向量库：把多个文档都索引进去
print("\n[准备] 重建向量库，加入多个文档...")
all_chunks = []
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=40,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)
for filepath in ["data/long_document.txt", "data/company_policy.md"]:
    docs = TextLoader(filepath, encoding="utf-8").load()
    chunks = splitter.split_documents(docs)
    all_chunks.extend(chunks)
    print("  已加载: {} ({} 块)".format(filepath, len(chunks)))

vector_store = FAISS.from_documents(all_chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# ========== 提问 ==========
questions = [
    "星辰科技2025年的营收是多少？",
    "星辰科技的远程办公政策是怎样的？",
    "入职4年的员工有多少天年假？",
    "星辰科技未来的重点发展方向是什么？",
]

for question in questions:
    print("\n" + "=" * 60)
    print("问题:", question)
    print("=" * 60)

    # ---- 步骤 1: 检索 ----
    print("\n[1] 检索相关文档...")
    docs = retriever.invoke(question)
    for i, doc in enumerate(docs):
        print("  文档{}: {}...".format(i+1, doc.page_content[:60]))

    # ---- 步骤 2: 增强 ----
    print("\n[2] 构建增强提示...")
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = """你是一个基于知识库回答问题的助手。
请根据以下参考资料回答用户的问题。
如果参考资料不足以回答问题，请如实说明。

参考资料：
{context}

问题：{question}

请用中文回答：""".format(context=context, question=question)

    # ---- 步骤 3: 生成 ----
    print("[3] 生成回答...")
    response = llm.invoke([("human", prompt)])
    print("\n回答:", response.content)

# ========== 对比总结 ==========
print("\n\n" + "=" * 60)
print("阶段 6 总结")
print("=" * 60)
print("""
  RAG 的三个核心步骤：
    R - Retrieval（检索）：从知识库中找到相关文档
    A - Augmented（增强）：将文档拼接到 prompt 中
    G - Generation（生成）：LLM 基于文档回答
  
  相比纯 LLM，RAG 的优势：
    + 回答基于事实，减少幻觉
    + 知识可随时更新（换文档即可）
    + 可溯源（知道答案来自哪篇文档）
  
  下一阶段（进阶）：对话式 RAG、重排序、查询改写
""")
