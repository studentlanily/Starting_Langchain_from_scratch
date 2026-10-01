"""
阶段 5: 检索 (Retrieval)
========================
目标：从向量库中检索最相关的文档片段

这是 RAG 的 "R"（Retrieval）环节！
给定一个问题，找到知识库中最相关的几段文本。
"""

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)

# 加载之前保存的向量库
vector_store = FAISS.load_local(
    "data/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)

print("=" * 60)
print("阶段 5: 检索 (Retrieval)")
print("=" * 60)

# ========== 1. 基础语义检索 ==========
print("\n[1] 基础语义检索 similarity_search")
print("-" * 40)

query = "星辰科技的年营收是多少？"
results = vector_store.similarity_search(query, k=2)

print("问题:", query)
print("检索到 {} 条结果:\n".format(len(results)))
for i, doc in enumerate(results):
    print("--- 结果 {} (来源: {}) ---".format(i+1, doc.metadata.get("source")))
    print(doc.page_content)
    print()

# ========== 2. 带分数的检索 ==========
print("[2] 带分数的检索 similarity_search_with_score")
print("-" * 40)

results_with_score = vector_store.similarity_search_with_score(query, k=3)
print("问题:", query)
for i, (doc, score) in enumerate(results_with_score):
    print("  结果 {}: 距离={:.4f}, 内容前50字: {}...".format(
        i+1, score, doc.page_content[:50]
    ))
print("  (距离越小 = 越相关)")

# ========== 3. MMR 检索（多样性） ==========
print("\n[3] MMR 检索（增加多样性，避免结果太相似）")
print("-" * 40)

query2 = "星辰科技有哪些技术优势？"
mmr_results = vector_store.max_marginal_relevance_search(query2, k=3, fetch_k=10)
print("问题:", query2)
for i, doc in enumerate(mmr_results):
    print("  结果 {}: {}...".format(i+1, doc.page_content[:80]))

# ========== 4. 创建 Retriever 对象 ==========
print("\n[4] 创建 Retriever 对象（供后续 RAG 流水线使用）")
print("-" * 40)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
print("Retriever 已创建！")
print("  search_type=similarity, k=3")

# 测试 retriever
retrieved_docs = retriever.invoke("星辰科技的创始人是谁？")
print("\n测试检索 '星辰科技的创始人是谁？':")
for doc in retrieved_docs:
    print("  -", doc.page_content[:80])

# ========== 总结 ==========
print("\n" + "=" * 60)
print("阶段 5 总结")
print("=" * 60)
print("""
  检索方式：
    - similarity_search: 最基础的相似度检索
    - similarity_search_with_score: 带距离分数的检索
    - max_marginal_relevance_search: 兼顾相关性和多样性
  
  Retriever 对象是 RAG 流水线的关键组件：
    - 将向量库包装成 retriever 接口
    - 支持 search_type（similarity / mmr）
    - 支持 k（返回结果数）
  
  下一阶段：把检索 + 生成组合成完整的 RAG 流水线！
""")
