"""
阶段 4: 嵌入 & 向量存储 (Embeddings + Vector Store)
====================================================
目标：将文本转换为向量，存入向量数据库

核心概念：
  - Embedding（嵌入）：把文本变成一串固定长度的数字（向量）
  - 语义相近的文本 → 向量距离也相近
  - Vector Store（向量数据库）：存储向量并支持快速相似性搜索
"""

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

# 使用 nomic-embed-text 作为嵌入模型（274MB，本地运行）
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)

print("=" * 60)
print("阶段 4: 嵌入 & 向量存储")
print("=" * 60)

# ========== 1. 把文本变成向量 ==========
print("\n[1] Embedding 演示：文本 → 向量")
print("-" * 40)

texts_for_test = [
    "今天天气真好，适合出去散步",
    "明天可能会下雨，记得带伞",
    "星辰科技是一家AI公司，专注于自然语言处理",
]

vectors = embeddings.embed_documents(texts_for_test)
for text, vec in zip(texts_for_test, vectors):
    print("文本: {} -> 向量维度: {}, 前5个值: {}".format(
        text[:20], len(vec), [round(v, 4) for v in vec[:5]]
    ))

# 计算向量相似度（余弦相似度）
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("\n向量相似度对比：")
sim_12 = cosine_similarity(vectors[0], vectors[1])  # 天气相关的两句
sim_13 = cosine_similarity(vectors[0], vectors[2])  # 天气 vs AI公司
print("  天气 vs 下雨: {:.4f} (语义相近，相似度高)".format(sim_12))
print("  天气 vs AI公司: {:.4f} (语义不同，相似度低)".format(sim_13))

# ========== 2. 构建 FAISS 向量库 ==========
print("\n[2] 构建 FAISS 向量库")
print("-" * 40)

# 加载文档并分块
loader = TextLoader("data/long_document.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=40,
    separators=["\n\n", "\n", "。", "，", " ", ""],
)
chunks = splitter.split_documents(documents)
print("文档分块数:", len(chunks))

# 构建向量库（这步会自动调用 embedding 模型）
vector_store = FAISS.from_documents(chunks, embeddings)
print("向量库构建完成！")

# 保存到本地（以后可以直接加载，不用重新计算）
vector_store.save_local("data/faiss_index")
print("向量库已保存到 data/faiss_index/")

# ========== 3. 验证：加载已保存的向量库 ==========
print("\n[3] 加载向量库并验证")
print("-" * 40)

loaded_store = FAISS.load_local(
    "data/faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,  # 仅本地安全时使用
)
print("向量库加载成功！")
print("向量总数:", loaded_store.index.ntotal)

# ========== 总结 ==========
print("\n" + "=" * 60)
print("阶段 4 总结")
print("=" * 60)
print("""
  Embedding 模型：将文本转为向量
    - nomic-embed-text（本地，274MB）
    - text-embedding-ada-002（OpenAI API）
    - bge-large-zh（中文专用，本地）
  
  向量数据库（Vector Store）：
    - FAISS：本地内存式，简单快速
    - Chroma：本地持久化，功能丰富
    - Pinecone / Milvus：生产级分布式方案
  
  下一阶段：从向量库中检索最相关的文档片段
""")
