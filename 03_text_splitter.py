"""
阶段 3: 文本分块 (Text Splitting)
==================================
目标：理解为什么需要分块，掌握不同分块策略

为什么分块？
  - LLM 上下文窗口有限（如 qwen2.5:7b 是 32K tokens）
  - 检索时，小块比整篇文档更容易匹配到相关段落
  - 小块可以提高检索精度（命中率）

常见的分块方式：
  1. CharacterTextSplitter      - 按字符数切割（简单粗暴）
  2. RecursiveCharacterTextSplitter - 递归按分隔符切割（效果最好，最常用）
  3. TokenTextSplitter         - 按 token 数切割（精确但慢）
"""

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

# 加载长文档
with open("data/long_document.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("原始文档长度:", len(text), "字符")
print("=" * 60)

# ========== 1. CharacterTextSplitter ==========
print("\n[1] CharacterTextSplitter (按字符数切割)")
print("-" * 40)

splitter1 = CharacterTextSplitter(
    chunk_size=100,       # 每块最多 100 字符
    chunk_overlap=20,     # 块之间重叠 20 字符
    separator="。\n",      # 按句号+换行切割
)
chunks1 = splitter1.split_text(text)
print("分块数:", len(chunks1))
for i, chunk in enumerate(chunks1[:3]):
    print("  块{}: {}...".format(i+1, chunk[:60]))

# ========== 2. RecursiveCharacterTextSplitter (推荐) ==========
print("\n[2] RecursiveCharacterTextSplitter (递归切割，推荐)")
print("-" * 40)

splitter2 = RecursiveCharacterTextSplitter(
    chunk_size=200,       # 每块约 200 字符
    chunk_overlap=40,     # 重叠 40 字符
    separators=["\n\n", "\n", "。", "，", " ", ""],  # 优先级从高到低
)
chunks2 = splitter2.split_text(text)
print("分块数:", len(chunks2))
for i, chunk in enumerate(chunks2):
    print("  块{} ({}字): {}".format(i+1, len(chunk), chunk[:80] + "..."))

# ========== 3. 使用文档对象分块 ==========
print("\n[3] 分块时保留 metadata")
print("-" * 40)

from langchain_community.document_loaders import TextLoader

loader = TextLoader("data/long_document.txt", encoding="utf-8")
documents = loader.load()

chunks_docs = splitter2.split_documents(documents)
print("文档分块数:", len(chunks_docs))
print("第一块 metadata:", chunks_docs[0].metadata)
print("第一块内容前60字:", chunks_docs[0].page_content[:60])

# ========== 总结 ==========
print("\n" + "=" * 60)
print("阶段 3 总结")
print("=" * 60)
print("""
  RecursiveCharacterTextSplitter 是最常用的分块方式：
    - chunk_size:    每块的目标大小（字符数或 token 数）
    - chunk_overlap: 块之间重叠的部分，保持上下文连贯
    - separators:    分隔符优先级列表
  
  经验法则：
    - 中文通常 chunk_size=200~500，overlap=20~50
    - 英文通常 chunk_size=500~1000，overlap=100~200
    - 根据文档类型和检索需求调整
  
  下一阶段：把分块后的文本变成向量，存入向量数据库
""")
