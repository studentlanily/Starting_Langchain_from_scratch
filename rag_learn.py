from langchain_classic.chains.qa_with_sources.map_reduce_prompt import question_prompt_template
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import vector_stores

#链接LLM
llm = ChatOpenAI(
    model="qwen2.5:32b",
    base_url="http://localhost:11434/v1",
    api_key="default",
    temperature=0.7,
)

# # 手动检索
# with open("data/company_policy.md","r",encoding="utf-8") as f:
#     document = f.read()

# prompt = f"""请根据以下参考资料回答用户的问题。
#
# 参考资料：
# {document}
#
# 问题：星辰科技公司的年假政策是什么？入职4年的员工有多少天年假？
#
# 请用中文回答："""
#
# response = llm.invoke([("human", prompt)])
# print(response.content)

# # 文档加载
# loader1 = TextLoader("data/readme.txt",encoding = "utf-8")
# docs1 = loader1.load()
# #返回List[Document]
# # Document对象只有两个字段：
# # page_content——文本
# # metadata——字典，记录来源路径，页码等
#
# #loader2 = Docx2txtLoader()
#
# doc =docs1[0]
# print("来源：",doc.metadata.get("source"))
# print("内容:",doc.page_content)

# # 文本分块
# # 概念
# # 为什么不直接把整篇文档扔给 LLM？
# # 1.上下文窗口有限（qwen2.5:7b 约 32K tokens）
# # 2.检索时，小块比整篇文档更容易命中你想找的那个知识点
# # 3.比如你问"年假"，只有文档里「年假政策」那一小段有用，整篇文档塞进去反而稀释了注意力
# text="""星辰科技公司成立于2018年，总部位于深圳。
# 公司年营收达到50亿元人民币。
# 核心产品是星云大模型（XingYun-LLM）。
# 年假政策：正式员工每年15天，满3年额外加3天。
# 培训补贴：每人每年8000元。"""
#
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=50,
#     chunk_overlap=10,   #重叠字数：确保句子的完整性，确保关键信息不被切碎
#     separators=["\n","。"," ","，"],
# )
#
# chunks = splitter.split_text(text)
# print("分成了 {} 块：\n".format(len(chunks)))
# for i, chunk in enumerate(chunks):
#     print("块{} ({}字): {}".format(i+1, len(chunk), chunk))
#
# #  RecursiveCharacterTextSplitter 的逻辑：
# # 先按 \n 切，再把小段合并不超过 chunk_size=50 字
# # 块1：两行合并（38字）
# # 块2：两行合并（49字）——年假政策被保留在同一个 chunk 里
# # 块3：最后一行（15字）
# # 效果：
# # 分成了 3 块：
# #
# # 块1 (38字): 星辰科技公司成立于2018年，总部位于深圳。
# # 公司年营收达到50亿元人民币。
# # 块2 (49字): 核心产品是星云大模型（XingYun-LLM）。
# # 年假政策：正式员工每年15天，满3年额外加3天。
# # 块3 (15字): 培训补贴：每人每年8000元。

# 阶段4：嵌入&向量存储
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)

# texts = [
#     "今天天气真好，适合出去散步",
#     "明天可能会下雨，记得带伞",
#     "星辰科技年营收达到50亿元",
# ]
#
# vectors = embeddings.embed_documents(texts)
#
# for text,vec in zip(texts,vectors):
#     print("文本：{}->向量维度：{}，前5个值：{}".format(
#
#         text,len(vec),[round(v,4) for v in vec[:5]]
#     ))

# 阶段 5：向量存储与检索

# # 1. 加载文档并分块
# loader = TextLoader("data/long_document.txt",encoding="utf-8")
# documents = loader.load()
#
# splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=40)
# chunks = splitter.split_documents(documents)
# print("分成了{}块".format(len(chunks)))
#
# # 2. 构建 FAISS 向量库
# vector_stores = FAISS.from_documents(chunks,embeddings)
# print("向量库构建完成！共有{}个向量".format(vector_stores.index.ntotal))#vector_store.index.ntotal是直接访问底层 FAISS 索引的属性
# # 你的 chunk 文本 → embedding 变向量 → 存入 FAISS。查的时候，把你的问题也变向量，FAISS 用余弦相似度或欧几里得距离快速找到最接近的 N 个 chunk。
# # FAISS 的特点是：
# # 快——百万级向量检索毫秒级
# # 内存中运行（也可以保存到硬盘）
# # 只存向量，不存原始文本（原始文本由你自己维护）
#
# # 3. 检索
# query = "星辰科技2025年的营收是多少？"
# results = vector_stores.similarity_search(query,k=2)
# #返回前 2 个最匹配的 chunk
#
# print("\n问题:", query)
# print("检索结果:")
# for i, doc in enumerate(results):
#     print("--- 结果 {} ---".format(i+1))
#     print(doc.page_content[:100])

#阶段 6：完整的 RAG 流水线！
# 核心
# 检索可能有噪音，没关系。我们把检索到的内容 + 问题一起交给 LLM，让 LLM 来综合判断、提取正确答案。
# 这就是 RAG 的最后一步 G（Generation）

# ===== 1. 构建向量库 =====
# loader = TextLoader("data/long_document.txt",encoding='utf-8')
#chunks =splitter.split_documents(loader.load())
#vector_store = FAISS.from_documents(chunks,embeddings)
all_chunks = []
splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=40)
for path in ["data/long_document.txt", "data/company_policy.md"]:
    loader = TextLoader(path, encoding="utf-8")
    chunks = splitter.split_documents(loader.load())
    all_chunks.extend(chunks)
    print("已加载: {} ({}块)".format(path, len(chunks)))
vector_store = FAISS.from_documents(all_chunks, embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k":5})
#把向量库包装成一个统一的「检索器对象」。后面可以直接 retriever.invoke(query) 来检索。其中K表示「返回最相关的几个结果」的个数。

# ===== 2. RAG 问答 =====
question = "星辰科技2025年的营收是多少？"
question2 = "星辰科技的专利奖励制度是怎样的？"

# R - 检索
docs = retriever.invoke(question)
context = "\n\n".join([doc.page_content for doc in docs])
#这行是把检索到的多个文档片段拼接成一个完整的上下文字符串。遍历 docs（检索到的文档列表），提取每个文档的文本内容，组装成列表。用两个换行符 \n\n 把列表里的所有文本串起来，变成一个长字符串。
docs2 = retriever.invoke(question2)
context2 = "\n\n".join([doc.page_content for doc in docs2])

#A+G - 增强并并生成
prompt = f"""请根据以下参考资料回答用户的问题。
如果参考资料不足以回答，请如实说明。

参考资料：
{context}

问题：{question}

请用中文回答："""

prompt2 = f"""请根据以下参考资料回答用户的问题。
如果参考资料不足以回答，请如实说明。

参考资料：
{context2}

问题：{question2}

请用中文回答："""

response = llm.invoke([("human",prompt)])
print("回答:", response.content)

response2 = llm.invoke([("human", prompt2)])
print("回答2:", response2.content)
