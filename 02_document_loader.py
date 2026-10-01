"""
阶段 2: 文档加载 (Document Loading)
===================================
目标：学会从多种来源加载文档内容

RAG 的第一步是获取知识来源。LangChain 提供了多种加载器。

本节课涵盖：
  1. TextLoader   - 加载 txt 和 md 文件
  2. PyPDFLoader  - 加载 PDF 文件
  3. WebBaseLoader - 加载网页内容
"""

from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
import os

print("=" * 60)
print("阶段 2: 文档加载演示")
print("=" * 60)

# ========== 1. TextLoader：加载文本文件 ==========
print("\n[1] TextLoader - 加载 txt 文件")
print("-" * 40)

loader = TextLoader("data/readme.txt", encoding="utf-8")
docs = loader.load()
doc = docs[0]
print("来源:", doc.metadata.get("source"))
print("内容长度:", len(doc.page_content), "字符")
print("前100字:", doc.page_content[:100])

print("\n[1b] TextLoader - 加载 markdown 文件")
print("-" * 40)

loader_md = TextLoader("data/company_policy.md", encoding="utf-8")
docs_md = loader_md.load()
doc_md = docs_md[0]
print("来源:", doc_md.metadata.get("source"))
print("内容长度:", len(doc_md.page_content), "字符")

# ========== 2. PyPDFLoader：加载 PDF 文件 ==========
print("\n[2] PyPDFLoader - 加载 PDF 文件")
print("-" * 40)

# 先生成一个示例 PDF
pdf_path = "data/sample.pdf"
if not os.path.exists(pdf_path):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, text="XingChen AI - 2025 Annual Summary", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(200, 10, text="", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 10, text="Revenue reached 5 billion RMB, up 35% YoY.\nEmployee count exceeded 2,000.\nLaunched 3 new AI products, obtained 15 patents.")
    pdf.output(pdf_path)
    print("已创建示例 PDF:", pdf_path)

loader_pdf = PyPDFLoader(pdf_path)
docs_pdf = loader_pdf.load()
print("PDF 页数:", len(docs_pdf))
for i, page in enumerate(docs_pdf):
    print("  第{}页内容: {}".format(i+1, page.page_content.replace("\n", " ")))

# ========== 3. WebBaseLoader：加载网页 ==========
print("\n[3] WebBaseLoader - 加载网页")
print("-" * 40)

try:
    loader_web = WebBaseLoader("https://example.com")
    docs_web = loader_web.load()
    print("网页标题内容:", docs_web[0].page_content[:150])
except Exception as e:
    print("网页加载失败（可能需要联网）:", e)

# ========== 总结 ==========
print("\n" + "=" * 60)
print("阶段 2 总结")
print("=" * 60)
print("""
  Document 的结构：
    - page_content: 文档文本内容
    - metadata:     文档元数据（来源、页码等）
  
  常见的加载器：
    - TextLoader       -> txt, md, csv 等纯文本
    - PyPDFLoader      -> PDF 文件
    - WebBaseLoader    -> 网页
    - CSVLoader        -> CSV 文件
    - DirectoryLoader  -> 批量加载目录下所有文档
  
  下一阶段：文本分块 —— 把长文档切成适合检索的小块
""")
