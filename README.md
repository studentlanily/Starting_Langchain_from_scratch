# Starting RAG from Scratch

从零学习 RAG（检索增强生成）与 LangChain Agent 的分阶段示例项目。

## RAG 分阶段脚本

| 文件 | 阶段 |
|------|------|
| `01_concept_demo.py` | RAG 概念演示 |
| `02_document_loader.py` | 文档加载（Document Loader） |
| `03_text_splitter.py` | 文本切分（Text Splitter） |
| `04_vector_store.py` | 向量库构建（FAISS） |
| `05_retrieval.py` | 检索（Retrieval） |
| `06_rag_pipeline.py` | 完整 RAG 流水线（检索 + 生成） |
| `07_advanced_rag.py` | 进阶 RAG |

## LangChain Agent

`learnLangChain/agent.py` — 基于 `create_agent` 的智能体示例，包含工具调用（`@tool`）与动态模型中间件（`wrap_model_call`）。

## 环境

- Python 3.12 + LangChain
- 本地模型服务：[Ollama](https://ollama.com)（`nomic-embed-text` 向量模型、`qwen2.5` 系列对话模型）
- 向量库：FAISS

```bash
# 启动 Ollama 后运行示例
python 06_rag_pipeline.py
```
