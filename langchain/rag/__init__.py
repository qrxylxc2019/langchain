"""
RAG 知识库模块

使用方式:
    from rag.rag_engine import RAGEngine

    engine = RAGEngine()
    engine.load_pdfs()  # 加载 PDF 文档
    result = engine.query("你的问题")  # 查询知识库
"""

from .rag_engine import RAGEngine, VectorStore, PDFLoader, TextSplitter, EmbeddingService

__all__ = ["RAGEngine", "VectorStore", "PDFLoader", "TextSplitter", "EmbeddingService"]
