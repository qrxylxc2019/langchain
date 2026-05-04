"""
RAG (Retrieval-Augmented Generation) 学习脚本
=============================================
适合零基础学习 RAG 的完整流程，从文档加载到问答

RAG 核心流程：
1. 加载文档 (Load)
2. 分割文本 (Split)
3. 向量化 (Embed)
4. 存储 (Store)
5. 检索 (Retrieve)
6. 生成回答 (Generate)

运行方式: python learn_rag.py
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ============================================================
# 第一部分：文档加载 (Document Loading)
# ============================================================

def learn_document_loading():
    """学习如何加载 PDF 文档"""
    print("=" * 60)
    print("📚 第一部分：文档加载 (Document Loading)")
    print("=" * 60)

    from rag.rag_engine import PDFLoader

    pdf_dir = r"D:\ai\ai agent项目\langchain\pdf"

    # 获取所有 PDF 文件
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

    print(f"\n📁 PDF 目录: {pdf_dir}")
    print(f"📄 发现 {len(pdf_files)} 个 PDF 文件:")
    for f in pdf_files:
        print(f"   - {f}")

    if pdf_files:
        # 加载第一个 PDF
        pdf_path = os.path.join(pdf_dir, pdf_files[0])
        print(f"\n🔄 正在加载: {pdf_files[0]}")

        pages = PDFLoader.load_pdf(pdf_path)

        print(f"\n📖 加载结果:")
        print(f"   总页数: {len(pages)}")

        if pages:
            print(f"\n📄 第一页内容预览 (前 200 字符):")
            print(f"   {pages[0]['content'][:200]}...")


# ============================================================
# 第二部分：文本分割 (Text Splitting)
# ============================================================

def learn_text_splitting():
    """学习如何将长文本分割成小块"""
    print("\n" + "=" * 60)
    print("📚 第二部分：文本分割 (Text Splitting)")
    print("=" * 60)

    from rag.rag_engine import TextSplitter

    # 示例长文本
    sample_text = """
    人工智能（Artificial Intelligence，简称 AI）是指由人制造出来的系统所表现出来的智能。
    通常人工智能是指通过普通计算机程序来呈现人类智能的技术。

    机器学习是人工智能的一个分支，它让计算机能够从数据中学习，而不需要明确编程。
    深度学习是机器学习的一个子集，使用神经网络来模拟人脑的工作方式。

    自然语言处理（NLP）是人工智能和语言学领域的分支学科，
    研究如何实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
    """

    print("\n📝 示例文本:")
    print(f"   总长度: {len(sample_text)} 字符")

    # 创建分割器
    splitter = TextSplitter(chunk_size=100, chunk_overlap=20)

    print(f"\n✂️ 分割参数:")
    print(f"   块大小: {splitter.chunk_size}")
    print(f"   重叠大小: {splitter.chunk_overlap}")

    # 分割文本
    chunks = splitter.split_text(sample_text, source="示例文档", page=1)

    print(f"\n📦 分割结果: 共 {len(chunks)} 个文本块")

    for i, chunk in enumerate(chunks, 1):
        print(f"\n   块 {i}:")
        print(f"   ID: {chunk.id}")
        print(f"   长度: {len(chunk.content)} 字符")
        print(f"   内容: {chunk.content[:80]}...")


# ============================================================
# 第三部分：向量化 (Embedding)
# ============================================================

def learn_embedding():
    """学习如何将文本转换为向量"""
    print("\n" + "=" * 60)
    print("📚 第三部分：向量化 (Embedding)")
    print("=" * 60)

    from rag.rag_engine import EmbeddingService

    print("\n🤔 什么是 Embedding？")
    print("   Embedding 是将文本转换为高维向量的技术")
    print("   语义相似的文本，其向量距离也更近")

    # 初始化 Embedding 服务
    embedding_service = EmbeddingService()

    # 示例文本
    texts = [
        "人工智能是计算机科学的一个分支",
        "机器学习是人工智能的重要技术",
        "今天天气很好，适合去公园散步"
    ]

    print(f"\n📝 示例文本:")
    for i, text in enumerate(texts, 1):
        print(f"   {i}. {text}")

    # 向量化
    print("\n🔄 正在向量化...")
    embeddings = embedding_service.embed_texts(texts)

    print(f"\n📊 向量化结果:")
    for i, (text, embedding) in enumerate(zip(texts, embeddings), 1):
        print(f"   文本 {i}: 向量维度 = {len(embedding)}")
        print(f"   向量前 5 个值: {[round(x, 4) for x in embedding[:5]]}")

    # 计算相似度
    print("\n📐 文本相似度计算:")

    def cosine_similarity(a, b):
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0

    sim_1_2 = cosine_similarity(embeddings[0], embeddings[1])
    sim_1_3 = cosine_similarity(embeddings[0], embeddings[2])

    print(f"   '人工智能...' vs '机器学习...' 相似度: {sim_1_2:.4f} (应该较高)")
    print(f"   '人工智能...' vs '今天天气...' 相似度: {sim_1_3:.4f} (应该较低)")


# ============================================================
# 第四部分：向量存储 (Vector Store)
# ============================================================

def learn_vector_store():
    """学习如何使用 SQLite 存储向量"""
    print("\n" + "=" * 60)
    print("📚 第四部分：向量存储 (Vector Store)")
    print("=" * 60)

    from rag.rag_engine import VectorStore, DocumentChunk

    db_path = r"D:\ai\ai agent项目\langchain\rag\test.db"

    print(f"\n💾 数据库路径: {db_path}")

    # 创建向量存储
    store = VectorStore(db_path)

    # 创建示例文档块
    chunks = [
        DocumentChunk(
            id="chunk_1",
            content="人工智能是计算机科学的一个分支",
            source="AI入门.pdf",
            page=1,
            chunk_index=0,
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
        ),
        DocumentChunk(
            id="chunk_2",
            content="机器学习是人工智能的重要技术",
            source="AI入门.pdf",
            page=2,
            chunk_index=1,
            embedding=[0.15, 0.25, 0.35, 0.45, 0.55]
        ),
        DocumentChunk(
            id="chunk_3",
            content="今天天气很好，适合去公园散步",
            source="日记.pdf",
            page=1,
            chunk_index=0,
            embedding=[0.9, 0.8, 0.7, 0.6, 0.5]
        ),
    ]

    print(f"\n📦 准备存储 {len(chunks)} 个文档块")

    # 存储
    store.add_chunks(chunks)

    # 查询
    print("\n🔍 执行相似度搜索...")
    query_embedding = [0.12, 0.22, 0.32, 0.42, 0.52]  # 接近 chunk_1 和 chunk_2
    results = store.search(query_embedding, top_k=2)

    print(f"\n📑 搜索结果 (Top 2):")
    for i, result in enumerate(results, 1):
        chunk = result["chunk"]
        print(f"   {i}. [{chunk.source} 第 {chunk.page} 页]")
        print(f"      内容: {chunk.content}")
        print(f"      相似度: {result['similarity']:.4f}")

    # 清理测试数据库
    os.remove(db_path)
    print("\n✅ 测试完成，已清理测试数据库")


# ============================================================
# 第五部分：完整 RAG 流程
# ============================================================

def learn_full_rag():
    """学习完整的 RAG 流程"""
    print("\n" + "=" * 60)
    print("📚 第五部分：完整 RAG 流程演示")
    print("=" * 60)

    from rag.rag_engine import RAGEngine

    print("\n🔄 RAG 流程:")
    print("   1. 加载 PDF 文档")
    print("   2. 分割文本为小块")
    print("   3. 向量化文本块")
    print("   4. 存储到 SQLite 数据库")
    print("   5. 接收用户问题")
    print("   6. 问题向量化")
    print("   7. 检索相似文档块")
    print("   8. 结合 LLM 生成回答")

    print("\n" + "-" * 60)
    print("💡 提示: 运行 rag_engine.py 的交互模式来体验完整功能")
    print("   命令: python -m rag.rag_engine")
    print("-" * 60)


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("🎓 RAG (检索增强生成) 零基础学习脚本")
    print("本脚本会逐步演示 RAG 的核心流程\n")

    # 按顺序学习各个部分
    learn_document_loading()
    print("\n")

    learn_text_splitting()
    print("\n")

    learn_embedding()
    print("\n")

    learn_vector_store()
    print("\n")

    learn_full_rag()
    print("\n")

    print("=" * 60)
    print("🎉 学习完成！")
    print("=" * 60)
    print("\n下一步建议:")
    print("  1. 运行 python rag/rag_engine.py 体验交互式知识库")
    print("  2. 运行 python rag_api.py 启动 API 服务")
    print("  3. 查看 rag/rag_engine.py 源码深入学习")
