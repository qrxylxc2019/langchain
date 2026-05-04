"""
RAG (Retrieval-Augmented Generation) 知识库引擎
===============================================
功能：
1. 加载 PDF 文档
2. 文本分割（Chunking）
3. 向量化（使用 OpenAI Embedding）
4. 存储到 SQLite（本地向量数据库）
5. 相似度检索
6. 结合 LLM 进行问答

技术栈：
- LangChain 1.x
- SQLite + sqlite-vec（向量扩展）
- OpenAI Embedding API
"""

import os
import sqlite3
import json
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from dotenv import load_dotenv

# LangChain 相关导入
from langchain_openai import ChatOpenAI
try:
    from langchain_community.embeddings import DashScopeEmbeddings
except ImportError:
    DashScopeEmbeddings = None
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# 加载环境变量
load_dotenv()

# ==================== 配置 ====================

PDF_DIR = r"D:\ai\ai agent项目\langchain\pdf"
DB_PATH = r"D:\ai\ai agent项目\langchain\rag\rag.db"

# Embedding 模型配置
EMBEDDING_MODEL = "text-embedding-v3"  # 阿里云百炼 embedding 模型
EMBEDDING_DIM = 1024  # text-embedding-v3 的维度

# 文本分割配置
CHUNK_SIZE = 500      # 每个文本块的大小（字符数）
CHUNK_OVERLAP = 100   # 文本块之间的重叠大小


# ==================== 数据模型 ====================

@dataclass
class DocumentChunk:
    """文档片段（Chunk）数据类"""
    id: str
    content: str
    source: str           # 来源文件
    page: int             # 页码
    chunk_index: int      # 在文档中的块索引
    embedding: Optional[List[float]] = None
    created_at: Optional[str] = None


# ==================== SQLite 向量数据库 ====================

class VectorStore:
    """
    基于 SQLite 的向量存储
    使用简单的向量相似度计算（余弦相似度）
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """初始化数据库表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 创建文档块表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                page INTEGER,
                chunk_index INTEGER,
                embedding TEXT,  -- JSON 格式的向量
                created_at TEXT
            )
        """)

        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source ON document_chunks(source)
        """)

        conn.commit()
        conn.close()
        print(f"✅ 数据库初始化完成: {self.db_path}")

    def add_chunks(self, chunks: List[DocumentChunk]):
        """添加文档块到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for chunk in chunks:
            embedding_json = json.dumps(chunk.embedding) if chunk.embedding else None
            cursor.execute("""
                INSERT OR REPLACE INTO document_chunks
                (id, content, source, page, chunk_index, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.id,
                chunk.content,
                chunk.source,
                chunk.page,
                chunk.chunk_index,
                embedding_json,
                chunk.created_at or datetime.now().isoformat()
            ))

        conn.commit()
        conn.close()
        print(f"✅ 已存储 {len(chunks)} 个文档块")

    def get_all_chunks(self) -> List[DocumentChunk]:
        """获取所有文档块"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM document_chunks")
        rows = cursor.fetchall()
        conn.close()

        chunks = []
        for row in rows:
            embedding = json.loads(row[5]) if row[5] else None
            chunks.append(DocumentChunk(
                id=row[0],
                content=row[1],
                source=row[2],
                page=row[3],
                chunk_index=row[4],
                embedding=embedding,
                created_at=row[6]
            ))
        return chunks

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        向量相似度搜索
        使用余弦相似度计算
        """
        chunks = self.get_all_chunks()

        if not chunks:
            return []

        # 计算余弦相似度
        results = []
        for chunk in chunks:
            if chunk.embedding:
                similarity = self._cosine_similarity(query_embedding, chunk.embedding)
                results.append({
                    "chunk": chunk,
                    "similarity": similarity
                })

        # 按相似度排序
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:top_k]

    def clear_all(self):
        """清空所有数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM document_chunks")
        conn.commit()
        conn.close()
        print("✅ 数据库已清空")

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        import math

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


# ==================== PDF 加载器 ====================

class PDFLoader:
    """PDF 文档加载器"""

    @staticmethod
    def load_pdf(pdf_path: str) -> List[Dict[str, Any]]:
        """
        加载 PDF 文件，返回每页的内容
        返回: [{"page": 1, "content": "..."}, ...]
        """
        try:
            import pymupdf  # fitz
        except ImportError:
            print("❌ 请先安装 pymupdf: pip install pymupdf")
            return []

        documents = []
        doc = pymupdf.open(pdf_path)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                documents.append({
                    "page": page_num + 1,
                    "content": text.strip()
                })

        doc.close()
        print(f"✅ PDF 加载完成: {os.path.basename(pdf_path)}, 共 {len(documents)} 页")
        return documents


# ==================== 文本分割器 ====================

class TextSplitter:
    """文本分割器 - 将长文本分割成小块"""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str, source: str, page: int) -> List[DocumentChunk]:
        """
        将文本分割成重叠的块
        """
        chunks = []

        # 按段落分割，保持语义完整
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]

        current_chunk = ""
        chunk_index = 0

        for paragraph in paragraphs:
            # 如果当前段落加上已有内容不超过 chunk_size，就合并
            if len(current_chunk) + len(paragraph) < self.chunk_size:
                current_chunk += paragraph + "\n"
            else:
                # 保存当前块
                if current_chunk.strip():
                    chunk_id = hashlib.md5(
                        f"{source}_{page}_{chunk_index}".encode()
                    ).hexdigest()

                    chunks.append(DocumentChunk(
                        id=chunk_id,
                        content=current_chunk.strip(),
                        source=source,
                        page=page,
                        chunk_index=chunk_index
                    ))
                    chunk_index += 1

                # 开始新块，保留部分重叠内容
                if len(paragraph) > self.chunk_size:
                    # 段落本身就很长，需要进一步分割
                    for i in range(0, len(paragraph), self.chunk_size - self.chunk_overlap):
                        sub_text = paragraph[i:i + self.chunk_size]
                        if sub_text.strip():
                            chunk_id = hashlib.md5(
                                f"{source}_{page}_{chunk_index}".encode()
                            ).hexdigest()

                            chunks.append(DocumentChunk(
                                id=chunk_id,
                                content=sub_text.strip(),
                                source=source,
                                page=page,
                                chunk_index=chunk_index
                            ))
                            chunk_index += 1
                    current_chunk = ""
                else:
                    current_chunk = paragraph + "\n"

        # 保存最后一个块
        if current_chunk.strip():
            chunk_id = hashlib.md5(
                f"{source}_{page}_{chunk_index}".encode()
            ).hexdigest()

            chunks.append(DocumentChunk(
                id=chunk_id,
                content=current_chunk.strip(),
                source=source,
                page=page,
                chunk_index=chunk_index
            ))

        return chunks


# ==================== Embedding 服务 ====================

class EmbeddingService:
    """文本向量化服务"""

    def __init__(self):
        api_key = os.getenv("DASHSCOPE_API_KEY")

        if not api_key:
            raise ValueError("环境变量 DASHSCOPE_API_KEY 未设置，请检查 .env 文件")

        # 使用 DashScopeEmbeddings (阿里云百炼官方支持)
        if DashScopeEmbeddings is not None:
            self.embeddings = DashScopeEmbeddings(
                model=EMBEDDING_MODEL,
                dashscope_api_key=api_key,
            )
        else:
            # 备选方案：使用 OpenAIEmbeddings 兼容模式
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                api_key=api_key,
                base_url=os.getenv("DASHSCOPE_API_BASE"),
            )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """将文本列表转换为向量"""
        print(f"🔄 正在向量化 {len(texts)} 个文本块...")
        vectors = self.embeddings.embed_documents(texts)
        print(f"✅ 向量化完成")
        return vectors

    def embed_query(self, query: str) -> List[float]:
        """将查询文本转换为向量"""
        return self.embeddings.embed_query(query)


# ==================== RAG 引擎 ====================

class RAGEngine:
    """
    RAG 知识库引擎
    整合所有组件，提供完整的知识库功能
    """

    def __init__(self):
        self.vector_store = VectorStore(DB_PATH)
        self.pdf_loader = PDFLoader()
        self.text_splitter = TextSplitter()
        self.embedding_service = EmbeddingService()

        # 初始化 LLM
        self.llm = ChatOpenAI(
            model="deepseek-v4-pro",
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_API_BASE"),
            temperature=0.7,
            max_tokens=2048,
            extra_body={"enable_thinking": False},
        )

        # RAG Prompt 模板
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的知识库助手。请根据以下检索到的参考资料回答用户的问题。

参考资料：
{context}

回答要求：
1. 基于参考资料回答，不要编造信息
2. 如果参考资料不足以回答问题，请明确说明
3. 回答要简洁、准确
4. 可以适当引用参考资料中的内容"""),
            ("human", "{question}"),
        ])

    def load_pdfs(self):
        """
        加载 PDF 目录下的所有文档，构建知识库
        """
        print("\n" + "=" * 50)
        print("📚 开始构建知识库")
        print("=" * 50)

        # 获取所有 PDF 文件
        pdf_files = [
            f for f in os.listdir(PDF_DIR)
            if f.lower().endswith('.pdf')
        ]

        if not pdf_files:
            print(f"❌ 在 {PDF_DIR} 中没有找到 PDF 文件")
            return

        print(f"📄 发现 {len(pdf_files)} 个 PDF 文件")

        all_chunks = []

        for pdf_file in pdf_files:
            pdf_path = os.path.join(PDF_DIR, pdf_file)
            print(f"\n📖 正在处理: {pdf_file}")

            # 1. 加载 PDF
            pages = self.pdf_loader.load_pdf(pdf_path)

            # 2. 分割文本
            for page_info in pages:
                chunks = self.text_splitter.split_text(
                    text=page_info["content"],
                    source=pdf_file,
                    page=page_info["page"]
                )
                all_chunks.extend(chunks)
                print(f"   第 {page_info['page']} 页 -> {len(chunks)} 个文本块")

        print(f"\n📊 总共生成 {len(all_chunks)} 个文本块")

        # 3. 向量化
        if all_chunks:
            texts = [chunk.content for chunk in all_chunks]
            embeddings = self.embedding_service.embed_texts(texts)

            # 将向量附加到 chunk
            for chunk, embedding in zip(all_chunks, embeddings):
                chunk.embedding = embedding

            # 4. 存储到数据库
            self.vector_store.add_chunks(all_chunks)

        print("\n✅ 知识库构建完成！")

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        查询知识库

        流程：
        1. 将问题向量化
        2. 在向量数据库中检索最相似的文档块
        3. 将检索结果作为上下文，结合 LLM 生成回答
        """
        print(f"\n🔍 用户问题: {question}")

        # 1. 问题向量化
        query_embedding = self.embedding_service.embed_query(question)

        # 2. 检索相似文档
        results = self.vector_store.search(query_embedding, top_k=top_k)

        if not results:
            return {
                "answer": "知识库中没有找到相关资料，请先加载文档。",
                "sources": [],
                "success": False
            }

        print(f"📑 检索到 {len(results)} 个相关文档块")

        # 3. 构建上下文
        context_parts = []
        sources = []

        for i, result in enumerate(results, 1):
            chunk = result["chunk"]
            similarity = result["similarity"]

            context_parts.append(
                f"[文档 {i}] 来源: {chunk.source} (第 {chunk.page} 页)\n"
                f"相似度: {similarity:.4f}\n"
                f"内容: {chunk.content}\n"
            )

            sources.append({
                "source": chunk.source,
                "page": chunk.page,
                "similarity": round(similarity, 4),
                "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            })

        context = "\n".join(context_parts)

        # 4. 调用 LLM 生成回答
        print("🤖 正在生成回答...")
        messages = self.rag_prompt.format_messages(
            context=context,
            question=question
        )

        response = self.llm.invoke(messages)
        answer = response.content

        return {
            "answer": answer,
            "sources": sources,
            "success": True
        }

    def clear_knowledge_base(self):
        """清空知识库"""
        self.vector_store.clear_all()
        print("✅ 知识库已清空")

    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        chunks = self.vector_store.get_all_chunks()

        sources = set(chunk.source for chunk in chunks)

        return {
            "total_chunks": len(chunks),
            "total_sources": len(sources),
            "sources": list(sources),
            "db_path": DB_PATH
        }


# ==================== 命令行交互 ====================

def interactive_mode():
    """交互式命令行模式"""
    print("\n" + "=" * 50)
    print("🤖 RAG 知识库系统")
    print("=" * 50)

    engine = RAGEngine()

    while True:
        print("\n请选择操作:")
        print("1. 加载 PDF 文档到知识库")
        print("2. 查询知识库")
        print("3. 查看知识库统计")
        print("4. 清空知识库")
        print("5. 退出")

        choice = input("\n请输入选项 (1-5): ").strip()

        if choice == "1":
            engine.load_pdfs()

        elif choice == "2":
            question = input("\n请输入问题: ").strip()
            if question:
                result = engine.query(question)
                print("\n" + "-" * 50)
                print("📝 回答:")
                print(result["answer"])
                print("-" * 50)
                print("📚 参考来源:")
                for source in result.get("sources", []):
                    print(f"  - {source['source']} (第 {source['page']} 页) [相似度: {source['similarity']}]")

        elif choice == "3":
            stats = engine.get_stats()
            print(f"\n📊 知识库统计:")
            print(f"  总文档块数: {stats['total_chunks']}")
            print(f"  来源文档数: {stats['total_sources']}")
            print(f"  文档列表: {', '.join(stats['sources']) if stats['sources'] else '无'}")
            print(f"  数据库路径: {stats['db_path']}")

        elif choice == "4":
            confirm = input("确定要清空知识库吗？(yes/no): ").strip().lower()
            if confirm == "yes":
                engine.clear_knowledge_base()

        elif choice == "5":
            print("👋 再见！")
            break

        else:
            print("❌ 无效选项，请重新选择")


if __name__ == "__main__":
    interactive_mode()
