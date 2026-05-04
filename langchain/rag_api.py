"""
RAG 知识库 API 服务
==================
基于 Flask 的 REST API，提供知识库管理、查询等功能

接口列表:
- POST /api/rag/load       - 加载 PDF 文档到知识库
- POST /api/rag/query      - 查询知识库
- GET  /api/rag/stats      - 获取知识库统计信息
- POST /api/rag/clear      - 清空知识库
- GET  /api/rag/health     - 健康检查
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import sys

# 加载环境变量
load_dotenv()

# 确保可以导入 rag 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.rag_engine import RAGEngine

app = Flask(__name__)
CORS(app)  # 允许跨域

# 初始化 RAG 引擎（单例模式）
rag_engine = None


def get_engine() -> RAGEngine:
    """获取 RAG 引擎实例（懒加载）"""
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()
    return rag_engine


# ==================== API 路由 ====================

@app.route("/")
def index():
    """首页，显示 API 文档"""
    return jsonify({
        "status": "running",
        "service": "RAG 知识库 API",
        "endpoints": {
            "/api/rag/load": {
                "method": "POST",
                "description": "加载 PDF 文档到知识库",
                "body": None,
                "response": {
                    "success": True,
                    "message": "知识库构建完成",
                    "stats": {
                        "total_chunks": 100,
                        "total_sources": 2
                    }
                }
            },
            "/api/rag/query": {
                "method": "POST",
                "description": "查询知识库",
                "body": {"question": "你的问题", "top_k": 5},
                "response": {
                    "success": True,
                    "answer": "回答内容",
                    "sources": [
                        {
                            "source": "文档.pdf",
                            "page": 1,
                            "similarity": 0.95,
                            "content": "相关内容摘要..."
                        }
                    ]
                }
            },
            "/api/rag/stats": {
                "method": "GET",
                "description": "获取知识库统计信息",
                "response": {
                    "total_chunks": 100,
                    "total_sources": 2,
                    "sources": ["文档1.pdf", "文档2.pdf"],
                    "db_path": "..."
                }
            },
            "/api/rag/clear": {
                "method": "POST",
                "description": "清空知识库",
                "response": {
                    "success": True,
                    "message": "知识库已清空"
                }
            },
            "/api/rag/health": {
                "method": "GET",
                "description": "健康检查",
                "response": {
                    "status": "healthy",
                    "engine_ready": True
                }
            }
        }
    })


@app.route("/api/rag/load", methods=["POST"])
def load_documents():
    """
    加载 PDF 文档到知识库
    自动扫描 langchain/pdf 目录下的所有 PDF 文件
    """
    try:
        engine = get_engine()
        engine.load_pdfs()

        stats = engine.get_stats()

        return jsonify({
            "success": True,
            "message": "知识库构建完成",
            "stats": {
                "total_chunks": stats["total_chunks"],
                "total_sources": stats["total_sources"],
                "sources": stats["sources"]
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/rag/query", methods=["POST"])
def query_knowledge_base():
    """
    查询知识库
    请求体: {"question": "你的问题", "top_k": 5}
    """
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({
            "success": False,
            "error": "请提供 question 字段"
        }), 400

    question = data.get("question", "").strip()
    top_k = data.get("top_k", 5)

    if not question:
        return jsonify({
            "success": False,
            "error": "问题不能为空"
        }), 400

    try:
        engine = get_engine()
        result = engine.query(question, top_k=top_k)

        return jsonify({
            "success": result["success"],
            "answer": result["answer"],
            "sources": result.get("sources", [])
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/rag/stats", methods=["GET"])
def get_stats():
    """获取知识库统计信息"""
    try:
        engine = get_engine()
        stats = engine.get_stats()

        return jsonify({
            "success": True,
            "total_chunks": stats["total_chunks"],
            "total_sources": stats["total_sources"],
            "sources": stats["sources"],
            "db_path": stats["db_path"]
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/rag/clear", methods=["POST"])
def clear_knowledge_base():
    """清空知识库"""
    try:
        engine = get_engine()
        engine.clear_knowledge_base()

        return jsonify({
            "success": True,
            "message": "知识库已清空"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/api/rag/health", methods=["GET"])
def health_check():
    """健康检查"""
    try:
        engine = get_engine()
        return jsonify({
            "status": "healthy",
            "engine_ready": True
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


# ==================== 运行服务 ====================

if __name__ == "__main__":
    port = int(os.getenv("RAG_API_PORT", 5001))
    debug = os.getenv("RAG_API_DEBUG", "True").lower() == "true"

    print(f"🚀 RAG 知识库 API 启动在 http://127.0.0.1:{port}")
    print(f"📖 API 文档: http://127.0.0.1:{port}/")
    print("\n可用接口:")
    print(f"  POST http://127.0.0.1:{port}/api/rag/load   - 加载文档")
    print(f"  POST http://127.0.0.1:{port}/api/rag/query  - 查询知识库")
    print(f"  GET  http://127.0.0.1:{port}/api/rag/stats   - 查看统计")
    print(f"  POST http://127.0.0.1:{port}/api/rag/clear  - 清空知识库")

    app.run(host="0.0.0.0", port=port, debug=debug)
