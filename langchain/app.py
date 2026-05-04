"""
Flask + LangChain 后端服务
用于学习 LangChain 的核心组件：Model、Prompt、Chain 等
LangChain 1.x 版本
新增：RAG 知识库检索功能
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os
import sys

# LangChain 1.x 相关导入
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 加载环境变量
load_dotenv()

# ==================== 配置管理 ====================

PROVIDER = os.getenv("PROVIDER", "deepseek").lower()

if PROVIDER == "dashscope":
    API_KEY = os.getenv("DASHSCOPE_API_KEY")
    API_BASE = os.getenv("DASHSCOPE_API_BASE")
    MODEL = os.getenv("DASHSCOPE_MODEL", "deepseek-v4-pro")
    EMBEDDING_MODEL = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")
    PROVIDER_NAME = "阿里云百炼"
else:
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    API_BASE = os.getenv("DEEPSEEK_API_BASE")
    MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    EMBEDDING_MODEL = None  # DeepSeek 不提供 embedding，需要单独处理
    PROVIDER_NAME = "DeepSeek"

print(f"🔧 当前 AI 提供商: {PROVIDER_NAME} ({PROVIDER})")
print(f"🤖 使用模型: {MODEL}")

# 确保可以导入 rag 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rag.rag_engine import RAGEngine

app = Flask(__name__)
CORS(app)  # 允许跨域，方便前端调用

# ==================== 初始化 RAG 引擎 ====================
rag_engine = None

def get_rag_engine():
    """获取 RAG 引擎实例（懒加载）"""
    global rag_engine
    if rag_engine is None:
        rag_engine = RAGEngine()
    return rag_engine

# RAG Prompt 模板（带知识库上下文的对话）
rag_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个专业的 AI 助手。请根据以下知识库资料回答用户问题。

知识库资料：
{context}

回答要求：
1. 优先基于知识库资料回答
2. 如果知识库资料不足，结合你的知识补充回答
3. 回答要简洁、准确、有帮助
4. 可以适当引用资料来源"""),
    ("human", "{question}"),
])

# ==================== 初始化 LangChain 组件 ====================

# 初始化 LLM (大语言模型)
llm_kwargs = {
    "model": MODEL,
    "api_key": API_KEY,
    "base_url": API_BASE,
    "temperature": 0.7,
    "max_tokens": 2048,
}

# 阿里云百炼特有参数
if PROVIDER == "dashscope":
    llm_kwargs["extra_body"] = {"enable_thinking": False}

llm = ChatOpenAI(**llm_kwargs)

# 2. 手动管理对话历史（LangChain 1.x 推荐方式）
conversation_history = []


def build_messages(user_input: str) -> list:
    """构建包含历史记录的消息列表"""
    messages = []
    for msg in conversation_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "ai":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=user_input))
    return messages


# ==================== API 路由 ====================

@app.route("/")
def index():
    """首页，检查服务是否运行"""
    return jsonify({
        "status": "running",
        "provider": PROVIDER,
        "provider_name": PROVIDER_NAME,
        "model": MODEL,
        "message": f"LangChain + Flask + RAG 后端服务已启动 ({PROVIDER_NAME})",
        "endpoints": [
            "/api/chat                - 普通对话",
            "/api/chat/stream         - 流式对话",
            "/api/chat/rag            - RAG 知识库对话",
            "/api/prompt-demo         - Prompt 模板演示",
            "/api/clear-memory        - 清空对话记忆",
            "/api/rag/load            - 加载 PDF 到知识库",
            "/api/rag/query           - 查询知识库",
            "/api/rag/stats           - 知识库统计",
            "/api/rag/clear           - 清空知识库",
        ]
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    普通对话接口
    请求体: {"message": "你好"}
    返回: {"reply": "..."}
    """
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "消息不能为空"}), 400

    try:
        # 构建包含历史的消息列表
        messages = build_messages(user_message)

        # 调用模型
        response = llm.invoke(messages)
        reply = response.content

        # 保存到历史记录
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "ai", "content": reply})

        # 限制历史长度（保留最近 20 轮）
        while len(conversation_history) > 40:
            conversation_history.pop(0)
            conversation_history.pop(0)

        return jsonify({
            "reply": reply,
            "success": True
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """
    流式对话接口（SSE 格式）
    前端可以逐字显示 AI 回复
    """
    from flask import Response, stream_with_context

    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "消息不能为空"}), 400

    def generate():
        """生成流式响应"""
        try:
            # 使用 llm.stream 获取流式输出
            messages = build_messages(user_message)
            full_reply = ""
            for chunk in llm.stream(messages):
                content = chunk.content if hasattr(chunk, "content") else ""
                if content:
                    full_reply += content
                    yield f"data: {content}\n\n"

            # 保存到历史记录
            conversation_history.append({"role": "user", "content": user_message})
            conversation_history.append({"role": "ai", "content": full_reply})

            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.route("/api/prompt-demo", methods=["POST"])
def prompt_demo():
    """
    Prompt 模板演示
    展示如何使用 LangChain 的 PromptTemplate
    """
    data = request.get_json()
    topic = data.get("topic", "人工智能")
    style = data.get("style", "幽默")

    try:
        # 创建 Prompt 模板
        template = ChatPromptTemplate.from_messages([
            ("system", "你是一位{style}风格的专家，擅长用{style}的方式解释各种概念。"),
            ("human", "请用{style}的风格解释一下：{topic}"),
        ])

        # 填充模板并调用模型
        messages = template.format_messages(topic=topic, style=style)
        response = llm.invoke(messages)

        return jsonify({
            "reply": response.content,
            "topic": topic,
            "style": style,
            "success": True
        })
    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/clear-memory", methods=["POST"])
def clear_memory():
    """清空对话记忆"""
    global conversation_history
    conversation_history = []
    return jsonify({"message": "对话记忆已清空", "success": True})


# ==================== RAG 知识库接口 ====================

@app.route("/api/chat/rag", methods=["POST"])
def chat_with_rag():
    """
    RAG 知识库对话接口
    会先检索知识库，再结合 LLM 生成回答
    请求体: {"message": "问题", "top_k": 5}
    返回: {"reply": "...", "sources": [...]}
    """
    data = request.get_json()
    user_message = data.get("message", "").strip()
    top_k = data.get("top_k", 5)

    if not user_message:
        return jsonify({"error": "消息不能为空", "success": False}), 400

    try:
        engine = get_rag_engine()

        # 1. 查询知识库
        rag_result = engine.query(user_message, top_k=top_k)

        if not rag_result["success"]:
            # 知识库为空，使用普通对话
            messages = build_messages(user_message)
            response = llm.invoke(messages)
            reply = response.content

            conversation_history.append({"role": "user", "content": user_message})
            conversation_history.append({"role": "ai", "content": reply})

            return jsonify({
                "reply": reply,
                "sources": [],
                "rag_enabled": False,
                "success": True
            })

        # 2. 构建带上下文的 Prompt
        context_parts = []
        for i, source in enumerate(rag_result.get("sources", []), 1):
            context_parts.append(
                f"[资料 {i}] 来源: {source['source']} (第 {source['page']} 页)\n"
                f"{source['content'][:300]}..."
            )
        context = "\n\n".join(context_parts)

        # 3. 调用 LLM 生成回答
        messages = rag_chat_prompt.format_messages(
            context=context,
            question=user_message
        )
        response = llm.invoke(messages)
        reply = response.content

        # 4. 保存对话历史
        conversation_history.append({"role": "user", "content": user_message})
        conversation_history.append({"role": "ai", "content": reply})

        # 限制历史长度
        while len(conversation_history) > 40:
            conversation_history.pop(0)
            conversation_history.pop(0)

        return jsonify({
            "reply": reply,
            "sources": rag_result.get("sources", []),
            "rag_enabled": True,
            "success": True
        })

    except Exception as e:
        return jsonify({"error": str(e), "success": False}), 500


@app.route("/api/rag/load", methods=["POST"])
def rag_load_documents():
    """加载 PDF 文档到知识库"""
    try:
        engine = get_rag_engine()
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
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/rag/query", methods=["POST"])
def rag_query():
    """查询知识库"""
    data = request.get_json()
    question = data.get("question", "").strip()
    top_k = data.get("top_k", 5)

    if not question:
        return jsonify({"success": False, "error": "问题不能为空"}), 400

    try:
        engine = get_rag_engine()
        result = engine.query(question, top_k=top_k)

        return jsonify({
            "success": result["success"],
            "answer": result["answer"],
            "sources": result.get("sources", [])
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/rag/stats", methods=["GET"])
def rag_stats():
    """获取知识库统计信息"""
    try:
        engine = get_rag_engine()
        stats = engine.get_stats()

        return jsonify({
            "success": True,
            "total_chunks": stats["total_chunks"],
            "total_sources": stats["total_sources"],
            "sources": stats["sources"],
            "db_path": stats["db_path"]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/rag/clear", methods=["POST"])
def rag_clear():
    """清空知识库"""
    try:
        engine = get_rag_engine()
        engine.clear_knowledge_base()
        return jsonify({"success": True, "message": "知识库已清空"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ==================== 运行服务 ====================

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    print(f"🚀 服务启动在 http://127.0.0.1:{port}")
    print("📖 API 文档: http://127.0.0.1:{port}/")
    app.run(host="0.0.0.0", port=port, debug=debug)
