"""
Flask + LangChain 后端服务
用于学习 LangChain 的核心组件：Model、Prompt、Chain 等
LangChain 1.x 版本
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os

# LangChain 1.x 相关导入
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 加载环境变量
load_dotenv()

app = Flask(__name__)
CORS(app)  # 允许跨域，方便前端调用

# ==================== 初始化 LangChain 组件 ====================

# 1. 初始化 LLM (大语言模型)
# 阿里云百炼 DeepSeek-v4-pro，兼容 OpenAI API 格式
llm = ChatOpenAI(
    model="deepseek-v4-pro",         # 阿里云百炼 DeepSeek-v4-pro
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_API_BASE"),
    temperature=0.7,                 # 创造性程度 (0-2)
    max_tokens=2048,
    extra_body={"enable_thinking": False},  # 启用深度思考（阿里云百炼特有参数）
)

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
        "message": "LangChain + Flask 后端服务已启动",
        "endpoints": [
            "/api/chat          - 普通对话",
            "/api/chat/stream   - 流式对话",
            "/api/prompt-demo   - Prompt 模板演示",
            "/api/clear-memory  - 清空对话记忆",
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


# ==================== 运行服务 ====================

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    print(f"🚀 服务启动在 http://127.0.0.1:{port}")
    print("📖 API 文档: http://127.0.0.1:{port}/")
    app.run(host="0.0.0.0", port=port, debug=debug)
