"""
LangChain 学习脚本
适合零基础入门，逐步学习各个组件
LangChain 1.x 版本
运行方式: python learn_langchain.py
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ============================================================
# 第一部分：Model (模型)
# ============================================================

def learn_model():
    """学习如何使用 LangChain 调用大模型"""
    print("=" * 50)
    print("📚 第一部分：Model (模型)")
    print("=" * 50)

    from langchain_openai import ChatOpenAI

    # 初始化模型
    # model: 阿里云百炼 DeepSeek-v4-pro
    # temperature: 控制创造性，0=保守，2=很有创意
    # extra_body: 额外参数，enable_thinking 启用深度思考
    llm = ChatOpenAI(
        model="deepseek-v4-pro",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_API_BASE"),
        temperature=0.7,
        extra_body={"enable_thinking": True},
    )

    # 最简单的调用方式
    print("\n1. 直接调用模型：")
    response = llm.invoke("你好，请介绍一下你自己")
    print(f"AI回复: {response.content}\n")

    # 流式输出
    print("2. 流式输出（逐字显示）：")
    print("AI回复: ", end="", flush=True)
    for chunk in llm.stream("讲一个短笑话"):
        print(chunk.content, end="", flush=True)
    print("\n")


# ============================================================
# 第二部分：Prompt (提示词)
# ============================================================

def learn_prompt():
    """学习如何使用 Prompt 模板"""
    print("=" * 50)
    print("📚 第二部分：Prompt (提示词)")
    print("=" * 50)

    from langchain_core.prompts import ChatPromptTemplate

    # 方式1：从字符串创建模板
    template1 = ChatPromptTemplate.from_template(
        "请用{style}的风格写一段关于{topic}的介绍"
    )
    messages1 = template1.format_messages(style="幽默", topic="编程")
    print(f"\n1. 字符串模板生成的消息：\n{messages1}\n")

    # 方式2：从消息列表创建模板（推荐）
    template2 = ChatPromptTemplate.from_messages([
        ("system", "你是一个{role}，擅长{skill}。"),
        ("human", "请帮我{task}"),
    ])
    messages2 = template2.format_messages(
        role="Python专家",
        skill="数据分析",
        task="写一个读取CSV文件的代码"
    )
    print(f"2. 消息列表模板生成的消息：\n{messages2}\n")


# ============================================================
# 第三部分：Chain (链)
# ============================================================

def learn_chain():
    """学习如何使用 Chain 组合组件"""
    print("=" * 50)
    print("📚 第三部分：Chain (链)")
    print("=" * 50)

    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatOpenAI(
        model="deepseek-v4-pro",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_API_BASE"),
        temperature=0.7,
        extra_body={"enable_thinking": True},
    )

    # 创建 Prompt 模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个翻译助手，将中文翻译成英文。"),
        ("human", "请翻译：{text}"),
    ])

    # LangChain 1.x 推荐：使用管道语法 (LCEL)
    print("\n1. 使用 LCEL 管道语法（推荐）：")
    chain = prompt | llm
    result = chain.invoke({"text": "你好，世界"})
    print(f"翻译结果: {result.content}\n")


# ============================================================
# 第四部分：Memory (记忆)
# ============================================================

def learn_memory():
    """学习如何使用 Memory 让 AI 记住对话"""
    print("=" * 50)
    print("📚 第四部分：Memory (记忆)")
    print("=" * 50)

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, AIMessage

    llm = ChatOpenAI(
        model="deepseek-v4-pro",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url=os.getenv("DASHSCOPE_API_BASE"),
        temperature=0.7,
        extra_body={"enable_thinking": True},
    )

    # 手动管理对话历史（LangChain 1.x 推荐方式）
    history = []

    print("\n开始对话：\n")

    def chat_with_history(user_input: str) -> str:
        """带历史记录的对话"""
        messages = list(history)  # 复制历史消息
        messages.append(HumanMessage(content=user_input))

        response = llm.invoke(messages)
        reply = response.content

        # 保存到历史
        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=reply))

        return reply

    print("用户: 我叫小明")
    response1 = chat_with_history("我叫小明")
    print(f"AI: {response1}\n")

    print("用户: 我叫什么名字？")
    response2 = chat_with_history("我叫什么名字？")
    print(f"AI: {response2}\n")

    print("查看历史消息数量：", len(history))


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("🎓 LangChain 零基础学习脚本")
    print("本脚本会逐步演示 LangChain 的核心组件\n")

    # 按顺序学习各个组件
    learn_model()
    print("\n")

    learn_prompt()
    print("\n")

    learn_chain()
    print("\n")

    learn_memory()
    print("\n")

    print("=" * 50)
    print("🎉 学习完成！建议打开代码文件 learn_langchain.py 仔细阅读注释")
    print("=" * 50)
