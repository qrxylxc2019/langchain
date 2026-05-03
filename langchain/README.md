# LangChain 后端服务

这是一个基于 Flask + LangChain 的后端项目，适合零基础学习 LangChain。

## 📁 项目结构

```
langchain/
├── app.py              # Flask 主服务
├── learn_langchain.py  # 学习脚本（零基础入门）
├── requirements.txt    # Python 依赖
├── .env               # 环境变量配置
└── README.md          # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd langchain
pip install -r requirements.txt
```

### 2. 运行学习脚本（推荐先运行这个）

```bash
python learn_langchain.py
```

这个脚本会逐步演示：
- **Model**: 如何调用大语言模型
- **Prompt**: 如何使用提示词模板
- **Chain**: 如何将组件串联起来
- **Memory**: 如何让 AI 记住对话上下文

### 3. 启动 Flask 服务

```bash
python app.py
```

服务启动后访问 http://127.0.0.1:5000 查看 API 文档。

## 📡 API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务状态 |
| `/api/chat` | POST | 普通对话 |
| `/api/chat/stream` | POST | 流式对话 |
| `/api/prompt-demo` | POST | Prompt 模板演示 |
| `/api/clear-memory` | POST | 清空对话记忆 |

### 请求示例

```bash
# 普通对话
curl -X POST http://127.0.0.1:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好"}'

# Prompt 演示
curl -X POST http://127.0.0.1:5000/api/prompt-demo \
  -H "Content-Type: application/json" \
  -d '{"topic": "量子计算", "style": "通俗易懂"}'
```

## 📚 LangChain 核心概念

### 1. Model (模型)
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="your-api-key",
    base_url="https://api.deepseek.com/v1"
)
response = llm.invoke("你好")
```

### 2. Prompt (提示词)
```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}"),
    ("human", "{question}"),
])
messages = prompt.format_messages(role="医生", question="我感冒了怎么办")
```

### 3. Chain (链)
```python
# 管道语法
chain = prompt | llm
result = chain.invoke({"role": "医生", "question": "我感冒了"})
```

### 4. Memory (记忆)
```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)
```
