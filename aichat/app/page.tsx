"use client";

import { useState, useRef, useEffect } from "react";

// 消息类型定义
interface Message {
  id: string;
  role: "user" | "ai";
  content: string;
  isLoading?: boolean;
}

// API 配置
const API_BASE_URL = "http://127.0.0.1:5000";

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // 自动调整输入框高度
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        120
      )}px`;
    }
  }, [input]);

  // 发送消息
  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
    };

    const loadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: "ai",
      content: "",
      isLoading: true,
    };

    setMessages((prev) => [...prev, userMessage, loadingMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: userMessage.content }),
      });

      const data = await response.json();

      if (data.success) {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === loadingMessage.id
              ? { ...msg, content: data.reply, isLoading: false }
              : msg
          )
        );
      } else {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === loadingMessage.id
              ? { ...msg, content: "出错了，请稍后重试", isLoading: false }
              : msg
          )
        );
      }
    } catch (error) {
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === loadingMessage.id
            ? { ...msg, content: "连接失败，请检查后端服务是否启动", isLoading: false }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  // 处理键盘事件
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // 清空对话
  const clearChat = async () => {
    setMessages([]);
    try {
      await fetch(`${API_BASE_URL}/api/clear-memory`, {
        method: "POST",
      });
    } catch {
      // 忽略错误
    }
  };

  return (
    <div className="chat-container">
      {/* 头部 */}
      <header className="chat-header">
        <h1>🤖 AI 聊天助手</h1>
        <p>基于 LangChain + DeepSeek + Flask</p>
        {messages.length > 0 && (
          <button className="clear-button" onClick={clearChat}>
            清空对话
          </button>
        )}
      </header>

      {/* 消息列表 */}
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="empty-state">
            <h2>👋 欢迎使用 AI 聊天助手</h2>
            <p>在下方输入框发送消息，开始与 AI 对话</p>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={`message ${msg.role}`}
            >
              <div className="avatar">
                {msg.role === "user" ? "👤" : "🤖"}
              </div>
              <div
                className={`message-content ${msg.isLoading ? "loading" : ""}`}
              >
                {msg.content || (msg.isLoading ? "思考中" : "")}
              </div>
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 */}
      <div className="chat-input-area">
        <div className="input-container">
          <textarea
            ref={textareaRef}
            className="chat-input"
            placeholder="输入消息，按 Enter 发送..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={1}
            disabled={isLoading}
          />
          <button
            className="send-button"
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
          >
            {isLoading ? "发送中" : "发送"}
          </button>
        </div>
      </div>
    </div>
  );
}
