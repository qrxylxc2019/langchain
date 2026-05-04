"use client";

import { useState, useRef, useEffect } from "react";

// 消息类型定义
interface Source {
  source: string;
  page: number;
  similarity: number;
  content: string;
}

interface Message {
  id: string;
  role: "user" | "ai";
  content: string;
  isLoading?: boolean;
  sources?: Source[];
  ragEnabled?: boolean;
}

// API 配置
const API_BASE_URL = "http://127.0.0.1:5000";

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [ragEnabled, setRagEnabled] = useState(false);
  const [kbStats, setKbStats] = useState<{ total_chunks: number; sources: string[] } | null>(null);
  const [providerInfo, setProviderInfo] = useState<{ provider: string; provider_name: string; model: string } | null>(null);
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

  // 获取知识库状态和后端配置
  useEffect(() => {
    fetchKbStats();
    fetchProviderInfo();
  }, []);

  const fetchKbStats = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/rag/stats`);
      const data = await res.json();
      if (data.success) {
        setKbStats({ total_chunks: data.total_chunks, sources: data.sources });
      }
    } catch {
      // 忽略错误
    }
  };

  const fetchProviderInfo = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/`);
      const data = await res.json();
      if (data.provider) {
        setProviderInfo({
          provider: data.provider,
          provider_name: data.provider_name,
          model: data.model,
        });
      }
    } catch {
      // 忽略错误
    }
  };

  // 加载知识库
  const loadKnowledgeBase = async () => {
    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE_URL}/api/rag/load`, { method: "POST" });
      const data = await res.json();
      if (data.success) {
        setKbStats({ total_chunks: data.stats.total_chunks, sources: data.stats.sources });
        alert(`知识库加载完成！\n共 ${data.stats.total_chunks} 个文档块\n来源: ${data.stats.sources.join(", ")}`);
      } else {
        alert("加载失败: " + data.error);
      }
    } catch (error) {
      alert("连接失败，请检查后端服务");
    } finally {
      setIsLoading(false);
    }
  };

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
      const endpoint = ragEnabled ? "/api/chat/rag" : "/api/chat";
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
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
              ? {
                  ...msg,
                  content: data.reply,
                  isLoading: false,
                  sources: data.sources,
                  ragEnabled: data.rag_enabled,
                }
              : msg
          )
        );
      } else {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === loadingMessage.id
              ? { ...msg, content: "出错了: " + (data.error || "请稍后重试"), isLoading: false }
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
        <p>
          基于 LangChain + Flask + RAG
          {providerInfo && (
            <span className="provider-badge">
              {providerInfo.provider_name} / {providerInfo.model}
            </span>
          )}
        </p>
        <div className="header-actions">
          {messages.length > 0 && (
            <button className="clear-button" onClick={clearChat}>
              清空对话
            </button>
          )}
        </div>
      </header>

      {/* 工具栏 */}
      <div className="toolbar">
        <div className="rag-toggle">
          <label className="switch">
            <input
              type="checkbox"
              checked={ragEnabled}
              onChange={(e) => setRagEnabled(e.target.checked)}
            />
            <span className="slider"></span>
          </label>
          <span className="rag-label">
            📚 知识库检索 {ragEnabled ? "已开启" : "已关闭"}
          </span>
        </div>

        <div className="kb-info">
          {kbStats ? (
            <span className="kb-stats">
              知识库: {kbStats.total_chunks} 块文档
              {kbStats.sources.length > 0 && (
                <span className="kb-sources"> ({kbStats.sources.join(", ")})</span>
              )}
            </span>
          ) : (
            <span className="kb-stats">知识库: 未连接</span>
          )}
          <button
            className="load-kb-button"
            onClick={loadKnowledgeBase}
            disabled={isLoading}
          >
            {isLoading ? "加载中..." : "🔄 加载文档"}
          </button>
        </div>
      </div>

      {/* 消息列表 */}
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="empty-state">
            <h2>👋 欢迎使用 AI 聊天助手</h2>
            <p>在下方输入框发送消息，开始与 AI 对话</p>
            <div className="rag-hint">
              <p>💡 开启「知识库检索」可以让 AI 基于 PDF 文档回答</p>
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.role}`}>
              <div className="avatar">
                {msg.role === "user" ? "👤" : "🤖"}
              </div>
              <div className="message-wrapper">
                <div
                  className={`message-content ${msg.isLoading ? "loading" : ""}`}
                >
                  {msg.content || (msg.isLoading ? "思考中" : "")}
                </div>
                {/* 显示 RAG 来源 */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources-box">
                    <div className="sources-title">📚 参考来源</div>
                    {msg.sources.map((source, idx) => (
                      <div key={idx} className="source-item">
                        <span className="source-badge">
                          {source.source} (第{source.page}页)
                        </span>
                        <span className="source-similarity">
                          相似度: {source.similarity}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
                {msg.ragEnabled === false && msg.role === "ai" && !msg.isLoading && (
                  <div className="rag-off-hint">
                    ⚠️ 知识库为空，使用普通对话模式
                  </div>
                )}
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
            placeholder={ragEnabled ? "输入消息，AI将基于知识库回答..." : "输入消息，按 Enter 发送..."}
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
