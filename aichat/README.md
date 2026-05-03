# AI Chat 前端项目

基于 Next.js + TypeScript 的 AI 聊天应用前端。

## 📁 项目结构

```
aichat/
├── app/
│   ├── layout.tsx      # 根布局
│   ├── page.tsx        # 聊天页面（主页面）
│   └── globals.css     # 全局样式
├── package.json        # 项目依赖
├── tsconfig.json       # TypeScript 配置
├── next.config.mjs     # Next.js 配置
└── README.md          # 本文件
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd aichat
npm install
```

### 2. 启动开发服务器

```bash
npm run dev
```

访问 http://localhost:3000 查看应用。

> ⚠️ 确保后端服务（Flask）已启动在 http://127.0.0.1:5000

## 📝 功能说明

- 💬 **发送消息**: 在输入框输入内容，按 Enter 发送
- 🔄 **Shift+Enter**: 换行
- 🧹 **清空对话**: 点击顶部"清空对话"按钮
- ⏳ **加载状态**: AI 回复时显示"思考中"

## 🔌 API 对接

前端通过 `fetch` 调用后端接口：

```typescript
// 发送消息
const response = await fetch("http://127.0.0.1:5000/api/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ message: "你好" }),
});

const data = await response.json();
console.log(data.reply); // AI 回复内容
```

## 🎨 样式说明

使用纯 CSS 实现，包含：
- 响应式布局
- 消息气泡样式
- 加载动画
- 渐变色头部
