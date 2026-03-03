# MCP (Model Context Protocol)

## 什么是 MCP？

MCP（Model Context Protocol）是 Anthropic 推出的一个开放协议，目标是标准化 AI 模型和外部工具/数据之间的交互方式。

它解决了一个核心问题：让 AI 助手知道如何"调用外部能力"，包括文件系统操作、数据库查询、API 调用等，而不是把所有工具的细节都写死在 prompt 里。

## 核心概念

### Server（服务端）
MCP Server 是一个独立进程，暴露一组"能力"给 AI 客户端使用。每个 Server 可以提供：
- **Tools**：AI 可以主动调用的函数（类似 function calling）
- **Resources**：AI 可以读取的数据源（类似只读文件）
- **Prompts**：预定义的提示模板

### Client（客户端）
MCP Client 嵌入在 AI 应用（如 Claude Desktop、Cursor）里，负责发现并调用 Server 的能力。

### Transport
MCP 支持两种传输方式：
1. **stdio**：通过标准输入输出通信，适合本地工具
2. **SSE（Server-Sent Events）**：HTTP 流，适合远程服务

## 和 Function Calling 的区别

| 特性 | Function Calling | MCP |
|------|-----------------|-----|
| 工具定义位置 | 每次请求时传入 | Server 独立维护 |
| 跨应用复用 | ❌ | ✅ |
| 标准化 | 各厂商不同 | 统一协议 |
| 工具发现 | 手动 | 自动（list_tools） |

## 为什么 MCP 对 Agent 开发很重要？

1. **解耦工具和模型**：工具可以独立开发、部署、版本管理。
2. **生态复用**：一套 MCP Server 可以被 Claude、GPT、本地模型共用。
3. **安全边界**：MCP Server 可以做权限控制，限制 AI 可以访问的资源范围。

## 使用示例

```python
# 在 Claude Desktop 的 config.json 里配置 MCP Server
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/my/project"]
    }
  }
}
```

## 学了 MCP 之后能做什么？

- 把你自己写的 Python 工具包装成 MCP Server，供任何 AI 客户端使用
- 接入数据库、内部系统、私有 API
- 在 Phase 4 里：把 MCP Server 作为 Multi-Agent 系统的"工具层"
