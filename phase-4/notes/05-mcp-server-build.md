# 构建你自己的 MCP Server

## 什么时候需要自己写 MCP Server？

- 你有内部工具/数据库，想让任何 AI 客户端都能调用
- 你在 Phase 4 写的 Agent 逻辑，想复用给 Claude Desktop / Cursor
- 你想把 `search_knowledge_base` 发布成标准 MCP Server，让团队共享

## 最简单的 MCP Server（stdlib 版）

```python
# mcp_server.py
from mcp.server.stdio import StdioServerSession
from mcp import Server, Tool
import asyncio

app = Server("my-tools")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="search_notes",
            description="搜索本地笔记知识库",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> str:
    if name == "search_notes":
        from retriever import search_notes
        return search_notes(arguments["query"])
    raise ValueError(f"Unknown tool: {name}")

async def main():
    async with StdioServerSession(app) as session:
        await session.run()

if __name__ == "__main__":
    asyncio.run(main())
```

## 在 Claude Desktop 使用

在 `claude_desktop_config.json` 里添加：

```json
{
  "mcpServers": {
    "my-agent-tools": {
      "command": "uv",
      "args": ["run", "python", "mcp_server.py"],
      "cwd": "/path/to/agent-programming-101"
    }
  }
}
```

重启 Claude Desktop，就能在聊天框里通过 `<tool_use>` 调用你的工具了。

## Phase 4 的 MCP 化思路

```
现在（Phase 4）         未来（MCP化后）
───────────────────    ──────────────────────────
tools.py               → MCP Server (独立进程)
researcher.py (内嵌)   → Claude Desktop 里的 Researcher
supervisor.py          → 任何 MCP Client 可调用
```

把 `search_knowledge_base` 包装成 MCP Server，
让 Claude Desktop、Cursor、或你自己写的任何 Agent 都能直接调用它——
这就是 MCP 的价值：**工具只写一次，到处可用**。

## 安装 MCP SDK

```bash
uv add mcp
```

然后参考官方文档：https://modelcontextprotocol.io/docs/python
