"""
Phase 4 - Shared Tools

Tools shared by all agents in the team:
    - search_knowledge_base: searches the Phase 3 ChromaDB (notes knowledge base)
    - web_search_mock: simulates a web search (returns hardcoded results for demo)

In a real project:
    - search_knowledge_base → use Phase 3's retriever.py (already built!)
    - web_search_mock → replace with Tavily, SerpAPI, or Bing Search API

Key insight:
    Tools are NOT tied to any specific agent. They live in a shared module.
    Each agent decides independently whether to call them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Reuse Phase 3's retriever
PHASE3_DIR = Path(__file__).parent.parent / "phase-3"
sys.path.insert(0, str(PHASE3_DIR))

try:
    from retriever import search_notes as _search_notes  # type: ignore
    _HAS_PHASE3 = True
except Exception:
    _HAS_PHASE3 = False

EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


# ---------------------------------------------------------------------------
# Tool 1: Search Knowledge Base（复用 Phase 3）
# ---------------------------------------------------------------------------
def search_knowledge_base(query: str, top_k: int = 3) -> str:
    """Search the Phase 3 local notes ChromaDB."""
    if not _HAS_PHASE3:
        return (
            "[knowledge_base] Phase 3 ChromaDB not found. "
            "Run 'uv run python phase-3/ingest.py' first."
        )
    return _search_notes(query, top_k=top_k)


SEARCH_KB_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "在本地笔记知识库中检索 AI Agent 相关知识（MCP、LangGraph、Function Calling 等）",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词或问题"},
                "top_k": {"type": "integer", "description": "返回结果数量", "default": 3},
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Tool 2: Web Search（mock for demo, replace with real API）
# ---------------------------------------------------------------------------
_MOCK_WEB_RESULTS = {
    "default": (
        "搜索结果（模拟）：\n"
        "1. AI Agent 技术综述 - 2024年 Agent 技术快速发展，主要方向包括 RAG、Multi-Agent、Tool Use 等。\n"
        "2. LangGraph 实战案例 - 多家企业已落地 LangGraph 生产环境，典型场景：客服、代码审查、研报生成。\n"
        "3. MCP 生态发展 - Anthropic 开源 MCP 后，超过 500 个 Server 被社区贡献。\n"
    ),
}

def web_search(query: str) -> str:
    """Simulated web search. Replace with real search API in production."""
    for keyword, result in _MOCK_WEB_RESULTS.items():
        if keyword in query.lower():
            return result
    return _MOCK_WEB_RESULTS["default"]


WEB_SEARCH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "搜索互联网获取最新信息（当知识库中没有相关内容时使用）",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索词"},
            },
            "required": ["query"],
        },
    },
}


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------
ALL_TOOLS = [SEARCH_KB_SCHEMA, WEB_SEARCH_SCHEMA]

def run_tool(name: str, args: dict) -> str:
    if name == "search_knowledge_base":
        return search_knowledge_base(args.get("query", ""), args.get("top_k", 3))
    elif name == "web_search":
        return web_search(args.get("query", ""))
    return f"[tools] Unknown tool: {name}"
