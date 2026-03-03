"""
Phase 3 - Step 2: Retrieval tool

This module exposes:
    - search_notes(query, top_k) -> str   (plain function for testing)
    - SEARCH_TOOL_SCHEMA                  (JSON schema for LLM tool calling)
    - run_search_tool(args_dict) -> str   (dispatcher used by agent.py)
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "notes"

# Must match the embedding function used in ingest.py
_embed_fn = DefaultEmbeddingFunction()


def _get_collection() -> chromadb.Collection:
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return chroma.get_collection(COLLECTION_NAME, embedding_function=_embed_fn)



def search_notes(query: str, top_k: int = 3) -> str:
    """Search the local notes ChromaDB by semantic similarity.

    Returns a formatted string of top-k relevant passages with their sources.
    ChromaDB auto-embeds the query using the same local embedding function as ingest.
    """
    collection = _get_collection()
    results = collection.query(
        query_texts=[query],  # ChromaDB auto-embeds using _embed_fn
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        return "知识库中没有找到相关内容。"

    parts = []
    for doc, meta, dist in zip(docs, metas, distances):
        source = meta.get("source", "unknown")
        similarity = round(1 - dist, 3)  # ChromaDB L2 distance → rough similarity
        parts.append(f"[来源: {source} | 相关度: {similarity}]\n{doc}")

    return "\n\n---\n\n".join(parts)


# --- Tool schema for OpenAI function calling ---
SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_notes",
        "description": (
            "在个人笔记知识库中进行语义搜索，返回最相关的笔记片段。"
            "当用户询问关于 MCP、Function Calling、LangGraph、Embedding、Agent 设计模式等学习内容时使用此工具。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询，用自然语言描述你要找的内容",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的相关片段数量，默认 3，最大 5",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}


def run_search_tool(args: dict) -> str:
    """Dispatcher called by the agent when the LLM requests this tool."""
    query = args.get("query", "")
    top_k = int(args.get("top_k", 3))
    return search_notes(query, top_k=top_k)


if __name__ == "__main__":
    # Quick test
    print(search_notes("MCP 是什么？"))
