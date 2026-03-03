"""
Phase 3 - Step 3: RAG Agent (LangGraph + LLM + Tool Calling)

Usage:
    uv run python phase-3/agent.py

Architecture:
    User message
        → classify (rule: does it need knowledge?)
        → [if yes] llm_with_tools (LLM decides to call search_notes)
        → tool_executor (run search_notes → get passages)
        → llm_answer (LLM generates final answer with retrieved context)
        → [if no]  llm_answer (direct LLM answer)

Key learning:
    Phase 1: You manually decided when to call tools.
    Phase 3: The LLM itself decides when to call search_notes.
             This is the core of a real RAG Agent.
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Any

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI
from pydantic import BaseModel, Field

from retriever import SEARCH_TOOL_SCHEMA, run_search_tool

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "MiniMax-M2.5")


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class RAGState(BaseModel):
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)
    # Whether the current turn needs the knowledge base
    needs_retrieval: bool = False


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是一个聪明的学习助手，专门解答 AI Agent 开发相关问题。

你有一个工具：search_notes —— 可以检索用户的个人学习笔记（内容包括 MCP、Function Calling、LangGraph、Embeddings、Agent 设计模式等）。

规则：
1. 如果用户的问题和 AI Agent 开发知识相关，优先调用 search_notes 检索笔记再回答。
2. 如果是闲聊、问候等，直接回答，不需要检索。
3. 检索结果中的"相关度"分数越高越可信。
4. 回答时要引用笔记中的具体内容，注明来源文件。"""


def llm_with_tools(state: RAGState) -> dict[str, Any]:
    """
    Let the LLM decide whether to call search_notes.

    This is the CORE difference from Phase 1 and Phase 2:
    - Phase 1: YOU write 'if tool_call: run_tool()'
    - Phase 3: LLM decides 'I want to call search_notes' based on function calling
    """
    client = get_client()

    messages_for_api = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in state.messages:
        role = getattr(msg, "type", "user")
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        content = getattr(msg, "content", str(msg))
        messages_for_api.append({"role": role, "content": content})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages_for_api,
        tools=[SEARCH_TOOL_SCHEMA],
        tool_choice="auto",  # LLM自己决定是否调用工具
    )

    choice = response.choices[0]
    msg = choice.message

    # Build a serializable message dict (not langchain object)
    ai_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}

    if choice.finish_reason == "tool_calls" and msg.tool_calls:
        # LLM wants to call a tool — store the tool call info for the executor
        ai_msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
        return {
            "messages": [ai_msg],
            "needs_retrieval": True,
        }

    # LLM answered directly (no tool call needed)
    return {
        "messages": [ai_msg],
        "needs_retrieval": False,
    }


def tool_executor(state: RAGState) -> dict[str, Any]:
    """
    Execute tool calls that the LLM requested.

    Notice: this is almost identical to what you wrote manually in Phase 1!
    LangGraph just separates it into a clean node.
    """
    tool_messages = []

    # Find the latest AI message with tool_calls
    for msg in reversed(state.messages):
        raw_tool_calls = getattr(msg, "additional_kwargs", {}).get("tool_calls") or getattr(
            msg, "tool_calls", None
        )
        if not raw_tool_calls:
            # Try if msg is dict-like
            if isinstance(msg, dict) and msg.get("tool_calls"):
                raw_tool_calls = msg["tool_calls"]

        if raw_tool_calls:
            for tc in raw_tool_calls:
                if isinstance(tc, dict):
                    tc_id = tc["id"]
                    name = tc["function"]["name"]
                    args_str = tc["function"]["arguments"]
                else:
                    tc_id = tc.id
                    name = tc.function.name
                    args_str = tc.function.arguments

                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                result = run_search_tool(args)
                tool_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result,
                    "name": name,
                })
            break

    return {"messages": tool_messages}


def llm_answer(state: RAGState) -> dict[str, Any]:
    """
    Generate the final answer after tools have run.
    LLM now has the retrieved passages in its context.
    """
    client = get_client()

    messages_for_api = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in state.messages:
        role = getattr(msg, "type", "user")
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        content = getattr(msg, "content", str(msg))
        extra = getattr(msg, "additional_kwargs", {})

        api_msg: dict[str, Any] = {"role": role, "content": content}
        if extra.get("tool_calls"):
            api_msg["tool_calls"] = extra["tool_calls"]
        if role == "tool":
            api_msg["tool_call_id"] = getattr(msg, "tool_call_id", "")
        messages_for_api.append(api_msg)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages_for_api,
    )
    answer = response.choices[0].message.content
    return {"messages": [{"role": "assistant", "content": answer}]}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def route_after_llm(state: RAGState) -> str:
    """If LLM decided to call a tool, go execute it; otherwise we're done."""
    return "tool_executor" if state.needs_retrieval else END


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------
def build_app():
    g = StateGraph(RAGState)

    g.add_node("llm_with_tools", llm_with_tools)
    g.add_node("tool_executor", tool_executor)
    g.add_node("llm_answer", llm_answer)

    g.set_entry_point("llm_with_tools")
    g.add_conditional_edges(
        "llm_with_tools",
        route_after_llm,
        {"tool_executor": "tool_executor", END: END},
    )
    g.add_edge("tool_executor", "llm_answer")
    g.add_edge("llm_answer", END)

    return g.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------
def main():
    print("Phase 3 RAG Agent - 输入 quit 退出")
    print("提示：可以问我关于 MCP、LangGraph、Function Calling、Embedding 的问题\n")

    app = build_app()
    config = {"configurable": {"thread_id": "rag-demo"}}
    cursor = 0

    while True:
        user_text = input("User: ").strip()
        if user_text.lower() in ("quit", "exit", "q"):
            break

        out = app.invoke(
            {"messages": [{"role": "user", "content": user_text}]},
            config=config,
        )

        for msg in out.get("messages", [])[cursor:]:
            msg_type = getattr(msg, "type", None)
            if msg_type in ("ai", "assistant"):
                content = getattr(msg, "content", "")
                if content:
                    print(f"Agent: {content}")
        cursor = len(out.get("messages", []))


if __name__ == "__main__":
    main()
