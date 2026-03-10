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
import sys
from typing import Annotated, Any

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI
from pydantic import BaseModel, Field

from retriever import SEARCH_TOOL_SCHEMA, run_search_tool

load_dotenv(override=True)

for stream_name in ("stdin", "stdout", "stderr"):
    stream = getattr(sys, stream_name, None)
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(errors="replace")
        except Exception:
            pass

CHAT_MODEL = os.getenv("CHAT_MODEL", "MiniMax-M2.5")

# Debug: 确认当前使用的配置
print("--- [DEBUG CONFIG] ---")
raw_key = os.getenv("OPENAI_API_KEY", "")
raw_url = os.getenv("OPENAI_BASE_URL", "")
masked_key = f"{raw_key[:12]}...{raw_key[-8:]}" if raw_key else "NONE"
print(f"API_KEY: {masked_key}")
print(f"BASE_URL: {raw_url}")
print(f"MODEL: {CHAT_MODEL}")
print("----------------------\n")


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


def _normalize_role(role: Any) -> str:
    if role == "human":
        return "user"
    if role == "ai":
        return "assistant"
    return role or "user"


def _message_to_api(msg: Any) -> dict[str, Any]:
    if isinstance(msg, dict):
        role = _normalize_role(msg.get("role") or msg.get("type"))
        api_msg: dict[str, Any] = {"role": role, "content": msg.get("content", "")}
        if msg.get("tool_calls"):
            api_msg["tool_calls"] = msg["tool_calls"]
        if role == "tool":
            api_msg["tool_call_id"] = msg.get("tool_call_id", "")
            if msg.get("name"):
                api_msg["name"] = msg["name"]
        return api_msg

    role = _normalize_role(getattr(msg, "type", "user"))
    api_msg = {"role": role, "content": getattr(msg, "content", "")}
    extra = getattr(msg, "additional_kwargs", {}) or {}
    tool_calls = getattr(msg, "tool_calls", None) or extra.get("tool_calls")
    if tool_calls:
        api_msg["tool_calls"] = tool_calls
    if role == "tool":
        api_msg["tool_call_id"] = getattr(msg, "tool_call_id", "")
        name = getattr(msg, "name", None)
        if name:
            api_msg["name"] = name
    return api_msg


def _coerce_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _extract_choice_message(response: Any) -> tuple[Any | None, str | None]:
    choices = getattr(response, "choices", None)
    if not choices:
        return None, f"响应中缺少 choices，实际类型: {type(response).__name__}"

    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        return None, "响应中缺少 message 字段"

    return choice, None


def _tool_messages_from_state(messages: list[Any]) -> list[dict[str, str]]:
    tool_messages: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = _normalize_role(msg.get("role") or msg.get("type"))
            if role == "tool":
                tool_messages.append({
                    "name": str(msg.get("name") or "search_notes"),
                    "content": _coerce_text_content(msg.get("content")),
                })
            continue

        if _normalize_role(getattr(msg, "type", None)) != "tool":
            continue
        tool_messages.append({
            "name": str(getattr(msg, "name", None) or "search_notes"),
            "content": _coerce_text_content(getattr(msg, "content", "")),
        })
    return tool_messages


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
        messages_for_api.append(_message_to_api(msg))

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages_for_api,
            tools=[SEARCH_TOOL_SCHEMA],
            tool_choice="auto",  # LLM自己决定是否调用工具
        )
    except Exception as e:
        return {
            "messages": [{"role": "assistant", "content": f"[系统报错] 调用大模型 API 发生异常: {str(e)}"}],
            "needs_retrieval": False,
        }

    # 兼容处理：防备代理/中转 API 抽风，返回了包含错误信息的普通字符串而不是标准 JSON
    if isinstance(response, str):
        return {
            "messages": [{"role": "assistant", "content": f"[API中转报错] 返回了无效对象: {response}"}],
            "needs_retrieval": False,
        }

    choice, error = _extract_choice_message(response)
    if error:
        return {
            "messages": [{"role": "assistant", "content": f"[API返回异常] {error}"}],
            "needs_retrieval": False,
        }

    msg = choice.message

    # Build a serializable message dict (not langchain object)
    ai_msg: dict[str, Any] = {"role": "assistant", "content": _coerce_text_content(msg.content)}

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
    Made robust to handle different message formats from LangGraph/Proxies.
    """
    tool_messages = []

    # Find the latest AI message with tool_calls
    for msg in reversed(state.messages):
        # LangGraph message objects often store tool_calls in .tool_calls
        # but sometimes it's in additional_kwargs
        raw_tool_calls = getattr(msg, "tool_calls", None) or getattr(msg, "additional_kwargs", {}).get("tool_calls")
        
        # If it's a dict, try to get tool_calls key
        if not raw_tool_calls and isinstance(msg, dict):
            raw_tool_calls = msg.get("tool_calls")

        if raw_tool_calls:
            print(f"\n[系统日志] 发现 {len(raw_tool_calls)} 个工具调用请求...")
            for tc in raw_tool_calls:
                # 1. 提取 ID
                tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None)
                
                # 2. 提取函数名和参数 (支持多种嵌套格式)
                name = None
                args_str = None
                
                if isinstance(tc, dict):
                    # 格式 A: OpenAI 官方 {"function": {"name": "...", "arguments": "..."}}
                    if "function" in tc:
                        name = tc["function"].get("name")
                        args_str = tc["function"].get("arguments")
                    # 格式 B: 扁平化格式 {"name": "...", "args": {...}}
                    else:
                        name = tc.get("name")
                        args_str = tc.get("args") or tc.get("arguments")
                else:
                    # 格式 C: 对象格式 (OpenAI SDK / LangChain Message)
                    name = getattr(tc.function, "name", None) if hasattr(tc, "function") else getattr(tc, "name", None)
                    args_str = getattr(tc.function, "arguments", None) if hasattr(tc, "function") else getattr(tc, "args", None)

                if not name:
                    print(f"  [警告] 无法从工具请求中解析出函数名: {tc}")
                    continue

                # 3. 执行
                print(f"  -> 执行工具: {name} (参数: {args_str})")
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                if not args: args = {}
                
                try:
                    result = run_search_tool(args)
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": result,
                        "name": name,
                    })
                except Exception as e:
                    print(f"  [系统异常] 执行工具 {name} 失败: {e}")
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": f"工具执行报错: {str(e)}",
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

    tool_messages = _tool_messages_from_state(state.messages)
    messages_for_api = [{"role": "system", "content": SYSTEM_PROMPT}]

    if tool_messages:
        latest_user_message = ""
        for msg in reversed(state.messages):
            api_msg = _message_to_api(msg)
            if api_msg["role"] == "user":
                latest_user_message = _coerce_text_content(api_msg.get("content"))
                break

        retrieved_context = "\n\n".join(
            f"[{tool_msg['name']}]\n{tool_msg['content']}" for tool_msg in tool_messages if tool_msg["content"]
        )
        messages_for_api.append(
            {
                "role": "user",
                "content": (
                    f"用户问题：{latest_user_message or '请基于检索结果回答。'}\n\n"
                    "以下是 search_notes 的检索结果。请基于这些内容回答，并尽量引用来源文件；"
                    "如果检索结果不足以回答，就明确说明。\n\n"
                    f"{retrieved_context}"
                ),
            }
        )
    else:
        for msg in state.messages:
            messages_for_api.append(_message_to_api(msg))

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages_for_api,
        )
    except Exception as e:
        return {"messages": [{"role": "assistant", "content": f"[系统报错] 生成最终回答发生异常: {str(e)}"}]}

    if isinstance(response, str):
        return {"messages": [{"role": "assistant", "content": f"[API中转报错] 返回了无效对象: {response}"}]}

    choice, error = _extract_choice_message(response)
    if error:
        return {"messages": [{"role": "assistant", "content": f"[API返回异常] {error}"}]}

    answer = _coerce_text_content(choice.message.content) or "（模型未返回文本内容）"
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
        # Windows 终端编码清洗：忽略无法编码的非法字符（surrogates）
        try:
            user_text = user_text.encode("utf-8", errors="replace").decode("utf-8")
        except Exception:
            pass
        
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
