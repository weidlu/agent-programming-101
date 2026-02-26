from __future__ import annotations

import re
import uuid
from typing import Annotated, Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field


class CustomerServiceState(BaseModel):
    # LangGraph will aggregate messages with add_messages. We can feed in dict/tuple/str and it
    # will be converted into langchain-core message objects in state.
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)

    # Information extracted from the user (e.g. order_id).
    user_info: dict[str, Any] = Field(default_factory=dict)

    # High-level routing fields.
    intent: Literal["refund", "consult", "unknown"] = "unknown"
    needs_human: bool = False

    # Refund flow state (kept explicit so nodes can be idempotent/re-entrant).
    refund_decision: Literal["pending", "approved", "declined"] = "pending"
    refund_transaction_id: str | None = None


_ORDER_ID_RE = re.compile(r"(?:订单|order)[^0-9]*([0-9]{3,})", re.IGNORECASE)
_ANGRY_WORDS = (
    "生气",
    "愤怒",
    "垃圾",
    "投诉",
    "差评",
    "骗子",
    "要告你",
    "气死了",
)


def _extract_user_info(text: str) -> dict[str, Any]:
    info: dict[str, Any] = {}
    m = _ORDER_ID_RE.search(text)
    if m:
        info["order_id"] = m.group(1)
    return info


def _is_angry(text: str) -> bool:
    return any(w in text for w in _ANGRY_WORDS)


def classify_intent(state: CustomerServiceState) -> dict[str, Any]:
    """Node A: intent recognition + basic info extraction (rule-based, no LLM)."""
    last_user_text = ""
    for msg in reversed(state.messages):
        if getattr(msg, "type", None) in ("human", "user"):
            last_user_text = getattr(msg, "content", "") or ""
            break

    info_update = _extract_user_info(last_user_text)
    merged_info = {**state.user_info, **info_update}

    text_l = last_user_text.lower()
    if "退款" in last_user_text or "refund" in text_l:
        intent: Literal["refund", "consult", "unknown"] = "refund"
    elif last_user_text.strip():
        intent = "consult"
    else:
        intent = "unknown"

    needs_human = _is_angry(last_user_text)

    # If the user asks for a refund again, we reset the decision to "pending"
    # so the graph can re-enter the confirmation step. If we've already issued
    # a refund (transaction_id exists), we keep the decision as-is.
    refund_decision = state.refund_decision
    if intent == "refund" and state.refund_transaction_id is None:
        refund_decision = "pending"

    return {
        "user_info": merged_info,
        "intent": intent,
        "needs_human": needs_human,
        "refund_decision": refund_decision,
    }


def human_handoff(state: CustomerServiceState) -> dict[str, Any]:
    """Node C: human-in-the-loop handoff (simulated)."""
    return {
        "messages": [
            {
                "role": "assistant",
                "content": "我理解你的情绪。我先为你转人工客服处理（模拟）。你可以补充订单号/问题细节。",
            }
        ]
    }


def confirm_refund(state: CustomerServiceState) -> dict[str, Any]:
    """Pause before refunding. This node is intentionally side-effect free.

    Important LangGraph behavior: when you resume after an interrupt, the node will
    re-run from the start. So do not put irreversible side effects in this node.
    """
    if state.refund_decision != "pending":
        return {}

    order_id = state.user_info.get("order_id")
    prompt = {
        "type": "confirm_refund",
        "question": "是否确认要发起退款？请输入 yes/no。",
        "context": {"order_id": order_id},
    }
    decision = interrupt(prompt)  # Resume value comes from Command(resume=...)

    approved = False
    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
    else:
        approved = bool(decision)

    if not approved:
        return {
            "refund_decision": "declined",
            "messages": [{"role": "assistant", "content": "好的，我不会发起退款。如需继续，请告诉我你的诉求。"}],
        }

    return {
        "refund_decision": "approved",
        "messages": [{"role": "assistant", "content": "收到，我将为你发起退款（模拟）。"}],
    }


def _issue_refund_tool(order_id: str | None) -> str:
    """Node B tool: simulate an external refund API call."""
    # Real systems would call a payment/refund API here.
    return f"refund_{order_id or 'unknown'}_{uuid.uuid4().hex[:8]}"


def process_refund(state: CustomerServiceState) -> dict[str, Any]:
    """Node B: refund processing. Must be idempotent."""
    if state.refund_decision != "approved":
        return {}

    # Idempotency guard: if the node is re-run (e.g. due to retries), do not issue a second refund.
    if state.refund_transaction_id:
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"退款已处理过（模拟）。退款单号：{state.refund_transaction_id}",
                }
            ]
        }

    order_id = state.user_info.get("order_id")
    tx_id = _issue_refund_tool(order_id=order_id)
    return {
        "refund_transaction_id": tx_id,
        "messages": [{"role": "assistant", "content": f"已发起退款（模拟）。退款单号：{tx_id}"}],
    }


def refund_status(state: CustomerServiceState) -> dict[str, Any]:
    """If we already have a refund transaction id, report it."""
    if not state.refund_transaction_id:
        return {}
    return {
        "messages": [
            {
                "role": "assistant",
                "content": f"你的退款已在处理中（模拟）。退款单号：{state.refund_transaction_id}",
            }
        ]
    }


def answer_consult(state: CustomerServiceState) -> dict[str, Any]:
    """Non-refund path (placeholder)."""
    return {
        "messages": [
            {
                "role": "assistant",
                "content": "我可以帮你处理咨询类问题（示例）。如果你要退款，请直接说“我要退款”，并附上订单号。",
            }
        ]
    }


def _route_after_classify(state: CustomerServiceState) -> str:
    if state.needs_human:
        return "human_handoff"
    if state.intent == "refund":
        if state.refund_transaction_id:
            return "refund_status"
        return "confirm_refund"
    return "answer_consult"


def _route_after_confirm(state: CustomerServiceState) -> str:
    if state.refund_decision == "approved":
        return "process_refund"
    return "end"


def build_app():
    graph = StateGraph(CustomerServiceState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("human_handoff", human_handoff)
    graph.add_node("confirm_refund", confirm_refund)
    graph.add_node("process_refund", process_refund)
    graph.add_node("refund_status", refund_status)
    graph.add_node("answer_consult", answer_consult)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {
            "human_handoff": "human_handoff",
            "confirm_refund": "confirm_refund",
            "refund_status": "refund_status",
            "answer_consult": "answer_consult",
        },
    )

    graph.add_conditional_edges(
        "confirm_refund",
        _route_after_confirm,
        {"process_refund": "process_refund", "end": END},
    )

    graph.add_edge("human_handoff", END)
    graph.add_edge("refund_status", END)
    graph.add_edge("answer_consult", END)
    graph.add_edge("process_refund", END)

    return graph.compile(checkpointer=InMemorySaver())


def _render_ai_messages(messages: list[Any], start_at: int) -> int:
    """Print only new assistant messages and return the new cursor."""
    for msg in messages[start_at:]:
        msg_type = getattr(msg, "type", None)
        if msg_type in ("ai", "assistant"):
            print(f"Agent: {getattr(msg, 'content', str(msg))}")
    return len(messages)


def main():
    app = build_app()
    thread_id = "phase2-demo"
    config = {"configurable": {"thread_id": thread_id}}

    cursor = 0
    print("Customer Service Agent (LangGraph) - 输入 quit 退出")

    while True:
        user_text = input("User: ").strip()
        if user_text.lower() in ("quit", "exit", "q"):
            break

        out = app.invoke({"messages": [{"role": "user", "content": user_text}]}, config=config)

        # Handle human-in-the-loop interrupts.
        while "__interrupt__" in out:
            intr = out["__interrupt__"][0]
            payload = intr.value

            if isinstance(payload, dict) and payload.get("type") == "confirm_refund":
                raw = input(f"[确认] {payload.get('question')} ").strip().lower()
                approved = raw in ("y", "yes", "是", "确认", "ok")
                out = app.invoke(Command(resume={"approved": approved}), config=config)
            else:
                # Generic fallback: resume with whatever the user types.
                raw = input("[需要输入以继续] ").strip()
                out = app.invoke(Command(resume=raw), config=config)

        cursor = _render_ai_messages(out.get("messages", []), cursor)


if __name__ == "__main__":
    main()
