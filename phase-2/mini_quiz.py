from __future__ import annotations

import uuid
from typing import Annotated, Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field


# ----------------------------
# 1) State
# ----------------------------
class MiniState(BaseModel):
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)

    intent: Literal["refund", "consult", "unknown"] = "unknown"
    refund_decision: Literal["pending", "approved", "declined"] = "pending"
    refund_transaction_id: str | None = None


# ----------------------------
# 2) Nodes
# ----------------------------
def classify(state: MiniState) -> dict[str, Any]:
    """Decide intent based on last user message."""
    # TODO 1:
    # - 找到最后一条用户消息文本 last_user_text
    # - 如果包含“退款” -> intent="refund"，否则如果非空 -> "consult"，否则 "unknown"
    # - 如果 intent="refund" 且 refund_transaction_id is None，把 refund_decision 重置为 "pending"
    # - return 对应字段更新 dict
    ...


def confirm_refund(state: MiniState) -> dict[str, Any]:
    """Ask yes/no. Must be side-effect free."""
    if state.refund_decision != "pending":
        return {}

    prompt = {
        "type": "confirm_refund",
        "question": "确认要退款吗？yes/no",
    }

    # TODO 2:
    # - decision = interrupt(prompt)
    # - decision 可能是 dict {"approved": True/False}，也可能是 bool/str
    # - 解析出 approved: bool
    # - 如果 not approved: return refund_decision="declined" + 一条 assistant message
    # - 如果 approved: return refund_decision="approved" + 一条 assistant message
    ...


def process_refund(state: MiniState) -> dict[str, Any]:
    """Issue refund (simulated). Must be idempotent."""
    if state.refund_decision != "approved":
        return {}

    # TODO 3:
    # - 如果 refund_transaction_id 已存在：不要再生成新的，直接回一条 assistant message 告知已处理过 + 单号
    # - 否则生成 tx_id = "refund_" + uuid4 8位，写入 refund_transaction_id，并回 assistant message
    ...


def refund_status(state: MiniState) -> dict[str, Any]:
    """Report existing tx_id."""
    if not state.refund_transaction_id:
        return {}

    # TODO 4:
    # - return 一条 assistant message，内容包含退款单号
    ...


def consult(state: MiniState) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "assistant",
                "content": "这是咨询路径。若要退款请说“我要退款”。",
            }
        ]
    }


# ----------------------------
# 3) Routing functions
# ----------------------------
def route_after_classify(state: MiniState) -> str:
    # TODO 5:
    # - 如果 intent == "refund":
    #     - 如果 refund_transaction_id 已存在：去 refund_status
    #     - 否则去 confirm_refund
    # - 如果 intent != "refund": 去 consult
    ...


def route_after_confirm(state: MiniState) -> str:
    # TODO 6:
    # - 如果 refund_decision == "approved": 去 process_refund
    # - 否则 end
    ...


# ----------------------------
# 4) Build graph
# ----------------------------
def build_app():
    g = StateGraph(MiniState)

    g.add_node("classify", classify)
    g.add_node("confirm_refund", confirm_refund)
    g.add_node("process_refund", process_refund)
    g.add_node("refund_status", refund_status)
    g.add_node("consult", consult)

    g.set_entry_point("classify")

    g.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "confirm_refund": "confirm_refund",
            "refund_status": "refund_status",
            "consult": "consult",
        },
    )

    g.add_conditional_edges(
        "confirm_refund",
        route_after_confirm,
        {"process_refund": "process_refund", "end": END},
    )

    g.add_edge("process_refund", END)
    g.add_edge("refund_status", END)
    g.add_edge("consult", END)

    return g.compile(checkpointer=InMemorySaver())


def render_new_ai(messages: list[Any], start_at: int) -> int:
    for msg in messages[start_at:]:
        if getattr(msg, "type", None) in ("ai", "assistant"):
            print("Agent:", getattr(msg, "content", str(msg)))
    return len(messages)


def main():
    app = build_app()
    config = {"configurable": {"thread_id": "mini-refund"}}

    cursor = 0
    print("Mini Refund Graph - 输入 quit 退出")

    while True:
        user_text = input("User: ").strip()
        if user_text.lower() in ("quit", "exit", "q"):
            break

        out = app.invoke({"messages": [{"role": "user", "content": user_text}]}, config=config)

        while "__interrupt__" in out:
            intr = out["__interrupt__"][0]
            payload = intr.value
            if isinstance(payload, dict) and payload.get("type") == "confirm_refund":
                raw = input(f"[确认] {payload['question']} ").strip().lower()
                approved = raw in ("y", "yes", "是", "确认", "ok")
                out = app.invoke(Command(resume={"approved": approved}), config=config)
            else:
                raw = input("[继续] ").strip()
                out = app.invoke(Command(resume=raw), config=config)

        cursor = render_new_ai(out.get("messages", []), cursor)


if __name__ == "__main__":
    """
    1. 为什么 confirm 节点不能调用真实退款 API？
    2. resume 后为什么会打印两条 Agent？分别来自哪个节点？
    3. refund_transaction_id 的幂等 guard 是在保护什么场景？
    4. conditional edge 的路由函数用的 state 是“更新后的”还是“更新前的”？
    5. messages 为什么在 node 里传 dict 最后 state 里变成带 .type 的对象？
    """
    main()