"""
Phase 4 - Supervisor: Multi-Agent Orchestration with LangGraph

This is the MAIN file of Phase 4. It wires Researcher + Writer + Supervisor
into a LangGraph state machine.

Graph structure:
    researcher ──→ supervisor_check ──→ writer ──→ supervisor_check ──→ END
                        ↑ (loop back if quality check fails)
                        └──────────────────────────────────────────────┘

Usage:
    uv run python phase-4/supervisor.py

Key learning:
    - How to combine existing Agent functions into a graph
    - State machine for quality gates (Supervisor pattern)
    - Revision loops with safety limits (max_revisions)
    - Each node is isolated: Researcher doesn't know Writer exists!
"""

from __future__ import annotations

import os
from typing import Annotated, Any

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from researcher import research
from writer import write_article

load_dotenv()

# ---------------------------------------------------------------------------
# State: shared across all nodes
# ---------------------------------------------------------------------------
class TeamState(BaseModel):
    # Input
    topic: str = ""

    # Pipeline data
    research_notes: str = ""   # Filled by Researcher
    article_draft: str = ""    # Filled by Writer

    # Supervisor control
    stage: str = "research"    # "research" | "writing" | "done"
    revision_feedback: str = ""
    researcher_revisions: int = 0
    writer_revisions: int = 0

    # Audit trail (append-only)
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)

    # Config
    MAX_RESEARCHER_REVISIONS: int = Field(default=2, exclude=True)
    MAX_WRITER_REVISIONS: int = Field(default=2, exclude=True)


# ---------------------------------------------------------------------------
# Node: Researcher
# ---------------------------------------------------------------------------
def researcher_node(state: TeamState) -> dict[str, Any]:
    """
    Call the Researcher agent with the topic (and any revision feedback).
    """
    topic = state.topic
    if state.revision_feedback and state.researcher_revisions > 0:
        # Revision: append feedback to topic to guide re-research
        topic = f"{topic}（改进方向：{state.revision_feedback}）"

    print(f"\n🔬 [Researcher] 开始研究: {topic}")
    notes = research(topic)
    print(f"   ✅ 研究完成，{len(notes)} 字")

    return {
        "research_notes": notes,
        "stage": "research",
        "researcher_revisions": state.researcher_revisions + 1,
        "revision_feedback": "",
        "messages": [{"role": "researcher", "content": f"研究完成: {len(notes)} 字"}],
    }


# ---------------------------------------------------------------------------
# Node: Writer
# ---------------------------------------------------------------------------
def writer_node(state: TeamState) -> dict[str, Any]:
    """
    Call the Writer agent with research notes (and any revision feedback).
    """
    print(f"\n✍️  [Writer] 开始写作: {state.topic}")
    draft = write_article(
        topic=state.topic,
        research_notes=state.research_notes,
        revision_feedback=state.revision_feedback,
    )
    print(f"   ✅ 写作完成，{len(draft)} 字")

    return {
        "article_draft": draft,
        "stage": "writing",
        "writer_revisions": state.writer_revisions + 1,
        "revision_feedback": "",
        "messages": [{"role": "writer", "content": f"初稿完成: {len(draft)} 字"}],
    }


# ---------------------------------------------------------------------------
# Node: Supervisor (quality gate — no LLM, pure rules)
# ---------------------------------------------------------------------------
def supervisor_node(state: TeamState) -> dict[str, Any]:
    """
    Quality gate. Checks current output and decides whether to
    approve (move on) or request revision.

    This uses simple rules, not LLM — fast and predictable.
    Real projects: replace _check_research / _check_draft with LLM judge calls.
    """
    if state.stage == "research":
        ok, feedback = _check_research(state.research_notes)
        if ok or state.researcher_revisions >= state.MAX_RESEARCHER_REVISIONS:
            if not ok:
                print(f"\n👔 [Supervisor] 研究质量未达标，但已达最大修订次数，强制通过")
            else:
                print(f"\n👔 [Supervisor] 研究质量通过 ✅")
            return {"stage": "writing", "messages": [{"role": "supervisor", "content": "研究通过，进入写作"}]}
        else:
            print(f"\n👔 [Supervisor] 研究需改进: {feedback}")
            return {
                "stage": "research",
                "revision_feedback": feedback,
                "messages": [{"role": "supervisor", "content": f"要求修改研究: {feedback}"}],
            }

    elif state.stage == "writing":
        ok, feedback = _check_draft(state.article_draft)
        if ok or state.writer_revisions >= state.MAX_WRITER_REVISIONS:
            if not ok:
                print(f"\n👔 [Supervisor] 文章质量未达标，但已达最大修订次数，强制完成")
            else:
                print(f"\n👔 [Supervisor] 文章质量通过 ✅")
            return {"stage": "done", "messages": [{"role": "supervisor", "content": "文章通过，输出最终结果"}]}
        else:
            print(f"\n👔 [Supervisor] 文章需改进: {feedback}")
            return {
                "stage": "writing",
                "revision_feedback": feedback,
                "messages": [{"role": "supervisor", "content": f"要求修改文章: {feedback}"}],
            }

    return {"stage": "done"}


def _check_research(notes: str) -> tuple[bool, str]:
    """Simple rule-based research quality check."""
    if len(notes) < 150:
        return False, "研究内容太短，请补充更多细节（至少 150 字）"
    if "## 关键要点" not in notes:
        return False, "缺少'关键要点'章节，请按格式输出"
    if notes.count("-") < 2:
        return False, "关键要点太少，至少需要 2 条"
    return True, ""


def _check_draft(draft: str) -> tuple[bool, str]:
    """Simple rule-based draft quality check."""
    if len(draft) < 400:
        return False, "文章太短，请扩展到至少 400 字"
    if "## 引言" not in draft and "## 核心" not in draft:
        return False, "文章结构不完整，需要包含引言和核心内容章节"
    return True, ""


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------
def route_supervisor(state: TeamState) -> str:
    """After supervisor check, decide which node to go to next."""
    if state.stage == "research":
        return "researcher"   # Supervisor asked for revision
    elif state.stage == "writing":
        return "writer"       # Supervisor asked for revision
    else:
        return END            # "done"


def route_after_researcher(state: TeamState) -> str:
    """After researcher finishes, always go to supervisor."""
    return "supervisor"


def route_after_writer(state: TeamState) -> str:
    """After writer finishes, always go to supervisor."""
    return "supervisor"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_team():
    g = StateGraph(TeamState)

    g.add_node("researcher", researcher_node)
    g.add_node("writer", writer_node)
    g.add_node("supervisor", supervisor_node)

    g.set_entry_point("researcher")
    g.add_edge("researcher", "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {"researcher": "researcher", "writer": "writer", END: END},
    )
    g.add_edge("writer", "supervisor")

    return g.compile(checkpointer=InMemorySaver())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Phase 4: Multi-Agent 研报生成系统")
    print("=" * 60)

    topic = input("\n请输入研究主题（回车使用默认）: ").strip()
    if not topic:
        topic = "LangGraph 在企业级 AI Agent 中的应用"

    print(f"\n📋 主题: {topic}")
    print("-" * 60)

    team = build_team()
    config = {"configurable": {"thread_id": "team-demo"}}

    final = team.invoke({"topic": topic}, config=config)

    print("\n" + "=" * 60)
    print("📄 最终文章")
    print("=" * 60)
    print(final["article_draft"])
    print(f"\n📊 统计: 研究修订 {final['researcher_revisions']} 次，写作修订 {final['writer_revisions']} 次")


if __name__ == "__main__":
    main()
