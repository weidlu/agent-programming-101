"""
Phase 4 Quiz - 练习骨架：自己实现一个最小多 Agent 系统

目标：不看 supervisor.py / researcher.py / writer.py，独立组装一个 Mini 研报生成器

运行方式（先完成 TODO 才能运行）：
    uv run python phase-4/quiz.py

完成标准：
    - 输入"MCP 协议"，能看到 Researcher 输出研究笔记
    - Supervisor 能判断质量并决定是否打回重做
    - Writer 能基于研究笔记生成文章
"""

from __future__ import annotations

import os
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "MiniMax-M2.5")


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


# ============================================================
# State
# ============================================================
class MiniTeamState(BaseModel):
    topic: str = ""
    research_notes: str = ""
    article_draft: str = ""
    stage: Literal["research", "writing", "done"] = "research"
    revision_feedback: str = ""
    researcher_revisions: int = 0
    writer_revisions: int = 0
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)


# ============================================================
# TODO 1: 实现 researcher_node
# ============================================================
def researcher_node(state: MiniTeamState) -> dict[str, Any]:
    """
    调用 LLM 对 state.topic 进行研究，返回结构化的 Markdown 笔记。

    提示：
    - 构造 system_prompt，要求 LLM 输出"## 摘要 / ## 关键要点 / ## 信息来源"格式
    - 如果 state.revision_feedback 非空，把它附加到 prompt 里（"改进方向：xxx"）
    - LLM 调用完后返回：
      {
        "research_notes": <LLM 输出>,
        "stage": "research",
        "researcher_revisions": state.researcher_revisions + 1,
        "revision_feedback": "",  # 清除之前的反馈
      }

    注意：这里不需要 Tool Calling，直接让 LLM 生成研究摘要即可（简化版）
    """
    # TODO 1
    ...


# ============================================================
# TODO 2: 实现 writer_node
# ============================================================
def writer_node(state: MiniTeamState) -> dict[str, Any]:
    """
    基于 state.research_notes 生成文章草稿。

    提示：
    - system_prompt 要求：# 标题 / ## 引言 / ## 核心内容 / ## 总结
    - 如果 state.revision_feedback 非空，要求 LLM 根据反馈修改
    - 返回：
      {
        "article_draft": <LLM 输出>,
        "stage": "writing",
        "writer_revisions": state.writer_revisions + 1,
        "revision_feedback": "",
      }
    """
    # TODO 2
    ...


# ============================================================
# TODO 3: 实现 supervisor_node（规则判断，无 LLM）
# ============================================================
def supervisor_node(state: MiniTeamState) -> dict[str, Any]:
    """
    质量把关节点（不调用 LLM，用规则判断）。

    逻辑：
    if state.stage == "research":
        - 如果 research_notes 长度 > 100 字 且包含"关键要点"→ 进入 writing 阶段
        - 否则（且 researcher_revisions < 2）→ 返回 stage="research" + revision_feedback
        - 如果已达最大修订次数 → 强制进入 writing
    elif state.stage == "writing":
        - 如果 article_draft 长度 > 300 字 → stage="done"
        - 否则（且 writer_revisions < 2）→ 返回 stage="writing" + revision_feedback
        - 如果已达最大修订次数 → stage="done"

    需要返回 {"stage": ..., "revision_feedback": ..., "messages": [...]}
    """
    # TODO 3
    ...


# ============================================================
# TODO 4: 实现路由函数
# ============================================================
def route_supervisor(state: MiniTeamState) -> str:
    """
    根据 state.stage 决定下一个节点：
    - "research" → "researcher"（需要修改研究）
    - "writing" → "writer"（需要修改文章）
    - "done" → END

    提示：from langgraph.graph import END
    """
    # TODO 4
    ...


# ============================================================
# TODO 5: 组装 LangGraph 图
# ============================================================
def build_team():
    """
    组装多 Agent 图：
    entry_point → researcher → supervisor → (条件路由)
                                           ├── researcher（打回）
                                           ├── writer
                                           │     └──→ supervisor（条件路由）
                                           └── END

    步骤：
    1. g = StateGraph(MiniTeamState)
    2. add_node 三个节点
    3. set_entry_point("researcher")
    4. add_edge("researcher", "supervisor")
    5. add_conditional_edges("supervisor", route_supervisor, {...})
    6. add_edge("writer", "supervisor")
    7. return g.compile(checkpointer=InMemorySaver())
    """
    # TODO 5
    ...


# ============================================================
# 思考题
# ============================================================
"""
1. 为什么 Supervisor 用规则判断而不用 LLM？在什么场景下你会换成 LLM？

2. researcher_revisions 计数器如果不加，会发生什么？

3. Writer 在修改时收到的 revision_feedback 和第一次写作相比，
   哪些信息他需要保留？如何避免"改了这里，忘了那里"的问题？

4. 如果 Researcher 和 Writer 可以并行运行（不互相依赖），
   LangGraph 怎么实现并行节点？（提示：看 LangGraph 的 fan-out 文档）

5. 这个系统目前的工具调用（search_knowledge_base）在哪个节点里？
   如果要把 Writer 也加上检索能力，需要改哪些地方？
"""


def main():
    team = build_team()
    if team is None:
        print("⚠️  请先完成 TODO 5 再运行")
        return

    topic = input("研究主题(回车使用默认): ").strip() or "Function Calling 技术详解"
    config = {"configurable": {"thread_id": "quiz-team"}}

    print(f"\n🚀 开始研究: {topic}\n")
    result = team.invoke({"topic": topic}, config=config)

    print("\n" + "=" * 50)
    print("📄 最终文章:")
    print("=" * 50)
    print(result.get("article_draft", "（未生成）"))


if __name__ == "__main__":
    main()
