"""
Phase 4 - Writer Agent

Responsibility: Given research notes and a topic, write a well-structured
article draft. No tool usage — pure LLM generation based on given context.

Key design decisions:
    1. Writer does NOT use tools — it only knows how to write
    2. Input: research_notes (from Researcher) + topic
    3. Output: a structured article in Markdown
    4. Supports revision: if given feedback, incorporates it in a rewrite

Separation of concerns:
    Researcher knows how to FIND information.
    Writer knows how to COMMUNICATE information.
    Keeping these separate makes each Agent easier to improve independently.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "MiniMax-M2.5")

WRITER_SYSTEM_PROMPT = """你是一个专业的技术文章写作者，擅长把复杂的技术内容写成清晰易读的文章。

文章结构要求：
# [文章标题]

## 引言
（吸引人的开头，说明文章解决什么问题）

## 核心内容
（2-3 个小节，每节有标题，深入解释研究笔记中的关键要点）

## 实践启示
（对读者的具体建议：下一步怎么做？）

## 总结
（一句话核心观点）

写作风格：
- 深入浅出，技术准确但不晦涩
- 用例子和类比辅助解释
- 每个小节 150-250 字
- 总字数 600-900 字"""


def write_article(topic: str, research_notes: str, revision_feedback: str = "") -> str:
    """
    Write an article based on research notes.

    Args:
        topic: The article topic.
        research_notes: Structured notes from the Researcher agent.
        revision_feedback: If non-empty, this is a rewrite request with feedback
                           from the Supervisor.

    Returns:
        The article as a Markdown string.
    """
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    if revision_feedback:
        user_content = (
            f"请根据以下反馈修改文章。\n\n"
            f"主题：{topic}\n\n"
            f"研究笔记：\n{research_notes}\n\n"
            f"修改意见：{revision_feedback}\n\n"
            f"请给出修改后的完整文章。"
        )
    else:
        user_content = (
            f"主题：{topic}\n\n"
            f"研究笔记：\n{research_notes}\n\n"
            f"请根据以上研究笔记写一篇技术文章。"
        )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": WRITER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content or "（写作者未生成内容）"


if __name__ == "__main__":
    # Quick test with fake research notes
    fake_notes = """
## 摘要
MCP 是 Anthropic 推出的开放协议，用于标准化 AI 模型和外部工具的交互。

## 关键要点
- MCP 定义了 Server/Client/Transport 三层架构
- 相比 Function Calling，MCP 支持跨应用工具复用
- 已有 500+ 社区 Server 贡献
"""
    article = write_article("MCP 协议详解", fake_notes)
    print(article)
