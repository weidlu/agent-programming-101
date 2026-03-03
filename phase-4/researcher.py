"""
Phase 4 - Researcher Agent

Responsibility: Given a topic, search for information using tools,
synthesize findings into structured research notes.

Key design decisions:
    1. Uses BOTH search_knowledge_base (local notes) AND web_search (internet)
    2. Runs a mini tool-use loop (like Phase 1!) but contained within one function
    3. Returns structured Markdown: ## Summary + ## Key Points + ## Sources
    4. Has no awareness of Writer or Supervisor — stays focused on its one job
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import ALL_TOOLS, run_tool

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "MiniMax-M2.5")

RESEARCHER_SYSTEM_PROMPT = """你是一个专业的 AI 研究员，擅长从多个来源收集和总结信息。

你的任务：
1. 用 search_knowledge_base 工具从本地笔记知识库检索相关内容
2. 用 web_search 工具补充最新的网络信息
3. 整合所有信息，输出结构化的研究摘要

输出格式必须是 Markdown：
## 摘要
（2-3 句话的核心结论）

## 关键要点
- 要点 1
- 要点 2
- 要点 3（至少 3 条）

## 信息来源
- 来源 1
- 来源 2

注意：
- 必须调用至少一次工具，不要凭空编造
- 如果知识库没有相关内容，用 web_search 补充
- 保持客观，引用具体来源
"""


def research(topic: str, max_tool_calls: int = 4) -> str:
    """
    Run the researcher agent on a given topic.
    Returns structured research notes as Markdown.

    This is basically Phase 1's tool loop, wrapped in a clean function!
    """
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    messages = [
        {"role": "system", "content": RESEARCHER_SYSTEM_PROMPT},
        {"role": "user", "content": f"请研究以下主题并给出结构化摘要：{topic}"},
    ]

    tool_calls_made = 0

    # Mini tool-use loop (Phase 1 style!)
    while tool_calls_made < max_tool_calls:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=ALL_TOOLS,
            tool_choice="auto",
        )
        choice = response.choices[0]
        msg = choice.message

        if choice.finish_reason == "tool_calls" and msg.tool_calls:
            # LLM wants to call tools
            tool_calls_made += len(msg.tool_calls)

            # Add the LLM's tool call request to messages
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })

            # Execute each tool and add results
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = run_tool(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                    "name": tc.function.name,
                })

        else:
            # LLM finished — return the research notes
            return msg.content or "（研究员未返回内容）"

    # Forced stop after max tool calls
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages + [
            {"role": "user", "content": "请现在整合你收集到的信息，给出最终的结构化摘要。"}
        ],
    )
    return response.choices[0].message.content or "（超出工具调用限制，未生成摘要）"


if __name__ == "__main__":
    notes = research("MCP 协议在 AI Agent 中的应用")
    print(notes)
