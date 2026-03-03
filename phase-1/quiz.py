"""
Phase 1 Quiz — 从零手撸一个最小 ReAct Agent

目标：不看 simple_agent.py，独立实现一个能完成以下任务的 Agent：
    "深圳今天天气怎么样？顺便算一下 999 乘以 123 是多少？"

Agent 需要：
1. 调用 get_weather 工具查天气
2. 调用 multiply 工具做乘法
3. 拿到两个工具的结果后，输出一段自然语言回答

运行方式（先完成 TODO 才能运行）：
    uv run python phase-1/quiz.py

完成标准：
    - 终端里能看到 Agent 打印出两次 [工具调用] 日志
    - 最后打印 Agent: 的自然语言回答，包含天气和计算结果
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ============================================================
# 工具函数（已实现，不需要修改）
# ============================================================
def get_weather(city: str) -> str:
    """模拟天气查询（写死数据）。"""
    data = {
        "北京": "晴，25°C",
        "上海": "阴有小雨，22°C",
        "深圳": "多云，28°C",
        "广州": "晴，30°C",
    }
    result = data.get(city, f"未知城市 [{city}]，无法获取天气")
    print(f"  [工具调用] get_weather({city!r}) → {result}")
    return result


def multiply(a: float, b: float) -> float:
    """乘法计算。"""
    result = a * b
    print(f"  [工具调用] multiply({a}, {b}) → {result}")
    return result


# 工具分发表（已实现）
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "multiply": multiply,
}


# ============================================================
# TODO 1: 定义 Tool Schema（给 LLM 看的"说明书"）
# ============================================================
# 参考格式：
# {
#     "type": "function",
#     "function": {
#         "name": "...",
#         "description": "...",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "city": {"type": "string", "description": "..."},
#             },
#             "required": ["city"],
#         },
#     },
# }
#
# 需要定义两个 tool：get_weather 和 multiply
TOOLS: list[dict] = []  # TODO 1: 填充这个列表


# ============================================================
# TODO 2: 初始化 OpenAI 客户端
# ============================================================
# 提示：
# - api_key = os.getenv("OPENAI_API_KEY")
# - base_url = os.getenv("OPENAI_BASE_URL")  ← 使用自定义 proxy
def make_client() -> OpenAI:
    # TODO 2: 返回一个配置好的 OpenAI 客户端
    ...


# ============================================================
# TODO 3: 实现工具执行函数
# ============================================================
def execute_tool(tool_call) -> str:
    """
    接收 LLM 返回的 tool_call 对象，执行对应函数，返回字符串结果。

    步骤：
    1. 从 tool_call.function.name 获取函数名
    2. 用 json.loads(tool_call.function.arguments) 获取参数 dict
    3. 从 TOOL_REGISTRY 里查找函数
    4. 调用函数（用 **kwargs 解包参数）
    5. 返回 str(结果)

    如果函数名不在 TOOL_REGISTRY 里，返回 "未知工具: {name}"
    """
    # TODO 3: 实现工具执行逻辑
    ...


# ============================================================
# TODO 4: 实现 Agent 主循环
# ============================================================
def run_agent(user_query: str, model: str = "MiniMax-M2.5") -> None:
    """
    ReAct Agent 核心循环：Think → Act → Observe → Think → ...

    步骤：
    1. 初始化 messages = [system_prompt, user_message]
    2. 进入 while 循环（带 max_steps 护栏，建议 max=8）
    3. 每次循环：
       a. 调用 LLM（传入 tools=TOOLS, tool_choice="auto"）
       b. 拿到 response_message = response.choices[0].message
       c. 如果 finish_reason == "tool_calls"（或 response_message.tool_calls 非空）：
          - 把 response_message 追加到 messages
          - 对每个 tool_call：调用 execute_tool → 追加 tool message
            格式：{"role": "tool", "tool_call_id": ..., "name": ..., "content": ...}
       d. 否则（LLM 直接回答了）：
          - 打印 "Agent: {response_message.content}"
          - break

    TODO 4: 实现这个函数
    """
    client = make_client()
    print(f"User: {user_query}\n")
    # TODO 4: ...
    ...


# ============================================================
# TODO 5: 护栏自测
# ============================================================
# 在 run_agent 里加入以下护栏后，取消注释下面的测试用例验证：
#
# (a) max_steps 护栏：循环超过 N 次后强制退出并打印警告
# (b) 工具异常捕获：如果 execute_tool 抛出异常，把异常信息作为 tool message 内容传回 LLM
#
# 验证方式（可以故意触发）：
#   - 把 max_steps 改成 1，看看 Agent 会不会停在工具调用没完成的地方
#   - 在 TOOL_REGISTRY 里删掉 "multiply"，看看 Agent 如何处理未知工具

# ============================================================
# 思考题
# ============================================================
"""
1. 为什么每次调用 LLM 之前，messages 里要包含之前所有的 tool_call 和 tool 消息？
   如果只传最新一条会发生什么？

2. LLM 返回的 tool_call 对象里，tool_call_id 有什么用？
   如果两次工具调用有相同的 id 会怎样？

3. max_steps 护栏应该放在 while 条件里（while step < max_steps）
   还是在循环体里做 if 判断？两种写法有什么区别？

4. 这个 Agent 能同时处理"天气 + 乘法"两个问题，
   是因为 LLM 一次返回了多个 tool_call，还是分两轮循环的？
   加一行打印 len(tool_calls) 验证一下。

5. 如果把 tool_choice="auto" 改成 tool_choice="none"，Agent 会有什么变化？
"""


if __name__ == "__main__":
    run_agent("深圳今天天气怎么样？顺便算一下 999 乘以 123 是多少？")
