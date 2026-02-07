from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件，override=True 覆盖已有环境变量

import json
import os
from openai import OpenAI

# ----------------------
# 0. 配置客户端
# ----------------------
# NewAPI 兼容 OpenAI API，直接用 OpenAI SDK 即可
# 只需要将 base_url 指向 NewAPI 的地址
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

# ----------------------
# 1. 定义工具 (Python 函数)
# ----------------------
# 这是 Agent 的“手”。LLM 无法直接运行它，需要我们帮它运行。
def multiply(a, b):
    """
    计算两个数的乘积
    """
    print(f"\n[系统日志] 正在调用本地函数 multiply: {a} * {b} ...")
    return a * b

# ----------------------
# 2. 定义工具描述 (Schema)
# ----------------------
# 这是给 LLM 看的“说明书”。告诉它有哪些工具可用，参数是什么。
tools = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "计算两个数字的乘积。当用户询问数学计算时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "第一个数字"
                    },
                    "b": {
                        "type": "number",
                        "description": "第二个数字"
                    }
                },
                "required": ["a", "b"]
            }
        }
    }
]

# 映射表：字符串 -> 真实函数
available_functions = {
    "multiply": multiply
}

# ----------------------
# 3. Agent 核心逻辑 (The Loop)
# ----------------------
def run_agent(user_query):
    # 初始化对话历史
    messages = [
        {"role": "system", "content": "你是一个有用的助手。如果遇到计算问题，必须调用工具。"},
        {"role": "user", "content": user_query}
    ]

    print(f"用户: {user_query}")

    # --- 开始循环 ---
    while True:
        # Step A: 思考 (调用 LLM)
        response = client.chat.completions.create(
            model="deepseek-chat",  # 可以换成 deepseek-chat, claude-3-5-sonnet 等 NewAPI 支持的模型
            messages=messages,
            tools=tools,
            tool_choice="auto"  # 让模型自己决定是用工具还是直接说话
        )
        
        # 调试：打印完整的 API 响应
        print(f"\n[DEBUG] 完整响应: {response}")
        print(f"[DEBUG] Model: {response.model}")
        print(f"[DEBUG] Usage: {response.usage}")
        
        response_message = response.choices[0].message
        print(f"[DEBUG] Message: {response_message}")
        
        # Step B: 检查 LLM 是否想调用工具
        tool_calls = response_message.tool_calls

        if tool_calls:
            # LLM 想调用工具！
            # 1. 必须把 LLM 的“思考结果”加到历史记录里，否则它会“失忆”
            messages.append(response_message) 

            # 2. 遍历所有它想调用的工具（它可能一次想调好几个）
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Step C: 行动 (执行 Python 代码)
                if function_name in available_functions:
                    function_to_call = available_functions[function_name]
                    function_response = function_to_call(
                        a=function_args.get("a"),
                        b=function_args.get("b")
                    )
                    
                    # Step D: 观察 (把结果封装成 Tool Message 塞回去)
                    # 这里的 tool_call_id 非常重要，LLM 靠它知道这个结果对应哪次调用
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": str(function_response),
                        }
                    )
            
            # 循环继续... 带着工具的结果，进入下一次 while 循环，再次请求 LLM
        else:
            # LLM 没有调用工具，直接输出了文本，说明任务结束
            print(f"Agent: {response_message.content}")
            break

# ----------------------
# 4. 运行测试
# ----------------------
if __name__ == "__main__":
    # 测试 1: 不需要工具
    # run_agent("你好，你是谁？")
    
    # 测试 2: 需要工具
    run_agent("3829 乘以 812 是多少？")