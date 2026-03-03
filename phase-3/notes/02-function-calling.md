# Function Calling（函数调用）

## 核心机制

Function Calling 是 LLM 的一种能力：模型在输出时，不仅能返回文本，还能返回"我想调用哪个函数、参数是什么"的结构化信息。

```
用户输入 → LLM → [文本 OR 函数调用请求]
                         ↓ (如果是函数调用)
                  本地执行函数 → 结果传回 LLM → 最终回答
```

## Tool Schema 定义

每个工具必须提供 JSON Schema 描述，让 LLM 知道怎么调用：

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如'北京'或'Shanghai'"
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

## The Loop（核心循环）

```python
messages = [{"role": "user", "content": user_input}]

while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools
    )
    
    choice = response.choices[0]
    
    if choice.finish_reason == "tool_calls":
        # 执行工具
        tool_call = choice.message.tool_calls[0]
        result = execute_tool(tool_call.function.name, tool_call.function.arguments)
        
        # 把工具结果追加到 messages
        messages.append(choice.message)  # 模型的"我要调用工具"消息
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })
        # 继续循环，让 LLM 根据工具结果继续思考
    
    elif choice.finish_reason == "stop":
        print(choice.message.content)
        break
```

## 关键细节

### finish_reason
- `"stop"`: 模型正常结束，输出文本
- `"tool_calls"`: 模型要调用工具
- `"length"`: 超出 max_tokens 限制

### 工具结果的格式
工具结果必须以 `role: "tool"` 的消息格式传回，并附上 `tool_call_id`（用于匹配是哪次调用的结果）。

### 护栏（必须加！）
```python
MAX_STEPS = 10  # 防止死循环
step = 0
while step < MAX_STEPS:
    step += 1
    ...
```

## Parallel Tool Calls（并行工具调用）

新版模型支持一次返回多个工具调用请求（如同时查两个城市的天气）：

```python
# response.choices[0].message.tool_calls 是一个列表
for tool_call in choice.message.tool_calls:
    result = execute_tool(tool_call.function.name, ...)
    messages.append({"role": "tool", "tool_call_id": tool_call.id, ...})
```

## 在 RAG 中的应用

Phase 3 里，`search_notes` 就是一个 Tool：
```python
# LLM "决定"要检索知识库时，会返回：
{
  "function": "search_notes",
  "arguments": {"query": "MCP是什么", "top_k": 3}
}
```
这比硬编码"每条消息都去检索"要智能得多——LLM自己判断什么时候需要检索。
