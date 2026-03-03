# Agent 间通信方式

## 三种通信模式

### 模式一：Shared State（共享状态）—— 我们 Phase 4 用的方式

所有 Agent 读写同一个 `TeamState` 对象。

```
Researcher → TeamState.research_notes="..."
               ↓ (Supervisor 读取)
Supervisor → TeamState.stage="writing"
               ↓ (Writer 读取)
Writer → TeamState.article_draft="..."
```

**优点**：实现简单，状态一目了然  
**缺点**：所有 Agent 紧耦合，难以独立部署

### 模式二：Message Passing（消息传递）

Agent 之间发送结构化消息，像聊天一样：

```python
# Researcher 完成后发消息
message = {
    "from": "researcher",
    "to": "supervisor",
    "type": "research_complete",
    "payload": {"notes": "...", "sources": [...]}
}
```

**优点**：松耦合，每个 Agent 可以独立运行  
**缺点**：需要消息队列（如 Kafka、Redis Streams）

### 模式三：Handoff（交接）

一个 Agent 完成后，直接把控制权 + 上下文转交给下一个 Agent。
OpenAI Agents SDK 原生支持：

```python
from agents import Agent, handoff

researcher = Agent(
    name="Researcher",
    tools=[search_tool],
    handoffs=[writer],  # 完成后交给 Writer
)
```

**优点**：直观，像流水线一样  
**缺点**：交接后 context 可能丢失，需要设计好 handoff payload

## 我们的选择：为什么用 Shared State？

1. **学习阶段**：Shared State 是最透明的，状态变化一目了然，容易调试
2. **LangGraph 原生**：State 是 LangGraph 的核心抽象，不需要额外中间件
3. **进阶路径**：理解 Shared State 后，迁移到 Message Passing 或 Handoff 非常容易

## 通信中的关键问题：上下文窗口管理

多轮对话后，messages 列表会越来越长，最终超出 LLM 的 token 限制。

### 解决方案：摘要压缩

```python
def compress_messages(messages: list) -> str:
    """把旧消息压缩成摘要，替换掉原始对话历史。"""
    client = OpenAI()
    response = client.chat.completions.create(
        model="MiniMax-M2.5",
        messages=[
            {"role": "system", "content": "把以下对话历史压缩成要点摘要，200字以内"},
            {"role": "user", "content": str(messages)}
        ]
    )
    return response.choices[0].message.content
```

这正是 **OpenViking** 做的事：它的 `Compressor` 模块自动压缩会话历史！
当你学完 Phase 4，再去看 OpenViking 的 `session.py` 会有强烈的"原来如此"感。
