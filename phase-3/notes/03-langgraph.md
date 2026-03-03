# LangGraph 核心概念

## 为什么需要 LangGraph？

手写 while loop 适合线性流程，但遇到以下场景会变得混乱：
- 多个条件分支（退款/咨询/转人工）
- 需要暂停等待人类确认（Human-in-the-loop）
- 跨轮对话保持状态
- 节点需要幂等性保证

LangGraph 把这些问题用"有向图"的方式优雅解决。

## 四大核心概念

### 1. State（状态）
图中所有节点共享一个 State 对象，用 Pydantic 定义：

```python
from pydantic import BaseModel, Field
from typing import Annotated, Any
from langgraph.graph.message import add_messages

class MyState(BaseModel):
    messages: Annotated[list[Any], add_messages] = Field(default_factory=list)
    intent: str = "unknown"
    user_info: dict = Field(default_factory=dict)
```

`Annotated[list, add_messages]` 表示这个字段用"追加"而不是"覆盖"来合并更新。

### 2. Nodes（节点）
每个节点是一个普通 Python 函数，接收 State 并返回要更新的字段：

```python
def my_node(state: MyState) -> dict:
    # 处理逻辑...
    return {"intent": "refund"}  # 只需返回要更新的字段
```

### 3. Edges（边）
连接节点的路径，分两种：
- **固定边**：`graph.add_edge("node_a", "node_b")` — 总是从 A 到 B
- **条件边**：`graph.add_conditional_edges("node_a", routing_fn, {...})` — 根据函数返回值路由

### 4. Checkpointer（检查点）
保存每一步执行后的状态，支持暂停/恢复：

```python
from langgraph.checkpoint.memory import InMemorySaver
app = graph.compile(checkpointer=InMemorySaver())
```

## interrupt()：Human-in-the-loop 的核心

```python
from langgraph.types import interrupt, Command

def confirm_action(state):
    decision = interrupt({"question": "是否继续？"})  # 暂停执行
    # resume 后从这里继续（注意：整个函数会重跑）
    return {"approved": decision.get("approved")}

# 恢复执行
app.invoke(Command(resume={"approved": True}), config=config)
```

### ⚠️ 重要陷阱
`interrupt` 之后节点**整个重跑**，不是从中断行继续。
所以 `interrupt` 之前的代码必须是**幂等的**（重跑不产生副作用）。

## Thread ID：会话隔离

```python
config = {"configurable": {"thread_id": "user-123"}}
app.invoke({"messages": [...]}, config=config)
```

同一个 `thread_id` 的所有调用共享状态，实现多轮对话记忆。

## 典型图结构

```
entry_point
    ↓
classify_intent
    ↓ (conditional)
├── answer_consult → END
├── confirm_refund → (conditional)
│       ├── process_refund → END
│       └── END (declined)
└── human_handoff → END
```

## LangGraph 和 LangChain 的关系

- LangGraph 是独立包，不依赖 LangChain
- 但原生支持 LangChain 的消息格式（`HumanMessage`, `AIMessage` 等）
- `add_messages` 会把 `dict` 自动转换成 LangChain 消息对象
