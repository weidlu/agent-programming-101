# Multi-Agent 系统概述

## 为什么需要多个 Agent？

单一 Agent 的天花板：
- **上下文窗口限制**：处理不了超长的研究任务
- **专注度不足**：同时做"搜索+写作+质量审查"容易出错
- **不可并行**：只能顺序思考，速度慢

Multi-Agent 系统让每个 Agent 只做一件事，做精、做专。

## 三种角色

### Worker Agent（工人）
专注执行具体任务，不关心全局。例如：
- **Researcher**：搜索信息，总结要点
- **Writer**：把摘要写成文章
- **Coder**：根据需求写代码

### Supervisor Agent（主管）
负责任务分配和质量把控。收到 Worker 的输出后，判断：
- 质量够了 → 发给下一个 Worker 或输出最终结果
- 不够好 → 打回去重做，附上改进意见

### Orchestrator（编排器）
更高层的调度器，管理多个 Supervisor 或者复杂的工作流。在 LangGraph 里通常就是图本身。

## 通信方式

### Shared State（共享状态）
所有 Agent 读写同一个 State 对象（LangGraph 的方式）。
```python
class TeamState(BaseModel):
    topic: str
    research_notes: str = ""
    draft: str = ""
    revision_count: int = 0
```

### Message Passing（消息传递）
Agent 之间发送结构化消息。适合松耦合、分布式场景。

### Handoff（交接）
一个 Agent 完成任务后，把控制权 + 上下文"交接"给下一个 Agent。
OpenAI Agents SDK 对此有原生支持（`handoff()` 函数）。

## Multi-Agent 的典型陷阱

### 1. 无限循环
Supervisor 不满意 → 打回 Researcher → 还是不满意 → 循环...

**解决**：设置最大修订次数（`max_revisions`）。

### 2. 上下文爆炸
多轮研究后，messages 历史越来越长，超出 token 限制。

**解决**：Compressor（压缩摘要），或只传最新 N 轮。

### 3. 职责不清
两个 Agent 都在做同一件事，或者互相等对方。

**解决**：每个 Agent 只有一个明确的职责，在 System Prompt 里写清楚边界。
