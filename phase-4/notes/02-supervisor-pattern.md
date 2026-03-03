# Supervisor 模式详解

## 模式结构

```
用户输入
   ↓
Supervisor（协调者）
   ↓ 分配任务
Researcher ──→ [研究结果] ──→ Supervisor（质量把关）
                                    ↓ 不满意：重研究
                                    ↓ 满意：交给 Writer
                             Writer ──→ [文章草稿] ──→ Supervisor
                                                           ↓ 不满意：重写
                                                           ↓ 满意：输出
```

## 在 LangGraph 里实现

LangGraph 天然支持这个模式，因为图的边可以是条件的：

```python
def supervisor_route(state: TeamState) -> str:
    if state.stage == "research":
        if is_good_enough(state.research_notes):
            return "writer"      # 研究够了，交给 Writer
        elif state.revision_count >= MAX_REVISIONS:
            return "writer"      # 超过最大修订次数，强制继续
        else:
            return "researcher"  # 打回去重研究
    elif state.stage == "writing":
        if is_good_enough(state.draft):
            return END           # 完成
        else:
            return "writer"      # 打回去重写
```

## 质量评判：怎么判断"够不够"？

### 方法一：规则检查
```python
def is_research_good_enough(notes: str) -> bool:
    return len(notes) > 200 and notes.count("\n") > 3
```
简单粗暴，但速度快、成本低。适合原型阶段。

### 方法二：LLM 评判
```python
def supervisor_judge(notes: str) -> str:
    """Returns 'continue' or 'revise'"""
    response = llm.chat([
        {"role": "system", "content": "你是严格的质量评审官..."},
        {"role": "user", "content": f"评估以下研究内容：\n{notes}"}
    ])
    return "continue" if "通过" in response else "revise"
```
更智能，但每次都消耗 LLM token。

### 方法三：混合策略（推荐）
先用规则快速过滤（<100字？直接打回），再用 LLM 做精细判断。

## 防止死循环：修订计数器

```python
class TeamState(BaseModel):
    researcher_revisions: int = 0
    writer_revisions: int = 0
    MAX_RESEARCHER_REVISIONS: int = 3
    MAX_WRITER_REVISIONS: int = 2
```

每次打回时 `revision_count += 1`，超出上限强制进入下一步。

## Phase 4 项目里的 Supervisor 实现

我们的 Supervisor 是 LangGraph 的一个节点，它：
1. 读取 `TeamState.stage`（"research" or "writing"）
2. 读取 `TeamState.quality_check`（规则检查结果）
3. 返回下一个节点的名称（路由决策）

Supervisor 本身**不调用 LLM**——因为我们用规则就够了。
真实项目里可以换成 LLM 评判，只需修改 `supervision_route` 函数。
