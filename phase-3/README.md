# Phase 3：RAG + 向量数据库 — 个人笔记问答助手

## 学习目标

完成本阶段后，你将能够：
- 理解 Embedding 是什么，为什么能做语义搜索
- 独立构建一个"笔记向量化"管道
- 把检索功能封装成 LLM Tool，让 AI 自动决定何时查知识库
- 评测 RAG 质量并进行迭代优化

---

## 三步走

### 第一步：建索引（ingest.py）

```bash
uv run python phase-3/ingest.py
```

会读取 `notes/` 下所有 `.md` 文件，切片向量化，存入本地 ChromaDB。
**首次运行会调用 embedding API，约 1~2 分钟。**

输出示例：
```
[ingest] 01-mcp.md: 8 chunks
[ingest] 02-function-calling.md: 9 chunks
...
[ingest] Done! Stored 42 chunks into 'notes'
```

### 第二步：对话（agent.py）

```bash
uv run python phase-3/agent.py
```

**试试这些问题：**
| 问题 | 预期行为 |
|------|---------|
| `MCP 是什么？` | 触发检索 → 返回笔记内容 |
| `LangGraph 的 interrupt 有什么坑？` | 触发检索 → 引用 03-langgraph.md |
| `今天天气好吗？` | 直接回答，不检索 |
| `什么是余弦相似度？` | 触发检索 → 返回 04-embeddings.md 内容 |

> 💡 关键观察：你没有写"如果问 MCP 就检索"这样的规则——是 LLM 自己决定的！

### 第三步：评测（eval.py）

```bash
uv run python phase-3/eval.py
```

自动运行 15 道题，输出命中率报告。目标：命中率 > 80%。

**如果命中率低，尝试修改 `ingest.py` 里的参数：**
```python
CHUNK_SIZE = 300    # 减小 → 更精准但可能丢失上下文
CHUNK_OVERLAP = 150 # 增大 → 减少边界截断问题
```
修改后重新运行 ingest.py，再跑 eval.py 对比效果。

---

## 课后练习（quiz.py）

```bash
uv run python phase-3/quiz.py
```

**5 个 TODO，从零实现一个 RAG pipeline：**

| TODO | 知识点 |
|------|--------|
| TODO 1 | Fixed-size Chunking with overlap |
| TODO 2 | Ingest pipeline（读文件 → embedding → ChromaDB） |
| TODO 3 | 相似度检索（query embedding → vector search） |
| TODO 4 | OpenAI Tool Schema 定义 |
| TODO 5 | Agent while loop（手写版，不用 LangGraph） |

---

## 文件结构说明

```
phase-3/
├── notes/                 # 5 篇 Markdown 学习笔记（知识库）
├── ingest.py              # 数据管道：笔记 → ChromaDB
├── retriever.py           # 检索工具（search_notes + Tool Schema）
├── agent.py               # 完整 RAG Agent（LangGraph + LLM）
├── eval.py                # 评测脚本
├── quiz.py                # 练习骨架（自己填 TODO）
└── chroma_db/             # ChromaDB 数据目录（运行后自动生成）
```

---

## 核心知识图谱

```
用户问题
   ↓ embedding
查询向量 ── 余弦相似度 ──→ ChromaDB ──→ Top-3 相关片段
                                              ↓
                          [系统 Prompt + 检索结果] → LLM → 最终回答
```

---

## 衔接 Phase 4

Phase 3 给了 Agent 一双"眼睛"（能读知识库）。

Phase 4 要做的是给 Agent 一个"团队"（多 Agent 协作）：
- **Researcher**：负责搜索、总结信息（用 Phase 3 的 RAG Tool）
- **Writer**：把研究结果写成文章
- **Supervisor**：协调 Researcher 和 Writer，做质量把关
