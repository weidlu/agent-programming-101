# Phase 4：Multi-Agent 研报生成系统

## 学习目标

完成本阶段后，你将能够：
- 理解 Supervisor-Worker 多 Agent 设计模式
- 用 LangGraph 协调多个 Agent 的工作流
- 实现带质量把关和修订循环的 Agent 系统
- 理解 Agent 最小授权、Handoff、Prompt Injection 防护

---

## 系统架构

```
用户输入主题
     ↓
 Researcher ──→ Supervisor ──→ Writer ──→ Supervisor ──→ 输出文章
     ↑  质量不达标，打回重研究        ↑  质量不达标，打回重写
     └──────────────────────────────┘
```

每个 Agent 只做一件事：
- **Researcher**：用工具搜索 → 输出结构化研究笔记
- **Writer**：读研究笔记 → 生成 Markdown 文章
- **Supervisor**：规则判断质量 → 路由到下一步或打回

---

## 快速开始

```bash
uv run python phase-4/supervisor.py
```

按提示输入研究主题，如：`"LangGraph 在企业级 AI Agent 中的应用"`

**提前准备**（确保 Phase 3 索引已建）：
```bash
uv run python phase-3/ingest.py
```

---

## 文件结构

```
phase-4/
├── notes/                        # 5 篇学习笔记
│   ├── 01-multi-agent-overview.md
│   ├── 02-supervisor-pattern.md
│   ├── 03-agent-communication.md
│   ├── 04-handoff-patterns.md
│   └── 05-mcp-server-build.md
├── tools.py      # 共享工具（复用 Phase 3 的 RAG 检索）
├── researcher.py # Researcher Agent（带 Tool Use 循环）
├── writer.py     # Writer Agent（纯 LLM 生成）
├── supervisor.py # LangGraph 编排入口（主文件）
└── quiz.py       # 练习骨架（5 个 TODO）
```

---

## 课后练习（quiz.py）

```bash
uv run python phase-4/quiz.py
```

| TODO | 考点 |
|------|------|
| TODO 1 | 实现 researcher_node（LLM 生成研究摘要） |
| TODO 2 | 实现 writer_node（基于研究笔记写文章） |
| TODO 3 | 实现 supervisor_node（规则质量判断） |
| TODO 4 | 实现路由函数（stage → 下一节点） |
| TODO 5 | 组装 LangGraph 图 |

---

## 四个阶段的进化路线

```
Phase 1: while loop + Function Calling
         ↓ 学会：LLM 如何"决定"调用工具
Phase 2: LangGraph + Human-in-the-loop
         ↓ 学会：有状态、可暂停的工作流
Phase 3: RAG + LLM Tool Calling
         ↓ 学会：给 Agent 接入私有知识库
Phase 4: Multi-Agent + Supervisor
         ↓ 学会：多 Agent 协作 + 质量把关
         ↓
     下一步：OpenViking（生产级 Agent 上下文管理系统）
```

Phase 4 学完，你就具备了读懂 **OpenViking** 所有核心模块的知识储备：
- `Session` → Phase 2 的 State 管理
- `Compressor` → Phase 4 的消息历史压缩
- `HierarchicalRetriever` → Phase 3 的向量检索进阶版
- `MemoryExtractor` → Phase 4 的 Supervisor 提取关键信息的思路
