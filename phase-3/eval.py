"""
Phase 3 - Step 4: RAG Evaluation Script

Usage:
    # First ingest notes:
    uv run python phase-3/ingest.py

    # Then evaluate:
    uv run python phase-3/eval.py

What it does:
    Runs 15 predefined questions against the RAG retriever and checks
    whether the relevant source file appears in the results.
    Outputs a hit-rate report and suggestions for improvement.

Learning goal:
    RAG is NOT "plug and play". You must measure and iterate.
    This script teaches you: how to think about RAG quality.
"""

from __future__ import annotations

from retriever import search_notes

# ---------------------------------------------------------------------------
# 15 Q&A pairs: (question, expected_source_file)
# A "hit" means the expected source appears in the retrieved results.
# ---------------------------------------------------------------------------
QA_PAIRS = [
    # MCP
    ("MCP 是什么协议？", "01-mcp.md"),
    ("MCP 的 Server 和 Client 分别做什么？", "01-mcp.md"),
    ("MCP 和 Function Calling 有什么区别？", "01-mcp.md"),
    ("什么是 MCP Transport？", "01-mcp.md"),
    # Function Calling
    ("Function Calling 的核心循环是怎么运作的？", "02-function-calling.md"),
    ("finish_reason 是什么意思？", "02-function-calling.md"),
    ("如何防止 Agent 死循环？", "02-function-calling.md"),
    ("什么是 Parallel Tool Calls？", "02-function-calling.md"),
    # LangGraph
    ("LangGraph 中的 State 是什么？", "03-langgraph.md"),
    ("interrupt() 有什么注意事项？", "03-langgraph.md"),
    ("Thread ID 的作用是什么？", "03-langgraph.md"),
    # Embeddings
    ("什么是向量数据库？", "04-embeddings.md"),
    ("Chunking 的 overlap 有什么作用？", "04-embeddings.md"),
    ("余弦相似度是用来做什么的？", "04-embeddings.md"),
    # Agent Patterns
    ("Supervisor-Worker 模式是什么？", "05-agent-patterns.md"),
]


def evaluate(top_k: int = 3) -> None:
    hits = 0
    misses = []

    print(f"{'='*60}")
    print(f"RAG 评测  (top_k={top_k}, 共 {len(QA_PAIRS)} 题)")
    print(f"{'='*60}\n")

    for question, expected_source in QA_PAIRS:
        result = search_notes(question, top_k=top_k)
        hit = expected_source in result
        status = "✅ HIT " if hit else "❌ MISS"
        print(f"{status} | Q: {question[:40]:<40} | expected: {expected_source}")
        if hit:
            hits += 1
        else:
            misses.append((question, expected_source, result[:200]))

    hit_rate = hits / len(QA_PAIRS) * 100
    print(f"\n{'='*60}")
    print(f"命中率: {hits}/{len(QA_PAIRS)} = {hit_rate:.1f}%")
    print(f"{'='*60}")

    if hit_rate < 60:
        print("\n💡 命中率偏低，建议:")
        print("  1. 减小 CHUNK_SIZE（更细粒度的切片）")
        print("  2. 增大 top_k")
        print("  3. 在 ingest.py 里尝试 query rewrite（改写问题再检索）")
    elif hit_rate < 80:
        print("\n💡 命中率一般，可以尝试:")
        print("  1. 增大 CHUNK_OVERLAP 减少边界截断")
        print("  2. 尝试增大 top_k 到 5")
    else:
        print("\n🎉 命中率不错！尝试把 top_k 减到 2 看能否保持精度")

    if misses:
        print(f"\n--- 未命中的题目 ({len(misses)} 道) ---")
        for q, src, snippet in misses:
            print(f"\nQ: {q}")
            print(f"Expected: {src}")
            print(f"Got: {snippet[:150]}...")


if __name__ == "__main__":
    evaluate(top_k=3)
