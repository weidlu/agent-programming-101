"""
Phase 3 Quiz - 练习骨架：自己实现一个最小 RAG 管道

目标：不看 ingest.py / retriever.py / agent.py，从头实现关键逻辑

运行方式（先完成 TODO 才能运行）：
    uv run python phase-3/quiz.py

完成标准：
    - 输入"MCP是什么"，能返回笔记里的相关内容
    - 输入"今天天气好吗"，Agent 直接回答而不去检索
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

NOTES_DIR = Path(__file__).parent / "notes"
CHROMA_DIR = Path(__file__).parent / "chroma_db_quiz"  # 用单独目录，不影响主库
COLLECTION_NAME = "quiz_notes"
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
CHAT_MODEL = os.getenv("CHAT_MODEL", "MiniMax-M2.5")


def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )


# ============================================================
# TODO 1: 实现文本切分
# ============================================================
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    把长文本切成有重叠的小块。

    提示：
    - 用 while 循环，start 从 0 开始
    - 每次 end = start + chunk_size
    - 下一个 start = start + chunk_size - overlap（注意：减去 overlap 保证连续性）
    - 记得 strip() 去掉空白，且只保留非空 chunk
    """
    # TODO 1: 实现这个函数
    ...


# ============================================================
# TODO 2: 构建向量库（Ingest）
# ============================================================
def build_index() -> chromadb.Collection:
    """
    读取 notes/ 下所有 .md 文件，切片后向量化存入 ChromaDB。

    步骤：
    1. client.get_or_create_collection(COLLECTION_NAME)
    2. 如果 collection.count() > 0，直接返回（已建索引就不重复建）
    3. 遍历 NOTES_DIR.glob("*.md")，读取文本，调用 chunk_text 切片
    4. 批量调用 client.embeddings.create(model=EMBED_MODEL, input=chunks)
    5. collection.add(ids=[...], documents=[...], embeddings=[...], metadatas=[...])

    提示：
    - id 可以用 f"{file.stem}_chunk_{i}" 格式
    - metadata 里放 {"source": file.name} 方便后续追踪来源
    """
    client = get_client()
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma.get_or_create_collection(COLLECTION_NAME)

    if collection.count() > 0:
        print(f"[quiz] 已有 {collection.count()} 个 chunk，跳过建索引")
        return collection

    # TODO 2: 实现 ingest 逻辑
    ...

    return collection


# ============================================================
# TODO 3: 实现检索函数
# ============================================================
def search(query: str, collection: chromadb.Collection, top_k: int = 3) -> str:
    """
    用 query 在 collection 里做相似度检索，返回格式化字符串。

    步骤：
    1. 用 client.embeddings.create 给 query 生成向量
    2. collection.query(query_embeddings=[...], n_results=top_k, include=["documents","metadatas"])
    3. 把检索结果格式化成：
       [来源: xxx.md]
       片段内容...

       ---
       [来源: yyy.md]
       ...

    提示：results["documents"][0] 是文档列表，results["metadatas"][0] 是元数据列表
    """
    client = get_client()
    # TODO 3: 实现检索
    ...


# ============================================================
# TODO 4: Tool Schema 定义
# ============================================================
# 定义一个 OpenAI function calling schema，让 LLM 知道有 search_notes 这个工具。
#
# 参考格式：
# {
#     "type": "function",
#     "function": {
#         "name": "search_notes",
#         "description": "...",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "query": {"type": "string", "description": "..."},
#             },
#             "required": ["query"],
#         },
#     },
# }
SEARCH_TOOL_SCHEMA: dict = {}  # TODO 4: 填充这个 schema


# ============================================================
# TODO 5: 实现 Agent 主循环
# ============================================================
def agent_loop(collection: chromadb.Collection) -> None:
    """
    实现一个简单的 RAG Agent 循环（不用 LangGraph，用 while loop）：

    每轮循环：
    1. 读取用户输入
    2. 把消息发给 LLM，同时传入 tools=[SEARCH_TOOL_SCHEMA]，tool_choice="auto"
    3. 如果 finish_reason == "tool_calls"：
       a. 解析 tool_calls，拿到 query 参数
       b. 调用 search(query, collection) 拿到检索结果
       c. 把工具结果以 {"role": "tool", ...} 追加到 messages
       d. 再次调用 LLM（这次不传 tools），让它基于检索结果回答
    4. 如果 finish_reason == "stop"：直接打印回答

    提示：
    - messages 列表需要跨轮保持（多轮对话记忆）
    - 别忘了把每轮的用户消息和助手消息都追加到 messages
    """
    client = get_client()
    messages = []

    print("Quiz RAG Agent - 输入 quit 退出")
    while True:
        user_text = input("User: ").strip()
        if user_text.lower() in ("quit", "exit", "q"):
            break

        messages.append({"role": "user", "content": user_text})

        # TODO 5: 实现 agent loop
        ...


# ============================================================
# 思考题
# ============================================================
"""
1. 为什么 ingest 时要用 "get_or_create_collection" 并检查 count() > 0？
   如果每次都重新 ingest 会有什么问题？

2. 在 agent_loop 里，向 LLM 的第二次调用（拿到工具结果后）为什么不传 tools 参数？
   如果传了会怎样？

3. CHUNK_SIZE=500 和 CHUNK_SIZE=2000 分别有什么优缺点？

4. 如果笔记里没有某个问题的答案，RAG Agent 应该怎么回答？
   在 prompt 里应该怎么写规则？

5. 现在的 search() 用的是向量相似度（L2距离）。
   如果换成 BM25（关键词匹配），两种方式各适合什么场景？
"""


def main():
    print("[quiz] 正在建索引（首次运行会调用 embedding API）...")
    collection = build_index()
    agent_loop(collection)


if __name__ == "__main__":
    main()
