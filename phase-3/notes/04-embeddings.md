# Embeddings（向量嵌入）

## 什么是 Embedding？

Embedding 是把文本转换成高维数值向量的过程。语义相似的文本，向量之间的距离也近。

```
"MCP 是什么协议？" → [0.12, -0.34, 0.78, ...]  ← 1536维向量
"Model Context Protocol 介绍" → [0.13, -0.32, 0.80, ...]  ← 方向接近！
"今天天气怎么样？" → [-0.45, 0.21, -0.60, ...]  ← 方向完全不同
```

## 用 OpenAI API 生成 Embedding

```python
from openai import OpenAI

client = OpenAI()
response = client.embeddings.create(
    model="text-embedding-3-small",  # 1536维，性价比高
    input="MCP 是什么？"
)
vector = response.data[0].embedding  # list of 1536 floats
```

## 相似度计算：余弦相似度

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 值域：-1 到 1，越接近 1 越相似
score = cosine_similarity(query_vector, doc_vector)
```

## 向量数据库：ChromaDB

ChromaDB 是本地轻量级向量数据库，不需要额外服务，一行代码安装：

```bash
pip install chromadb
```

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")  # 持久化到本地
collection = client.get_or_create_collection("notes")

# 存入文档
collection.add(
    ids=["doc1", "doc2"],
    documents=["MCP 是一个标准协议...", "Function Calling 是..."],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]]  # 预计算好的向量
)

# 查询
results = collection.query(
    query_embeddings=[[0.1, 0.2, ...]],  # 查询向量
    n_results=3
)
print(results["documents"])
```

## Chunking（文本切分）

为什么需要切分？
- 文档太长，超出 embedding 模型的 token 限制
- 检索时需要精准的局部片段，而不是整篇文章

### Fixed-size Chunking

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # overlap 保证上下文连续
    return chunks
```

### Overlap 的作用

```
原文: AAAAAABBBBBBCCCCCC
chunk1: AAAAAABB      (包含AB交界)
chunk2:     BBBBBBCC  (包含BC交界)
```

没有 overlap，如果关键信息恰好在两个 chunk 的交界处，可能被切断导致检索失败。

## RAG 完整流程

### 构建阶段（Ingestion）
```
Markdown文件 → 读取文本 → Chunking → Embedding → 存入ChromaDB
```

### 检索阶段（Retrieval）
```
用户问题 → Embedding → 向量相似度搜索 → Top-K 片段 → 拼入 Prompt
```

### 关键参数
- `chunk_size`：每块大小（字符数），影响检索精度
- `overlap`：重叠大小，防止关键信息被截断
- `top_k`：返回最相关的 K 个片段
