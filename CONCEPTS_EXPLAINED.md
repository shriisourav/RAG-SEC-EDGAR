# 📚 RAG Concepts Explained
## Interview-Ready Reference Guide

---

## 🧩 1. Chunking vs Tokens

### What is a Token?
A **token** is the smallest unit of text that an AI model processes. It's NOT the same as a word.

| Text | Tokens | Why? |
|------|--------|------|
| "hello" | 1 token | Common word |
| "JPMorgan" | 2 tokens | "JP" + "Morgan" |
| "cryptocurrency" | 3 tokens | "crypt" + "o" + "currency" |
| "10-K" | 3 tokens | "10" + "-" + "K" |

**Rule of thumb:** 1 token ≈ 0.75 words (or 4 characters)

### What is Chunking?
**Chunking** = Splitting a large document into smaller pieces (chunks)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Large 10-K Document (1.2 MB)                 │
│   "JPMorgan Chase & Co. is a financial holding company..."     │
└─────────────────────────────────────────────────────────────────┘
                              ↓ CHUNKING
┌──────────────┐ ┌──────────────┐ ┌──────────────┐     ┌──────────────┐
│  Chunk 1     │ │  Chunk 2     │ │  Chunk 3     │ ... │  Chunk 508   │
│  ~600 tokens │ │  ~600 tokens │ │  ~600 tokens │     │  ~600 tokens │
└──────────────┘ └──────────────┘ └──────────────┘     └──────────────┘
```

### Why Chunk?
1. **LLM context limits**: GPT-4 can only read ~128K tokens at once
2. **Precise retrieval**: Find specific relevant sections, not entire docs
3. **Cost efficiency**: Process less tokens = cheaper API calls

### Chunk Parameters
| Parameter | Value Used | Purpose |
|-----------|------------|---------|
| **Chunk Size** | 600 tokens | Balance of context vs precision |
| **Overlap** | 100 tokens | Prevent cutting sentences mid-thought |
| **Min Size** | 100 tokens | Don't create tiny useless chunks |

### Overlap Explained
```
Chunk 1:  [==================]
Chunk 2:        [==================]  ← 100 tokens overlap with Chunk 1
Chunk 3:              [==================]
```
Overlap ensures important information at chunk boundaries isn't lost.

---

## 🧠 2. Embeddings & Dimensions

### What is an Embedding?
An **embedding** converts text into numbers (a vector) that captures **meaning**.

```
Text: "The bank manages credit risk"
         ↓ Embedding Model
Vector: [0.23, -0.45, 0.12, 0.89, -0.34, ...]  ← 384 numbers
```

### The Magic: Similar Meanings → Similar Vectors
```
"The bank manages credit risk"     → [0.23, -0.45, 0.12, ...]
"Credit risk management at banks"  → [0.25, -0.43, 0.14, ...]  ← SIMILAR! (close numbers)
"I love pizza"                     → [-0.82, 0.67, -0.91, ...] ← DIFFERENT! (far apart)
```

### What are Dimensions?
Each number in the vector represents one "dimension" of meaning:

| Dimension | What it might capture (simplified) |
|-----------|-----------------------------------|
| Dim 1 | Is it about finance? (+) or food? (-) |
| Dim 2 | Is it positive? (+) or negative? (-) |
| Dim 3 | Is it about risk? (+) or opportunity? (-) |
| ... | ... |
| Dim 384 | Some other learned pattern |

### Dimension Comparison
| Model | Dimensions | Quality | Speed | Cost |
|-------|------------|---------|-------|------|
| all-MiniLM-L6-v2 | 384 | Good | ⚡ Fast | Free |
| all-mpnet-base-v2 | 768 | Better | Medium | Free |
| text-embedding-3-small | 1536 | Great | Fast | $ |
| text-embedding-3-large | 3072 | Best | Slow | $$ |

**We used 384 dimensions** - best balance of speed and quality for local RAG.

---

## 🗄️ 3. Vector Database Explained

### What is a Vector Database?
A database optimized to store and search **vectors** (lists of numbers).

### The Problem It Solves
```
You have: 2,332 chunks, each with 384 numbers
User asks: "What is JPM's credit risk?"
Need: Find the most similar chunks FAST

Naive approach: Compare query to ALL 2,332 chunks = SLOW ❌
Vector DB: Use smart algorithms (HNSW, IVF) = FAST ✅
```

### How Similarity Search Works
```
User Query: "What is JPM's credit risk?"
     ↓
Convert to vector: [0.31, -0.22, 0.77, ...]
     ↓
┌─────────────────────────────────────────┐
│         VECTOR DATABASE                 │
│   Search using cosine similarity        │
│   Find vectors "closest" to query       │
└─────────────────────────────────────────┘
     ↓
Top 5 most similar chunks:
1. Chunk 847 (similarity: 0.89) ← About credit risk!
2. Chunk 849 (similarity: 0.85)
3. ...
```

### Similarity Metrics
| Metric | Formula | Best For |
|--------|---------|----------|
| **Cosine** | angle between vectors | Text similarity (most common) |
| **Euclidean** | straight-line distance | When magnitude matters |
| **Dot Product** | vector multiplication | Normalized embeddings |

---

## 🆚 4. ChromaDB vs FAISS vs Other Vector DBs

### Quick Comparison

| Feature | ChromaDB | FAISS | Pinecone | Weaviate | Qdrant |
|---------|----------|-------|----------|----------|--------|
| **Type** | Local/Cloud | Local only | Cloud only | Cloud/Local | Cloud/Local |
| **Setup** | 1 line | Medium | Easy | Complex | Medium |
| **Metadata** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Persistence** | ✅ Yes | Manual | ✅ Yes | ✅ Yes | ✅ Yes |
| **Cost** | Free | Free | Paid | Freemium | Freemium |
| **Scale** | Medium | Massive | Massive | Large | Large |
| **Best For** | Prototyping, small-medium apps | Pure speed, research | Production, enterprise | Complex queries | Performance |

### Detailed Breakdown

#### ChromaDB (What we used) ✅
```python
pip install chromadb

import chromadb
client = chromadb.PersistentClient(path="./db")
collection = client.create_collection("docs")
collection.add(documents=["text"], embeddings=[[0.1, 0.2, ...]], ids=["id1"])
```
**Pros:**
- Dead simple setup
- Built-in embedding functions
- Stores metadata with vectors
- Persists to disk
- Python-native

**Cons:**
- Not for billion-scale data
- Newer, less battle-tested

#### FAISS (Facebook AI)
```python
pip install faiss-cpu

import faiss
index = faiss.IndexFlatL2(384)  # 384 dimensions
index.add(vectors)
distances, indices = index.search(query_vector, k=5)
```
**Pros:**
- Blazing fast (C++ core)
- Handles billions of vectors
- GPU support
- Research-proven

**Cons:**
- No metadata storage
- No persistence (manual save/load)
- Steeper learning curve

#### Pinecone (Cloud)
```python
pip install pinecone-client

import pinecone
pinecone.init(api_key="key")
index = pinecone.Index("my-index")
index.upsert(vectors=[("id1", [0.1, 0.2, ...], {"key": "value"})])
```
**Pros:**
- Fully managed (no infra)
- Massive scale
- Real-time updates
- Great dashboard

**Cons:**
- Paid service
- Requires internet
- Data leaves your system

#### When to Use What?

| Scenario | Recommendation |
|----------|----------------|
| Learning/Prototyping | **ChromaDB** |
| Speed-critical research | **FAISS** |
| Production with budget | **Pinecone** or **Qdrant Cloud** |
| Self-hosted production | **Weaviate** or **Qdrant** |
| Existing Postgres | **pgvector** |

---

## 🔄 5. Complete RAG Pipeline

### What is RAG?
**R**etrieval-**A**ugmented **G**eneration = Give the LLM relevant context before asking it to answer.

### Why RAG?
| Problem | Without RAG | With RAG |
|---------|-------------|----------|
| LLM knowledge cutoff | "I don't know about 2024" | "Based on the 2024 10-K..." |
| Hallucination | Makes up facts | Cites actual documents |
| Specificity | Generic answers | Company-specific answers |
| Verifiability | "Trust me" | "Source: JPM 10-K page 47" |

### RAG Pipeline Architecture
```
                        ┌─────────────────────────────────┐
                        │         RAG PIPELINE            │
                        └─────────────────────────────────┘

INDEXING (Offline - Done Once):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   10-K      │ →→ │   Chunk     │ →→ │  Embedding  │ →→ │  Vector     │
│   Documents │    │   (split)   │    │   Model     │    │   Database  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

QUERYING (Online - Per Question):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   User      │ →→ │  Embedding  │ →→ │  Vector     │ →→ │   Top-K     │
│   Question  │    │   Model     │    │   Search    │    │   Chunks    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                               ↓
┌─────────────┐    ┌─────────────────────────────────────────────────┐
│   Answer    │ ←← │  LLM (GPT-4/Gemini): "Based on the context:     │
│   + Source  │    │  JPM's main risks include..." [Source: ...]     │
└─────────────┘    └─────────────────────────────────────────────────┘
```

### Key Components

| Component | Our Implementation | Purpose |
|-----------|-------------------|---------|
| **Document Loader** | `A_SEC_EDGAR.py` | Download 10-K filings |
| **Chunker** | `B_Chunking_Indexing.py` | Split into 600-token chunks |
| **Embedding Model** | all-MiniLM-L6-v2 | Convert text → 384D vectors |
| **Vector Store** | ChromaDB | Store and search vectors |
| **Retriever** | `C_Retrieval.py` | Find top-5 relevant chunks |
| **Generator** | `D_Generation.py` | LLM generates answer with citations |
| **API** | `api.py` (FastAPI) | REST API for queries |

### Our Numbers

| Metric | Value |
|--------|-------|
| Documents | 3 (JPM, GS, UBS 10-Ks) |
| Total Chunks | 2,332 |
| Chunk Size | 600 tokens (~450 words) |
| Vector Dimensions | 384 |
| Retrieval Time | ~38ms |
| Top-K Retrieved | 5 chunks per query |

---

## 💡 Interview Tips

### Common Questions & Answers

**Q: Why not just use the full document?**
> A: LLMs have context limits (GPT-4: 128K tokens). A 10-K can have 300K+ tokens. Also, retrieving specific chunks is more precise and cost-effective.

**Q: Why 600 tokens per chunk?**
> A: Sweet spot between context (enough info to be useful) and precision (specific enough to match queries). Industry standard is 500-800.

**Q: Why overlap chunks?**
> A: Sentences at chunk boundaries might get cut off. 100-token overlap ensures continuity.

**Q: ChromaDB vs FAISS?**
> A: ChromaDB for ease of use + metadata. FAISS for pure speed at massive scale. We prioritized developer experience.

**Q: How do you prevent hallucination?**
> A: 1) System prompt enforces "only answer from context", 2) Citations required, 3) Confidence scoring, 4) "I don't know" responses when context insufficient.

**Q: How would you scale this?**
> A: 1) Cloud vector DB (Pinecone/Qdrant), 2) Async processing, 3) Caching frequent queries, 4) Multiple retrieval strategies (hybrid search).

---

## 📊 Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG QUICK REFERENCE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TOKEN: Smallest unit LLM processes (~0.75 words)               │
│  CHUNK: Document piece (~600 tokens) for retrieval              │
│  EMBEDDING: Text → Vector (list of numbers)                     │
│  DIMENSION: Each number in vector (384 for our model)           │
│  VECTOR DB: Database optimized for similarity search            │
│  SIMILARITY: How "close" two vectors are (0-1 scale)            │
│  TOP-K: Number of chunks to retrieve (we use 5)                 │
│  RAG: Retrieval-Augmented Generation                            │
│                                                                 │
│  FORMULA:                                                       │
│  Query → Embed → Search VectorDB → Get Chunks → LLM → Answer    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Last updated: January 2026*
*Project: SEC EDGAR 10-K RAG Pipeline*
