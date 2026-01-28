# 🏆 AI & LLM Complete Learning Guide 2026
## The Ultimate Interview Reference & Learning Journey

Author: Sourav Shrivastava  
Last Updated: January 2026  
Project: SEC EDGAR 10-K RAG Pipeline

---

# 📖 How to Use This Guide

This guide tells the story of building an AI system from scratch. Each concept builds on the previous one, following a natural learning journey:

PART I: Foundations (Tokens → Embeddings → Attention → Transformers)  
PART II: Large Language Models (LLMs → Fine-tuning → Prompting)  
PART III: RAG Systems (Chunking → Vector DBs → Retrieval → Generation)  
PART IV: Production (Agents → Infrastructure → Azure/Databricks)  
PART V: Interview Prep (Q&A → Quick Reference)

---

# PART I: FOUNDATIONS

## Chapter 1: Tokenization - Breaking Text into Pieces

### The Story
Before an AI can understand "JPMorgan's credit risk increased by 15%", it must break this sentence into digestible pieces. This is tokenization - the very first step.

### What Is It?
Tokenization converts raw text into tokens - the atomic units an AI model processes.

Example:
```
"JPMorgan's credit risk" → ["JP", "Morgan", "'s", " credit", " risk"]
                              ↓      ↓       ↓       ↓         ↓
                          Token 1  Token 2  Token 3  Token 4   Token 5
```

### Types of Tokenization

Word-level: Split on spaces → "Hello world" → ["Hello", "world"]

Character-level: Each char is a token → "Hi" → ["H", "i"]

Subword (BPE): Frequent patterns → "playing" → ["play", "ing"]

SentencePiece: Language-agnostic subword (used by LLaMA, T5)

### TikToken - OpenAI's Tokenizer

```python
import tiktoken

# Get tokenizer for GPT-4
enc = tiktoken.encoding_for_model("gpt-4")

text = "What are JPMorgan's main risk factors?"
tokens = enc.encode(text)
print(f"Token count: {len(tokens)}")  # Output: 9
```

### Token Rules of Thumb

- 1 token ≈ 0.75 words ≈ 4 characters
- Common words = 1 token
- "JPMorgan" = 2 tokens (JP + Morgan)
- Numbers: each digit often = 1 token
- Punctuation = 1 token each

### Why Tokens Matter

1. COST: You pay per token (GPT-4o: $2.50 per 1M input tokens)
2. CONTEXT LIMITS: GPT-4 = 128K tokens, Claude = 200K tokens
3. CHUNKING: Must fit chunks within these limits

### Interview Q&A

Q: Why don't we just use words as tokens?
A: Three reasons: (1) Vocabulary explosion - millions of words, (2) OOV problem - new words wouldn't exist, (3) Morphology - "running/ran/runs" are related but different.

Q: What is BPE (Byte Pair Encoding)?
A: BPE starts with characters and iteratively merges frequent pairs. "lo" + "w" → "low". Creates subword units balancing vocabulary size and meaning.

---

## Chapter 2: Embeddings - Giving Text Meaning

### The Story
Now that we have tokens, we need to give them meaning. Computers don't understand "credit" or "risk" - they need numbers.

### What Is It?
Embedding = Converting text into dense numerical vectors that capture semantic meaning.

Example:
```
"bank" (financial) → [0.8, -0.2, 0.5, ...]   ← 384 dimensions
"bank" (river)     → [-0.1, 0.7, -0.3, ...]  ← Different meaning!
"finance"          → [0.75, -0.15, 0.45, ...] ← Similar to financial bank!
```

### The Magic: Similar Meanings → Similar Vectors

```
"credit risk" ↔ "loan default"    = 0.89 similarity (HIGH)
"credit risk" ↔ "market risk"     = 0.72 similarity (MEDIUM)
"credit risk" ↔ "pizza recipe"    = 0.12 similarity (LOW)
```

### Code Example

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

texts = ["JPMorgan's credit risk", "Bank loan default", "Pizza recipe"]
embeddings = model.encode(texts)

print(f"Embedding shape: {embeddings.shape}")  # (3, 384)
```

### Embedding Model Comparison

all-MiniLM-L6-v2: 384 dims, Very Fast, Good quality, Free - Best for prototyping

all-mpnet-base-v2: 768 dims, Fast, Better quality, Free - Balanced choice

text-embedding-3-small: 1536 dims, Fast, Great quality, Paid - Production

text-embedding-3-large: 3072 dims, Slow, Best quality, Paid - High accuracy

### Interview Q&A

Q: How do embeddings capture semantic meaning?
A: During training, embeddings are optimized so similar texts have vectors close in high-dimensional space. The model learns patterns from billions of examples.

Q: Why do dimensions matter?
A: More dimensions = more capacity for nuance, but also more compute/storage. 384 dims work for most cases; 3072 for specialized domains.

---

## Chapter 3: Attention Mechanism - Understanding Context

### The Story
We can represent words as numbers, but how does AI understand that "bank" in "river bank" differs from "bank account"? The answer is attention.

### What Is It?
Attention allows the model to focus on relevant parts of input when processing each word.

Example:
```
"The animal didn't cross the street because it was too tired"
                                            ↑
                                      What does "it" refer to?
                                      
Attention reveals: "it" → attends to → "animal" (not "street")
```

### Self-Attention Formula

```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V

Where:
  Q (Query): "What am I looking for?"
  K (Key):   "What do I contain?"
  V (Value): "What information do I provide?"
  d_k:       Dimension of keys (for scaling)
```

### Multi-Head Attention

```
Input → [Head 1: syntax patterns  ]
      → [Head 2: semantic meaning ] → Concat → Linear → Output
      → [Head 3: position relations]
      → [Head 4: entity references ]

8-12 heads is typical for base models
```

### Interview Q&A

Q: What problem does attention solve that RNNs couldn't?
A: RNNs process sequentially, losing information over long distances. Attention provides direct connections between any positions, enabling O(1) path length.

Q: Why divide by √d_k?
A: Without scaling, dot products grow large for high dimensions, pushing softmax into regions with tiny gradients.

---

## Chapter 4: Transformer Architecture

### The Story
Attention was revolutionary, but it's just one component. The Transformer combines attention with other innovations to power GPT, BERT, and every modern LLM.

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                   TRANSFORMER                    │
├─────────────────────┬───────────────────────────┤
│      ENCODER        │        DECODER             │
│                     │                            │
│  Input Embedding    │  Output Embedding          │
│  + Positional Enc   │  + Positional Encoding     │
│        ↓            │         ↓                  │
│  Multi-Head         │  Masked Multi-Head         │
│  Self-Attention     │  Self-Attention            │
│        ↓            │         ↓                  │
│  Add & Norm         │  Add & Norm                │
│        ↓            │         ↓                  │
│  Feed-Forward       │  Cross-Attention           │
│        ↓            │  (attends to encoder)      │
│  Add & Norm         │         ↓                  │
│        ↓            │  Feed-Forward              │
│   (× N layers)      │   (× N layers)             │
└─────────────────────┴───────────────────────────┘
```

### Transformer Variants

Encoder-only (BERT, RoBERTa): Bidirectional, for classification, NER

Decoder-only (GPT-4, LLaMA): Autoregressive, for text generation

Encoder-Decoder (T5, BART): Seq2Seq, for translation, summarization

### Interview Q&A

Q: Why are transformers better than LSTMs?
A: (1) Parallelization, (2) No vanishing gradients, (3) Direct attention to any position, (4) Better scaling.

Q: What are positional encodings?
A: Attention is permutation-invariant (doesn't know word order). Positional encodings add position information using sine/cosine or learned embeddings.

---

# PART II: LARGE LANGUAGE MODELS

## Chapter 5: LLMs - The Power of Scale

### The Story
With transformers and self-supervision, we can scale up. Large Language Models are transformers trained on massive text, capable of human-like text generation.

### LLM Size Comparison

```
BERT-base:     110M parameters    ████
GPT-2:         1.5B parameters    ████████
GPT-3:         175B parameters    ████████████████████████
GPT-4:         ~1.8T parameters*  ████████████████████████████
LLaMA-2 70B:   70B parameters     ████████████████
```
*estimated

### Key LLM Capabilities

In-context learning: Few examples → new task (pattern matching)

Chain-of-thought: Step-by-step reasoning (scratchpad)

Instruction following: "Summarize this" (RLHF alignment)

Code generation: Write Python (trained on code repos)

### Interview Q&A

Q: What are "emergent capabilities"?
A: Abilities that appear suddenly at certain scales - not present in smaller models. Examples: arithmetic (10B+), chain-of-thought (100B+).

Q: How do LLMs "know" facts?
A: Facts stored in parameters during pre-training. But: (1) Frozen at cutoff, (2) Can be incorrect, (3) Hard to update. RAG solves this.

---

## Chapter 6: LLM Parameters - Temperature, Top-P & More

### Parameter Overview

TEMPERATURE (0.0 - 2.0): Controls randomness
- 0.0 = Deterministic (same output each time)
- 0.7 = Balanced (default)
- 2.0 = Very random (may be incoherent)

TOP_P / Nucleus Sampling (0.0 - 1.0): Limits token pool
- 0.1 = Only highest probability tokens
- 0.9 = Most tokens considered

MAX_TOKENS: Maximum output length

FREQUENCY_PENALTY: Penalizes repeated tokens

PRESENCE_PENALTY: Encourages new topics

### Temperature Visualization

```
Temperature = 0 (Deterministic):
  "bank"     ████████████████ 80%  ← Always picks "bank"
  "company"  ███              15%
  "firm"     █                 5%

Temperature = 1.0 (Balanced):
  "bank"     ████████████ 50%
  "company"  ████████     33%      ← Sometimes "company"
  "firm"     ████         17%
```

### Best Practices by Use Case

RAG/Factual Q&A: Temperature 0.0-0.3, Top_P 0.9 (accuracy first)

Code Generation: Temperature 0.0-0.2, Top_P 0.95 (precision needed)

Creative Writing: Temperature 0.7-1.0, Top_P 0.9 (variety desired)

Trade Surveillance: Temperature 0.0-0.2, Top_P 0.9 (compliance = accuracy)

---

## Chapter 7: Fine-tuning vs RAG

### Fine-tuning Spectrum

Full Fine-tuning: All params (most compute, best quality)

LoRA/QLoRA: ~0.1% params (low compute, nearly same quality)

Adapter Layers: 1-5% params (add small trainable modules)

Prompt Tuning: <0.01% params (only tune soft prompts)

### LoRA Explained

```
Original:  W (frozen) → Output
LoRA:      W (frozen) + B×A → Output
                 ↑
           Low-rank decomposition
           A: d × r  (r << d)
           B: r × d
           
Instead of updating 10M params, update 100K!
```

### When to Use What

Use RAG when:
- Data changes frequently
- Need source citations
- Factual accuracy critical
- Limited compute budget
- Explainability required

Use Fine-tuning when:
- Data is static
- Need style/tone change
- Domain-specific jargon
- Want faster inference
- Smaller model needed

Use BOTH when:
- Fine-tune for domain understanding
- RAG for current, verifiable facts

### Cost Comparison

RAG only: ~$50 upfront, ~$0.01/query, ~$0.50 to update

Fine-tune only: ~$500-5000 upfront, ~$0.001/query, ~$500 to retrain

RAG + Fine-tune: Best of both worlds

---

# PART III: RAG SYSTEMS

## Chapter 8: Chunking - Splitting Documents

### The Story
A 10-K filing has 300K+ tokens. LLMs have limits (GPT-4: 128K). We need to split documents into manageable chunks.

### Chunking Visualization

```
┌─────────────────────────────────────────────────────────┐
│            Large 10-K Document (1.2 MB)                 │
│  "JPMorgan Chase & Co. is a financial holding..."      │
└─────────────────────────────────────────────────────────┘
                          ↓ CHUNKING
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Chunk 1     │ │  Chunk 2     │ │  Chunk 3     │ ...
│  ~600 tokens │ │  ~600 tokens │ │  ~600 tokens │
└──────────────┘ └──────────────┘ └──────────────┘
```

### Chunk Parameters

Chunk Size: 600 tokens - Balance of context vs precision

Overlap: 100 tokens - Prevent cutting sentences mid-thought

Min Size: 100 tokens - Don't create tiny useless chunks

### Overlap Explained

```
Chunk 1:  [==================]
Chunk 2:        [==================]  ← 100 token overlap
Chunk 3:              [==================]
```
Overlap ensures information at chunk boundaries isn't lost.

### Interview Q&A

Q: Why not use the full document?
A: (1) Context limits, (2) Cost (more tokens = more $), (3) Precision (specific chunks match better).

Q: Why 600 tokens?
A: Sweet spot between context (enough info) and precision (specific enough). Industry standard: 500-800.

---

## Chapter 9: Vector Databases

### The Story
We have 2,332 chunks, each with 384-dimensional embeddings. We need to find the most similar chunks FAST. Vector databases do this.

### The Problem

```
You have: 2,332 chunks, each with 384 numbers
User asks: "What is JPM's credit risk?"
Need: Find most similar chunks FAST

Naive approach: Compare to ALL 2,332 = SLOW ❌
Vector DB: Smart algorithms (HNSW, IVF) = FAST ✅
```

### Vector DB Comparison

ChromaDB: Local/Cloud, Has metadata, Medium scale, Free - Best for prototyping (we used this)

FAISS: Local only, No metadata, Massive scale, Free - Best for pure speed

Pinecone: Cloud only, Has metadata, Massive scale, Paid - Best for production

Qdrant: Both, Has metadata, Large scale, Freemium - Best for self-hosted production

### Similarity Metrics

Cosine: Angle between vectors - Best for text (most common)

Euclidean: Straight-line distance - When magnitude matters

Dot Product: Vector multiplication - For normalized embeddings

### Interview Q&A

Q: How does HNSW work?
A: Hierarchical Navigable Small World graphs. Multi-layer graph where each layer is a "zoom level." O(log N) search complexity.

Q: ChromaDB vs FAISS?
A: ChromaDB for ease + metadata. FAISS for pure speed at massive scale.

---

## Chapter 10: The Complete RAG Pipeline

### RAG Architecture

```
INDEXING (Offline - Done Once):
Documents → Chunk → Embed → Store in Vector DB

QUERYING (Online - Per Question):
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Question │ →→ │  Embed   │ →→ │  Search  │
│          │    │  Query   │    │ VectorDB │
└──────────┘    └──────────┘    └────┬─────┘
                                     ↓
                             ┌──────────────┐
                             │  Top-K Docs   │
                             └──────┬───────┘
                                    ↓
┌──────────────────────────────────────────────────┐
│  PROMPT = "Context: {docs}\n\nQuestion: {q}"   │
└──────────────────────────────────────────────────┘
                                    ↓
                             ┌──────────────┐
                             │     LLM      │
                             └──────┬───────┘
                                    ↓
                       ┌────────────────────────┐
                       │  Answer + Citations    │
                       └────────────────────────┘
```

### Our Implementation (SEC EDGAR Project)

Document Loader: A_SEC_EDGAR.py - Download 10-K filings

Chunker: B_Chunking_Indexing.py - 600-token chunks

Embedding Model: all-MiniLM-L6-v2 - 384D vectors

Vector Store: ChromaDB - Store and search

Retriever: C_Retrieval.py - Top-5 chunks

Generator: D_Generation.py - LLM with citations

API: api.py (FastAPI) - REST endpoint

### The RAG Code Location

```python
# D_Generation.py - RAGEngine.query() method

def query(self, question: str, k: int = 5) -> RAGResponse:
    # STEP 1: RETRIEVAL (R)
    chunks = self.retriever.retrieve(query=question, k=k)
    
    # STEP 2: AUGMENTATION (A)
    context = self._format_context(chunks)
    
    # STEP 3: GENERATION (G)
    answer = self.llm.generate(SYSTEM_PROMPT, context + question)
    
    return RAGResponse(answer=answer, citations=...)
```

### Our Numbers

Documents: 3 (JPM, GS, UBS 10-Ks)

Total Chunks: 2,332

Chunk Size: 600 tokens (~450 words)

Vector Dimensions: 384

Retrieval Time: ~38ms

Top-K Retrieved: 5 chunks per query

---

## Chapter 11: Preventing Hallucination

### Types of Hallucination

Factual: "JPMorgan was founded in 1750" (wrong date)

Fabrication: Citing a paper that doesn't exist

Conflation: Mixing up Goldman Sachs and Morgan Stanley

Extrapolation: "Q4 2025 revenue will be..." (future prediction)

### Multi-Layer Prevention

```
┌─────────────────────────────────────────────────────────┐
│            HALLUCINATION PREVENTION STACK               │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Layer 1: RETRIEVAL                                     │
│  - Only use high-similarity chunks (threshold > 0.4)   │
│  - Include source metadata with every chunk            │
│                                                         │
│  Layer 2: PROMPT ENGINEERING                            │
│  - System prompt: "ONLY answer from provided context"  │
│  - Require: "If not in context, say 'I don't know'"    │
│  - Force citation format: [Source: Company - Section]  │
│                                                         │
│  Layer 3: POST-PROCESSING                               │
│  - Verify citations exist in retrieved chunks          │
│  - Confidence scoring (HIGH/MEDIUM/LOW)                │
│                                                         │
│  Layer 4: EVALUATION                                    │
│  - Hallucination-trigger test cases                    │
│  - Human-in-the-loop for critical responses            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Our System Prompt

```python
SYSTEM_PROMPT = """
CRITICAL RULES:
1. ONLY answer based on provided context from 10-K filings
2. If context doesn't have info, say "I cannot find this"
3. NEVER make up information
4. ALWAYS cite sources: [Source: Company - Section]
5. If asked about unknown company, state you don't have it
"""
```

---

# PART IV: PRODUCTION & ENTERPRISE

## Chapter 12: RAG vs MCP

### What is MCP?
MCP (Model Context Protocol) = Anthropic's standard for connecting LLMs to external tools/data.

### Key Difference

```
RAG:
Query → Embed → Search → Get Docs → Add to Prompt → LLM
(Pre-retrieval, static pipeline)

MCP:
Query → LLM → "I need data from X" → Tool Call → Get Data → Continue
(Dynamic, on-demand tool use)
```

### When to Use Each

RAG: Document Q&A, known corpus, simple retrieval

MCP: Multi-step research, dynamic data, complex agentic workflows

Both: MCP can USE RAG as one of its tools

---

## Chapter 13: Multi-Agent Systems

### What is an Agent?
Agent = LLM + Tools + Memory + Goal

```
┌─────────────────────────────────────────────────────────┐
│                   ANATOMY OF AN AGENT                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  GOAL:   "Research JPM's risk factors and compare"     │
│                     ↓                                   │
│  LLM:    Reasoning engine (GPT-4, Claude)              │
│          Decides what to do next                        │
│                     ↓                                   │
│  TOOLS:  • RAG search    • Web browse                  │
│          • Calculator    • Code execution              │
│          • API calls     • File read/write             │
│                     ↓                                   │
│  MEMORY: • Conversation history                        │
│          • Previous findings                           │
│          • Task state                                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Agent Communication Patterns

Hierarchical: Manager → Worker agents

Peer-to-Peer: Agents debate/collaborate

Sequential: Pipeline of specialized agents

### Interview Q&A

Q: Chain vs Agent?
A: Chain = fixed sequence. Agent = LLM decides which steps dynamically.

Q: Risks of autonomous agents?
A: Runaway costs, hallucinated actions, security (prompt injection), unexpected behavior.

---

## Chapter 14: LLM Frameworks

### Framework Landscape

ORCHESTRATION:
- LangChain: Most popular, "Swiss Army knife"
- LlamaIndex: Focused on data/retrieval
- Haystack: Production-ready pipelines

AGENT FRAMEWORKS:
- LangGraph: Stateful multi-agent graphs
- AutoGen: Microsoft's multi-agent
- CrewAI: Role-based agent teams

LOCAL LLM RUNNING:
- Ollama: Easiest local runner
- LM Studio: GUI for local models
- vLLM: High-performance inference

### When to Use What

Building a chatbot? → LangChain

Building RAG system? → LlamaIndex or Raw (like we did)

Building multi-agent? → LangGraph or CrewAI

Running locally? → Ollama

Need max control? → Build from scratch (our approach)

---

## Chapter 15: GPU vs CPU

### Why GPUs for AI?

```
CPU: 4-64 powerful cores → Sequential processing
     Great for: Logic, branching, single-threaded

GPU: 1000s of cores → Parallel processing
     Great for: Matrix math (LLMs are matrix operations!)
```

### When to Use What

LLM Training: GPU (A100, H100) - Massive parallel ops

LLM Inference (large): GPU - 70B+ models need VRAM

LLM Inference (small): CPU or GPU - 7B quantized runs on CPU

Embedding Generation: CPU often fine - Small models

Vector Search: CPU - Memory-bound, not compute-bound

Our RAG Pipeline: CPU + API - Retrieval on CPU, LLM via API

### GPU Memory Requirements

7B params: FP16: 14GB, INT8: 7GB, INT4: 4GB

13B params: FP16: 26GB, INT8: 13GB, INT4: 7GB

70B params: FP16: 140GB, INT8: 70GB, INT4: 35GB

### Quantization

```
FP32 (32-bit): 1.234567890123456 → Most accurate, 4 bytes
FP16 (16-bit): 1.234567          → Good balance, 2 bytes
INT8 (8-bit):  1.23              → 2x smaller, slight loss
INT4 (4-bit):  1.2               → 4x smaller, more loss
```

For most RAG: INT4 or INT8 is sufficient

---

## Chapter 16: Azure + Databricks + Copilot

### Architecture for Trade Surveillance

```
┌─────────────────────────────────────────────────────────┐
│                    AZURE CLOUD                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │
│  │Azure OpenAI │ │ Azure Blob  │ │ Azure SQL   │      │
│  │(GPT-4)      │ │ Storage     │ │ Database    │      │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘      │
│         └───────────────┼───────────────┘              │
│                         ↓                               │
│              ┌─────────────────────┐                   │
│              │     DATABRICKS      │                   │
│              │  • Delta Lake       │                   │
│              │  • Vector Search    │                   │
│              │  • Unity Catalog    │                   │
│              │  • MLflow           │                   │
│              └─────────────────────┘                   │
│                         ↓                               │
│              ┌─────────────────────┐                   │
│              │   GITHUB COPILOT    │                   │
│              │  • Code assistance  │                   │
│              │  • SQL generation   │                   │
│              └─────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

### Trade Surveillance Use Cases

- "Explain why this trade was flagged"
- "What regulatory rules apply to this pattern?"
- "Summarize communications around this trade"
- "Compare to similar historical cases"

### Security & Compliance

DATA: Unity Catalog access control, column-level masking, encryption

LLM: Azure OpenAI in private VNet, no training on your data, prompt logging

REGULATORY: MiFID II, MAR, SEC Rule 17a-4, FINRA, SOC 2 Type II

---

# PART V: INTERVIEW PREPARATION

## Master Q&A Collection

### Fundamentals

Q: What is a token?
A: Smallest unit LLM processes (~0.75 words). "JPMorgan" = 2 tokens.

Q: What is an embedding?
A: Text → Dense vector capturing semantic meaning. 384-3072 dimensions.

Q: What is attention?
A: Mechanism to focus on relevant parts of input. Formula: softmax(QK^T/√d_k) × V

### RAG

Q: What is RAG?
A: Retrieval-Augmented Generation. Retrieve docs → Add to prompt → Generate with citations.

Q: How do you prevent hallucination?
A: Multi-layer: (1) High similarity threshold, (2) System prompt "only from context", (3) Required citations, (4) Test suite.

Q: Where is RAG in your code?
A: D_Generation.py → RAGEngine.query() → Retrieval + Augment + Generate

### Production

Q: RAG vs Fine-tuning?
A: RAG for dynamic data + citations. Fine-tune for style/tone + static knowledge. Often use both.

Q: GPU vs CPU for RAG?
A: CPU for embedding + retrieval. GPU for large LLM inference. API for simplicity.

Q: How to scale RAG?
A: Horizontal scaling (more servers), cloud vector DB, caching (Redis), queue for LLM calls.

### Parameters

Q: What is temperature?
A: Controls randomness. 0 = deterministic, 0.7 = balanced, 2.0 = creative/random.

Q: Temperature 0 vs 0.7?
A: 0 for factual accuracy (RAG, compliance). 0.7+ for creativity.

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────┐
│           AI/LLM INTERVIEW QUICK REFERENCE              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  TOKEN:        ~0.75 words, smallest LLM unit          │
│  EMBEDDING:    Text → Vector (semantic meaning)        │
│  ATTENTION:    Focus on relevant input parts           │
│  TRANSFORMER:  Attention-based architecture            │
│  LLM:          Large neural net for text               │
│  FINE-TUNING:  Specialize on new data                  │
│  LoRA:         Efficient fine-tuning (0.1% params)     │
│  RAG:          Retrieve → Augment → Generate           │
│  VECTOR DB:    Fast similarity search                  │
│  AGENT:        LLM + Tools + Goals                     │
│  CoT:          "Think step by step" = better reasoning │
│  TEMPERATURE:  0=accurate, 0.7=balanced, 2=creative    │
│  QUANTIZATION: 32→4 bit for efficiency                 │
│                                                         │
│  FORMULAS:                                              │
│  Attention = softmax(QK^T / √d_k) × V                  │
│  Cosine Sim = (A·B) / (||A|| × ||B||)                  │
│                                                         │
│  OUR PROJECT:                                           │
│  ChromaDB + all-MiniLM-L6-v2 + Gemini/OpenAI          │
│  D_Generation.py → RAGEngine.query()                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Study Roadmap

Week 1-2: Foundations (Tokenization, Embeddings, Attention)

Week 3-4: Transformers + implement from scratch

Week 5-6: LLMs + Fine-tuning + LoRA

Week 7-8: RAG + Vector DBs + Build your own

Week 9-10: Agents + Tool use + MCP

Week 11-12: Optimization (Quantization, GPU/CPU)

Week 13+: Papers + Interview practice

---

# Holy Grail Summary

```
┌─────────────────────────────────────────────────────────┐
│            AI/LLM COMPLETE LEARNING GUIDE               │
│                     January 2026                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  FUNDAMENTALS:                                          │
│  ✓ Tokens, Embeddings, Attention, Transformers         │
│  ✓ TikToken tokenization & counting                    │
│                                                         │
│  LLMs:                                                  │
│  ✓ Temperature, top_p, parameters                      │
│  ✓ Fine-tuning vs RAG decision                         │
│  ✓ LoRA efficient fine-tuning                          │
│                                                         │
│  RAG:                                                   │
│  ✓ Chunking, Vector DBs, Retrieval                     │
│  ✓ Hallucination prevention                            │
│  ✓ Complete pipeline implementation                    │
│                                                         │
│  AGENTS:                                                │
│  ✓ Agent architecture (LLM + Tools + Memory)           │
│  ✓ Multi-agent patterns                                │
│  ✓ LangChain, LangGraph, Ollama                        │
│                                                         │
│  PRODUCTION:                                            │
│  ✓ GPU vs CPU, Quantization                            │
│  ✓ Scaling, Latency optimization                       │
│  ✓ Azure + Databricks + Trade Surveillance             │
│                                                         │
│  PROJECT:                                               │
│  ✓ SEC EDGAR 10-K RAG Pipeline                         │
│  ✓ D_Generation.py → RAGEngine.query()                 │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

Author: Sourav Shrivastava  
Project: SEC EDGAR 10-K RAG Pipeline  
Reference: AI/LLM Interview Holy Grail  
Enterprise: Azure + Databricks + Trade Surveillance

*Your complete guide to AI mastery* 🏆
