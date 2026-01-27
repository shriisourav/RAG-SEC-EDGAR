# SEC EDGAR 10-K RAG Pipeline v1.0

## Overview
This pipeline processes SEC 10-K filings for financial analysis using RAG (Retrieval-Augmented Generation).

## Pipeline Components

### Day 1: Data Collection (`A_SEC_EDGAR.py`)
- Downloads 10-K filings for JPMorgan Chase, Goldman Sachs, and UBS
- Stores raw HTML and converts to clean text

### Day 2: Chunking & Indexing (`B_Chunking_Indexing.py`)
- Splits documents into ~600 token chunks with 100 token overlap
- Generates embeddings using all-MiniLM-L6-v2
- Stores in ChromaDB vector database

### Day 3: Retrieval (`C_Retrieval.py`)
- Semantic search with top-k retrieval
- Company and section filtering
- Parameter tuning utilities

### Day 4: Generation (`D_Generation.py`)
- LLM integration (Gemini/OpenAI)
- Citation-enforced responses
- Hallucination prevention

### Day 5: Evaluation (`E_Evaluation.py`)
- 20 gold test questions
- Automated evaluation metrics
- Failure analysis

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Set API key (choose one)
export GEMINI_API_KEY='your-key'
# OR
export OPENAI_API_KEY='your-key'

# Run the full pipeline
python A_SEC_EDGAR.py      # Download data (if needed)
python B_Chunking_Indexing.py  # Index documents
python C_Retrieval.py      # Test retrieval
python D_Generation.py     # Test generation
python E_Evaluation.py     # Run evaluation
```

## Configuration
See `data/pipeline_v1_frozen.json` for frozen configuration.

## Frozen: 2026-01-25 22:26:41
