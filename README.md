---
title: SEC EDGAR 10-K RAG
emoji: 📊
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Ask questions about JPMorgan, Goldman Sachs & UBS 10-K filings
---

# SEC EDGAR 10-K RAG System

Ask questions about 10-K filings from major financial institutions using AI-powered retrieval.

## 🏦 Available Companies
- **JPMorgan Chase (JPM)**
- **Goldman Sachs (GS)**
- **UBS Group (UBS)**

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/docs` | GET | Interactive API documentation |
| `/query` | GET/POST | Ask a question |
| `/health` | GET | Check system status |
| `/companies` | GET | List available companies |

## 🔍 Example Query

```bash
curl "https://YOUR-SPACE.hf.space/query?question=What%20are%20JPM%27s%20risk%20factors?"
```

## 🛠️ Built With
- FastAPI
- ChromaDB
- Sentence Transformers
- Gemini/OpenAI LLMs
