"""
FastAPI wrapper for SEC EDGAR 10-K RAG System
Run with: uvicorn api:app --reload
"""

import os
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Import RAG components
from D_Generation import RAGEngine, AnswerConfidence

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str = Field(..., description="The question to ask about 10-K filings", min_length=3)
    company: Optional[str] = Field(None, description="Filter by company: JPM, GS, or UBS")
    section: Optional[str] = Field(None, description="Filter by section: Risk Factors, Liquidity, etc.")
    k: int = Field(5, description="Number of chunks to retrieve", ge=1, le=20)

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What are JPMorgan's main risk factors?",
                "company": "JPM",
                "k": 5
            }
        }


class Citation(BaseModel):
    """Citation model."""
    company: str
    section: str
    source_file: str


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    question: str
    answer: str
    confidence: str
    citations: List[Citation]
    num_chunks_retrieved: int
    companies_in_context: List[str]
    model_used: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    total_chunks: int
    available_companies: List[str]
    llm_provider: str


# ============================================================================
# FASTAPI APP
# ============================================================================

# Global RAG engine instance
rag_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG engine on startup."""
    global rag_engine
    print("🚀 Initializing RAG Engine...")
    rag_engine = RAGEngine()
    print("✅ RAG Engine ready!")
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="SEC EDGAR 10-K RAG API",
    description="""
    ## 📊 Analyze SEC 10-K Filings with AI
    
    This API allows you to ask questions about 10-K filings from:
    - **JPMorgan Chase (JPM)**
    - **Goldman Sachs (GS)**  
    - **UBS Group (UBS)**
    
    ### Features:
    - 🔍 Semantic search across 2,300+ document chunks
    - 🎯 Filter by company or section
    - 📚 Source citations included
    - 🛡️ Hallucination prevention
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Welcome endpoint."""
    return {
        "message": "SEC EDGAR 10-K RAG API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health_check():
    """Check API health and show available data."""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    return HealthResponse(
        status="healthy",
        total_chunks=rag_engine.retriever.collection.count(),
        available_companies=["JPM", "GS", "UBS"],
        llm_provider=rag_engine.llm_provider
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_rag(request: QueryRequest):
    """
    Ask a question about 10-K filings.
    
    The system will:
    1. Search for relevant document chunks
    2. Pass them to the LLM for analysis
    3. Return an answer with citations
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        response = rag_engine.query(
            question=request.question,
            k=request.k,
            company=request.company,
            section=request.section
        )
        
        return QueryResponse(
            question=response.query,
            answer=response.answer,
            confidence=response.confidence.value,
            citations=[
                Citation(
                    company=c.company,
                    section=c.section,
                    source_file=c.source_file
                )
                for c in response.citations
            ],
            num_chunks_retrieved=response.num_chunks_retrieved,
            companies_in_context=response.companies_in_context,
            model_used=response.model_used
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query", response_model=QueryResponse, tags=["Query"])
async def query_rag_get(
    question: str = Query(..., description="The question to ask", min_length=3),
    company: Optional[str] = Query(None, description="Filter by company: JPM, GS, or UBS"),
    k: int = Query(5, description="Number of chunks to retrieve", ge=1, le=20)
):
    """
    Ask a question (GET method for easy browser testing).
    
    Example: /query?question=What are JPM's risk factors?&company=JPM
    """
    request = QueryRequest(question=question, company=company, k=k)
    return await query_rag(request)


@app.get("/companies", tags=["Info"])
async def list_companies():
    """List available companies and their details."""
    return {
        "companies": [
            {
                "ticker": "JPM",
                "name": "JPMorgan Chase & Co",
                "filing_type": "10-K",
                "fiscal_year": "2024"
            },
            {
                "ticker": "GS", 
                "name": "Goldman Sachs Group Inc",
                "filing_type": "10-K",
                "fiscal_year": "2024"
            },
            {
                "ticker": "UBS",
                "name": "UBS Group AG",
                "filing_type": "10-K (20-F)",
                "fiscal_year": "2024"
            }
        ]
    }


@app.get("/sections", tags=["Info"])
async def list_sections():
    """List available document sections."""
    return {
        "sections": [
            "Risk Factors",
            "Liquidity", 
            "Credit Risk",
            "Market Risk",
            "Operational Risk",
            "Financial Statements",
            "MD&A",
            "Business Overview",
            "General"
        ]
    }


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
