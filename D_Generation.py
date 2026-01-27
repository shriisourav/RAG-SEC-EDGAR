"""
Day 4: RAG Generation Pipeline
Passes retrieved chunks to LLM, enforces grounded answers with citations.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import google.generativeai as genai
from enum import Enum

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM Configuration (supports OpenAI and Gemini)
LLM_PROVIDER = "gemini"  # "openai" or "gemini"
OPENAI_MODEL = "gpt-4o-mini"
GEMINI_MODEL = "gemini-1.5-flash-latest"

# Retrieval settings
DEFAULT_K = 5
MIN_SIMILARITY = 0.35


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

SYSTEM_PROMPT = """You are a financial analyst assistant specializing in SEC 10-K filings analysis.

CRITICAL RULES - YOU MUST FOLLOW THESE:
1. ONLY answer based on the provided context from 10-K filings
2. If the context doesn't contain enough information to answer, say "I cannot find this information in the provided documents"
3. NEVER make up information or hallucinate facts
4. ALWAYS cite your sources using [Source: Company - Section] format
5. If asked about a company not in the context, clearly state you don't have that information
6. Be precise with numbers, dates, and financial figures - quote them exactly as they appear
7. If information seems contradictory between sources, note the discrepancy

CONTEXT FORMAT:
You will receive context chunks from 10-K filings with metadata including:
- Company name (JPMorgan Chase, Goldman Sachs, or UBS)
- Section (Risk Factors, Liquidity, Credit Risk, etc.)
- Source file

YOUR RESPONSE FORMAT:
1. Provide a clear, concise answer
2. Include specific citations for each fact
3. If relevant, compare across companies when multiple are mentioned
4. End with a brief list of sources used"""

ANSWER_NOT_FOUND_PROMPT = """Based on my review of the provided 10-K filing excerpts, I cannot find sufficient information to answer this question.

Possible reasons:
- The specific information may be in a different section of the 10-K not included in the retrieved context
- The question may relate to information not typically disclosed in 10-K filings
- The question may be about a company or time period not covered in our documents

Suggestion: Please try rephrasing your question or asking about a specific aspect of {companies}'s risk factors, financial statements, or business operations."""


# ============================================================================
# DATA CLASSES
# ============================================================================

class AnswerConfidence(Enum):
    HIGH = "high"           # Direct answer found in context
    MEDIUM = "medium"       # Partial information found
    LOW = "low"            # Inferred from context
    NOT_FOUND = "not_found" # Cannot answer from context


@dataclass
class Citation:
    """A citation to a source document."""
    company: str
    section: str
    source_file: str
    chunk_id: str
    text_excerpt: str  # First 100 chars of cited text
    
    def __str__(self) -> str:
        return f"[{self.company} - {self.section}]"


@dataclass
class RAGResponse:
    """Complete RAG response with answer and metadata."""
    query: str
    answer: str
    confidence: AnswerConfidence
    citations: List[Citation]
    context_used: str
    num_chunks_retrieved: int
    companies_in_context: List[str]
    model_used: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": self.confidence.value,
            "citations": [
                {
                    "company": c.company,
                    "section": c.section,
                    "source_file": c.source_file,
                    "chunk_id": c.chunk_id
                }
                for c in self.citations
            ],
            "num_chunks_retrieved": self.num_chunks_retrieved,
            "companies_in_context": self.companies_in_context,
            "model_used": self.model_used
        }
    
    def print_response(self):
        """Pretty print the response."""
        print(f"\n{'='*70}")
        print(f"📝 QUERY: {self.query}")
        print(f"{'='*70}")
        print(f"\n💬 ANSWER (Confidence: {self.confidence.value}):\n")
        print(self.answer)
        print(f"\n{'─'*70}")
        print(f"📚 CITATIONS ({len(self.citations)} sources):")
        for i, citation in enumerate(self.citations, 1):
            print(f"  [{i}] {citation.company} - {citation.section}")
            print(f"      File: {citation.source_file}")
        print(f"\n🔍 Retrieved {self.num_chunks_retrieved} chunks from: {', '.join(self.companies_in_context)}")
        print(f"🤖 Model: {self.model_used}")
        print("="*70)


# ============================================================================
# LLM CLIENTS
# ============================================================================

class LLMClient:
    """Abstract LLM client interface."""
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """OpenAI GPT client."""
    
    def __init__(self, model: str = OPENAI_MODEL):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for factual responses
            max_tokens=1500
        )
        return response.choices[0].message.content


class GeminiClient(LLMClient):
    """Google Gemini client."""
    
    def __init__(self, model: str = GEMINI_MODEL):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.model_name = model
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Combine system and user prompts for Gemini
        full_prompt = f"{system_prompt}\n\n---\n\n{user_prompt}"
        response = self.model.generate_content(
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1500
            )
        )
        return response.text


class MockLLMClient(LLMClient):
    """Mock LLM for testing without API keys."""
    
    def __init__(self):
        self.model_name = "mock-llm"
    
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        # Extract context and question from user prompt
        if "CONTEXT:" in user_prompt and "QUESTION:" in user_prompt:
            context_start = user_prompt.find("CONTEXT:") + 8
            question_start = user_prompt.find("QUESTION:")
            context = user_prompt[context_start:question_start].strip()[:500]
            question = user_prompt[question_start + 9:].strip()
            
            return f"""Based on the provided 10-K filing excerpts, here is my analysis:

The documents discuss various aspects of the companies' operations and risk management practices. 

Key findings from the context:
- The financial institutions maintain comprehensive risk management frameworks
- Credit risk, market risk, and operational risk are primary concerns
- Regulatory capital requirements under Basel III are closely monitored

[Source: Based on retrieved 10-K excerpts]

Note: This is a mock response for testing. Set GEMINI_API_KEY or OPENAI_API_KEY for real responses."""
        return "Mock response - no context provided."


# ============================================================================
# RETRIEVER (from Day 3)
# ============================================================================

class Retriever:
    """Semantic retrieval engine using ChromaDB."""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_collection(name="sec_10k_filings")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    
    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_K,
        company: Optional[str] = None,
        section: Optional[str] = None,
        min_similarity: float = MIN_SIMILARITY
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks."""
        # Build filter - ChromaDB requires $and for multiple conditions
        where_filter = None
        conditions = []
        
        if company:
            conditions.append({"company": company.upper()})
        if section:
            conditions.append({"section": section})
        
        if len(conditions) == 1:
            where_filter = conditions[0]
        elif len(conditions) > 1:
            where_filter = {"$and": conditions}
        
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = []
        for i in range(len(results['ids'][0])):
            similarity = 1 - results['distances'][0][i]
            if similarity >= min_similarity:
                chunks.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "similarity": similarity
                })
        
        return chunks


# ============================================================================
# RAG ENGINE
# ============================================================================

class RAGEngine:
    """
    Complete RAG engine combining retrieval and generation.
    """
    
    def __init__(self, llm_provider: str = LLM_PROVIDER):
        print("Initializing RAG Engine...")
        
        # Initialize retriever
        self.retriever = Retriever()
        print(f"✓ Retriever initialized ({self.retriever.collection.count()} chunks)")
        
        # Initialize LLM client
        self.llm_provider = llm_provider
        try:
            if llm_provider == "openai":
                self.llm = OpenAIClient()
                self.model_name = OPENAI_MODEL
            elif llm_provider == "gemini":
                self.llm = GeminiClient()
                self.model_name = GEMINI_MODEL
            else:
                raise ValueError(f"Unknown LLM provider: {llm_provider}")
            print(f"✓ LLM client initialized ({self.model_name})")
        except ValueError as e:
            print(f"⚠️ {e}")
            print("  Using mock LLM client for testing...")
            self.llm = MockLLMClient()
            self.model_name = "mock-llm"
    
    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into context string."""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk["metadata"]
            context_parts.append(
                f"[Source {i}: {metadata['company_name']} - {metadata['section']}]\n"
                f"{chunk['text']}\n"
            )
        
        return "\n---\n".join(context_parts)
    
    def _create_user_prompt(self, query: str, context: str) -> str:
        """Create the user prompt with context and question."""
        return f"""CONTEXT:
{context}

---

QUESTION: {query}

Please answer based ONLY on the context above. Include citations in [Source: Company - Section] format."""
    
    def _extract_citations(self, chunks: List[Dict[str, Any]]) -> List[Citation]:
        """Extract citation objects from chunks."""
        citations = []
        for chunk in chunks:
            metadata = chunk["metadata"]
            citations.append(Citation(
                company=metadata.get("company_name", metadata.get("company")),
                section=metadata.get("section", "Unknown"),
                source_file=metadata.get("source_file", "Unknown"),
                chunk_id=chunk["id"],
                text_excerpt=chunk["text"][:100]
            ))
        return citations
    
    def _assess_confidence(
        self, 
        answer: str, 
        chunks: List[Dict[str, Any]]
    ) -> AnswerConfidence:
        """Assess confidence level of the answer."""
        # Check for "cannot find" or similar phrases
        no_answer_phrases = [
            "cannot find",
            "not found",
            "no information",
            "not mentioned",
            "don't have",
            "insufficient information"
        ]
        
        answer_lower = answer.lower()
        
        if any(phrase in answer_lower for phrase in no_answer_phrases):
            return AnswerConfidence.NOT_FOUND
        
        # Check if answer includes citations
        has_citations = "[source" in answer_lower or "[" in answer
        
        # Check similarity scores
        avg_similarity = sum(c["similarity"] for c in chunks) / len(chunks) if chunks else 0
        
        if avg_similarity > 0.6 and has_citations:
            return AnswerConfidence.HIGH
        elif avg_similarity > 0.4:
            return AnswerConfidence.MEDIUM
        else:
            return AnswerConfidence.LOW
    
    def query(
        self,
        question: str,
        k: int = DEFAULT_K,
        company: Optional[str] = None,
        section: Optional[str] = None
    ) -> RAGResponse:
        """
        Execute a RAG query: retrieve context and generate answer.
        
        Args:
            question: The user's question
            k: Number of chunks to retrieve
            company: Optional company filter (JPM, GS, UBS)
            section: Optional section filter
            
        Returns:
            RAGResponse with answer, citations, and metadata
        """
        # Step 1: Retrieve relevant chunks
        chunks = self.retriever.retrieve(
            query=question,
            k=k,
            company=company,
            section=section
        )
        
        if not chunks:
            # No relevant chunks found
            companies = company or "the companies"
            return RAGResponse(
                query=question,
                answer=ANSWER_NOT_FOUND_PROMPT.format(companies=companies),
                confidence=AnswerConfidence.NOT_FOUND,
                citations=[],
                context_used="",
                num_chunks_retrieved=0,
                companies_in_context=[],
                model_used=self.model_name
            )
        
        # Step 2: Format context
        context = self._format_context(chunks)
        
        # Step 3: Create prompt
        user_prompt = self._create_user_prompt(question, context)
        
        # Step 4: Generate answer
        answer = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        
        # Step 5: Extract metadata
        citations = self._extract_citations(chunks)
        companies_in_context = list(set(c.company for c in citations))
        confidence = self._assess_confidence(answer, chunks)
        
        return RAGResponse(
            query=question,
            answer=answer,
            confidence=confidence,
            citations=citations,
            context_used=context,
            num_chunks_retrieved=len(chunks),
            companies_in_context=companies_in_context,
            model_used=self.model_name
        )
    
    def query_with_hallucination_check(
        self,
        question: str,
        k: int = DEFAULT_K
    ) -> RAGResponse:
        """
        Query with additional hallucination prevention.
        Uses a two-pass approach: generate, then verify.
        """
        # First pass: normal query
        response = self.query(question, k=k)
        
        if response.confidence == AnswerConfidence.NOT_FOUND:
            return response
        
        # Second pass: verification prompt
        verification_prompt = f"""Review this answer for potential hallucinations:

ORIGINAL QUESTION: {question}

GENERATED ANSWER: {response.answer}

AVAILABLE CONTEXT (what we actually have):
{response.context_used[:2000]}...

TASK: 
1. Check if every claim in the answer is supported by the context
2. If any claim is NOT supported, mark it as [UNVERIFIED]
3. Return the corrected answer with only verified information

CORRECTED ANSWER:"""
        
        verified_answer = self.llm.generate(
            "You are a fact-checker. Only approve claims that are directly supported by the provided context.",
            verification_prompt
        )
        
        response.answer = verified_answer
        return response


# ============================================================================
# TESTING & EVALUATION
# ============================================================================

class RAGTester:
    """Test RAG system for hallucinations and accuracy."""
    
    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine
        self.test_results = []
    
    def test_grounding(self, question: str, expected_in_answer: List[str]) -> Dict:
        """Test if answer is properly grounded in sources."""
        response = self.rag.query(question)
        
        # Check if expected content appears in answer
        answer_lower = response.answer.lower()
        grounded = all(exp.lower() in answer_lower for exp in expected_in_answer)
        
        result = {
            "question": question,
            "grounded": grounded,
            "has_citations": len(response.citations) > 0,
            "confidence": response.confidence.value,
            "expected_found": [exp for exp in expected_in_answer if exp.lower() in answer_lower]
        }
        
        self.test_results.append(result)
        return result
    
    def test_hallucination_resistance(self) -> List[Dict]:
        """Test with questions designed to trigger hallucinations."""
        hallucination_tests = [
            {
                "question": "What was Apple's revenue in 2024?",
                "should_refuse": True,  # Apple not in our docs
                "reason": "Company not in context"
            },
            {
                "question": "What is JPMorgan's stock price prediction for 2025?",
                "should_refuse": True,  # 10-K doesn't have predictions
                "reason": "Information type not in 10-K"
            },
            {
                "question": "Who is the CEO of Goldman Sachs and what is their favorite color?",
                "should_refuse": True,  # Personal details not in 10-K
                "reason": "Personal info not in 10-K"
            },
            {
                "question": "What are JPMorgan's main risk factors?",
                "should_refuse": False,  # This should be answered
                "reason": "Valid question"
            },
        ]
        
        results = []
        print("\n" + "="*70)
        print("HALLUCINATION RESISTANCE TESTS")
        print("="*70)
        
        for test in hallucination_tests:
            response = self.rag.query(test["question"])
            
            # Check if system properly refused or answered
            refused = response.confidence == AnswerConfidence.NOT_FOUND or \
                     "cannot" in response.answer.lower() or \
                     "not found" in response.answer.lower() or \
                     "don't have" in response.answer.lower()
            
            passed = (test["should_refuse"] and refused) or \
                    (not test["should_refuse"] and not refused)
            
            result = {
                "question": test["question"],
                "should_refuse": test["should_refuse"],
                "did_refuse": refused,
                "passed": passed,
                "reason": test["reason"]
            }
            results.append(result)
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"\n{status}: {test['question'][:50]}...")
            print(f"   Expected: {'refuse' if test['should_refuse'] else 'answer'}")
            print(f"   Got: {'refused' if refused else 'answered'}")
        
        # Summary
        passed_count = sum(1 for r in results if r["passed"])
        print(f"\nHallucination Resistance: {passed_count}/{len(results)} tests passed")
        
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution with demo queries."""
    print("="*70)
    print("Day 4: RAG Generation Pipeline")
    print("="*70)
    
    # Initialize RAG engine
    rag = RAGEngine()
    
    # ========================================
    # Demo Queries
    # ========================================
    demo_questions = [
        "What are the main risk factors for JPMorgan Chase?",
        "How does Goldman Sachs manage credit risk?",
        "What are UBS's liquidity requirements?",
        "Compare the operational risk management across the three banks.",
        "What regulatory capital requirements do these banks face?",
    ]
    
    print("\n" + "="*70)
    print("DEMO QUERIES")
    print("="*70)
    
    for question in demo_questions:
        response = rag.query(question, k=5)
        response.print_response()
        print("\n")
    
    # ========================================
    # Filtered Query Demo
    # ========================================
    print("\n" + "="*70)
    print("FILTERED QUERY (JPM Only, Risk Factors Section)")
    print("="*70)
    
    response = rag.query(
        "What are the main risks?",
        k=5,
        company="JPM",
        section="Risk Factors"
    )
    response.print_response()
    
    # ========================================
    # Hallucination Tests
    # ========================================
    tester = RAGTester(rag)
    hallucination_results = tester.test_hallucination_resistance()
    
    # ========================================
    # Save sample responses
    # ========================================
    sample_responses = []
    for q in demo_questions[:3]:
        response = rag.query(q)
        sample_responses.append(response.to_dict())
    
    output_path = BASE_DIR / "data" / "sample_rag_responses.json"
    with open(output_path, 'w') as f:
        json.dump(sample_responses, f, indent=2)
    print(f"\n✓ Sample responses saved to: {output_path}")
    
    print("\n" + "="*70)
    print("✓ Day 4 Complete: RAG Generation Pipeline Ready!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Set GEMINI_API_KEY or OPENAI_API_KEY for real LLM responses")
    print("  2. Run: export GEMINI_API_KEY='your-api-key'")
    print("  3. Re-run this script for production-quality answers")


if __name__ == "__main__":
    main()
