"""
Day 5: Evaluation and Pipeline Freeze
Creates gold questions, runs comprehensive evaluation, logs failures, and freezes v1.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import chromadb
from sentence_transformers import SentenceTransformer

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
EVAL_DIR = BASE_DIR / "data" / "evaluation"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Ensure evaluation directory exists
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# GOLD QUESTION SET
# ============================================================================

GOLD_QUESTIONS = [
    # =========== RISK FACTORS (5 questions) ===========
    {
        "id": "RF-001",
        "question": "What are JPMorgan Chase's main risk factors?",
        "expected_company": "JPM",
        "expected_section": "Risk Factors",
        "expected_keywords": ["risk", "credit", "market", "operational"],
        "difficulty": "easy",
        "category": "risk_factors"
    },
    {
        "id": "RF-002",
        "question": "What credit risks does Goldman Sachs face?",
        "expected_company": "GS",
        "expected_section": "Credit Risk",
        "expected_keywords": ["credit", "counterparty", "default", "exposure"],
        "difficulty": "easy",
        "category": "risk_factors"
    },
    {
        "id": "RF-003",
        "question": "How does UBS describe its operational risks?",
        "expected_company": "UBS",
        "expected_section": "Operational Risk",
        "expected_keywords": ["operational", "process", "systems", "controls"],
        "difficulty": "medium",
        "category": "risk_factors"
    },
    {
        "id": "RF-004",
        "question": "What market risks are disclosed in the 10-K filings?",
        "expected_company": None,  # Any company
        "expected_section": "Market Risk",
        "expected_keywords": ["market", "trading", "volatility", "prices"],
        "difficulty": "medium",
        "category": "risk_factors"
    },
    {
        "id": "RF-005",
        "question": "Compare the regulatory risks across JPMorgan and Goldman Sachs",
        "expected_company": None,  # Multiple
        "expected_section": "Risk Factors",
        "expected_keywords": ["regulatory", "compliance", "Basel", "capital"],
        "difficulty": "hard",
        "category": "risk_factors"
    },
    
    # =========== LIQUIDITY (4 questions) ===========
    {
        "id": "LQ-001",
        "question": "How does JPMorgan manage its liquidity risk?",
        "expected_company": "JPM",
        "expected_section": "Liquidity",
        "expected_keywords": ["liquidity", "funding", "cash", "requirements"],
        "difficulty": "easy",
        "category": "liquidity"
    },
    {
        "id": "LQ-002",
        "question": "What are UBS's liquidity requirements under Basel III?",
        "expected_company": "UBS",
        "expected_section": "Liquidity",
        "expected_keywords": ["liquidity", "Basel", "LCR", "NSFR", "ratio"],
        "difficulty": "medium",
        "category": "liquidity"
    },
    {
        "id": "LQ-003",
        "question": "What liquidity buffers do the banks maintain?",
        "expected_company": None,
        "expected_section": "Liquidity",
        "expected_keywords": ["buffer", "reserve", "HQLA", "liquid assets"],
        "difficulty": "medium",
        "category": "liquidity"
    },
    {
        "id": "LQ-004",
        "question": "How do funding sources differ between the three banks?",
        "expected_company": None,
        "expected_section": "Liquidity",
        "expected_keywords": ["funding", "deposits", "borrowing", "sources"],
        "difficulty": "hard",
        "category": "liquidity"
    },
    
    # =========== CAPITAL & REGULATION (4 questions) ===========
    {
        "id": "CR-001",
        "question": "What are the regulatory capital requirements for JPMorgan?",
        "expected_company": "JPM",
        "expected_section": "General",
        "expected_keywords": ["capital", "regulatory", "CET1", "tier 1"],
        "difficulty": "easy",
        "category": "capital"
    },
    {
        "id": "CR-002",
        "question": "How does Goldman Sachs meet Basel III capital requirements?",
        "expected_company": "GS",
        "expected_section": "General",
        "expected_keywords": ["Basel", "capital", "ratio", "requirements"],
        "difficulty": "medium",
        "category": "capital"
    },
    {
        "id": "CR-003",
        "question": "What stress testing requirements do these banks face?",
        "expected_company": None,
        "expected_section": None,
        "expected_keywords": ["stress", "test", "CCAR", "scenario"],
        "difficulty": "medium",
        "category": "capital"
    },
    {
        "id": "CR-004",
        "question": "What are the G-SIB buffer requirements mentioned in the filings?",
        "expected_company": None,
        "expected_section": "General",
        "expected_keywords": ["G-SIB", "buffer", "systemic", "surcharge"],
        "difficulty": "hard",
        "category": "capital"
    },
    
    # =========== BUSINESS OPERATIONS (4 questions) ===========
    {
        "id": "BO-001",
        "question": "What are JPMorgan's main business segments?",
        "expected_company": "JPM",
        "expected_section": "General",
        "expected_keywords": ["segment", "consumer", "commercial", "investment"],
        "difficulty": "easy",
        "category": "business"
    },
    {
        "id": "BO-002",
        "question": "How does Goldman Sachs generate revenue?",
        "expected_company": "GS",
        "expected_section": "General",
        "expected_keywords": ["revenue", "trading", "fees", "investment banking"],
        "difficulty": "medium",
        "category": "business"
    },
    {
        "id": "BO-003",
        "question": "What geographic markets does UBS operate in?",
        "expected_company": "UBS",
        "expected_section": "General",
        "expected_keywords": ["geographic", "global", "region", "market"],
        "difficulty": "medium",
        "category": "business"
    },
    
    # =========== HALLUCINATION TESTS (3 questions) ===========
    {
        "id": "HT-001",
        "question": "What was Apple's revenue in 2024?",
        "expected_company": None,
        "expected_section": None,
        "expected_keywords": [],
        "should_refuse": True,
        "difficulty": "easy",
        "category": "hallucination_test"
    },
    {
        "id": "HT-002",
        "question": "What is JPMorgan's stock price prediction for 2026?",
        "expected_company": None,
        "expected_section": None,
        "expected_keywords": [],
        "should_refuse": True,
        "difficulty": "medium",
        "category": "hallucination_test"
    },
    {
        "id": "HT-003",
        "question": "What is the CEO's favorite restaurant?",
        "expected_company": None,
        "expected_section": None,
        "expected_keywords": [],
        "should_refuse": True,
        "difficulty": "easy",
        "category": "hallucination_test"
    },
]


# ============================================================================
# DATA CLASSES
# ============================================================================

class FailureType(Enum):
    RETRIEVAL_MISS = "retrieval_miss"           # Retrieved wrong chunks
    WRONG_COMPANY = "wrong_company"             # Got different company
    WRONG_SECTION = "wrong_section"             # Got different section
    MISSING_KEYWORDS = "missing_keywords"       # Expected keywords not in answer
    HALLUCINATION = "hallucination"             # Answered when should refuse
    FALSE_REFUSAL = "false_refusal"             # Refused when should answer
    LOW_SIMILARITY = "low_similarity"           # Retrieval quality too low
    EMPTY_RESPONSE = "empty_response"           # No answer generated


@dataclass
class EvaluationResult:
    """Result of evaluating a single question."""
    question_id: str
    question: str
    category: str
    difficulty: str
    passed: bool
    failures: List[str]
    retrieved_companies: List[str]
    retrieved_sections: List[str]
    avg_similarity: float
    top_similarity: float
    keywords_found: List[str]
    keywords_missing: List[str]
    answer_preview: str
    execution_time_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    timestamp: str
    total_questions: int
    passed: int
    failed: int
    pass_rate: float
    results_by_category: Dict[str, Dict[str, int]]
    results_by_difficulty: Dict[str, Dict[str, int]]
    failure_breakdown: Dict[str, int]
    avg_similarity: float
    avg_execution_time_ms: float
    individual_results: List[EvaluationResult]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['individual_results'] = [r.to_dict() for r in self.individual_results]
        return d


# ============================================================================
# LIGHTWEIGHT RETRIEVER (No LLM dependency)
# ============================================================================

class EvalRetriever:
    """Lightweight retriever for evaluation."""
    
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_collection(name="sec_10k_filings")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✓ Retriever loaded ({self.collection.count()} chunks)")
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = []
        for i in range(len(results['ids'][0])):
            chunks.append({
                "id": results['ids'][0][i],
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "similarity": 1 - results['distances'][0][i]
            })
        
        return chunks


# ============================================================================
# EVALUATOR
# ============================================================================

class RAGEvaluator:
    """Comprehensive RAG system evaluator."""
    
    def __init__(self):
        self.retriever = EvalRetriever()
        self.results: List[EvaluationResult] = []
    
    def evaluate_question(self, question_data: Dict) -> EvaluationResult:
        """Evaluate a single question."""
        import time
        start_time = time.time()
        
        question = question_data["question"]
        question_id = question_data["id"]
        
        # Retrieve chunks
        chunks = self.retriever.retrieve(question, k=5)
        
        # Calculate metrics
        if chunks:
            similarities = [c["similarity"] for c in chunks]
            avg_similarity = sum(similarities) / len(similarities)
            top_similarity = max(similarities)
            
            retrieved_companies = list(set(
                c["metadata"].get("company_name", c["metadata"].get("company")) 
                for c in chunks
            ))
            retrieved_sections = list(set(
                c["metadata"].get("section") for c in chunks
            ))
            
            # Create mock answer from chunks for keyword analysis
            answer_text = " ".join(c["text"] for c in chunks).lower()
        else:
            avg_similarity = 0
            top_similarity = 0
            retrieved_companies = []
            retrieved_sections = []
            answer_text = ""
        
        execution_time = (time.time() - start_time) * 1000
        
        # Check for failures
        failures = []
        
        # Check company match
        expected_company = question_data.get("expected_company")
        if expected_company:
            company_names = {
                "JPM": "JPMorgan Chase",
                "GS": "Goldman Sachs",
                "UBS": "UBS Group"
            }
            expected_name = company_names.get(expected_company, expected_company)
            if expected_name not in retrieved_companies and expected_company not in str(retrieved_companies):
                failures.append(FailureType.WRONG_COMPANY.value)
        
        # Check section match
        expected_section = question_data.get("expected_section")
        if expected_section:
            if not any(expected_section.lower() in s.lower() for s in retrieved_sections):
                failures.append(FailureType.WRONG_SECTION.value)
        
        # Check keywords
        expected_keywords = question_data.get("expected_keywords", [])
        keywords_found = [kw for kw in expected_keywords if kw.lower() in answer_text]
        keywords_missing = [kw for kw in expected_keywords if kw.lower() not in answer_text]
        
        if expected_keywords and len(keywords_found) < len(expected_keywords) * 0.5:
            failures.append(FailureType.MISSING_KEYWORDS.value)
        
        # Check similarity threshold
        if top_similarity < 0.4:
            failures.append(FailureType.LOW_SIMILARITY.value)
        
        # Check hallucination tests
        should_refuse = question_data.get("should_refuse", False)
        if should_refuse:
            # For hallucination tests, we want LOW similarity (meaning system can't find relevant info)
            if top_similarity > 0.5:
                failures.append(FailureType.HALLUCINATION.value)
            else:
                # Good! System correctly has no relevant information
                failures = []  # Clear any other failures for hallucination tests
        
        # Empty response check
        if not chunks:
            failures.append(FailureType.EMPTY_RESPONSE.value)
        
        passed = len(failures) == 0
        
        return EvaluationResult(
            question_id=question_id,
            question=question,
            category=question_data.get("category", "general"),
            difficulty=question_data.get("difficulty", "medium"),
            passed=passed,
            failures=failures,
            retrieved_companies=retrieved_companies,
            retrieved_sections=retrieved_sections,
            avg_similarity=avg_similarity,
            top_similarity=top_similarity,
            keywords_found=keywords_found,
            keywords_missing=keywords_missing,
            answer_preview=answer_text[:200] if answer_text else "",
            execution_time_ms=execution_time
        )
    
    def run_evaluation(self, questions: List[Dict] = None) -> EvaluationReport:
        """Run full evaluation on question set."""
        if questions is None:
            questions = GOLD_QUESTIONS
        
        print("\n" + "="*70)
        print("RAG SYSTEM EVALUATION")
        print("="*70)
        print(f"Questions to evaluate: {len(questions)}")
        print("="*70)
        
        self.results = []
        
        for i, q in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] Evaluating: {q['id']}")
            print(f"   Q: {q['question'][:60]}...")
            
            result = self.evaluate_question(q)
            self.results.append(result)
            
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"   {status} | Similarity: {result.top_similarity:.3f} | Companies: {result.retrieved_companies}")
            
            if not result.passed:
                print(f"   Failures: {result.failures}")
        
        # Generate report
        return self._generate_report()
    
    def _generate_report(self) -> EvaluationReport:
        """Generate comprehensive evaluation report."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        pass_rate = passed / len(self.results) if self.results else 0
        
        # Results by category
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {"passed": 0, "failed": 0}
            if r.passed:
                categories[r.category]["passed"] += 1
            else:
                categories[r.category]["failed"] += 1
        
        # Results by difficulty
        difficulties = {}
        for r in self.results:
            if r.difficulty not in difficulties:
                difficulties[r.difficulty] = {"passed": 0, "failed": 0}
            if r.passed:
                difficulties[r.difficulty]["passed"] += 1
            else:
                difficulties[r.difficulty]["failed"] += 1
        
        # Failure breakdown
        failure_breakdown = {}
        for r in self.results:
            for f in r.failures:
                failure_breakdown[f] = failure_breakdown.get(f, 0) + 1
        
        # Average metrics
        avg_similarity = sum(r.avg_similarity for r in self.results) / len(self.results)
        avg_time = sum(r.execution_time_ms for r in self.results) / len(self.results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(failure_breakdown, categories)
        
        return EvaluationReport(
            timestamp=datetime.now().isoformat(),
            total_questions=len(self.results),
            passed=passed,
            failed=failed,
            pass_rate=pass_rate,
            results_by_category=categories,
            results_by_difficulty=difficulties,
            failure_breakdown=failure_breakdown,
            avg_similarity=avg_similarity,
            avg_execution_time_ms=avg_time,
            individual_results=self.results,
            recommendations=recommendations
        )
    
    def _generate_recommendations(
        self, 
        failures: Dict[str, int],
        categories: Dict[str, Dict]
    ) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if failures.get(FailureType.LOW_SIMILARITY.value, 0) > 2:
            recommendations.append(
                "CHUNKING: Consider reducing chunk size or increasing overlap. "
                "Low similarity scores suggest chunks may not align well with question intent."
            )
        
        if failures.get(FailureType.WRONG_SECTION.value, 0) > 2:
            recommendations.append(
                "SECTION DETECTION: Improve section classification logic. "
                "Add more specific patterns for 10-K section headers."
            )
        
        if failures.get(FailureType.MISSING_KEYWORDS.value, 0) > 3:
            recommendations.append(
                "RETRIEVAL: Consider increasing k (number of chunks retrieved) "
                "or implementing query expansion/rewriting."
            )
        
        if failures.get(FailureType.HALLUCINATION.value, 0) > 0:
            recommendations.append(
                "HALLUCINATION: Strengthen the system prompt to enforce "
                "'I don't know' responses when context is insufficient."
            )
        
        # Category-specific recommendations
        for cat, stats in categories.items():
            if stats["failed"] > stats["passed"]:
                recommendations.append(
                    f"CATEGORY '{cat}': Pass rate below 50%. "
                    f"Review chunking and indexing for {cat}-related content."
                )
        
        if not recommendations:
            recommendations.append(
                "System performing well! Consider adding more diverse test questions."
            )
        
        return recommendations


# ============================================================================
# PIPELINE FREEZE
# ============================================================================

def freeze_pipeline_v1():
    """Freeze the v1 pipeline configuration."""
    print("\n" + "="*70)
    print("FREEZING PIPELINE v1")
    print("="*70)
    
    # Load existing configs
    retrieval_config_path = BASE_DIR / "data" / "retrieval_config.json"
    chunks_metadata_path = BASE_DIR / "data" / "chunks_metadata.json"
    
    retrieval_config = {}
    if retrieval_config_path.exists():
        with open(retrieval_config_path) as f:
            retrieval_config = json.load(f)
    
    # Get chunk stats
    chunk_count = 0
    if chunks_metadata_path.exists():
        with open(chunks_metadata_path) as f:
            chunks = json.load(f)
            chunk_count = len(chunks)
    
    # Create frozen config
    frozen_config = {
        "version": "1.0.0",
        "frozen_at": datetime.now().isoformat(),
        "description": "SEC EDGAR 10-K RAG Pipeline v1",
        
        "data_collection": {
            "companies": ["JPM", "GS", "UBS"],
            "filing_type": "10-K",
            "fiscal_year": "2024",
            "source": "SEC EDGAR",
            "raw_html_dir": "data/raw_html/",
            "raw_txt_dir": "data/raw_txt/"
        },
        
        "chunking": {
            "chunk_size_tokens": 600,
            "chunk_overlap_tokens": 100,
            "min_chunk_size_tokens": 100,
            "total_chunks": chunk_count
        },
        
        "embedding": {
            "model": "all-MiniLM-L6-v2",
            "dimensions": 384,
            "provider": "sentence-transformers"
        },
        
        "vector_store": {
            "type": "ChromaDB",
            "collection_name": "sec_10k_filings",
            "persist_directory": "data/chroma_db/",
            "similarity_metric": "cosine"
        },
        
        "retrieval": retrieval_config,
        
        "generation": {
            "supported_providers": ["openai", "gemini", "mock"],
            "default_provider": "gemini",
            "openai_model": "gpt-4o-mini",
            "gemini_model": "gemini-1.5-flash",
            "temperature": 0.1,
            "max_tokens": 1500
        },
        
        "evaluation": {
            "gold_questions_count": len(GOLD_QUESTIONS),
            "categories": list(set(q["category"] for q in GOLD_QUESTIONS))
        }
    }
    
    # Save frozen config
    frozen_path = BASE_DIR / "data" / "pipeline_v1_frozen.json"
    with open(frozen_path, 'w') as f:
        json.dump(frozen_config, f, indent=2)
    
    print(f"✓ Pipeline configuration frozen to: {frozen_path}")
    
    # Create README for the pipeline
    readme_content = f"""# SEC EDGAR 10-K RAG Pipeline v1.0

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

## Frozen: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_path = BASE_DIR / "PIPELINE_README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✓ Pipeline README created: {readme_path}")
    
    return frozen_config


# ============================================================================
# PRETTY REPORT PRINTER
# ============================================================================

def print_evaluation_report(report: EvaluationReport):
    """Print a formatted evaluation report."""
    print("\n" + "="*70)
    print("📊 EVALUATION REPORT")
    print("="*70)
    
    # Overall stats
    print(f"\n📈 OVERALL RESULTS")
    print(f"   Total Questions: {report.total_questions}")
    print(f"   Passed: {report.passed} ({report.pass_rate*100:.1f}%)")
    print(f"   Failed: {report.failed}")
    print(f"   Avg Similarity: {report.avg_similarity:.3f}")
    print(f"   Avg Execution Time: {report.avg_execution_time_ms:.1f}ms")
    
    # Results by category
    print(f"\n📁 RESULTS BY CATEGORY")
    for cat, stats in sorted(report.results_by_category.items()):
        total = stats['passed'] + stats['failed']
        rate = stats['passed'] / total * 100 if total else 0
        bar = "█" * int(rate/10) + "░" * (10 - int(rate/10))
        print(f"   {cat:20s} [{bar}] {rate:5.1f}% ({stats['passed']}/{total})")
    
    # Results by difficulty
    print(f"\n📊 RESULTS BY DIFFICULTY")
    for diff, stats in sorted(report.results_by_difficulty.items()):
        total = stats['passed'] + stats['failed']
        rate = stats['passed'] / total * 100 if total else 0
        print(f"   {diff:10s}: {stats['passed']}/{total} passed ({rate:.1f}%)")
    
    # Failure breakdown
    if report.failure_breakdown:
        print(f"\n❌ FAILURE BREAKDOWN")
        for failure_type, count in sorted(report.failure_breakdown.items(), key=lambda x: -x[1]):
            print(f"   {failure_type:25s}: {count}")
    
    # Failed questions detail
    failed_results = [r for r in report.individual_results if not r.passed]
    if failed_results:
        print(f"\n📋 FAILED QUESTIONS DETAIL")
        for r in failed_results:
            print(f"\n   {r.question_id}: {r.question[:50]}...")
            print(f"      Failures: {r.failures}")
            print(f"      Retrieved: {r.retrieved_companies}")
            print(f"      Similarity: {r.top_similarity:.3f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"   {i}. {rec}")
    
    print("\n" + "="*70)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run complete evaluation and freeze pipeline."""
    print("="*70)
    print("Day 5: Evaluation & Pipeline Freeze")
    print("="*70)
    
    # Run evaluation
    evaluator = RAGEvaluator()
    report = evaluator.run_evaluation(GOLD_QUESTIONS)
    
    # Print report
    print_evaluation_report(report)
    
    # Save detailed report
    report_path = EVAL_DIR / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    print(f"\n✓ Detailed report saved: {report_path}")
    
    # Save gold questions
    questions_path = EVAL_DIR / "gold_questions.json"
    with open(questions_path, 'w') as f:
        json.dump(GOLD_QUESTIONS, f, indent=2)
    print(f"✓ Gold questions saved: {questions_path}")
    
    # Freeze pipeline
    frozen_config = freeze_pipeline_v1()
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 DAY 5 COMPLETE - PIPELINE v1 FROZEN")
    print("="*70)
    print(f"""
Summary:
  • Evaluated {report.total_questions} gold questions
  • Pass rate: {report.pass_rate*100:.1f}%
  • Average retrieval similarity: {report.avg_similarity:.3f}
  • Pipeline configuration frozen
  
Files created:
  • {EVAL_DIR}/evaluation_report_*.json
  • {EVAL_DIR}/gold_questions.json
  • data/pipeline_v1_frozen.json
  • PIPELINE_README.md
  
Next steps:
  1. Review failed questions and recommendations
  2. Iterate on chunking/prompts if needed
  3. Add more gold questions for better coverage
  4. Set up API keys for production LLM testing
""")
    print("="*70)


if __name__ == "__main__":
    main()
