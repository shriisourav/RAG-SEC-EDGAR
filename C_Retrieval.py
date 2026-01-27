"""
Day 3: Retrieval System
Implements top-k semantic search with filtering, testing, and evaluation.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import chromadb
from sentence_transformers import SentenceTransformer
from tabulate import tabulate
import textwrap

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Default retrieval parameters
DEFAULT_K = 5               # Number of chunks to retrieve
MIN_SIMILARITY = 0.3        # Minimum similarity threshold


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RetrievalResult:
    """A single retrieval result with metadata."""
    rank: int
    chunk_id: str
    text: str
    company: str
    company_name: str
    section: str
    source_file: str
    similarity: float
    token_count: int
    
    def preview(self, max_chars: int = 200) -> str:
        """Get a preview of the text."""
        if len(self.text) <= max_chars:
            return self.text
        return self.text[:max_chars] + "..."
    
    def __str__(self) -> str:
        return (
            f"[Rank {self.rank}] {self.company} | {self.section} | "
            f"Similarity: {self.similarity:.3f}\n"
            f"{self.preview()}"
        )


@dataclass
class RetrievalResponse:
    """Complete response from a retrieval query."""
    query: str
    results: List[RetrievalResult]
    k: int
    filters_applied: Dict[str, str]
    
    def get_context(self, separator: str = "\n\n---\n\n") -> str:
        """Get combined context from all results for LLM."""
        return separator.join([r.text for r in self.results])
    
    def get_citations(self) -> List[Dict[str, str]]:
        """Get citation info for each result."""
        return [
            {
                "source": f"{r.company} - {r.section}",
                "file": r.source_file,
                "chunk_id": r.chunk_id
            }
            for r in self.results
        ]
    
    def print_results(self):
        """Pretty print the results."""
        print(f"\n{'='*70}")
        print(f"Query: \"{self.query}\"")
        print(f"Retrieved: {len(self.results)} chunks (k={self.k})")
        if self.filters_applied:
            print(f"Filters: {self.filters_applied}")
        print("="*70)
        
        for r in self.results:
            print(f"\n[{r.rank}] 📄 {r.company_name} | Section: {r.section}")
            print(f"    Similarity: {r.similarity:.3f} | Tokens: {r.token_count}")
            print(f"    Source: {r.source_file}")
            print(f"    ─" * 35)
            # Wrap text for readability
            wrapped = textwrap.fill(r.preview(300), width=70, initial_indent="    ", subsequent_indent="    ")
            print(wrapped)


# ============================================================================
# RETRIEVAL ENGINE
# ============================================================================

class Retriever:
    """
    Semantic retrieval engine using ChromaDB.
    Supports filtering by company, section, and similarity threshold.
    """
    
    def __init__(
        self, 
        persist_directory: Path = CHROMA_DIR,
        collection_name: str = "sec_10k_filings"
    ):
        """Initialize the retriever with ChromaDB connection."""
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection = self.client.get_collection(name=collection_name)
        
        # Initialize embedding model
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        print(f"✓ Retriever initialized")
        print(f"  Collection: {collection_name}")
        print(f"  Total documents: {self.collection.count()}")
    
    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_K,
        company: Optional[str] = None,
        section: Optional[str] = None,
        min_similarity: float = MIN_SIMILARITY
    ) -> RetrievalResponse:
        """
        Retrieve the top-k most relevant chunks for a query.
        
        Args:
            query: The search query
            k: Number of results to return
            company: Optional filter by company ticker (JPM, GS, UBS)
            section: Optional filter by section name
            min_similarity: Minimum similarity score (0-1)
            
        Returns:
            RetrievalResponse with ranked results
        """
        # Build filter
        where_filter = {}
        filters_applied = {}
        
        if company:
            where_filter["company"] = company.upper()
            filters_applied["company"] = company.upper()
        
        if section:
            where_filter["section"] = section
            filters_applied["section"] = section
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Query ChromaDB (get more results to filter by similarity)
        raw_results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k * 2,  # Get extra for filtering
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Process results
        results = []
        for i in range(len(raw_results['ids'][0])):
            # ChromaDB returns distance, convert to similarity
            distance = raw_results['distances'][0][i]
            similarity = 1 - distance
            
            # Apply similarity threshold
            if similarity < min_similarity:
                continue
            
            metadata = raw_results['metadatas'][0][i]
            
            result = RetrievalResult(
                rank=len(results) + 1,
                chunk_id=raw_results['ids'][0][i],
                text=raw_results['documents'][0][i],
                company=metadata.get('company', 'Unknown'),
                company_name=metadata.get('company_name', 'Unknown'),
                section=metadata.get('section', 'Unknown'),
                source_file=metadata.get('source_file', 'Unknown'),
                similarity=similarity,
                token_count=metadata.get('token_count', 0)
            )
            results.append(result)
            
            if len(results) >= k:
                break
        
        return RetrievalResponse(
            query=query,
            results=results,
            k=k,
            filters_applied=filters_applied
        )
    
    def retrieve_with_rerank(
        self,
        query: str,
        k: int = DEFAULT_K,
        initial_k: int = 20,
        company: Optional[str] = None,
        section: Optional[str] = None
    ) -> RetrievalResponse:
        """
        Two-stage retrieval: broad search + reranking.
        Useful for improving precision.
        """
        # Stage 1: Broad retrieval
        initial_results = self.retrieve(
            query=query,
            k=initial_k,
            company=company,
            section=section,
            min_similarity=0.2
        )
        
        # Stage 2: Rerank by combining similarity with keyword matching
        for result in initial_results.results:
            query_words = set(query.lower().split())
            text_words = set(result.text.lower().split())
            keyword_overlap = len(query_words & text_words) / len(query_words)
            
            # Boost similarity with keyword overlap
            result.similarity = 0.7 * result.similarity + 0.3 * keyword_overlap
        
        # Re-sort and take top-k
        initial_results.results.sort(key=lambda x: x.similarity, reverse=True)
        final_results = initial_results.results[:k]
        
        # Update ranks
        for i, result in enumerate(final_results):
            result.rank = i + 1
        
        return RetrievalResponse(
            query=query,
            results=final_results,
            k=k,
            filters_applied=initial_results.filters_applied
        )
    
    def multi_query_retrieve(
        self,
        queries: List[str],
        k_per_query: int = 3,
        deduplicate: bool = True
    ) -> RetrievalResponse:
        """
        Retrieve using multiple query variations.
        Useful for complex questions.
        """
        all_results = {}
        
        for query in queries:
            response = self.retrieve(query=query, k=k_per_query)
            for result in response.results:
                if result.chunk_id not in all_results:
                    all_results[result.chunk_id] = result
                else:
                    # Keep the higher similarity score
                    if result.similarity > all_results[result.chunk_id].similarity:
                        all_results[result.chunk_id] = result
        
        # Sort by similarity and update ranks
        sorted_results = sorted(all_results.values(), key=lambda x: x.similarity, reverse=True)
        for i, result in enumerate(sorted_results):
            result.rank = i + 1
        
        return RetrievalResponse(
            query=" | ".join(queries),
            results=sorted_results,
            k=len(sorted_results),
            filters_applied={}
        )
    
    def get_available_filters(self) -> Dict[str, List[str]]:
        """Get available filter values."""
        all_data = self.collection.get(include=["metadatas"])
        
        companies = set()
        sections = set()
        
        for metadata in all_data['metadatas']:
            companies.add(metadata.get('company'))
            sections.add(metadata.get('section'))
        
        return {
            "companies": sorted(list(companies)),
            "sections": sorted(list(sections))
        }


# ============================================================================
# TESTING & EVALUATION
# ============================================================================

class RetrievalTester:
    """Test and evaluate retrieval quality."""
    
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.test_results = []
    
    def test_query(
        self,
        query: str,
        expected_company: Optional[str] = None,
        expected_section: Optional[str] = None,
        k: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Test a single query and evaluate results.
        """
        response = self.retriever.retrieve(query=query, k=k)
        
        # Calculate metrics
        metrics = {
            "query": query,
            "num_results": len(response.results),
            "avg_similarity": 0,
            "top_similarity": 0,
            "company_match": False,
            "section_match": False,
        }
        
        if response.results:
            similarities = [r.similarity for r in response.results]
            metrics["avg_similarity"] = sum(similarities) / len(similarities)
            metrics["top_similarity"] = max(similarities)
            
            if expected_company:
                metrics["company_match"] = any(
                    r.company == expected_company.upper() 
                    for r in response.results
                )
            
            if expected_section:
                metrics["section_match"] = any(
                    expected_section.lower() in r.section.lower() 
                    for r in response.results
                )
        
        if verbose:
            response.print_results()
            print(f"\n📊 Metrics:")
            print(f"   Avg Similarity: {metrics['avg_similarity']:.3f}")
            print(f"   Top Similarity: {metrics['top_similarity']:.3f}")
            if expected_company:
                status = "✓" if metrics['company_match'] else "✗"
                print(f"   Company Match: {status}")
            if expected_section:
                status = "✓" if metrics['section_match'] else "✗"
                print(f"   Section Match: {status}")
        
        self.test_results.append(metrics)
        return metrics
    
    def run_test_suite(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """
        Run a suite of test cases.
        
        test_cases format: [
            {"query": "...", "expected_company": "JPM", "expected_section": "Risk"},
            ...
        ]
        """
        print("\n" + "="*70)
        print("RETRIEVAL TEST SUITE")
        print("="*70)
        
        results = []
        for i, test in enumerate(test_cases, 1):
            print(f"\n--- Test {i}/{len(test_cases)} ---")
            result = self.test_query(
                query=test.get("query"),
                expected_company=test.get("expected_company"),
                expected_section=test.get("expected_section"),
                verbose=True
            )
            results.append(result)
        
        # Aggregate metrics
        summary = {
            "total_tests": len(results),
            "avg_similarity": sum(r["avg_similarity"] for r in results) / len(results),
            "avg_top_similarity": sum(r["top_similarity"] for r in results) / len(results),
            "company_accuracy": sum(1 for r in results if r.get("company_match", True)) / len(results),
            "section_accuracy": sum(1 for r in results if r.get("section_match", True)) / len(results),
        }
        
        print("\n" + "="*70)
        print("TEST SUITE SUMMARY")
        print("="*70)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Avg Similarity: {summary['avg_similarity']:.3f}")
        print(f"Avg Top Similarity: {summary['avg_top_similarity']:.3f}")
        print(f"Company Match Rate: {summary['company_accuracy']*100:.1f}%")
        print(f"Section Match Rate: {summary['section_accuracy']*100:.1f}%")
        
        return summary
    
    def inspect_chunk(self, chunk_id: str):
        """Manually inspect a specific chunk."""
        result = self.retriever.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"]
        )
        
        if not result['ids']:
            print(f"Chunk not found: {chunk_id}")
            return
        
        print(f"\n{'='*70}")
        print(f"CHUNK INSPECTION: {chunk_id}")
        print("="*70)
        
        metadata = result['metadatas'][0]
        print(f"\nMetadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        print(f"\nFull Text ({len(result['documents'][0])} chars):")
        print("-"*70)
        print(result['documents'][0])
        print("-"*70)


class ParameterTuner:
    """Tune retrieval parameters for optimal performance."""
    
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
    
    def tune_k(
        self,
        test_queries: List[str],
        k_values: List[int] = [3, 5, 7, 10, 15],
        target_metric: str = "avg_similarity"
    ) -> Dict[int, float]:
        """
        Find optimal k by testing different values.
        """
        print("\n" + "="*70)
        print("TUNING: Optimal K Value")
        print("="*70)
        
        results = {}
        
        for k in k_values:
            metrics = []
            for query in test_queries:
                response = self.retriever.retrieve(query=query, k=k)
                if response.results:
                    similarities = [r.similarity for r in response.results]
                    metrics.append({
                        "avg_similarity": sum(similarities) / len(similarities),
                        "min_similarity": min(similarities),
                        "coverage": len(response.results)
                    })
            
            avg_metric = sum(m[target_metric] for m in metrics) / len(metrics)
            results[k] = avg_metric
            print(f"  k={k:2d}: {target_metric}={avg_metric:.4f}")
        
        best_k = max(results, key=results.get)
        print(f"\n✓ Recommended k: {best_k}")
        
        return results
    
    def tune_similarity_threshold(
        self,
        test_queries: List[str],
        thresholds: List[float] = [0.2, 0.3, 0.4, 0.5, 0.6]
    ) -> Dict[float, Dict]:
        """
        Find optimal similarity threshold.
        """
        print("\n" + "="*70)
        print("TUNING: Similarity Threshold")
        print("="*70)
        
        results = {}
        
        for threshold in thresholds:
            total_results = 0
            avg_similarity = []
            
            for query in test_queries:
                response = self.retriever.retrieve(
                    query=query, 
                    k=10, 
                    min_similarity=threshold
                )
                total_results += len(response.results)
                if response.results:
                    avg_similarity.extend([r.similarity for r in response.results])
            
            results[threshold] = {
                "avg_results_per_query": total_results / len(test_queries),
                "avg_similarity": sum(avg_similarity) / len(avg_similarity) if avg_similarity else 0
            }
            
            print(f"  threshold={threshold:.2f}: "
                  f"avg_results={results[threshold]['avg_results_per_query']:.1f}, "
                  f"avg_sim={results[threshold]['avg_similarity']:.3f}")
        
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution with tests and tuning."""
    print("="*70)
    print("Day 3: Retrieval System")
    print("="*70)
    
    # Initialize retriever
    retriever = Retriever()
    
    # Show available filters
    filters = retriever.get_available_filters()
    print(f"\nAvailable filters:")
    print(f"  Companies: {filters['companies']}")
    print(f"  Sections: {filters['sections']}")
    
    # ========================================
    # TEST 1: Basic Retrieval
    # ========================================
    print("\n" + "="*70)
    print("TEST 1: Basic Retrieval")
    print("="*70)
    
    basic_queries = [
        "What are the main risk factors for the bank?",
        "How does the company manage liquidity risk?",
        "What is the credit risk exposure?",
        "Describe the regulatory capital requirements",
        "What are the operational risk controls?",
    ]
    
    for query in basic_queries:
        response = retriever.retrieve(query, k=3)
        response.print_results()
    
    # ========================================
    # TEST 2: Filtered Retrieval
    # ========================================
    print("\n" + "="*70)
    print("TEST 2: Filtered Retrieval (JPM only)")
    print("="*70)
    
    response = retriever.retrieve(
        query="What are the liquidity risk management practices?",
        k=5,
        company="JPM"
    )
    response.print_results()
    
    # ========================================
    # TEST 3: Section-Specific Retrieval
    # ========================================
    print("\n" + "="*70)
    print("TEST 3: Section-Specific Retrieval (Risk Factors)")
    print("="*70)
    
    response = retriever.retrieve(
        query="regulatory compliance risks",
        k=5,
        section="Risk Factors"
    )
    response.print_results()
    
    # ========================================
    # TEST 4: Parameter Tuning
    # ========================================
    tuner = ParameterTuner(retriever)
    
    test_queries_for_tuning = [
        "credit risk management",
        "liquidity requirements",
        "market risk exposure",
        "operational controls",
        "capital adequacy",
    ]
    
    # Tune k
    k_results = tuner.tune_k(test_queries_for_tuning)
    
    # Tune similarity threshold
    threshold_results = tuner.tune_similarity_threshold(test_queries_for_tuning)
    
    # ========================================
    # TEST 5: Full Test Suite
    # ========================================
    tester = RetrievalTester(retriever)
    
    test_suite = [
        {
            "query": "What are JPMorgan's main risk factors?",
            "expected_company": "JPM",
            "expected_section": "Risk"
        },
        {
            "query": "Goldman Sachs credit risk exposure",
            "expected_company": "GS",
            "expected_section": "Credit"
        },
        {
            "query": "UBS liquidity management",
            "expected_company": "UBS",
            "expected_section": "Liquidity"
        },
        {
            "query": "Market risk and trading activities",
            "expected_section": "Market"
        },
        {
            "query": "Regulatory capital requirements Basel III",
            "expected_section": "Risk"
        },
    ]
    
    # Run test suite
    summary = tester.run_test_suite(test_suite)
    
    # ========================================
    # MANUAL INSPECTION EXAMPLE
    # ========================================
    print("\n" + "="*70)
    print("MANUAL INSPECTION: Sample Chunk")
    print("="*70)
    
    # Get a sample chunk to inspect
    sample_response = retriever.retrieve("credit risk", k=1)
    if sample_response.results:
        tester.inspect_chunk(sample_response.results[0].chunk_id)
    
    # ========================================
    # SAVE RETRIEVAL CONFIG
    # ========================================
    config = {
        "default_k": 5,
        "min_similarity": 0.35,
        "embedding_model": EMBEDDING_MODEL,
        "collection_name": "sec_10k_filings",
        "available_companies": filters['companies'],
        "available_sections": filters['sections'],
    }
    
    config_path = BASE_DIR / "data" / "retrieval_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✓ Retrieval config saved to: {config_path}")
    
    print("\n" + "="*70)
    print("✓ Day 3 Complete: Retrieval System Ready!")
    print("="*70)


if __name__ == "__main__":
    main()
