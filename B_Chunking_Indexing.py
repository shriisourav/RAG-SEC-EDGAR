"""
Day 2: Chunking and Indexing Pipeline
Splits 10-K documents into chunks, generates embeddings, and stores in ChromaDB.
"""

import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import tiktoken

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent
RAW_TXT_DIR = BASE_DIR / "data" / "raw_txt"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"

# Chunking parameters
CHUNK_SIZE = 600          # Target tokens per chunk (500-800 range)
CHUNK_OVERLAP = 100       # Overlap tokens between chunks
MIN_CHUNK_SIZE = 100      # Minimum tokens for a valid chunk

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast, good quality, 384 dimensions

# Company metadata mapping
COMPANY_INFO = {
    "JPM": {"name": "JPMorgan Chase", "sector": "Banking", "ticker": "JPM"},
    "GS": {"name": "Goldman Sachs", "sector": "Investment Banking", "ticker": "GS"},
    "UBS": {"name": "UBS Group", "sector": "Investment Banking", "ticker": "UBS"},
    "BAC": {"name": "Bank of America", "sector": "Banking", "ticker": "BAC"},
    "MS": {"name": "Morgan Stanley", "sector": "Investment Banking", "ticker": "MS"},
    "C": {"name": "Citigroup", "sector": "Banking", "ticker": "C"},
}

# 10-K Section patterns for classification
SECTION_PATTERNS = {
    "Business Overview": [r"(?i)item\s*1[.\s]+business", r"(?i)^business\s*$"],
    "Risk Factors": [r"(?i)item\s*1a[.\s]+risk\s*factors", r"(?i)^risk\s*factors"],
    "Properties": [r"(?i)item\s*2[.\s]+properties"],
    "Legal Proceedings": [r"(?i)item\s*3[.\s]+legal\s*proceedings"],
    "MD&A": [r"(?i)item\s*7[.\s]+management", r"(?i)management.s\s+discussion"],
    "Financial Statements": [r"(?i)item\s*8[.\s]+financial\s*statements"],
    "Controls & Procedures": [r"(?i)item\s*9a[.\s]+controls"],
    "Executive Compensation": [r"(?i)item\s*11[.\s]+executive\s*compensation"],
    "Liquidity": [r"(?i)liquidity\s*(and|&)?\s*capital", r"(?i)^liquidity"],
    "Credit Risk": [r"(?i)credit\s*risk"],
    "Market Risk": [r"(?i)market\s*risk"],
    "Operational Risk": [r"(?i)operational\s*risk"],
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    id: str
    text: str
    company: str
    company_name: str
    sector: str
    section: str
    source_file: str
    chunk_index: int
    total_chunks: int
    token_count: int
    char_start: int
    char_end: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# TOKENIZATION & CHUNKING
# ============================================================================

class TokenCounter:
    """Count tokens using tiktoken (OpenAI tokenizer)."""
    
    def __init__(self, model: str = "cl100k_base"):
        self.encoding = tiktoken.get_encoding(model)
    
    def count(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def decode(self, tokens: List[int]) -> str:
        """Decode tokens back to text."""
        return self.encoding.decode(tokens)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to tokens."""
        return self.encoding.encode(text)


class TextChunker:
    """Chunk text into overlapping segments based on token count."""
    
    def __init__(
        self, 
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.token_counter = TokenCounter()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better chunk boundaries."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        Returns list of dicts with 'text', 'char_start', 'char_end', 'token_count'.
        """
        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0
        current_start = 0
        char_position = 0
        
        for sentence in sentences:
            sentence_tokens = self.token_counter.count(sentence)
            
            # If single sentence exceeds chunk size, split it
            if sentence_tokens > self.chunk_size:
                # Flush current chunk if exists
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'char_start': current_start,
                        'char_end': current_start + len(chunk_text),
                        'token_count': current_tokens
                    })
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence by words
                words = sentence.split()
                temp_chunk = []
                temp_tokens = 0
                for word in words:
                    word_tokens = self.token_counter.count(word + ' ')
                    if temp_tokens + word_tokens > self.chunk_size:
                        if temp_chunk:
                            chunk_text = ' '.join(temp_chunk)
                            chunks.append({
                                'text': chunk_text,
                                'char_start': char_position,
                                'char_end': char_position + len(chunk_text),
                                'token_count': temp_tokens
                            })
                        temp_chunk = [word]
                        temp_tokens = word_tokens
                    else:
                        temp_chunk.append(word)
                        temp_tokens += word_tokens
                
                if temp_chunk:
                    current_chunk = temp_chunk
                    current_tokens = temp_tokens
                    current_start = char_position
                
                char_position += len(sentence) + 1
                continue
            
            # Check if adding this sentence exceeds chunk size
            if current_tokens + sentence_tokens > self.chunk_size:
                # Save current chunk
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'char_start': current_start,
                        'char_end': current_start + len(chunk_text),
                        'token_count': current_tokens
                    })
                
                # Start new chunk with overlap
                overlap_tokens = 0
                overlap_sentences = []
                for s in reversed(current_chunk):
                    s_tokens = self.token_counter.count(s)
                    if overlap_tokens + s_tokens <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_tokens = overlap_tokens + sentence_tokens
                current_start = char_position - sum(len(s) + 1 for s in overlap_sentences)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
            
            char_position += len(sentence) + 1
        
        # Don't forget the last chunk
        if current_chunk and current_tokens >= self.min_chunk_size:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'char_start': current_start,
                'char_end': current_start + len(chunk_text),
                'token_count': current_tokens
            })
        
        return chunks


# ============================================================================
# SECTION DETECTION
# ============================================================================

def detect_section(text: str) -> str:
    """Detect the 10-K section based on text content."""
    # Check first 500 characters for section headers
    header_text = text[:500].lower()
    
    for section_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, header_text):
                return section_name
    
    # Check for keywords in the full text
    text_lower = text.lower()
    
    keyword_scores = {
        "Risk Factors": ["risk", "adverse", "uncertainty", "material"],
        "Liquidity": ["liquidity", "funding", "capital resources", "cash flow"],
        "MD&A": ["results of operations", "compared to", "fiscal year"],
        "Credit Risk": ["credit risk", "loan loss", "allowance"],
        "Market Risk": ["market risk", "var", "value at risk", "trading"],
        "Financial Statements": ["consolidated", "balance sheet", "income statement"],
    }
    
    best_section = "General"
    best_score = 0
    
    for section, keywords in keyword_scores.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_section = section
    
    return best_section if best_score >= 2 else "General"


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def generate_chunk_id(company: str, chunk_index: int, text: str) -> str:
    """Generate a unique ID for a chunk."""
    content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{company}_{chunk_index:04d}_{content_hash}"


def process_document(filepath: Path) -> List[Chunk]:
    """Process a single 10-K document into chunks."""
    print(f"\n{'─' * 60}")
    print(f"Processing: {filepath.name}")
    print("─" * 60)
    
    # Read the document
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Extract company ticker from filename (e.g., "JPM_10K_20241231.txt")
    filename = filepath.stem
    ticker = filename.split('_')[0].upper()
    
    # Get company info
    company_info = COMPANY_INFO.get(ticker, {
        "name": ticker,
        "sector": "Unknown",
        "ticker": ticker
    })
    
    print(f"  Company: {company_info['name']} ({ticker})")
    print(f"  Document size: {len(text):,} characters")
    
    # Create chunker and process
    chunker = TextChunker()
    raw_chunks = chunker.chunk_text(text)
    
    print(f"  Raw chunks created: {len(raw_chunks)}")
    
    # Create Chunk objects with metadata
    chunks = []
    for i, raw_chunk in enumerate(raw_chunks):
        # Detect section for this chunk
        section = detect_section(raw_chunk['text'])
        
        chunk = Chunk(
            id=generate_chunk_id(ticker, i, raw_chunk['text']),
            text=raw_chunk['text'],
            company=ticker,
            company_name=company_info['name'],
            sector=company_info['sector'],
            section=section,
            source_file=filepath.name,
            chunk_index=i,
            total_chunks=len(raw_chunks),
            token_count=raw_chunk['token_count'],
            char_start=raw_chunk['char_start'],
            char_end=raw_chunk['char_end']
        )
        chunks.append(chunk)
    
    # Section distribution
    section_counts = {}
    for chunk in chunks:
        section_counts[chunk.section] = section_counts.get(chunk.section, 0) + 1
    
    print(f"\n  Section distribution:")
    for section, count in sorted(section_counts.items(), key=lambda x: -x[1]):
        print(f"    • {section}: {count} chunks")
    
    # Token statistics
    token_counts = [c.token_count for c in chunks]
    print(f"\n  Token statistics:")
    print(f"    • Min: {min(token_counts)}")
    print(f"    • Max: {max(token_counts)}")
    print(f"    • Avg: {sum(token_counts) / len(token_counts):.1f}")
    
    return chunks


# ============================================================================
# VECTOR DATABASE
# ============================================================================

class VectorStore:
    """ChromaDB vector store for document chunks."""
    
    def __init__(self, persist_directory: Path, collection_name: str = "sec_10k_filings"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize ChromaDB with persistence
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        
        # Initialize embedding function using sentence-transformers
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"  ChromaDB initialized at: {persist_directory}")
        print(f"  Collection: {collection_name}")
        print(f"  Existing documents: {self.collection.count()}")
    
    def add_chunks(self, chunks: List[Chunk], batch_size: int = 100):
        """Add chunks to the vector store."""
        print(f"\n  Adding {len(chunks)} chunks to vector store...")
        
        # Process in batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Prepare data
            ids = [c.id for c in batch]
            texts = [c.text for c in batch]
            metadatas = [
                {
                    "company": c.company,
                    "company_name": c.company_name,
                    "sector": c.sector,
                    "section": c.section,
                    "source_file": c.source_file,
                    "chunk_index": c.chunk_index,
                    "total_chunks": c.total_chunks,
                    "token_count": c.token_count,
                }
                for c in batch
            ]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(texts).tolist()
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            
            print(f"    Batch {i // batch_size + 1}: Added {len(batch)} chunks")
        
        print(f"  ✓ Total documents in store: {self.collection.count()}")
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        company_filter: str = None,
        section_filter: str = None
    ) -> List[Dict[str, Any]]:
        """Search for similar chunks."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Build filter
        where_filter = {}
        if company_filter:
            where_filter["company"] = company_filter
        if section_filter:
            where_filter["section"] = section_filter
        
        # Query
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter if where_filter else None,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return formatted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        # Get all metadata
        all_data = self.collection.get(include=["metadatas"])
        
        stats = {
            "total_chunks": self.collection.count(),
            "companies": {},
            "sections": {},
        }
        
        for metadata in all_data['metadatas']:
            company = metadata.get('company', 'Unknown')
            section = metadata.get('section', 'Unknown')
            
            stats['companies'][company] = stats['companies'].get(company, 0) + 1
            stats['sections'][section] = stats['sections'].get(section, 0) + 1
        
        return stats


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("Day 2: Chunking and Indexing Pipeline")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Chunk size: {CHUNK_SIZE} tokens")
    print(f"  Chunk overlap: {CHUNK_OVERLAP} tokens")
    print(f"  Embedding model: {EMBEDDING_MODEL}")
    print(f"  Vector DB: ChromaDB at {CHROMA_DIR}")
    
    # Ensure directories exist
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find all text files
    txt_files = list(RAW_TXT_DIR.glob("*.txt"))
    print(f"\nFound {len(txt_files)} text files to process")
    
    if not txt_files:
        print("Error: No text files found in", RAW_TXT_DIR)
        return
    
    # Process all documents
    all_chunks = []
    for txt_file in txt_files:
        chunks = process_document(txt_file)
        all_chunks.extend(chunks)
    
    print(f"\n{'=' * 70}")
    print("CHUNKING SUMMARY")
    print("=" * 70)
    print(f"Total chunks created: {len(all_chunks)}")
    
    # Initialize vector store
    print(f"\n{'=' * 70}")
    print("VECTOR DATABASE INDEXING")
    print("=" * 70)
    
    vector_store = VectorStore(CHROMA_DIR)
    
    # Clear existing data if re-running
    if vector_store.collection.count() > 0:
        print(f"  Clearing existing {vector_store.collection.count()} documents...")
        vector_store.client.delete_collection(vector_store.collection_name)
        vector_store.collection = vector_store.client.create_collection(
            name=vector_store.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    # Add chunks
    vector_store.add_chunks(all_chunks)
    
    # Save chunk metadata for reference
    metadata_path = BASE_DIR / "data" / "chunks_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump([c.to_dict() for c in all_chunks], f, indent=2)
    print(f"\n  Chunk metadata saved to: {metadata_path}")
    
    # Print final statistics
    stats = vector_store.get_stats()
    print(f"\n{'=' * 70}")
    print("FINAL STATISTICS")
    print("=" * 70)
    print(f"Total chunks indexed: {stats['total_chunks']}")
    
    print(f"\nChunks by company:")
    for company, count in sorted(stats['companies'].items()):
        print(f"  • {company}: {count}")
    
    print(f"\nChunks by section:")
    for section, count in sorted(stats['sections'].items(), key=lambda x: -x[1]):
        print(f"  • {section}: {count}")
    
    # Test search
    print(f"\n{'=' * 70}")
    print("VALIDATION: Test Search")
    print("=" * 70)
    
    test_queries = [
        "What are the main risk factors for the company?",
        "How does the company manage liquidity risk?",
        "What is the company's credit risk exposure?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        results = vector_store.search(query, n_results=2)
        for i, result in enumerate(results):
            print(f"  Result {i+1} (similarity: {result['similarity']:.3f}):")
            print(f"    Company: {result['metadata']['company']}")
            print(f"    Section: {result['metadata']['section']}")
            print(f"    Text: {result['text'][:150]}...")
    
    print(f"\n{'=' * 70}")
    print("✓ Day 2 Complete: Documents chunked and indexed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
