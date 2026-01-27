#!/usr/bin/env python3
"""
Startup script for Hugging Face Spaces deployment.
Downloads 10-K filings and builds the index if not present.
"""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"

def setup_data():
    """Download and index data if not present."""
    
    # Check if ChromaDB exists
    if (CHROMA_DIR / "chroma.sqlite3").exists():
        print("✓ ChromaDB already exists, skipping setup")
        return True
    
    print("🔄 Setting up data (first run)...")
    
    # Create directories
    (DATA_DIR / "raw_html").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw_txt").mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download 10-K files
    print("📥 Downloading 10-K filings...")
    try:
        from A_SEC_EDGAR import main as download_main
        download_main()
    except Exception as e:
        print(f"⚠️ Download failed: {e}")
        # Continue anyway - might have cached files
    
    # Build index
    print("🔨 Building vector index...")
    try:
        from B_Chunking_Indexing import main as index_main
        index_main()
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        return False
    
    print("✅ Setup complete!")
    return True


if __name__ == "__main__":
    success = setup_data()
    if not success:
        print("Setup failed!")
        sys.exit(1)
    
    # Start the API server
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
