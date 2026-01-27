"""
SEC 10-K Filing Downloader and HTML to Text Processor
Downloads 10-K filings for JPM, GS, and UBS from SEC EDGAR and converts them to clean text.
"""

import os
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time

# Configuration
companies = {
    'JPM': {
        'name': 'JPMorgan Chase',
        'cik': '0000019617',
        'url': 'https://www.sec.gov/Archives/edgar/data/19617/000001961725000270/jpm-20241231.htm',
        'filing_date': '2024-12-31'
    },
    'GS': {
        'name': 'Goldman Sachs',
        'cik': '0000886982',
        'url': 'https://www.sec.gov/Archives/edgar/data/886982/000088698225000005/gs-20241231.htm',
        'filing_date': '2024-12-31'
    },
    'UBS': {
        'name': 'UBS Group',
        'cik': '0001610520',
        'url': 'https://www.sec.gov/Archives/edgar/data/1610520/000161052025000023/ubs-20241231.htm',
        'filing_date': '2024-12-31',
        'note': 'UBS files Form 20-F (foreign issuer) instead of 10-K'
    }
}

# Create directories
raw_html_dir = Path('data/raw_html')
raw_txt_dir = Path('data/raw_txt')
raw_html_dir.mkdir(parents=True, exist_ok=True)
raw_txt_dir.mkdir(parents=True, exist_ok=True)


def download_html(ticker, company_info):
    """Download the HTML 10-K filing from SEC EDGAR."""
    print(f"\n{'='*60}")
    print(f"Downloading {company_info['name']} ({ticker})")
    print(f"{'='*60}")
    
    url = company_info['url']
    filename = f"{ticker}_10K_{company_info['filing_date'].replace('-', '')}.html"
    filepath = raw_html_dir / filename
    
    # Skip if file already exists
    if filepath.exists():
        print(f"✓ File already exists: {filepath}")
        return filepath
    
    try:
        # SEC requires user-agent header
        headers = {
            'User-Agent': 'Research Project contact@example.com'
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Save original HTML
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"✓ Downloaded: {filepath}")
        print(f"  Size: {len(response.text):,} characters")
        
        # Be respectful to SEC servers
        time.sleep(0.5)
        
        return filepath
        
    except Exception as e:
        print(f"✗ Error downloading {ticker}: {str(e)}")
        return None


def html_to_clean_text(html_filepath):
    """Convert HTML 10-K to clean text, removing scripts and tables."""
    print(f"\nProcessing: {html_filepath.name}")
    
    try:
        with open(html_filepath, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'meta', 'link', 'noscript']):
            element.decompose()
        
        # Remove tables (as requested)
        for table in soup.find_all('table'):
            table.decompose()
        
        # Get text
        text = soup.get_text(separator='\n', strip=False)
        
        # Clean up text
        # Remove excessive whitespace while preserving paragraph structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Strip whitespace from line
            line = line.strip()
            # Keep non-empty lines
            if line:
                cleaned_lines.append(line)
        
        # Join with single newlines
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove excessive blank lines (more than 2 consecutive newlines)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        # Save cleaned text
        txt_filename = html_filepath.stem + '.txt'
        txt_filepath = raw_txt_dir / txt_filename
        
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        print(f"✓ Converted to text: {txt_filepath}")
        print(f"  Original HTML size: {len(html_content):,} chars")
        print(f"  Cleaned text size: {len(cleaned_text):,} chars")
        
        return txt_filepath
        
    except Exception as e:
        print(f"✗ Error processing {html_filepath.name}: {str(e)}")
        return None


def find_section(text, section_name):
    """Find and extract a specific section from the 10-K text."""
    # Common patterns for section headers in 10-Ks
    patterns = [
        rf"(?i)\bitem\s+\d+[a-z]?\.\s*{re.escape(section_name)}\b",
        rf"(?i)^{re.escape(section_name)}$",
        rf"(?i)\b{re.escape(section_name)}\b"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            start_pos = match.start()
            # Get approximately 1000 characters after the section header
            excerpt = text[start_pos:start_pos + 1000]
            return excerpt
    
    return None


def verify_readable_sections(txt_filepath):
    """Verify that key sections (Risk, Liquidity) are readable in the text file."""
    print(f"\nVerifying sections in: {txt_filepath.name}")
    
    try:
        with open(txt_filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Check for Risk Factors section
        risk_section = find_section(text, "Risk Factors")
        if risk_section:
            print("✓ Risk Factors section found:")
            print(f"  Preview: {risk_section[:200]}...")
        else:
            print("⚠ Risk Factors section not clearly identified")
        
        # Check for Liquidity section
        liquidity_section = find_section(text, "Liquidity")
        if liquidity_section:
            print("✓ Liquidity section found:")
            print(f"  Preview: {liquidity_section[:200]}...")
        else:
            print("⚠ Liquidity section not clearly identified")
        
        return True
        
    except Exception as e:
        print(f"✗ Error verifying sections: {str(e)}")
        return False


def main():
    """Main execution function."""
    print("="*60)
    print("SEC 10-K Filing Downloader and Processor")
    print("="*60)
    print(f"\nProcessing {len(companies)} companies: {', '.join(companies.keys())}")
    print(f"\nDirectory structure:")
    print(f"  Raw HTML: {raw_html_dir.absolute()}")
    print(f"  Clean Text: {raw_txt_dir.absolute()}")
    
    results = {}
    
    # Download and process each company
    for ticker, info in companies.items():
        # Download HTML
        html_path = download_html(ticker, info)
        
        if html_path:
            # Convert to clean text
            txt_path = html_to_clean_text(html_path)
            
            if txt_path:
                # Verify sections
                verify_readable_sections(txt_path)
                results[ticker] = {
                    'html': html_path,
                    'txt': txt_path,
                    'status': 'success'
                }
            else:
                results[ticker] = {'status': 'text_conversion_failed'}
        else:
            results[ticker] = {'status': 'download_failed'}
    
    # Summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    
    for ticker, result in results.items():
        status = result.get('status')
        print(f"\n{ticker} - {companies[ticker]['name']}:")
        
        if status == 'success':
            print(f"  ✓ HTML: {result['html'].name}")
            print(f"  ✓ Text: {result['txt'].name}")
        else:
            print(f"  ✗ Status: {status}")
    
    print("\n" + "="*60)
    print("Files saved in:")
    print(f"  {raw_html_dir.absolute()}")
    print(f"  {raw_txt_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()