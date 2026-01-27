#!/usr/bin/env python3
"""
SEC EDGAR 10-K RAG - Interactive Query Interface
Ask questions about JPMorgan, Goldman Sachs, and UBS 10-K filings.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from D_Generation import RAGEngine, RAGResponse

# ============================================================================
# COLORS FOR TERMINAL OUTPUT
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_banner():
    """Print welcome banner."""
    banner = f"""
{Colors.CYAN}{Colors.BOLD}
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║         📊 SEC EDGAR 10-K RAG System                                 ║
║         Ask questions about JPMorgan, Goldman Sachs & UBS            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
{Colors.END}
{Colors.YELLOW}Available companies:{Colors.END} JPMorgan Chase (JPM), Goldman Sachs (GS), UBS Group (UBS)
{Colors.YELLOW}Filing type:{Colors.END} 10-K Annual Reports (FY 2024)

{Colors.GREEN}Commands:{Colors.END}
  • Type your question and press Enter
  • Type {Colors.BOLD}'filter jpm'{Colors.END} to only search JPMorgan documents
  • Type {Colors.BOLD}'filter gs'{Colors.END} to only search Goldman Sachs documents  
  • Type {Colors.BOLD}'filter ubs'{Colors.END} to only search UBS documents
  • Type {Colors.BOLD}'filter clear'{Colors.END} to search all companies
  • Type {Colors.BOLD}'examples'{Colors.END} to see sample questions
  • Type {Colors.BOLD}'quit'{Colors.END} or {Colors.BOLD}'exit'{Colors.END} to exit

"""
    print(banner)


def print_examples():
    """Print example questions."""
    examples = f"""
{Colors.CYAN}{Colors.BOLD}📝 Example Questions:{Colors.END}

{Colors.GREEN}Risk Factors:{Colors.END}
  • What are JPMorgan's main risk factors?
  • What credit risks does Goldman Sachs face?
  • How does UBS manage operational risk?

{Colors.GREEN}Liquidity:{Colors.END}
  • How does JPMorgan manage liquidity risk?
  • What are UBS's Basel III liquidity requirements?
  • What funding sources do the banks use?

{Colors.GREEN}Capital & Regulation:{Colors.END}
  • What are the regulatory capital requirements?
  • How do these banks meet Basel III standards?
  • What are the G-SIB buffer requirements?

{Colors.GREEN}Business:{Colors.END}
  • What are JPMorgan's main business segments?
  • How does Goldman Sachs generate revenue?
  • What geographic markets does UBS operate in?

{Colors.GREEN}Comparisons:{Colors.END}
  • Compare credit risk management across the banks
  • What regulatory risks do all three banks face?

"""
    print(examples)


def print_response(response: RAGResponse):
    """Pretty print a RAG response."""
    # Confidence color
    conf_colors = {
        "high": Colors.GREEN,
        "medium": Colors.YELLOW,
        "low": Colors.RED,
        "not_found": Colors.RED
    }
    conf_color = conf_colors.get(response.confidence.value, Colors.END)
    
    print(f"\n{Colors.CYAN}{'─'*70}{Colors.END}")
    print(f"{Colors.BOLD}💬 Answer{Colors.END} {conf_color}[Confidence: {response.confidence.value}]{Colors.END}")
    print(f"{Colors.CYAN}{'─'*70}{Colors.END}\n")
    
    # Print answer with wrapping
    import textwrap
    wrapped = textwrap.fill(response.answer, width=70)
    print(wrapped)
    
    # Print citations
    if response.citations:
        print(f"\n{Colors.YELLOW}📚 Sources:{Colors.END}")
        seen = set()
        for c in response.citations:
            key = f"{c.company} - {c.section}"
            if key not in seen:
                print(f"   • {key}")
                seen.add(key)
    
    print(f"\n{Colors.CYAN}{'─'*70}{Colors.END}")
    print(f"{Colors.GREEN}Retrieved {response.num_chunks_retrieved} chunks from: {', '.join(response.companies_in_context)}{Colors.END}")
    print(f"{Colors.CYAN}{'─'*70}{Colors.END}\n")


def interactive_mode():
    """Run interactive query mode."""
    print_banner()
    
    # Initialize RAG engine
    print(f"{Colors.YELLOW}Loading RAG system...{Colors.END}")
    try:
        rag = RAGEngine()
        print(f"{Colors.GREEN}✓ System ready!{Colors.END}\n")
    except Exception as e:
        print(f"{Colors.RED}Error initializing RAG engine: {e}{Colors.END}")
        print("Make sure you have run B_Chunking_Indexing.py first!")
        return
    
    # Track active filter
    company_filter = None
    
    while True:
        # Show prompt with filter status
        if company_filter:
            prompt = f"{Colors.BOLD}[Filter: {company_filter}] Ask a question: {Colors.END}"
        else:
            prompt = f"{Colors.BOLD}Ask a question: {Colors.END}"
        
        try:
            user_input = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{Colors.CYAN}Goodbye!{Colors.END}")
            break
        
        if not user_input:
            continue
        
        # Handle commands
        lower_input = user_input.lower()
        
        if lower_input in ['quit', 'exit', 'q']:
            print(f"\n{Colors.CYAN}Goodbye!{Colors.END}")
            break
        
        if lower_input == 'examples':
            print_examples()
            continue
        
        if lower_input.startswith('filter '):
            filter_value = lower_input.replace('filter ', '').strip().upper()
            if filter_value == 'CLEAR':
                company_filter = None
                print(f"{Colors.GREEN}✓ Filter cleared. Searching all companies.{Colors.END}\n")
            elif filter_value in ['JPM', 'GS', 'UBS']:
                company_filter = filter_value
                print(f"{Colors.GREEN}✓ Now filtering by: {filter_value}{Colors.END}\n")
            else:
                print(f"{Colors.RED}Invalid filter. Use: jpm, gs, ubs, or clear{Colors.END}\n")
            continue
        
        if lower_input == 'help':
            print_banner()
            continue
        
        # Process query
        print(f"\n{Colors.YELLOW}Searching documents...{Colors.END}")
        
        try:
            response = rag.query(
                question=user_input,
                k=5,
                company=company_filter
            )
            print_response(response)
        except Exception as e:
            print(f"{Colors.RED}Error processing query: {e}{Colors.END}\n")


def single_query(question: str, company: str = None):
    """Run a single query (for scripting/testing)."""
    rag = RAGEngine()
    response = rag.query(question=question, k=5, company=company)
    print_response(response)
    return response


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SEC EDGAR 10-K RAG Query Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query.py                           # Interactive mode
  python query.py -q "What are JPM's risks?"  # Single query
  python query.py -q "Credit risk" -c GS    # Query with company filter
        """
    )
    
    parser.add_argument(
        '-q', '--query',
        type=str,
        help='Single query to run (non-interactive mode)'
    )
    
    parser.add_argument(
        '-c', '--company',
        type=str,
        choices=['JPM', 'GS', 'UBS', 'jpm', 'gs', 'ubs'],
        help='Filter by company'
    )
    
    args = parser.parse_args()
    
    if args.query:
        # Single query mode
        company = args.company.upper() if args.company else None
        single_query(args.query, company)
    else:
        # Interactive mode
        interactive_mode()
