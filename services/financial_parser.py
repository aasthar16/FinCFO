"""
Financial Data Parser - LLM-based parsing of unstructured financial data.
Converts CSV, Excel, text, or bank statements into validated Pydantic models.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, date
import pandas as pd

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from services.schemas import ParsedFinancialData, TransactionRecord
from settings import settings

logger = logging.getLogger(__name__)

# Initialize Groq LLM with structured output
llm = ChatGroq(
    model=settings.groq_model,
    api_key=settings.groq_api_key,
    temperature=0.0,
    max_tokens=8192,
)

# Create structured LLM for parsing
parser_llm = llm.with_structured_output(ParsedFinancialData)

def parse_financial_file(
    file_content: Union[str, bytes, pd.DataFrame],
    filename: str,
    file_type: str = None
) -> Dict[str, Any]:
    """
    Parse a single financial file into structured data.
    """
    
    content_text = _convert_to_text(file_content)
    
    if not content_text or len(content_text) < 10:
        return {
            "success": False,
            "error": f"File '{filename}' appears empty or unreadable",
            "data": None
        }
    
    # Build prompt for LLM
    system_prompt = f"""
You are a financial data parser. Extract structured financial data from the provided input.

**FILENAME:** {filename}
**FILE TYPE:** {file_type or 'unknown'}

**INPUT DATA:**
{content_text[:8000]}  { '...(truncated)' if len(content_text) > 8000 else '' }

**RULES:**
1. Identify all transactions (date, amount, description)
2. Amounts should be NEGATIVE for outflows (expenses), POSITIVE for inflows (revenue)
3. For category, use any of these or create your own: salary, rent, software, marketing, revenue, payroll, legal, office, travel, saas, other
4. If a date is missing, try to infer from context
5. Look for cash balance and monthly revenue if mentioned
6. Mark is_one_time as true for irregular/large expenses

**OUTPUT:** Return the extracted data matching the TransactionRecord structure.
"""
    
    try:
        # Call LLM with structured output
        result: ParsedFinancialData = parser_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Parse this financial data and return structured output."),
        ])
        
        # Validate result has at least some transactions
        if not result.transactions:
            return {
                "success": False,
                "error": f"No transactions found in '{filename}'. Please check the file format.",
                "data": None
            }
        
        logger.info(f"✅ Parsed {len(result.transactions)} transactions from '{filename}'")
        return {
            "success": True,
            "data": result.model_dump(),
            "error": None
        }
        
    except Exception as e:
        logger.error(f"❌ Parser failed for '{filename}': {e}")
        return {
            "success": False,
            "error": f"Parser error: {str(e)}",
            "data": None
        }

def parse_multiple_files(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse multiple financial files and merge results.
    
    Args:
        files: List of dicts with keys: 'content', 'filename', 'type'
    
    Returns:
        Dict with merged transactions, cash_balance, monthly_revenue
    """
    all_transactions = []
    parsed_files = []
    errors = []
    
    cash_balance = None
    monthly_revenue = None
    currency = "USD"
    
    for file_info in files:
        content = file_info.get('content')
        filename = file_info.get('filename', 'unknown')
        file_type = file_info.get('type')
        
        result = parse_financial_file(content, filename, file_type)
        
        if result["success"]:
            data = result["data"]
            
            # Extract transactions
            if data.get('transactions'):
                all_transactions.extend(data['transactions'])
            
            # Track cash_balance (prefer later files)
            if data.get('cash_balance') is not None:
                cash_balance = data['cash_balance']
            
            # Track monthly_revenue
            if data.get('monthly_revenue') is not None:
                monthly_revenue = data['monthly_revenue']
            
            # Track currency
            if data.get('currency'):
                currency = data['currency']
            
            parsed_files.append(filename)
        else:
            errors.append(result["error"])
    
    # Sort transactions by date
    if all_transactions:
        all_transactions = sorted(all_transactions, key=lambda x: x['date'] if x.get('date') else date.min)
    
    return {
        "success": len(all_transactions) > 0,
        "transactions": all_transactions,
        "cash_balance": cash_balance,
        "monthly_revenue": monthly_revenue,
        "currency": currency,
        "parsed_files": parsed_files,
        "errors": errors,
        "total_transactions": len(all_transactions),
    }


def _convert_to_text(content: Union[str, bytes, pd.DataFrame]) -> str:
    """Convert various input formats to text for LLM."""
    if isinstance(content, pd.DataFrame):
        # Convert DataFrame to CSV string (first 100 rows)
        return content.head(100).to_csv(index=False)
    elif isinstance(content, bytes):
        # Try to decode as string
        try:
            return content.decode('utf-8')
        except UnicodeDecodeError:
            return str(content)
    elif isinstance(content, str):
        return content
    elif isinstance(content, list) and content and isinstance(content[0], dict):
        return json.dumps(content[:100], indent=2)
    else:
        return str(content)