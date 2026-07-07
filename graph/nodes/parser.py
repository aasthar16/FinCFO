"""
Financial Data Parser Node - LLM-based parsing of uploaded files.
Runs before supervisor to convert raw files into structured data.
"""

import logging
from typing import Dict, Any
from langchain_core.messages import AIMessage

from graph.state import GlobalState
from services.financial_parser import parse_multiple_files
from config.langsmith import traced

logger = logging.getLogger(__name__)


@traced("parser_node", tags=["parser", "financial"])
def parser_node(state: GlobalState) -> Dict[str, Any]:
    """
    Financial Data Parser Node.
    Takes raw_files from state, parses them using LLM + Pydantic,
    and stores structured transactions in state.
    """
    
    # Get raw files from state
    raw_files = state.get("raw_files", [])
    
    if not raw_files:
        logger.info("No raw files to parse")
        return {
            "parsing_status": "no_files",
            "messages": [AIMessage(content="📄 No files uploaded. Please upload a CSV or Excel file to analyze.")],
        }
    
    # Check if already parsed
    if state.get("parsing_status") == "done":
        logger.info(f"Already parsed {len(state.get('transactions_data', []))} transactions")
        return {
            "messages": [],  # Valid state key, no-op
        }
    
    logger.info(f"📄 Parsing {len(raw_files)} file(s)...")
    
    try:
        # Parse all files
        result = parse_multiple_files(raw_files)
        
        if result["success"]:
            # Store parsed data in state
            transactions = result["transactions"]
            cash_balance = result["cash_balance"]
            monthly_revenue = result["monthly_revenue"]
            
            logger.info(f"✅ Parsed {len(transactions)} transactions")
            
            # Build summary message
            msg = f"""
📊 **File(s) Parsed Successfully!**

- **Files:** {', '.join(result['parsed_files'])}
- **Total Transactions:** {len(transactions)}
- **Date Range:** {transactions[0]['date'] if transactions else 'N/A'} to {transactions[-1]['date'] if transactions else 'N/A'}
"""
            
            if cash_balance:
                msg += f"\n- **Cash Balance:** ${cash_balance:,.0f}"
            if monthly_revenue:
                msg += f"\n- **Monthly Revenue:** ${monthly_revenue:,.0f}"
            
            if result["errors"]:
                msg += f"\n\n⚠️ **Warnings:** {', '.join(result['errors'][:2])}"
            
            return {
                "transactions_data": transactions,
                "cash_balance": cash_balance or state.get("cash_balance", 0),
                "monthly_revenue": monthly_revenue or state.get("monthly_revenue", 0),
                "parsing_status": "done",
                "messages": [AIMessage(content=msg)],
            }
        else:
            error_msg = "❌ **Failed to parse files.**\n\n"
            if result.get("errors"):
                error_msg += f"Errors: {', '.join(result['errors'])}"
            else:
                error_msg += "No transactions found in the uploaded files. Please check the format."
            
            return {
                "parsing_status": "failed",
                "messages": [AIMessage(content=error_msg)],
                "error_state": "parser_failed",
            }
            
    except Exception as e:
        logger.error(f"❌ Parser node failed: {e}", exc_info=True)
        return {
            "parsing_status": "failed",
            "messages": [AIMessage(content=f"❌ Error parsing files: {str(e)}")],
            "error_state": "parser_failed",
        }


def route_after_parser(state: GlobalState) -> str:
    """
    Router after parser node.
    - If parsing succeeded or already done → go to supervisor
    - If parsing failed → stay in parser (or end with error)
    """
    status = state.get("parsing_status")
    
    if status == "done":
        return "supervisor"
    elif status == "failed":
        # User can retry or end conversation
        return "end"
    else:
        # No files or parsing not done
        return "end"