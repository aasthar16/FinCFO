"""
File upload handler for CSV and Excel files with robust date parsing.
"""

import streamlit as st
import pandas as pd
from typing import Optional, Dict, Any
import re
from datetime import datetime


def parse_date_flexible(date_str):
    """
    Parse date from various formats with error handling.
    """
    if pd.isna(date_str) or date_str is None or str(date_str).strip() == '':
        return None
    
    date_str = str(date_str).strip()
    
    # Try different date formats
    formats = [
        '%Y-%m-%d',           # 2025-12-01
        '%m/%d/%Y',           # 12/01/2025
        '%m-%d-%Y',           # 12-01-2025
        '%d/%m/%Y',           # 01/12/2025
        '%Y/%m/%d',           # 2025/12/01
        '%b %d, %Y',          # Dec 1, 2025
        '%B %d, %Y',          # December 1, 2025
        '%d-%b-%Y',           # 01-Dec-2025
        '%d-%b-%y',           # 01-Dec-25
        '%m/%d/%y',           # 12/01/25
        '%d/%m/%y',           # 01/12/25
        '%b %d %Y',           # Dec 1 2025
        '%d %b %Y',           # 1 Dec 2025
        '%Y%m%d',             # 20251201
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    
    # Try to extract date using regex if all formats fail
    date_patterns = [
        r'(\d{4})-(\d{2})-(\d{2})',  # 2025-12-01
        r'(\d{2})/(\d{2})/(\d{4})',  # 12/01/2025
        r'(\d{2})/(\d{2})/(\d{2})',  # 12/01/25
        r'(\d{2})-(\d{2})-(\d{4})',  # 12-01-2025
        r'(\d{2})/(\d{2})/(\d{4})',  # 12/01/2025
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            try:
                groups = match.groups()
                if len(groups) == 3:
                    # Try to figure out which is year/month/day
                    parts = [int(g) for g in groups]
                    if parts[0] > 1000:  # YYYY-MM-DD
                        return datetime(parts[0], parts[1], parts[2]).date()
                    elif parts[2] > 1000:  # MM/DD/YYYY
                        return datetime(parts[2], parts[0], parts[1]).date()
                    elif parts[0] < 31 and parts[1] < 31:  # DD/MM/YY or MM/DD/YY
                        # Assume MM/DD/YY
                        year = 2000 + parts[2] if parts[2] < 70 else 1900 + parts[2]
                        return datetime(year, parts[0], parts[1]).date()
            except:
                continue
    
    return None


def parse_amount_flexible(amount_str):
    """
    Parse amount from various formats.
    """
    if pd.isna(amount_str) or amount_str is None:
        return None
    
    if isinstance(amount_str, (int, float)):
        return float(amount_str)
    
    amount_str = str(amount_str).strip()
    
    # Remove currency symbols and spaces
    amount_str = re.sub(r'[^\d\-.,]', '', amount_str)
    
    # Handle negative amounts
    is_negative = amount_str.startswith('-')
    amount_str = amount_str.replace('-', '').replace(',', '')
    
    try:
        amount = float(amount_str)
        return -amount if is_negative else amount
    except ValueError:
        return None


def handle_file_upload() -> Optional[pd.DataFrame]:
    """
    Handle CSV/Excel file upload and return DataFrame with cleaned data.
    """
    st.sidebar.markdown("### 📁 Upload Data")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your transaction data for analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Get file extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # Read file based on extension
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.sidebar.error("Unsupported file format")
                return None
            
            # Show raw data preview
            with st.sidebar.expander("📊 Raw Data Preview"):
                st.dataframe(df.head())
            
            # Try to find date and amount columns
            date_col = None
            amount_col = None
            
            # Look for date columns (case insensitive)
            date_keywords = ['date', 'day', 'transaction_date', 'trandate', 'posted', 'posting_date']
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in date_keywords):
                    date_col = col
                    break
            
            # Look for amount columns
            amount_keywords = ['amount', 'transaction_amount', 'trnamt', 'value', 'net', 'total']
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in amount_keywords):
                    amount_col = col
                    break
            
            # If no date/amount columns found, ask user
            if date_col is None or amount_col is None:
                st.sidebar.warning("⚠️ Could not automatically detect columns")
                
                if date_col is None:
                    date_col = st.sidebar.selectbox("Select Date Column", df.columns)
                
                if amount_col is None:
                    amount_col = st.sidebar.selectbox("Select Amount Column", df.columns)
            
            # Parse dates
            df['parsed_date'] = df[date_col].apply(parse_date_flexible)
            
            # Parse amounts
            df['parsed_amount'] = df[amount_col].apply(parse_amount_flexible)
            
            # Drop rows with invalid dates
            invalid_dates = df['parsed_date'].isna()
            if invalid_dates.any():
                st.sidebar.warning(f"⚠️ {invalid_dates.sum()} rows have invalid dates and will be skipped")
                df = df.dropna(subset=['parsed_date'])
            
            # Drop rows with invalid amounts
            invalid_amounts = df['parsed_amount'].isna()
            if invalid_amounts.any():
                st.sidebar.warning(f"⚠️ {invalid_amounts.sum()} rows have invalid amounts and will be skipped")
                df = df.dropna(subset=['parsed_amount'])
            
            if len(df) == 0:
                st.sidebar.error("❌ No valid data rows after parsing")
                return None
            
            # Create final DataFrame
            result_df = pd.DataFrame({
                'date': df['parsed_date'],
                'amount': df['parsed_amount'],
                'description': df[date_col].astype(str) + ' - ' + df[amount_col].astype(str)
            })
            
            # Add category column if it exists
            category_keywords = ['category', 'type', 'class', 'group']
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in category_keywords):
                    result_df['category'] = df[col]
                    break
            
            # If no category, use default
            if 'category' not in result_df.columns:
                result_df['category'] = 'other'
            
            # Add one_time flag (can be customized)
            result_df['one_time'] = False
            
            # Show success and preview
            st.sidebar.success(f"✅ Loaded {len(result_df)} rows")
            
            with st.sidebar.expander("📊 Processed Data Preview"):
                st.dataframe(result_df.head())
                
                # Show date range
                st.caption(f"Date Range: {result_df['date'].min()} to {result_df['date'].max()}")
                st.caption(f"Total Amount: ${result_df['amount'].sum():,.2f}")
            
            # Store in session state
            st.session_state.uploaded_data = result_df
            
            return result_df
            
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {str(e)}")
            return None
    
    return None