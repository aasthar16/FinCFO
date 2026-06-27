"""
Validation utilities for financial data.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime


def validate_transactions(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate transaction DataFrame structure.
    
    Returns:
        Dict with validation results and errors if any.
    """
    errors = []
    warnings = []
    
    # Required columns
    required = ['date', 'amount', 'type', 'category']
    missing = [col for col in required if col not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}
    
    # Date validation
    if 'date' in df.columns:
        try:
            pd.to_datetime(df['date'])
        except Exception as e:
            errors.append(f"Invalid date format: {e}")
    
    # Amount validation
    if 'amount' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['amount']):
            errors.append("Amount column must be numeric")
        
        # Check for unrealistic amounts
        outliers = df[df['amount'].abs() > df['amount'].abs().quantile(0.99)]
        if len(outliers) > 0:
            warnings.append(f"Found {len(outliers)} potential outliers in amount")
    
    # Category validation
    valid_categories = ['salary', 'rent', 'software', 'marketing', 'other']
    if 'category' in df.columns:
        invalid = df[~df['category'].isin(valid_categories)]
        if len(invalid) > 0:
            warnings.append(f"Found {len(invalid)} invalid categories")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "n_rows": len(df),
        "date_range": (
            df['date'].min() if 'date' in df.columns else None,
            df['date'].max() if 'date' in df.columns else None,
        ),
    }


def validate_scenario_overrides(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate scenario override parameters.
    """
    errors = []
    warnings = []
    
    valid_params = ['headcount_change', 'avg_salary', 'revenue_change', 
                    'one_time_expenses', 'ramp_months', 'pricing_change']
    
    invalid = [k for k in overrides.keys() if k not in valid_params]
    if invalid:
        warnings.append(f"Unknown parameters: {invalid}")
    
    # Validate numeric values
    for key, value in overrides.items():
        if not isinstance(value, (int, float)):
            errors.append(f"Parameter '{key}' must be numeric")
    
    # Validate ramp months
    if 'ramp_months' in overrides:
        if overrides['ramp_months'] < 0 or overrides['ramp_months'] > 12:
            warnings.append(f"Ramp months should be between 0 and 12")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


def validate_forecast_data(df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
    """
    Validate data for forecasting.
    """
    errors = []
    warnings = []
    
    # Check data sufficiency
    if len(df) < 3:
        errors.append(f"Insufficient data: {len(df)} points, need at least 3")
    
    # Check for missing values
    missing = df[target_column].isna().sum()
    if missing > 0:
        warnings.append(f"Found {missing} missing values in {target_column}")
    
    # Check for constant series
    if len(df) > 0 and df[target_column].std() == 0:
        warnings.append(f"Target column '{target_column}' has no variation")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "n_points": len(df),
        "missing_count": missing,
    }