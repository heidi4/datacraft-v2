import pandas as pd
import re

DATE_REGEX = re.compile(r'^\d{1,4}[-/.\s]\d{1,2}[-/.\s]\d{1,4}$')

def is_likely_date_column(series: pd.Series) -> bool:

    if not pd.api.types.is_object_dtype(series.dtype):
        return False
    
    sample = series.dropna().head(20)
    if sample.empty:
        return False

    match_count = sample.astype(str).str.match(DATE_REGEX).sum()
    return (match_count / len(sample)) > 0.75

def detect_data_type(series: pd.Series) -> str:
    series_cleaned = series.dropna()

    if series_cleaned.empty:
        return 'empty'

    sample = series_cleaned.head(1000)
    numeric_sample = pd.to_numeric(sample, errors='coerce')
    if numeric_sample.notna().sum() / len(sample) > 0.90:
        try:
            if (numeric_sample.dropna() == numeric_sample.dropna().astype(int)).all():
                if numeric_sample.nunique() / len(numeric_sample.dropna()) > 0.95:
                    return 'identifier'
                return 'integer'
            else:
                return 'float'
        except (ValueError, TypeError):
             return 'float'

    if is_likely_date_column(series_cleaned):
        try:
            pd.to_datetime(sample, errors='raise', format='mixed')
            return 'date'
        except (ValueError, TypeError, AttributeError):
            pass

    unique_count = series_cleaned.nunique()
    total_count = len(series_cleaned)
    unique_ratio = unique_count / total_count if total_count > 0 else 0

    if unique_ratio > 0.95:
        return 'identifier'
    
    if unique_ratio < 0.5 or unique_count < 50:
        return 'categorical'
        
    return 'text'