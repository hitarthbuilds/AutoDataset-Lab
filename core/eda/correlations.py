import pandas as pd
import numpy as np

def compute_correlation(pdf: pd.DataFrame):
    numeric_df = pdf.select_dtypes(include=["float64", "int64"])
    
    if numeric_df.shape[1] == 0:
        return None

    return numeric_df.corr()
