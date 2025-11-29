import pandas as pd
print("DEBUG: analyze.py loaded")

def analyze_dataframe(df: pd.DataFrame):
    """
    Full EDA analysis pack used by UI and report generator.
    """

    overview = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "columns_list": df.columns.tolist(),
    }

    missing = df.isna().mean().round(4).to_dict()

    dtypes = {col: str(df[col].dtype) for col in df.columns}

    column_types = {
        "numeric": df.select_dtypes(include=["int64", "float64"]).columns.tolist(),
        "categorical": df.select_dtypes(include=["object", "category"]).columns.tolist(),
        "boolean": df.select_dtypes(include=["bool"]).columns.tolist(),
        "datetime": df.select_dtypes(include=["datetime64"]).columns.tolist(),
    }

    numeric_summary = {}
    if column_types["numeric"]:
        numeric_summary = df[column_types["numeric"]].describe().to_dict()

    categorical_summary = {
        col: df[col].value_counts().to_dict()
        for col in column_types["categorical"]
    }

    return {
        "overview": overview,
        "missing": missing,
        "dtypes": dtypes,
        "column_types": column_types,
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
    }
