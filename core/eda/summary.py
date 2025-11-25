import polars as pl

def dataset_overview(df: pl.DataFrame):
    info = {
        "Rows": df.height(),
        "Columns": df.width(),
        "Column Names": df.columns,
        "Dtypes": {col: str(df[col].dtype) for col in df.columns},
    }
    return info
