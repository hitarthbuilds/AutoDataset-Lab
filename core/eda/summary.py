import polars as pl

def dataset_overview(df: pl.DataFrame) -> dict:
    return {
        "Rows": df.height(),
        "Columns": len(df.columns),
        "Column Names": df.columns,
        "Dtypes": [str(dt) for dt in df.dtypes],
        "Numeric Columns": [col for col, dt in zip(df.columns, df.dtypes) if dt in {pl.Int64, pl.Float64}],
        "Categorical Columns": [col for col, dt in zip(df.columns, df.dtypes) if dt == pl.Utf8],
    }
