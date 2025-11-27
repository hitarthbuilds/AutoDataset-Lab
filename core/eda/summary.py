import polars as pl

def dataset_overview(df: pl.DataFrame):
    return {
        "Rows": df.height,
        "Columns": df.width,
        "Column Names": df.columns,
        "Numeric Columns": [
            col for col, dt in zip(df.columns, df.dtypes)
            if pl.datatypes.is_numeric_dtype(dt)
        ],
        "Categorical Columns": [
            col for col, dt in zip(df.columns, df.dtypes)
            if dt == pl.Utf8
        ],
    }
