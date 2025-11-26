import polars as pl

def missing_summary(df: pl.DataFrame) -> pl.DataFrame:
    total_rows = df.height()

    data = {
        "Column": df.columns,
        "Missing Count": [df[col].null_count() for col in df.columns],
        "Missing %": [(df[col].null_count() / total_rows) * 100 for col in df.columns]
    }

    return pl.DataFrame(data)
