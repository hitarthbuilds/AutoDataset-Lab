import polars as pl

def impute_missing_values(df: pl.DataFrame) -> pl.DataFrame:
    return df.fill_null(strategy="mean")
