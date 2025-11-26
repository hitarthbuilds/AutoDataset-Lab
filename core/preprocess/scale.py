import polars as pl

def scale_features(df: pl.DataFrame) -> pl.DataFrame:
    num_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt in (pl.Int64, pl.Float64)]

    if not num_cols:
        return df

    for col in num_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        df = df.with_columns(((pl.col(col) - col_min) / (col_max - col_min)).alias(col))

    return df
