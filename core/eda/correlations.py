import polars as pl

def compute_correlations(df: pl.DataFrame) -> pl.DataFrame:
    num_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt in (pl.Int64, pl.Float64)]

    if len(num_cols) < 2:
        return pl.DataFrame({"message": ["Not enough numeric columns for correlation"]})

    pdf = df.select(num_cols).to_pandas()
    corr = pdf.corr()

    return pl.DataFrame(corr.reset_index())
