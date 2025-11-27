import polars as pl

def missing_summary(df: pl.DataFrame):
    total_rows = df.height
    result = []

    for col in df.columns:
        missing = df[col].null_count()
        result.append({
            "Column": col,
            "Missing": missing,
            "Missing %": round((missing / total_rows) * 100, 2)
        })

    return pl.DataFrame(result)
