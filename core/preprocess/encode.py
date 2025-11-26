import polars as pl

def encode_features(df: pl.DataFrame) -> pl.DataFrame:
    cat_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt == pl.Utf8]
    if not cat_cols:
        return df

    return df.to_dummies(columns=cat_cols)
