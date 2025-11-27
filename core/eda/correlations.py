import polars as pl
import matplotlib.pyplot as plt

def compute_correlations(df: pl.DataFrame):
    numeric_df = df.select([
        col for col, dt in zip(df.columns, df.dtypes)
        if pl.datatypes.is_numeric_dtype(dt)
    ])

    if numeric_df.width < 2:
        return None, None

    corr_df = numeric_df.to_pandas().corr()

    fig, ax = plt.subplots(figsize=(6, 4))
    cax = ax.matshow(corr_df, cmap="coolwarm")
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=90)
    plt.yticks(range(len(corr_df.columns)), corr_df.columns)
    fig.colorbar(cax)

    return corr_df, fig
