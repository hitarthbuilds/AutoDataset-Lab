import polars as pl
import matplotlib.pyplot as plt

def plot_numeric_distributions(df: pl.DataFrame):
    numeric_cols = [
        col for col, dt in zip(df.columns, df.dtypes)
        if dt in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                  pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                  pl.Float32, pl.Float64)
    ]

    if not numeric_cols:
        return None

    fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(8, 4 * len(numeric_cols)))

    if len(numeric_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, numeric_cols):
        ax.hist(df[col].to_list(), bins=20)
        ax.set_title(col)

    plt.tight_layout()
    return fig
