import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("DEBUG: visualize.py loaded")

def plot_numeric_distribution(df, col):
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    return plt.gcf()


def plot_categorical_distribution(df, col):
    plt.figure(figsize=(6, 4))
    df[col].value_counts().plot(kind="bar")
    plt.title(f"Frequency of {col}")
    plt.tight_layout()
    return plt.gcf()


def plot_correlation_heatmap(df, cols=None):
    """
    df: full dataframe
    cols: list of numeric columns OR None (auto-detect)
    """

    if cols is None:
        cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(cols) < 2:
        raise ValueError("Need at least 2 numeric columns for correlation heatmap.")

    plt.figure(figsize=(8, 6))
    sns.heatmap(df[cols].corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.tight_layout()

    return plt.gcf()
