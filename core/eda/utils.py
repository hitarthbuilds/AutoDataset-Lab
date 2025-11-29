import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

def plot_correlation_heatmap(df, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    plt.figure(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    return plt.gcf()
