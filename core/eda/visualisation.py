import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_distribution(pdf: pd.DataFrame, column: str):
    try:
        plt.figure(figsize=(8, 4))
        sns.histplot(pdf[column], kde=True)
        plt.title(f"Distribution of {column}")
        plt.tight_layout()
        return plt
    except:
        return None


def plot_heatmap(corr: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    return plt
