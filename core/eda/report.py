import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import pandas as pd
print("DEBUG: report.py loaded")

def generate_visual_eda_report(df: pd.DataFrame, path="eda_report.pdf"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # COVER PAGE
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 10, "AutoDataset-Lab - EDA Report", ln=True, align="C")

    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, "Automatically generated Exploratory Data Analysis report.")
    pdf.ln(10)

    # OVERVIEW
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Dataset Overview", ln=True)
    pdf.set_font("Arial", "", 12)

    pdf.cell(0, 8, f"Rows: {df.shape[0]}", ln=True)
    pdf.cell(0, 8, f"Columns: {df.shape[1]}", ln=True)

    pdf.ln(4)

    # NUMERIC DISTRIBUTIONS
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        img_path = f"temp_{col}.png"
        plt.savefig(img_path)
        plt.close()

        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"Distribution: {col}", ln=True)
        pdf.image(img_path, x=10, w=180)

    # CORRELATION MATRIX
    if len(numeric_cols) > 1:
        plt.figure(figsize=(6, 5))
        sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig("corr.png")
        plt.close()

        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Correlation Heatmap", ln=True)
        pdf.image("corr.png", x=10, w=180)

    pdf.output(path)
    return path
