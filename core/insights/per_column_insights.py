# core/insights/per_column_insights.py
import streamlit as st
import pandas as pd
import polars as pl
from core.utils.sessions import get_df
from core.eda.analyze import analyze_dataframe
from core.eda.visualize import plotly_histogram, plotly_bar, fig_to_datauri, plotly_heatmap
from core.eda.recommendations import generate_recommendations
from core.eda.feature_importance import get_tree_importance

def app():
    st.header("Per-column Insights")

    df_polars = get_df()
    if df_polars is None:
        st.warning("Upload dataset first.")
        return
    df = df_polars.to_pandas()

    analysis = analyze_dataframe(df, run_missing=True, run_anomaly=True, run_quality=True, run_drift=False)
    recs = generate_recommendations(analysis)

    cols = list(df.columns)
    col = st.selectbox("Select column", cols)

    st.subheader(f"Column: {col}")
    st.write("Type hint:", "numeric" if col in analysis["analyze"]["column_types"].get("numeric",[]) else "categorical")

    st.subheader("Preview & Stats")
    st.write(df[[col]].describe(include="all"))

    st.subheader("Visuals")
    if col in analysis["analyze"]["column_types"].get("numeric",[]):
        fig = plotly_histogram(df[col], title=f"Distribution — {col}")
        if fig: st.plotly_chart(fig, use_container_width=True)
    else:
        vc = df[col].value_counts().head(50)
        fig = plotly_bar(vc, title=f"Top categories — {col}")
        if fig: st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recommendations (auto)")
    st.json(recs["by_column"].get(col, {}))

    st.subheader("Feature importance (if target uploaded)")
    if "target" in st.session_state:
        target = st.session_state["target"]
        try:
            X = df.drop(columns=[target])
            y = df[target]
            imps = get_tree_importance(X.select_dtypes(include=["number"]).fillna(0), y)
            # show top 10
            sorted_imps = sorted(imps.items(), key=lambda x: x[1], reverse=True)[:10]
            st.table(pd.DataFrame(sorted_imps, columns=["feature","importance"]))
        except Exception as e:
            st.warning("Feature importance failed: "+str(e))

    # Export option
    if st.button("Download column as CSV"):
        buf = df[[col]].to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=buf, file_name=f"{col}.csv", mime="text/csv")
