import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Monthly Analyzer", layout="wide")

st.title("📊 6-Month Data Analyzer")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("File successfully uploaded!")
    
    st.subheader("📄 Raw Data Preview")
    st.dataframe(df.head())

    # ---------- Monthly Trend Plot ----------
    if "Month" in df.columns and "Value" in df.columns:

        st.subheader("📈 Month vs Value Trend")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Month"],
            y=df["Value"],
            mode='lines+markers'
        ))
        fig.update_layout(title="Monthly Trend")

        st.plotly_chart(fig, use_container_width=True)

        # Download figure as PNG (no kaleido needed)
        buf = BytesIO()
        fig.write_html(buf)
        st.download_button(
            "⬇ Download Chart (HTML)",
            buf.getvalue(),
            file_name="trend_chart.html"
        )

    # ---------- Heatmap (Matplotlib) ----------
    st.subheader("🔥 Value Distribution Heatmap")

    plt.figure(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True)
    st.pyplot(plt)

    buf2 = BytesIO()
    plt.savefig(buf2, format="png")
    st.download_button(
        "⬇ Download Heatmap (PNG)",
        buf2.getvalue(),
        file_name="heatmap.png"
    )

else:
    st.info("Please upload a CSV file to begin.")
