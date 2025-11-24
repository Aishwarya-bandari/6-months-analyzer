import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import base64

st.title("📊 6 Months Analyzer")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### 🔹 Data Preview")
    st.dataframe(df)

    # Numeric Summary
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:
        st.write("### 📈 Trend Line (First Numeric Column)")
        x = np.arange(len(df))
        y = df[numeric_cols[0]]

        fig = px.line(x=x, y=y, labels={'x': 'Record', 'y': numeric_cols[0]})
        st.plotly_chart(fig)

        # Export chart for PDF
        buf = BytesIO()
        try:
            fig.write_image(buf, format="png")
        except Exception:
            buf = None

        # Heatmap
        heat_fig = px.imshow(df[numeric_cols].corr(), text_auto=True)
        st.write("### 🔥 Correlation Heatmap")
        st.plotly_chart(heat_fig)

    # Generate PDF
    if st.button("Download PDF Report"):
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        flow = []

        flow.append(Paragraph("6 Month Report", styles["Title"]))
        flow.append(Spacer(1, 12))

        # Table of numeric summary
        if len(numeric_cols) > 0:
            summary = df[numeric_cols].describe().reset_index()
            data = [summary.columns.tolist()] + summary.values.tolist()

            tbl = Table(data)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.black),
            ]))
            flow.append(tbl)

        doc.build(flow)

        pdf_buffer.seek(0)
        b64 = base64.b64encode(pdf_buffer.read()).decode()

        st.download_button(
            "Download PDF",
            data=b64,
            file_name="report.pdf",
            mime="application/pdf",
        )

