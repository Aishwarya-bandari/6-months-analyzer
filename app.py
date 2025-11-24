import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import html
import io
import base64
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
import tempfile
import os

st.set_page_config(page_title="📈 Executive AutoInsights (Enterprise)", layout="wide")
st.title("📊 Executive AutoInsights — Enterprise")
st.write("Upload a CSV — get automated analytics, visualizations, executive summary & export options.")

# ====================================================================
# ------------------- HELPER FUNCTIONS -------------------------------
# ====================================================================
def convert_volume(val):
    if pd.isna(val):
        return None
    s = str(val).replace(",", "").strip()
    try:
        if s.lower().endswith("m"):
            return float(s[:-1]) * 1_000_000
        if s.lower().endswith("k"):
            return float(s[:-1]) * 1_000
        return float(s)
    except:
        cleaned = "".join(ch for ch in s if ch.isdigit() or ch == "." or ch == "-")
        return float(cleaned) if cleaned else None

def clean_stock_dataframe(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for col in df.columns:
        try:
            if "vol" in col.lower():
                df[col] = df[col].apply(convert_volume)
            else:
                df[col] = df[col].astype(str).str.replace(",", "").str.strip()
                df[col] = pd.to_numeric(df[col], errors="ignore")
        except:
            pass
    return df

def detect_column(df, keywords):
    for col in df.columns:
        cleaned = col.lower().replace(".", "").replace("_", "").replace(" ", "")
        for key in keywords:
            if key in cleaned:
                return col
    return None

def detect_date_column(df):
    for col in df.columns:
        if "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                return col
            except:
                continue
    return None

def filter_last_6_months(df, date_col):
    df = df.dropna(subset=[date_col]).sort_values(by=date_col)
    if df.empty:
        return df
    last_date = df[date_col].max()
    six_months_ago = last_date - pd.DateOffset(months=6)
    return df[df[date_col] >= six_months_ago].copy()

def compute_pct_change(series):
    series = series.dropna().astype(float)
    if series.size < 2:
        return None
    return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100.0

def volatility_index(series):
    try:
        s = series.dropna().astype(float)
        return float(s.std())
    except:
        return None

def momentum_rating(series):
    try:
        s = series.dropna().astype(float).values
        if s.size < 2:
            return None
        X = np.arange(len(s)).reshape(-1,1)
        model = LinearRegression().fit(X, s)
        slope = model.coef_[0]
        mean_p = s.mean()
        score = (slope / mean_p) * 1000
        score_clamped = max(min(50 + score*5, 100), 0)
        return float(score_clamped)
    except:
        return None

def risk_score(pct_change, vol_idx, momentum, corr):
    score = 30.0
    if vol_idx is not None:
        score += min(vol_idx * 3.0, 30)
    if pct_change is not None:
        if pct_change < 0:
            score += min(abs(pct_change) * 1.5, 25)
        else:
            score -= min(pct_change * 0.5, 10)
    if momentum is not None:
        score -= (momentum - 50) * 0.2
    if corr is not None:
        score += (1 - abs(corr)) * 10
    return float(max(min(score, 100), 0))

def sentiment_score(pct_change, corr, vol_idx):
    s = 0.0
    if pct_change is not None:
        s += np.tanh(pct_change/10.0) * 50
    if corr is not None:
        s += (corr - 0.5) * 20
    if vol_idx is not None:
        s -= min(vol_idx, 10) * 2
    return float(max(min(s, 100), -100) / 100)

def institutional_activity_estimate(volume_series):
    try:
        v = pd.to_numeric(volume_series, errors="coerce").dropna()
        if v.shape[0] < 10:
            return None
        recent = v.iloc[-5:].mean()
        prev = v.iloc[max(0, len(v)-60):len(v)-5].mean() if len(v) > 10 else v.mean()
        if prev == 0:
            return None
        ratio = recent / prev
        if ratio > 1.5:
            return ("High", ratio)
        elif ratio > 1.1:
            return ("Moderate", ratio)
        elif ratio < 0.7:
            return ("Low", ratio)
        else:
            return ("Normal", ratio)
    except:
        return None

def detect_anomalies_price(series):
    try:
        s = series.dropna().astype(float)
        if s.size < 5:
            return []
        mean = s.mean()
        std = s.std()
        if std == 0:
            return []
        z = (s - mean) / std
        return np.where(np.abs(z) > 3)[0].tolist()
    except:
        return []

def create_pdf_with_summary(pdf_path, summary_text, trend_bytes, heatmap_bytes):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "AutoInsights - Executive Summary")
    y -= 28

    t = c.beginText(margin, y - 10)
    t.setFont("Helvetica", 10)
    for line in summary_text.split("\n"):
        t.textLine(line[:120])
    c.drawText(t)

    y_img = 200

    if trend_bytes:
        img = ImageReader(trend_bytes)
        c.drawImage(img, margin, y_img, width - 2 * margin, 180, preserveAspectRatio=True)

    if heatmap_bytes:
        img2 = ImageReader(heatmap_bytes)
        c.drawImage(img2, margin, y_img - 190, width - 2 * margin, 180, preserveAspectRatio=True)

    c.showPage()
    c.save()

# ====================================================================
# --------------------------- UI START -------------------------------
# ====================================================================
uploaded_file = st.file_uploader("📂 Upload CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload any CSV containing financial/stock data.")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded_file)
except:
    st.error("Invalid CSV. Please re-upload.")
    st.stop()

df = clean_stock_dataframe(df_raw)

date_col = detect_date_column(df)
df_filtered = filter_last_6_months(df, date_col) if date_col else df.copy()

close_col = detect_column(df_filtered, ["close", "last", "price"])
vol_col = detect_column(df_filtered, ["vol", "volume"])
val_col = detect_column(df_filtered, ["value", "turnover"])

pct_change = compute_pct_change(df_filtered[close_col]) if close_col else None
vol_idx = volatility_index(df_filtered[close_col]) if close_col else None
momentum = momentum_rating(df_filtered[close_col]) if close_col else None

corr = None
if vol_col and val_col:
    try:
        numeric = df_filtered[[vol_col,val_col]].apply(pd.to_numeric, errors="coerce").dropna()
        corr = numeric.corr().iloc[0,1]
    except:
        pass

risk = risk_score(pct_change, vol_idx, momentum, corr)
sentiment = sentiment_score(pct_change, corr, vol_idx)
inst_activity = institutional_activity_estimate(df_filtered[vol_col]) if vol_col else None

# ====================================================================
# -------------------- EXECUTIVE CARD -------------------------------
# ====================================================================
st.subheader("📄 Executive Summary")

summary_text = f"""
Total Records: {df_filtered.shape[0]}
Price Change: {pct_change:.2f}% if available
Risk Score: {risk:.1f}
Sentiment Score: {sentiment:.2f}
"""

st.text(summary_text)

# ====================================================================
# ----------------------- TREND CHART -------------------------------
# ====================================================================
trend_png = None
if date_col and close_col:
    plotted = df_filtered.dropna(subset=[date_col, close_col])
    if plotted.shape[0] > 1:
        fig = px.line(plotted, x=date_col, y=close_col, title="Price Trend")
        st.plotly_chart(fig, use_container_width=True)

        buf = io.BytesIO()
        fig.write_image(buf, format="png")
        buf.seek(0)
        trend_png = buf

# ====================================================================
# ----------------------- HEATMAP -----------------------------------
# ====================================================================
heatmap_png = None
numeric = df_filtered.select_dtypes(include=np.number)
if numeric.shape[1] > 1:
    fig, ax = plt.subplots()
    sns.heatmap(numeric.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    heatmap_png = buf

# ====================================================================
# ------------------------- DOWNLOAD CSV -----------------------------
# ====================================================================
csv_buf = io.StringIO()
df_filtered.to_csv(csv_buf, index=False)
st.download_button("📥 Download Cleaned CSV", csv_buf.getvalue(), "cleaned.csv", "text/csv")

# ====================================================================
# ------------------------ DOWNLOAD PDF ------------------------------
# ====================================================================
if st.button("📄 Generate PDF Summary"):
    tmp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmp_dir, "summary.pdf")

    create_pdf_with_summary(pdf_path, summary_text, trend_png, heatmap_png)

    with open(pdf_path, "rb") as f:
        st.download_button("📥 Download PDF", f, "autoinsights_summary.pdf", "application/pdf")
