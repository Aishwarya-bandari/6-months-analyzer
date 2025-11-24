# app.py (Enterprise: PDF export, risk, patterns, anomaly detection)
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

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="📈 Executive AutoInsights (Enterprise)", layout="wide")
st.title("📊 Executive AutoInsights — Enterprise (6-Month Intelligence)")
st.write("Upload a CSV and get a dynamic executive summary, analytics, anomaly detection, pattern recognition, and export options.")

# -------------------------
# HELPERS (parsing, cleaning)
# -------------------------
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
        cleaned = "".join(ch for ch in s if (ch.isdigit() or ch == "." or ch == "-"))
        try:
            return float(cleaned) if cleaned else None
        except:
            return None

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
        except Exception:
            df[col] = df[col]
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
            except Exception:
                continue
    return None

def filter_last_6_months(df, date_col):
    df = df.dropna(subset=[date_col]).sort_values(by=date_col)
    if df.empty:
        return df
    last_date = df[date_col].max()
    six_months_ago = last_date - pd.DateOffset(months=6)
    return df[df[date_col] >= six_months_ago].copy()

# -------------------------
# ANALYTICS HELPERS (scores, patterns, anomalies)
# -------------------------
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
    # slope of linear regression normalized
    try:
        s = series.dropna().astype(float).values
        if s.size < 2:
            return None
        X = np.arange(len(s)).reshape(-1,1)
        model = LinearRegression().fit(X, s)
        slope = model.coef_[0]
        # normalize slope by mean price
        mean_p = s.mean()
        score = (slope / mean_p) * 1000  # scaled
        # map to 0-100
        score_clamped = max(min(50 + score*5, 100), 0)
        return float(score_clamped)
    except:
        return None

def risk_score(pct_change, vol_idx, momentum, corr):
    # combine indicators into a 0-100 risk (higher = riskier)
    # rough heuristic:
    # - high volatility increases risk
    # - large negative pct_change increases risk
    # - negative momentum increases risk
    # - weak correlation slightly increases risk
    score = 30.0
    if vol_idx is not None:
        score += min(vol_idx * 3.0, 30)
    if pct_change is not None:
        if pct_change < 0:
            score += min(abs(pct_change) * 1.5, 25)
        else:
            score -= min(pct_change * 0.5, 10)
    if momentum is not None:
        score -= (momentum - 50) * 0.2  # higher momentum reduces risk
    if corr is not None:
        score += (1 - abs(corr)) * 10  # weaker correlation -> more risk
    return float(max(min(score, 100), 0))

def sentiment_score(pct_change, corr, vol_idx):
    # combine to produce -100..100 sentiment, map to -1..1
    s = 0.0
    if pct_change is not None:
        s += np.tanh(pct_change/10.0) * 50
    if corr is not None:
        s += (corr - 0.5) * 20
    if vol_idx is not None:
        s -= min(vol_idx, 10) * 2
    # clamp -100..100
    s = max(min(s, 100), -100)
    # scale to -1..1
    return float(s / 100.0)

def institutional_activity_estimate(volume_series):
    # compare recent 5-day average to previous 60-day average
    try:
        v = pd.to_numeric(volume_series, errors="coerce").dropna()
        if v.shape[0] < 10:
            return None
        recent = v.iloc[-5:].mean()
        prev = v.iloc[max(0, len(v)-60):len(v)-5].mean() if len(v) > 10 else v.mean()
        if prev == 0 or np.isnan(prev):
            return None
        ratio = recent / prev
        # classify
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
    # z-score anomaly detection
    try:
        s = series.dropna().astype(float)
        if s.size < 5:
            return []
        mean = s.mean()
        std = s.std()
        if std == 0:
            return []
        z = (s - mean) / std
        anomalies_idx = np.where(np.abs(z) > 3)[0]
        return anomalies_idx.tolist()
    except:
        return []

def detect_patterns(df, date_col, close_col):
    patterns = []
    try:
        s = df.dropna(subset=[close_col])
        prices = s[close_col].astype(float)
        if len(prices) >= 200:
            ma50 = prices.rolling(window=50).mean()
            ma200 = prices.rolling(window=200).mean()
            # last crossover
            if ma50.iloc[-1] > ma200.iloc[-1] and ma50.iloc[-2] <= ma200.iloc[-2]:
                patterns.append("Golden Cross (50MA crossed above 200MA)")
            if ma50.iloc[-1] < ma200.iloc[-1] and ma50.iloc[-2] >= ma200.iloc[-2]:
                patterns.append("Death Cross (50MA crossed below 200MA)")
        # simple Doji detection (last candle small body)
        if close_col in df.columns:
            last_vals = df[close_col].dropna().astype(float).values
            if len(last_vals) >= 3:
                body = abs(last_vals[-1] - last_vals[-2])
                range_ = max(last_vals[-2], last_vals[-1]) - min(last_vals[-2], last_vals[-1])
                if range_ > 0 and (body / range_) < 0.1:
                    patterns.append("Doji-like candle (small body)")
    except:
        pass
    return patterns

# -------------------------
# PDF & download helpers
# -------------------------
def fig_to_image_bytes(fig):
    """Save plotly figure to PNG bytes"""
    buf = io.BytesIO()
    fig.write_image(buf, format="png", scale=2)
    buf.seek(0)
    return buf

def save_matplotlib_fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf

def create_pdf_with_summary(pdf_path, summary_html, trend_fig_bytes, heatmap_bytes):
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "AutoInsights - Executive Summary")
    y -= 28

    # summary HTML -> plain text fallback (strip tags)
    from re import sub
    summary_text = sub('<[^<]+?>', '', summary_html)
    # draw text wrapping
    text_object = c.beginText(margin, y - 10)
    text_object.setFont("Helvetica", 10)
    for line in summary_text.splitlines():
        # wrap lines to page width
        for chunk in [line[i:i+90] for i in range(0, len(line), 90)]:
            text_object.textLine(chunk)
    c.drawText(text_object)

    # add images below
    y_img = 200
    try:
        if trend_fig_bytes:
            img = ImageReader(trend_fig_bytes)
            c.drawImage(img, margin, y_img, width=width - 2*margin, height=180, preserveAspectRatio=True, anchor='c')
            y_img -= 190
        if heatmap_bytes:
            img2 = ImageReader(heatmap_bytes)
            c.drawImage(img2, margin, y_img, width=width - 2*margin, height=180, preserveAspectRatio=True, anchor='c')
    except Exception:
        pass

    c.showPage()
    c.save()

# -------------------------
# STYLE SELECTOR
# -------------------------
st.sidebar.header("Executive Card Style")
style_choice = st.sidebar.selectbox("Choose card style", ("Premium Blue", "Soft Grey", "Dark Mode", "Custom"))

if style_choice == "Custom":
    custom_bg = st.sidebar.text_input("Card background (CSS)", "#F7F7FF")
    custom_border = st.sidebar.text_input("Border color", "#4A90E2")
    custom_text = st.sidebar.text_input("Text color", "#0B3D91")
else:
    custom_bg = None
    custom_border = None
    custom_text = None

# -------------------------
# UPLOAD
# -------------------------
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])
if uploaded_file is None:
    st.info("Upload any CSV file (stock or financial). The app will auto-detect columns and generate the executive summary and exports.")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded_file)
except Exception:
    st.error("Could not read CSV — please save as UTF-8 CSV and try again.")
    st.stop()

df = clean_stock_dataframe(df_raw)

# -------------------------
# COMPUTE ANALYTICS
# -------------------------
# detect columns
date_col = detect_date_column(df)
if date_col:
    df_filtered = filter_last_6_months(df, date_col)
    if df_filtered.shape[0] < 2:
        df_filtered = df.copy()
else:
    df_filtered = df.copy()

close_col = detect_column(df_filtered, ["close","prevclose","last","price","closeprice","prev"])
vol_col = detect_column(df_filtered, ["vol","volume","tradedvolume","totalvolume"])
val_col = detect_column(df_filtered, ["value","turnover","totalvalue","marketvalue"])

n_rows, n_cols = df_filtered.shape

# compute core metrics
pct_change = None
if close_col:
    pct_change = compute_pct_change(df_filtered[close_col])

vol_idx = None
if close_col:
    vol_idx = volatility_index(df_filtered[close_col])

momentum = None
if close_col:
    momentum = momentum_rating(df_filtered[close_col])

# correlation
corr = None
if vol_col and val_col:
    try:
        numeric = df_filtered[[vol_col, val_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if numeric.shape[0] > 2:
            corr = numeric.corr().iloc[0,1]
    except:
        corr = None

risk = risk_score(pct_change, vol_idx, momentum if momentum is not None else 50, corr)
sentiment = sentiment_score(pct_change, corr, vol_idx)
inst_activity = institutional_activity_estimate(df_filtered[vol_col]) if vol_col else None

anomalies_idx = detect_anomalies_price(df_filtered[close_col]) if close_col else []
patterns = detect_patterns(df_filtered, date_col, close_col) if close_col else []

# BUILD a stronger summary (safe HTML, no inner <div>)
safe_close = html.escape(close_col) if close_col else "N/A"
safe_vol = html.escape(vol_col) if vol_col else "N/A"
safe_val = html.escape(val_col) if val_col else "N/A"

# Trend wording
if pct_change is None:
    trend_heading = "No trend information available."
    trend_reason = ""
else:
    if pct_change > 0:
        trend_heading = f"The <b>{safe_close}</b> shows a <b>📈 upward</b> trend, changing by <b>{pct_change:.2f}%</b> over the observed period."
        trend_reason = "This sustained upward movement signals strengthening investor conviction and accumulation."
    else:
        trend_heading = f"The <b>{safe_close}</b> shows a <b>📉 downward</b> trend, changing by <b>{pct_change:.2f}%</b> over the observed period."
        trend_reason = "This persistent downside reflects distribution or profit-taking and selling pressure."

# volume & corr wording
volume_info = ""
if vol_col:
    try:
        nv = pd.to_numeric(df_filtered[vol_col], errors="coerce").dropna()
        if not nv.empty:
            volume_info = f"Trading volumes ranged between <b>{nv.min():,}</b> and <b>{nv.max():,}</b>, averaging <b>{nv.mean():,.0f}</b>."
    except:
        volume_info = ""

correlation_info = ""
if corr is not None:
    correlation_info = f"The correlation between <b>{safe_vol}</b> and <b>{safe_val}</b> is <b>{corr:.2f}</b>, showing alignment between trading activity and market value."

# dynamic summary building (stronger and concrete)
summary_parts = []
if pct_change is not None:
    if pct_change > 8:
        summary_parts.append("Strong bullish momentum observed with meaningful price appreciation.")
    elif pct_change > 2:
        summary_parts.append("Upward bias is present and supported by recent price gains.")
    elif pct_change > -2:
        summary_parts.append("Price action is range-bound — likely consolidation.")
    elif pct_change > -8:
        summary_parts.append("Mild bearish pressure is present; monitor for follow-through.")
    else:
        summary_parts.append("Significant bearish pressure; risk sentiment elevated.")

if vol_idx is not None:
    if vol_idx < 1:
        summary_parts.append("Price volatility remains low — orderly trading.")
    elif vol_idx < 3:
        summary_parts.append("Moderate volatility suggests active but controlled trading.")
    else:
        summary_parts.append("Elevated volatility — expect rapid short-term swings.")

if inst_activity is not None:
    level, ratio = inst_activity
    summary_parts.append(f"Institutional activity: <b>{level}</b> (recent/prev ratio ≈ {ratio:.2f}).")

if corr is not None:
    if corr > 0.8:
        summary_parts.append("Strong volume–value correlation suggests coordinated market participation.")
    elif corr > 0.4:
        summary_parts.append("Moderate correlation indicates partial alignment between volume and value.")
    else:
        summary_parts.append("Weak correlation suggests fragmented market flows.")

if patterns:
    summary_parts.append("Detected patterns: " + "; ".join(patterns) + ".")

if anomalies_idx:
    summary_parts.append(f"Anomalies detected at {len(anomalies_idx)} points (z-score > 3).")

dynamic_summary = " ".join(summary_parts).strip()
if not dynamic_summary:
    dynamic_summary = "Insufficient data to derive a dynamic narrative."

# build summary_html (no nested div)
summary_html = (
    f"<h3 style='margin-bottom:8px;'>📄 Executive Summary (Latest 6 Months)</h3>"
    f"<p>The dataset contains <b>{n_rows}</b> records and <b>{n_cols}</b> columns.</p>"
    f"<h4 style='margin-top:6px;'>🔹 Trend</h4>"
    f"<p>{trend_heading}<br>{trend_reason}</p>"
    f"<h4 style='margin-top:6px;'>🔹 Volume & Correlation</h4>"
    f"<p>{volume_info}<br>{correlation_info}</p>"
    f"<h4 style='margin-top:6px;'>🔹 Summary</h4>"
    f"<p>{dynamic_summary}</p>"
)

# -------------------------
# PLOT & EXPORT PREP
# -------------------------
# Plotly trend figure (if available)
trend_fig = None
trend_png = None
if date_col and close_col and date_col in df_filtered.columns and close_col in df_filtered.columns:
    try:
        plotted = df_filtered.dropna(subset=[date_col, close_col]).sort_values(by=date_col)
        if plotted.shape[0] > 1:
            trend_fig = px.line(plotted, x=date_col, y=close_col, markers=True, title=close_col)
            # add moving averages
            try:
                plotted["MA20"] = plotted[close_col].astype(float).rolling(20).mean()
                plotted["MA50"] = plotted[close_col].astype(float).rolling(50).mean()
                trend_fig.add_scatter(x=plotted[date_col], y=plotted["MA20"], mode="lines", name="MA20")
                trend_fig.add_scatter(x=plotted[date_col], y=plotted["MA50"], mode="lines", name="MA50")
            except:
                pass
            # bytes for PDF
            try:
                buf = io.BytesIO()
                trend_fig.write_image(buf, format="png", scale=2)
                buf.seek(0)
                trend_png = buf
            except Exception:
                trend_png = None
    except:
        trend_fig = None

# Correlation heatmap (matplotlib)
heatmap_png = None
heatmap_fig = None
try:
    numeric = df_filtered.select_dtypes(include=np.number)
    if numeric.shape[1] > 1:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        heatmap_fig = fig
        buf2 = io.BytesIO()
        fig.savefig(buf2, format="png", dpi=150, bbox_inches="tight")
        buf2.seek(0)
        heatmap_png = buf2
        plt.close(fig)
except:
    heatmap_png = None

# -------------------------
# CARD STYLE VALUES
# -------------------------
if style_choice == "Premium Blue":
    card_bg = "linear-gradient(135deg, #EAF3FF 0%, #FFFFFF 100%)"
    border_color = "#1F77B4"
    text_color = "#0B3D91"
elif style_choice == "Soft Grey":
    card_bg = "#F7F8FA"
    border_color = "#4A90E2"
    text_color = "#1F2937"
elif style_choice == "Dark Mode":
    card_bg = "linear-gradient(135deg, #0f1724 0%, #111827 100%)"
    border_color = "#00E5FF"
    text_color = "#E6F7FF"
else:
    card_bg = custom_bg or "#F7F7FF"
    border_color = custom_border or "#4A90E2"
    text_color = custom_text or "#0B3D91"

card_html = f"""
<div style="
    background: {card_bg};
    padding: 24px;
    border-radius: 14px;
    border-left: 8px solid {border_color};
    box-shadow: 0 8px 28px rgba(8,18,40,0.06);
    color: { '#E6F7FF' if style_choice == 'Dark Mode' else '#0b1730'};
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
">
    {summary_html}
</div>
"""

# Render card
st.markdown(card_html, unsafe_allow_html=True)

# -------------------------
# Key Insights (unchanged)
# -------------------------
st.subheader("💡 Key Insights")
ins_list = []
if pct_change is not None:
    ins_list.append(f"📈 Price Change: {pct_change:.2f}%")
if vol_col:
    try:
        avg_vol = pd.to_numeric(df_filtered[vol_col], errors="coerce").mean()
        if not np.isnan(avg_vol):
            ins_list.append(f"📊 Avg Volume: {avg_vol:,.0f}")
    except:
        pass
if corr is not None:
    ins_list.append(f"🔗 Volume–Value Correlation: {corr:.2f}")

# Enterprise extra metrics
ins_list.append(f"⚠️ Risk Score (0-100): {risk:.1f}")
ins_list.append(f"📉 Volatility Index (std): {vol_idx:.3f}" if vol_idx is not None else "📉 Volatility Index: N/A")
ins_list.append(f"⚡ Momentum Rating (0-100): {momentum:.1f}" if momentum is not None else "⚡ Momentum Rating: N/A")
ins_list.append(f"💬 Sentiment Score (-1..1): {sentiment:.2f}")
if inst_activity is not None:
    inst_level, inst_ratio = inst_activity
    ins_list.append(f"🏦 Institutional Activity: {inst_level} (ratio {inst_ratio:.2f})")

for it in ins_list:
    st.markdown(f"- {html.escape(it)}")

# -------------------------
# Trend Chart & Download
# -------------------------
if trend_fig is not None:
    st.subheader("📉 Price Trend")
    st.plotly_chart(trend_fig, use_container_width=True)

    # download PNG of trend
    try:
        png_bytes = trend_png.getvalue()
        st.download_button("📥 Download trend chart (PNG)", data=png_bytes, file_name="trend_chart.png", mime="image/png")
    except Exception:
        pass

# -------------------------
# Correlation Heatmap & Download
# -------------------------
if heatmap_png is not None:
    st.subheader("📊 Correlation Heatmap")
    st.image(heatmap_png)
    st.download_button("📥 Download heatmap (PNG)", data=heatmap_png.getvalue(), file_name="heatmap.png", mime="image/png")

# -------------------------
# Patterns & Anomalies
# -------------------------
if patterns:
    st.subheader("🔍 Detected Patterns")
    for p in patterns:
        st.markdown(f"- {html.escape(p)}")

if anomalies_idx:
    st.subheader("🚨 Price Anomalies (z-score > 3)")
    st.markdown(f"- Number of anomalies detected: {len(anomalies_idx)}")
    # show the rows (if date_col available)
    try:
        idx_rows = [df_filtered.iloc[i] for i in anomalies_idx if i < len(df_filtered)]
        # present small table
        anomalies_df = pd.DataFrame(idx_rows)
        st.dataframe(anomalies_df.head(10))
    except Exception:
        pass

# -------------------------
# DOWNLOAD CLEANED CSV
# -------------------------
clean_buf = io.StringIO()
df_filtered.to_csv(clean_buf, index=False)
clean_bytes = clean_buf.getvalue().encode('utf-8')
st.download_button("📥 Download cleaned CSV", data=clean_bytes, file_name="cleaned_data.csv", mime="text/csv")

# -------------------------
# EXPORT PDF (Executive Summary + charts)
# -------------------------
st.subheader("📄 Export")
pdf_button = st.button("Generate & Download Executive PDF")

if pdf_button:
    # create temp files for PDF
    tmp_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(tmp_dir, "executive_summary.pdf")
    # create PDF embedding summary text (plain) and images if present
    try:
        # create image bytes for trend and heatmap if available
        trend_bytes = trend_png if trend_png is not None else None
        heatmap_bytes = heatmap_png if heatmap_png is not None else None
        create_pdf_with_summary(pdf_path, summary_html, trend_bytes, heatmap_bytes)
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        st.download_button("📥 Download Executive Summary PDF", data=pdf_data, file_name="executive_summary.pdf", mime="application/pdf")
    except Exception as e:
        st.error("Failed to generate PDF: " + str(e))

# -------------------------
# CLEANUP temp files (if any)
# -------------------------
# (temp files are in tmp_dir; system will clean after runtime)
