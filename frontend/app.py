import sys
import time
import streamlit as st

from pathlib import Path

# =========================================================
# IMPORT ENSEMBLE DETECTOR
# =========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

sys.path.append(
    str(ROOT_DIR / "training")
)

from ensemble import detect_phishing

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        to bottom right,
        #020617,
        #06122B,
        #020617
    );
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    text-align: center;
    font-size: 54px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #94A3B8;
    margin-bottom: 40px;
}

textarea {
    background-color: #111827 !important;
    color: white !important;
    border-radius: 14px !important;
    border: 1px solid #1E293B !important;
    font-size: 16px !important;
}

.stButton > button {
    width: 100%;
    height: 52px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(
        90deg,
        #2563EB,
        #1D4ED8
    );
    color: white;
    font-size: 18px;
    font-weight: 600;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.01);
}

.metric-card {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid #1E293B;
    border-radius: 18px;
    padding: 24px;
    text-align: center;
    min-height: 170px;
}

.metric-title {
    color: #94A3B8;
    font-size: 16px;
    margin-bottom: 12px;
}

.metric-value {
    color: white;
    font-size: 34px;
    font-weight: bold;
    margin-top: 10px;
}

.verdict-card {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid #1E293B;
    border-radius: 18px;
    padding: 30px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="main-title">
🛡️ PhishGuard AI
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="subtitle">
AI-Powered Phishing Detection using
SVM + DistilBERT Cascade Intelligence
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚙️ Detection Engine")

    st.markdown("---")

    st.markdown("""
### Core Technologies

- TF-IDF + SVM
- DistilBERT NLP
- Heuristic Analysis
- Threat Scoring
- Explainable AI
- Cascade Architecture
""")

    st.markdown("---")

    st.info(
        "Fast phishing detection using "
        "hybrid machine learning intelligence."
    )

# =========================================================
# INPUT
# =========================================================

st.subheader("📩 Email / Message Analysis")

email_text = st.text_area(
    "Paste suspicious email, SMS, or message:",
    height=240,
    placeholder="Paste suspicious message here..."
)

analyze = st.button(
    "🔍 Analyze Threat"
)

# =========================================================
# DETECTION
# =========================================================

if analyze:

    if not email_text.strip():

        st.warning(
            "Please enter a message."
        )

    else:

        start_time = time.time()

        with st.spinner(
            "Running AI threat analysis..."
        ):

            result = detect_phishing(
                email_text
            )

        end_time = time.time()

        inference_time = round(
            end_time - start_time,
            2
        )

        risk = float(
            result["confidence"]
        )

        # =================================================
        # RISK LEVEL
        # =================================================

        if risk >= 85:

            risk_color = "#DC2626"
            risk_label = "CRITICAL"

        elif risk >= 65:

            risk_color = "#F59E0B"
            risk_label = "HIGH"

        elif risk >= 40:

            risk_color = "#2563EB"
            risk_label = "MEDIUM"

        else:

            risk_color = "#10B981"
            risk_label = "MINIMAL"

        st.markdown("---")

        # =================================================
        # RESULT CARD
        # =================================================

        if result["label"] == "PHISHING":

            st.markdown(
                f"""
<div style="
background: linear-gradient(90deg,#450A0A,#7F1D1D);
border:1px solid #EF4444;
padding:30px;
border-radius:22px;
margin-bottom:20px;
">

<h1 style="
color:#FF5C5C;
margin-bottom:10px;
">
🚨 PHISHING DETECTED
</h1>

<h2 style="color:white;">
Threat Score: {risk:.2f}%
</h2>

</div>
""",
                unsafe_allow_html=True
            )

        else:

            safe_confidence = 100 - risk

            st.markdown(
                f"""
<div style="
background: linear-gradient(90deg,#052E16,#065F46);
border:1px solid #10B981;
padding:30px;
border-radius:22px;
margin-bottom:20px;
">

<h1 style="
color:#34D399;
margin-bottom:10px;
">
✅ LEGITIMATE MESSAGE
</h1>

<h2 style="color:white;">
Confidence: {safe_confidence:.2f}%
</h2>

</div>
""",
                unsafe_allow_html=True
            )

        # =================================================
        # RISK METER
        # =================================================

        st.subheader("📈 Threat Risk Meter")

        st.markdown(
            f"""
<div style="
width:100%;
background:#111827;
border-radius:14px;
padding:6px;
border:1px solid #1F2937;
margin-top:10px;
">

<div style="
width:{risk}%;
background:{risk_color};
height:22px;
border-radius:10px;
">
</div>

</div>
""",
            unsafe_allow_html=True
        )

        # =================================================
        # THREAT LEVEL
        # =================================================

        st.markdown(
            f"""
<div style="
margin-top:20px;
background:{risk_color};
padding:14px;
border-radius:12px;
text-align:center;
font-size:24px;
font-weight:600;
color:white;
">
Threat Level: {risk_label}
</div>
""",
            unsafe_allow_html=True
        )

        # =================================================
        # METRICS
        # =================================================

        st.subheader("📊 Detection Metrics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.markdown(
                f"""
<div class="metric-card">

<div class="metric-title">
SVM Score
</div>

<div class="metric-value">
{result['svm_score']}
</div>

</div>
""",
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                f"""
<div class="metric-card">

<div class="metric-title">
BERT Score
</div>

<div class="metric-value">
{result['bert_score']}
</div>

</div>
""",
                unsafe_allow_html=True
            )

        with col3:

            st.markdown(
                f"""
<div class="metric-card">

<div class="metric-title">
BERT Activated
</div>

<div class="metric-value">
{result['bert_used']}
</div>

</div>
""",
                unsafe_allow_html=True
            )

        with col4:

            st.markdown(
                f"""
<div class="metric-card">

<div class="metric-title">
Inference Time
</div>

<div class="metric-value">
{inference_time}s
</div>

</div>
""",
                unsafe_allow_html=True
            )

        # =================================================
        # THREAT INTELLIGENCE
        # =================================================

        st.subheader("⚠️ Threat Intelligence")

        reasons = result["threat_reasons"]

        if len(reasons) == 0:

            st.markdown(
                """
<div style="
background: rgba(16,185,129,0.15);
border:1px solid #10B981;
color:#6EE7B7;
padding:16px;
border-radius:14px;
margin-bottom:14px;
">
No major phishing indicators detected.
</div>
""",
                unsafe_allow_html=True
            )

        else:

            for reason in reasons:

                st.markdown(
                    f"""
<div style="
background: rgba(239,68,68,0.12);
border:1px solid rgba(239,68,68,0.4);
color:#FCA5A5;
padding:16px;
border-radius:14px;
margin-bottom:14px;
">
{reason}
</div>
""",
                    unsafe_allow_html=True
                )

        # =================================================
        # VERDICT
        # =================================================

        st.subheader("🧠 AI Security Verdict")

        if result["label"] == "PHISHING":

            verdict = (
                "The message contains suspicious phishing "
                "patterns detected through heuristic analysis, "
                "semantic AI processing, and domain intelligence."
            )

        else:

            verdict = (
                "The message appears legitimate based on "
                "machine learning classification and semantic "
                "threat analysis."
            )

        st.markdown(
            f"""
<div class="verdict-card">

<h2 style="
color:white;
margin-bottom:20px;
">
AI Verdict
</h2>

<p style="
color:#CBD5E1;
font-size:18px;
line-height:1.8;
">
{verdict}
</p>

</div>
""",
            unsafe_allow_html=True
        )