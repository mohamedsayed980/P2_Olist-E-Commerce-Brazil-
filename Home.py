# ─────────────────────────────────────────────
#  Repo_2 — Olist E-Commerce Analysis
#  Home.py  |  Multipage Launcher
#  Author : Mohamed · M3
# ─────────────────────────────────────────────
# =============================================================================
## path = streamlit run "G:\FINAL_PROJECTS\P2_Olist E-Commerce (Brazil)\Home.py"
# =============================================================================
import streamlit as st
import pathlib

# ── Page config ───────────────────────────────
st.set_page_config(
    page_title  = "Olist E-Commerce · M3",
    page_icon   = "🛒",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Logo ──────────────────────────────────────
LOGO = pathlib.Path(__file__).parent / "M3_logo.png"

# ── CLR palette ───────────────────────────────
CLR = {
    "primary"   : "#1565c0",
    "success"   : "#2e7d32",
    "warning"   : "#e65100",
    "danger"    : "#c62828",
    "teal"      : "#00695c",
    "accent"    : "#00695c",
    "secondary" : "#455a64",
    "light"     : "#e3f2fd",
    "dark"      : "#1a237e",
    "purple"    : "#6a1b9a",
    "amber"     : "#f57f17",
    "pink"      : "#ad1457",
    "indigo"    : "#283593",
    "cyan"      : "#00838f",
    "lime"      : "#558b2f",
    "brown"     : "#4e342e",
    "grey"      : "#546e7a",
    "white"     : "#ffffff",
    "black"     : "#212121",
}

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=70)
    st.markdown("### 🛒 Olist E-Commerce")
    st.markdown("**Author:** Mohamed · M3")
    st.markdown("---")
    st.markdown("#### 📂 Navigation")
    st.markdown("""
- **🏠 Home** ← You are here
- **📊 EDA Dashboard** → Stage 1
- **🤖 ML Models** → Stage 2
""")
    st.markdown("---")
    st.markdown("#### 📁 Dataset Info")
    st.markdown("""
- **Source:** Olist Brazil (Kaggle)
- **Rows:** 112,647
- **Columns:** 27
- **Period:** Sep 2016 – Sep 2018
""")
    st.markdown("---")
    st.caption("© M3 · Data Analysis Portfolio")

# ── Main Header ───────────────────────────────
st.markdown(f"""
<div style='background: linear-gradient(135deg, {CLR["dark"]} 0%, {CLR["primary"]} 100%);
            padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;'>
    <h1 style='color: white; margin: 0; font-size: 2.4rem;'>
        🛒 Olist E-Commerce Analysis
    </h1>
    <p style='color: #bbdefb; margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
        End-to-End Data Analysis & Machine Learning · Brazilian E-Commerce Platform
    </p>
</div>
""", unsafe_allow_html=True)

# ── Project Overview Cards ────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div style='background:{CLR["light"]}; padding:1.2rem; border-radius:12px;
                border-left: 5px solid {CLR["primary"]}; text-align:center;'>
        <h2 style='color:{CLR["primary"]}; margin:0;'>112,647</h2>
        <p style='color:{CLR["secondary"]}; margin:0; font-size:0.85rem;'>Total Orders</p>
    </div>""", unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style='background:{CLR["light"]}; padding:1.2rem; border-radius:12px;
                border-left: 5px solid {CLR["success"]}; text-align:center;'>
        <h2 style='color:{CLR["success"]}; margin:0;'>27</h2>
        <p style='color:{CLR["secondary"]}; margin:0; font-size:0.85rem;'>Features</p>
    </div>""", unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div style='background:{CLR["light"]}; padding:1.2rem; border-radius:12px;
                border-left: 5px solid {CLR["warning"]}; text-align:center;'>
        <h2 style='color:{CLR["warning"]}; margin:0;'>70+</h2>
        <p style='color:{CLR["secondary"]}; margin:0; font-size:0.85rem;'>Categories</p>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div style='background:{CLR["light"]}; padding:1.2rem; border-radius:12px;
                border-left: 5px solid {CLR["teal"]}; text-align:center;'>
        <h2 style='color:{CLR["teal"]}; margin:0;'>2 Years</h2>
        <p style='color:{CLR["secondary"]}; margin:0; font-size:0.85rem;'>2016 – 2018</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two Stages ────────────────────────────────
col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown(f"""
    <div style='background:white; border:1px solid #e0e0e0; border-radius:14px;
                padding:1.5rem; height:100%;'>
        <h3 style='color:{CLR["primary"]}; margin-top:0;'>📊 Stage 1 — EDA Dashboard</h3>
        <p style='color:{CLR["secondary"]};'>Deep exploratory analysis across 11 tabs:</p>
        <ul style='color:{CLR["black"]}; line-height:1.9;'>
            <li>Data Overview & Schema</li>
            <li>Univariate & Bivariate Analysis</li>
            <li>Correlation Analysis</li>
            <li>Feature Engineering</li>
            <li>Missing Values & Imputation</li>
            <li>Statistical Tests (ANOVA · Chi² · T-test)</li>
            <li>Business KPI Dashboard</li>
            <li>Category Deep Dive</li>
            <li>RFM-style Segmentation</li>
            <li>Insights & Recommendations</li>
        </ul>
    </div>""", unsafe_allow_html=True)

with col_b:
    st.markdown(f"""
    <div style='background:white; border:1px solid #e0e0e0; border-radius:14px;
                padding:1.5rem; height:100%;'>
        <h3 style='color:{CLR["success"]}; margin-top:0;'>🤖 Stage 2 — ML Models</h3>
        <p style='color:{CLR["secondary"]};'>Full machine learning pipeline across 5 tabs:</p>
        <ul style='color:{CLR["black"]}; line-height:1.9;'>
            <li>Regression Models (6) → predict <b>delivery days</b></li>
            <li>Classification Models (6) → predict <b>satisfaction</b></li>
            <li>Model Comparison & Report</li>
            <li>Predict New Data</li>
            <li>Final Insights & Report (PDF + Word)</li>
        </ul>
        <br>
        <p style='color:{CLR["secondary"]}; font-size:0.85rem;'>
            ✅ 12 models · Parallel training · Save/Load · Export
        </p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── How to Use ────────────────────────────────
st.markdown(f"""
<div style='background:{CLR["light"]}; border-radius:12px; padding:1.2rem 1.5rem;'>
    <h4 style='color:{CLR["dark"]}; margin-top:0;'>🚀 How to Use</h4>
    <ol style='color:{CLR["black"]}; line-height:2;'>
        <li>Go to <b>📊 EDA Dashboard</b> in the sidebar → upload <code>olist_full_clean.csv</code></li>
        <li>Explore all 11 EDA tabs — insights are auto-saved to session</li>
        <li>Go to <b>🤖 ML Models</b> → data flows automatically from Stage 1</li>
        <li>Train models, compare results, export the final report</li>
    </ol>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────
st.markdown(f"""
<div style='text-align:center; color:{CLR["grey"]}; font-size:0.8rem; padding:1rem;'>
    Built with ❤️ by Mohamed · M3 · Data Analysis Portfolio · Olist Brazil Dataset
</div>
""", unsafe_allow_html=True)
