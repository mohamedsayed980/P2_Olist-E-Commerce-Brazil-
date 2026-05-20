# 🛒 Olist E-Commerce — End-to-End Data Analysis & ML
> **Author:** Mohamed · M3 · Data Analysis Portfolio  
> **Dataset:** Brazilian E-Commerce Public Dataset by Olist (Kaggle)  
> **Stack:** Python · Streamlit · Scikit-learn · Pandas · Matplotlib · Seaborn

---

## 📌 Project Overview

A full end-to-end data analysis and machine learning project built on the **Olist Brazilian E-Commerce dataset** — one of the richest real-world retail datasets available, containing 9 interconnected CSV files covering orders, customers, sellers, products, payments, and reviews from 2016 to 2018.

This project demonstrates professional data analysis skills including multi-table merging, deep EDA, business KPI reporting, statistical testing, customer segmentation, and predictive modeling.

---
## 📥 Dataset Setup
1. Download from [Kaggle — Olist Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
2. Run the merge notebook → produces `olist_full_clean.csv`
3. Place in `/data` folder
4. Upload via the app file uploader
--- 

## 🗂️ Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) |
| Files merged | 9 CSV files |
| Final shape | 112,647 rows × 27 columns |
| Period | September 2016 → September 2018 |
| Geography | Brazil (27 states) |

### Merged Tables
```
order_items + products + payments + orders +
customers + reviews + sellers + geolocation + category_translation
→ olist_full_clean.csv (single analysis-ready file)
```

---

## 🏗️ Architecture

```
📁 Repo_2_Olist_Ecommerce/
├── Home.py                    ← Multipage launcher
├── M3_logo.png                ← Brand logo
├── requirements.txt
├── README.md
├── pages/
│   ├── EDA_dashboard.py       ← Stage 1 (Tabs 1–11)
│   └── ML_Models.py           ← Stage 2 (Tabs 9–13)
└── data/
    └── olist_full_clean.csv   ← Merged & cleaned dataset
```

**Run:**
```bash
streamlit run Home.py
```

---

## 📊 Stage 1 — EDA Dashboard (11 Tabs)

| Tab | Name | Description |
|-----|------|-------------|
| 1 | Data Overview & Correlation | Shape, dtypes, missing values, top correlations |
| 2 | Variables Analysis | Distributions, histograms, categorical counts |
| 3 | IQR Cleaning | Outlier detection and removal via IQR method |
| 4 | Outliers Lab | Z-score, box plots, outlier summary |
| 5 | Dashboard Summary | Visual summary of cleaned data |
| 6 | Missing Values | Heatmap, imputation strategies |
| 7 | Multicollinearity | VIF analysis, correlation matrix |
| 8 | Insights | Auto-generated EDA insights text |
| 9 | **Business KPI Dashboard** ⭐ | Revenue, delivery, satisfaction KPIs + monthly trend |
| 10 | **Category Deep Dive** ⭐ | Sales, price, freight by product category |
| 11 | **Statistical Tests** ⭐ | ANOVA, T-Test, Chi-Square — p-value confirmed insights |

> ⭐ = New tabs added beyond standard EDA — Olist-specific business analysis

---

## 🤖 Stage 2 — ML Models (5 Tabs)

| Tab | Name | Description |
|-----|------|-------------|
| 9 | Regression Models (6) | Predict **delivery days** |
| 10 | Classification Models (6) | Predict **customer satisfaction** (binary) |
| 11 | Comparison & Report | Side-by-side model comparison, best model selection |
| 12 | Predict New Data | Input new order data → get prediction |
| 13 | Final Insights & Report | Auto-generated report → export PDF + Word |

### Models Included
**Regression (6):** Linear Regression · Ridge · Lasso · Decision Tree · Random Forest · Gradient Boosting

**Classification (6):** Logistic Regression · KNN · Decision Tree · Random Forest · Gradient Boosting · SVM

### ML Targets
| Type | Target | Meaning |
|------|--------|---------|
| Regression | `delivery_days` | How many days until order is delivered |
| Classification | `is_satisfied` (0/1) | Will the customer leave a positive review? |

---

## 🔑 Key Business Insights

- **Total Revenue:** R$ 20,308,135 across 2 years
- **Avg Delivery Time:** 11.8 days (varies significantly by state)
- **Satisfaction Rate:** 75.5% of customers satisfied (review ≥ 4 stars)
- **Payment:** 75.6% use credit card · 20.3% boleto
- **Top State:** São Paulo (SP) = 42% of all orders
- **Top Category:** cama_mesa_banho (bed/bath) = 11,115 items
- **Statistical Tests:** Category significantly affects price (ANOVA p < 0.05) · State significantly affects delivery time (ANOVA p < 0.05)

---

## ⚙️ Features

| Feature | Detail |
|---------|--------|
| Multi-table merge | 9 CSV files → 1 clean DataFrame |
| Auto separator detection | `sep=None, engine="python"` |
| Parallel ML training | ThreadPoolExecutor · CPU slider |
| Save/Load models | joblib package |
| PDF export | reportlab |
| Word export | python-docx |
| Session bridge | EDA → ML via `st.session_state` |

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/Repo_2_Olist_Ecommerce.git
cd Repo_2_Olist_Ecommerce

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run Home.py

# 4. Upload olist_full_clean.csv in the EDA Dashboard
```

---

## 📦 Requirements

```
Python >= 3.10
streamlit · pandas · numpy · scipy · scikit-learn
matplotlib · seaborn · plotly · statsmodels
fpdf2 · reportlab · python-docx · psutil · joblib
```

---

## 🗺️ Portfolio Roadmap

| # | Project | Domain | Status |
|---|---------|--------|--------|
| P1 | House Prices Analysis | Real Estate | ✅ Complete |
| P2 | Olist E-Commerce | Retail / Business | ✅ Complete |
| P3 | HR Attrition | Human Resources | 🔨 In Progress |
| P4–P7 | Coming Soon... | Various Domains | 🔜 |

---

## 👤 Author

**Mohamed · M3**  
Mechanical Engineer → Data Analyst  
📊 Building a professional DA/DS portfolio — one dataset at a time.

---

*Built with ❤️ using Python & Streamlit*
