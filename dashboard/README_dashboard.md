# Personal Finance Analytics Dashboard

An interactive Plotly Dash dashboard that visualizes 2021 personal transaction data with live ARIMA forecasting and K-Means spending segmentation.

---

## Quickstart (under 2 minutes)

```bash
# 1. From the repo root, install dependencies
pip install -r dashboard/requirements.txt

# 2. Launch the dashboard
cd dashboard
python app.py
```

Then open **http://127.0.0.1:8050** in your browser.

---

## What's on the Dashboard

| Panel | Description |
|---|---|
| **KPI Cards** | Total Income, Total Expenses, Net Savings, Avg Monthly Savings Rate |
| **Monthly Trends** | Line chart — Income vs Expenses vs Savings by month |
| **Spending by Category** | Horizontal bar chart — total spend per category |
| **ARIMA Forecast** | Historical savings + 3-month ARIMA(1,1,1) projection (dashed line) |
| **K-Means Clusters** | Bubble scatter — expense categories segmented into Low / Medium / High spend |

---

## File Structure

```
dashboard/
├── app.py              # Full Dash application (single file)
├── requirements.txt    # Dashboard-specific dependencies
└── README_dashboard.md # This file

../personal finance data.csv  # Source data (one level up)
```

---

## Data Notes

- Source: `personal finance data.csv` — 225 transactions, January–December 2021
- Date format parsed: `%d %B %Y` (e.g., `01 January 2021`)
- Categories normalized to title case (handles `apparel` / `Apparel` / `salary` / `Salary`)
- ARIMA falls back to historical mean if model fitting fails on a given dataset

---

## Dependencies

| Package | Purpose |
|---|---|
| `dash` + `dash-bootstrap-components` | Web app framework + dark theme layout |
| `plotly` | Interactive charts |
| `pandas` / `numpy` | Data manipulation |
| `statsmodels` | ARIMA time-series forecasting |
| `scikit-learn` | K-Means clustering + StandardScaler |
