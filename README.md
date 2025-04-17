# 💰 Personal Finance Analyzer

A data-driven Python tool that analyzes financial behavior, forecasts future savings, and uncovers spending patterns using machine learning and time series techniques.

## 📌 Features

- 📊 **Monthly Financial Summary**: Calculates income, expenses, and savings month-over-month.
- 🔮 **Savings Forecasting**: Predicts future savings using ARIMA and Linear Regression models.
- 📉 **Spending Pattern Analysis**: Identifies spending behavior through K-Means clustering.
- 📈 **Visual Insights**: Generates intuitive visualizations for trends and category-wise expenses.

## 🧠 Technologies Used

- **Pandas, NumPy** – Data manipulation and analysis  
- **Matplotlib** – Visualizations  
- **scikit-learn** – Linear Regression, K-Means Clustering  
- **Statsmodels** – ARIMA time series forecasting  
- **StandardScaler, Train-Test Split** – Data preprocessing  
- **Jupyter Notebook** – Interactive analysis and visualization  

## 🚀 How It Works

1. **Ingests personal finance data** from a CSV file.
2. **Cleans and organizes** it into income and expenses by month.
3. **Calculates savings**, visualizes trends, and forecasts future values.
4. **Performs clustering** to reveal dominant expense categories.
5. **Presents results** via clean plots and performance metrics.

## 📂 File Structure

```bash
📁 personal-finance-analyzer/
├── personal_finance_analyzer.py
├── personal finance data.csv
└── README.md
