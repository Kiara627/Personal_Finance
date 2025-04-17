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

## 📈 Output 
📊 Monthly Income, Expenses, and Savings
![image](https://github.com/user-attachments/assets/5d5361a3-4ed2-4c90-9023-a4ef5dcb5ac6)
🔮 Savings Forecast (ARIMA)
![image](https://github.com/user-attachments/assets/73d17eb8-8093-4e6e-bd16-06364aa36695)
📌 K-Means Clustering of Expenses by Category
![image](https://github.com/user-attachments/assets/45b6df9b-cbe1-41de-9e3e-8b85c92c7098)
### 💬 Terminal Output Example

```bash
ARIMA Forecast: [415.23, 435.89, 452.74]
Mean Squared Error: 2800.54
Root Mean Squared Error: 52.91
R-squared Score: 0.89
Future Savings Predictions:
Month 1: 420.52
Month 2: 439.28
Month 3: 458.67
