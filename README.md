# Sales Forecasting using Machine Learning and Time Series Models

## 📌 Overview
This project forecasts sales using a mix of **machine learning** (Linear Regression, Random Forest) and **time series** (ARIMA) models. It evaluates and compares model performance using RMSE, enabling better decision-making in retail forecasting.

## 🧾 Dataset
- **Source**: Walmart historical sales dataset
- **Target**: `Sales`
- **Features**: Date components (Year, Month, Day, WeekOfYear), plus encoded categorical data

## ⚙️ Workflow
1. Load and clean data
2. Feature engineering from date
3. Encode categorical variables
4. Split into training/testing sets
5. Train & evaluate:
   - Linear Regression
   - Random Forest
   - ARIMA
6. Compare models via RMSE and visualize results

## 📊 Models & Evaluation
| Model             | Technique           | Metric |
|------------------|---------------------|--------|
| Linear Regression| Regression          | RMSE   |
| Random Forest    | Ensemble Learning   | RMSE   |
| ARIMA            | Time Series         | RMSE   |

## 🚀 How to Run

### Requirements
- Python 3.7+
- Install libraries:
  ```bash
  pip install pandas numpy matplotlib scikit-learn statsmodels
