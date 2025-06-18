# Walmart Sales Forecasting 📈

## 📝 Project Summary

This project aims to accurately forecast Walmart's weekly sales using advanced **machine learning techniques** combined with **time series feature engineering**. By leveraging historical sales data and engineered temporal patterns, we built highly accurate models that can help retailers make data-driven decisions for inventory planning, promotions, and resource allocation.

After several modeling approaches, **XGBoost** with carefully tuned hyperparameters outperformed simpler models like Linear Regression, Random Forest, and ARIMA, delivering a highly competitive forecasting solution.

---

## 📊 Dataset Description

* **Source**: Walmart historical sales dataset
* **Target Variable**: `Weekly_Sales`
* **Features Used**:

  * Date-based features: `Year`, `Month`, `Week`, `DayOfWeek`, `IsWeekend`
  * Store information (encoded)
  * Macroeconomic features: `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`
  * Time series lag features (1, 2, 3, 4, 7, 14, 28 weeks)
  * Rolling window statistics (mean & std) for multiple window sizes

---

## ⚙️ Workflow

1️⃣ **Data Loading & Cleaning**

* Imported raw CSV data
* Parsed date column, sorted data chronologically

2️⃣ **Feature Engineering**

* Generated multiple lag features to capture temporal sales dependencies
* Created rolling statistical features to reflect local trends and volatility
* Added weekend flags and extracted date parts (year, month, week, weekday)

3️⃣ **Preprocessing**

* Encoded categorical features (e.g., `Store`)
* Scaled continuous features using `StandardScaler`

4️⃣ **Train-Test Split**

* Time-based split using 80% of data for training and 20% for testing (preserving temporal ordering)

5️⃣ **Model Training & Hyperparameter Optimization**

* Used `Optuna` for automated hyperparameter tuning with 50 trials
* Models evaluated using **Mean Absolute Percentage Error (MAPE)**

---

## 🚀 Models Built

| Model             | Type              | Purpose                      | Notes                      |
| ----------------- | ----------------- | ---------------------------- | -------------------------- |
| Linear Regression | Baseline          | Simple benchmark             | Easy to interpret          |
| Random Forest     | Ensemble Learning | Captures non-linearity       | Limited temporal awareness |
| ARIMA             | Time Series       | Pure time series modeling    | Suitable for univariate    |
| **XGBoost**       | Gradient Boosting | Advanced supervised learning | Best performer             |

---

## 📈 Model Performance

After tuning and evaluating our final **XGBoost** model:

| Metric       | Value       |
| ------------ | ----------- |
| **MAE**      | `40,222.60` |
| **RMSE**     | `60,063.34` |
| **MAPE**     | `3.93%`     |
| **Accuracy** | **96.07%**  |

👉 **XGBoost** achieved an impressive **96% accuracy** on the test set, making it a highly reliable model for forecasting weekly sales in a retail setting.

This performance demonstrates how combining:

* Lag features
* Rolling statistics
* Advanced hyperparameter tuning
  can dramatically improve forecasting ability in time series problems.

---

## 🧰 Technologies Used

* **Python 3.7+**
* `pandas` — data processing
* `numpy` — numerical computations
* `scikit-learn` — preprocessing & metrics
* `xgboost` — machine learning model
* `optuna` — hyperparameter tuning
* `joblib` — model persistence

---

## 🔧 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/walmart-sales-forecasting.git
cd walmart-sales-forecasting
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost optuna joblib
```

---

## 💻 Running the Project

You can directly run the training script:

```bash
python train.py
```

The script will:

* Load and preprocess the data
* Engineer time series features
* Tune hyperparameters using Optuna
* Train final model with best params
* Evaluate on the test set
* Save trained model to disk (`xgb_model.joblib`)

---

## 📊 Future Improvements

* Incorporate **holiday effect modeling**
* Add **promotion data** (if available)
* Experiment with **deep learning models** (LSTM, GRU, Temporal Fusion Transformers)
* Deploy as an API service for real-time forecasting

---
