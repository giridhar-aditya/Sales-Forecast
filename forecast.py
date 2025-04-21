import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
df = pd.read_csv(r"D:\ML Projects\Sales Forecasting using Machine Learning\Walmart.csv")

# Convert date column to datetime format (adjust column name as needed)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by='Date')

# Fill missing values
df.fillna(method='ffill', inplace=True)

# Feature Engineering (Adjust column names based on actual dataset structure)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['WeekOfYear'] = df['Date'].dt.isocalendar().week

# Encoding categorical variables if any
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target
X = df[['Year', 'Month', 'Day', 'WeekOfYear']]
y = df['Sales']  # Adjust column name if necessary

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))

# Train ARIMA Model (Using entire series for time-series forecasting)
arima_model = ARIMA(y, order=(5,1,0))
arima_result = arima_model.fit()
arima_preds = arima_result.forecast(steps=len(y_test))
arima_rmse = np.sqrt(mean_squared_error(y_test, arima_preds))

# Print RMSE values
print(f"Linear Regression RMSE: {lr_rmse}")
print(f"Random Forest RMSE: {rf_rmse}")
print(f"ARIMA RMSE: {arima_rmse}")

# Plot predictions
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Sales', color='blue')
plt.plot(lr_preds, label='Linear Regression', linestyle='dashed', color='red')
plt.plot(rf_preds, label='Random Forest', linestyle='dotted', color='green')
plt.plot(arima_preds, label='ARIMA', linestyle='dashdot', color='purple')
plt.legend()
plt.title("Sales Forecasting Model Comparison")
plt.show()
