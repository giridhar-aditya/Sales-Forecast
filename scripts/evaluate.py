import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load model and preprocessors
model = joblib.load("xgb_model.joblib")
scaler = joblib.load("scaler.joblib")
le = joblib.load("label_encoder.joblib")
df = pd.read_csv("walmart_pp.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

for lag in [1, 2, 3, 4, 7, 14, 28]:
    df[f"Sales_Lag_{lag}"] = df.groupby('Store')['Weekly_Sales'].shift(lag)

for window in [3, 7, 14, 28]:
    df[f"Rolling_Mean_{window}"] = df.groupby('Store')['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(window=window).mean())
    df[f"Rolling_Std_{window}"] = df.groupby('Store')['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(window=window).std())

df = df.dropna().reset_index(drop=True)
df['Store'] = le.transform(df['Store'])

cont_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment'] + \
                [f"Sales_Lag_{lag}" for lag in [1, 2, 3, 4, 7, 14, 28]] + \
                [f"Rolling_Mean_{window}" for window in [3, 7, 14, 28]] + \
                [f"Rolling_Std_{window}" for window in [3, 7, 14, 28]]

df[cont_features] = scaler.transform(df[cont_features])

# Use same features as training
features = ['Store', 'Holiday_Flag', 'Year', 'Month', 'Week', 'DayOfWeek', 'IsWeekend'] + cont_features
target = 'Weekly_Sales'

preds = model.predict(df[features])

# Evaluate
mae = mean_absolute_error(df[target], preds)
rmse = np.sqrt(mean_squared_error(df[target], preds))
mape = mean_absolute_percentage_error(df[target], preds)
accuracy = 100 - mape*100

print("\n\u2714 Evaluation Results:")
print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)
print("Accuracy:", accuracy, "%")
