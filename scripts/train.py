import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
df = pd.read_csv("walmart_pp.csv")

# Feature Engineering
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

# Lag features
for lag in [1, 2, 3, 4, 7, 14, 28]:
    df[f"Sales_Lag_{lag}"] = df.groupby('Store')['Weekly_Sales'].shift(lag)

# Rolling features
for window in [3, 7, 14, 28]:
    df[f"Rolling_Mean_{window}"] = df.groupby('Store')['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(window=window).mean())
    df[f"Rolling_Std_{window}"] = df.groupby('Store')['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(window=window).std())

# Drop rows with NA due to lags
df = df.dropna().reset_index(drop=True)

# Encode Store
le = LabelEncoder()
df['Store'] = le.fit_transform(df['Store'])

# Scale continuous features
scaler = StandardScaler()
cont_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment'] + \
                [f"Sales_Lag_{lag}" for lag in [1, 2, 3, 4, 7, 14, 28]] + \
                [f"Rolling_Mean_{window}" for window in [3, 7, 14, 28]] + \
                [f"Rolling_Std_{window}" for window in [3, 7, 14, 28]]

df[cont_features] = scaler.fit_transform(df[cont_features])

# Train-test split (time-based)
split_date = df['Date'].quantile(0.8)
train = df[df['Date'] <= split_date]
test = df[df['Date'] > split_date]

features = ['Store', 'Holiday_Flag'] + cont_features
target = 'Weekly_Sales'

# Hyperparameter tuning
def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 300, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5)
    }
    model = xgb.XGBRegressor(**param, random_state=42, tree_method='hist')
    model.fit(train[features], train[target])
    preds = model.predict(test[features])
    mape = mean_absolute_percentage_error(test[target], preds)
    return mape

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

best_params = study.best_params

# Final model training
final_model = xgb.XGBRegressor(**best_params, random_state=42, tree_method='hist')
final_model.fit(train[features], train[target])

# Evaluate
preds = final_model.predict(test[features])
mae = mean_absolute_error(test[target], preds)
rmse = np.sqrt(mean_squared_error(test[target], preds))
mape = mean_absolute_percentage_error(test[target], preds)
accuracy = 100 - mape * 100

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.4f}")
print(f"Accuracy: {accuracy:.2f}%")

# Save model
joblib.dump(final_model, "xgb_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(le, "label_encoder.joblib")
