import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Load data
df = pd.read_csv('Walmart.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

# Feature Engineering
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)

# Lag features
df['Sales_Lag_1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
df['Sales_Lag_4'] = df.groupby('Store')['Weekly_Sales'].shift(4)
df['Rolling_Mean_4'] = df.groupby('Store')['Weekly_Sales'].transform(lambda x: x.shift(1).rolling(window=4).mean())

# Drop rows with any missing values (due to lag features)
df = df.dropna().reset_index(drop=True)

# Filter out stores with insufficient training data
min_required = 20  # you can adjust this threshold
store_counts = df['Store'].value_counts()
valid_stores = store_counts[store_counts >= min_required].index
df = df[df['Store'].isin(valid_stores)].reset_index(drop=True)

# Label Encode Store
le = LabelEncoder()
df['Store'] = le.fit_transform(df['Store'])

# Scale continuous features
scaler = StandardScaler()
cont_features = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Sales_Lag_1', 'Sales_Lag_4', 'Rolling_Mean_4']
df[cont_features] = scaler.fit_transform(df[cont_features])

# Save processed data
df.to_csv('walmart_pp.csv', index=False)

print("✅ Data preprocessing complete.")
