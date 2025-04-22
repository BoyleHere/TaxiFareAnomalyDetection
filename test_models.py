import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

# Load models
scaler = joblib.load("scaler.pkl")
surge_model = joblib.load("surge_rf.pkl")
price_model = joblib.load("price_rf.pkl")
demand_model = joblib.load("demand_xgb.pkl")  # XGBRegressor model

# Define column names
surge_features = ['fare_amount', 'trip_distance', 'hour', 'day', 'month']
price_features = ['trip_distance', 'hour', 'month']
demand_features = ['hour', 'day', 'month', 'trip_distance']

# Sample test inputs as Pandas DataFrames
test_input = pd.DataFrame([[10.0, 2.0, 14, 10, 3]], columns=surge_features)
price_input = pd.DataFrame([[2.0, 14, 3]], columns=price_features)
demand_input = pd.DataFrame([[14, 10, 3, 2.0]], columns=demand_features)

# Scale input using StandardScaler
scaled_input = scaler.transform(test_input)

# Predict surge
surge_pred = surge_model.predict(scaled_input)
print(f"Surge Prediction: {'Yes' if surge_pred[0] == 1 else 'No'}")

# Predict price
price_pred = price_model.predict(price_input)
print(f"Predicted Price: ${price_pred[0]:.2f}")

demand_input_gpu = xgb.DMatrix(demand_input, device="cuda")

# Predict demand using GPU
demand_pred = demand_model.predict(demand_input_gpu)
print(f"Predicted Demand: {demand_pred[0]:.2f}")

# ✅ Fix: Directly pass the Pandas DataFrame (NOT xgb.DMatrix)
demand_pred = demand_model.predict(demand_input)
print(f"Predicted Demand: {demand_pred[0]:.2f}")
