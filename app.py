import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score

# Set Streamlit page config (MUST be first)
st.set_page_config(page_title="Ride Prediction App", layout="wide")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_parquet("yellow_tripdata_2025-01.parquet", engine="pyarrow")
    
    # Convert datetime columns
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    # Feature Engineering
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day'] = df['tpep_pickup_datetime'].dt.day
    df['month'] = df['tpep_pickup_datetime'].dt.month
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    return df

df = load_data()

# Load models
@st.cache_resource
def load_models():
    models = {
        "scaler": joblib.load("scaler.pkl"),
        "surge_rf": joblib.load("surge_rf.pkl"),
        "price_rf": joblib.load("price_rf.pkl"),
        "demand_xgb": joblib.load("demand_xgb.pkl"),
    }
    return models

models = load_models()
scaler = models["scaler"]
rf_model = models["surge_rf"]
price_model = models["price_rf"]
demand_model = models["demand_xgb"]

# Streamlit UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Surge & Price Prediction", "Data Visualization"])

if page == "Surge & Price Prediction":
    st.title("🚖 Surge Prediction & Ride Price Estimator")
    
    fare_amount = st.number_input("Fare Amount", min_value=0.0)
    trip_distance = st.number_input("Trip Distance (miles)", min_value=0.0)
    hour = st.slider("Hour of the day", 0, 23, 12)
    day = st.slider("Day of the month", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)
    
    if st.button("Predict Surge"):
        input_data = np.array([[fare_amount, trip_distance, hour, day, month]])
        input_data = scaler.transform(input_data)
        y_probs = rf_model.predict_proba(input_data)[:, 1]
        prediction = (y_probs >= 0.5).astype(int)
        surge = "Yes" if prediction[0] == 1 else "No"
        st.write(f"Surge Pricing? {surge}")
    
    if st.button("Predict Price"):
        user_input_price = np.array([[trip_distance, hour, month]])
        predicted_price = price_model.predict(user_input_price)
        st.success(f"Estimated Ride Price: ${predicted_price[0]:.2f} 💰")

if page == "Data Visualization":
    st.title("📊 Data Visualization")

    # Check if trip_distance and fare_amount have valid values
    valid_data = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0)]

    # Remove outliers using percentiles
    fare_upper_limit = valid_data["fare_amount"].quantile(0.99)  # 99th percentile
    distance_upper_limit = valid_data["trip_distance"].quantile(0.99)  # 99th percentile
    filtered_data = valid_data[(valid_data["fare_amount"] <= fare_upper_limit) & 
                               (valid_data["trip_distance"] <= distance_upper_limit)]

    # Compute median fare amount
    median_fare = filtered_data["fare_amount"].median()

    # Scatter Plot: Trip Distance vs. Fare Amount
    st.subheader("Trip Distance vs. Fare Amount")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=filtered_data["trip_distance"], y=filtered_data["fare_amount"], alpha=0.5, ax=ax)

    # Add median fare as a horizontal line
    ax.axhline(median_fare, color="red", linestyle="dashed", label=f"Median Fare: ${median_fare:.2f}")

    # Dynamic Axis Scaling
    ax.set_xlim(0, filtered_data["trip_distance"].max())  # Set X-axis range dynamically
    ax.set_ylim(0, filtered_data["fare_amount"].max())  # Set Y-axis range dynamically

    # Labels & Title
    ax.set_xlabel("Trip Distance (miles)")
    ax.set_ylabel("Fare Amount ($)")
    ax.set_title("Trip Distance vs. Fare Amount")
    ax.legend()  # Show legend with median value

    st.pyplot(fig)


    # Line Chart: Ride Demand Over Time
    st.subheader("Ride Demand Over Time")
    ride_counts = df["hour"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=ride_counts.index, y=ride_counts.values, marker="o", ax=ax)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Rides")
    ax.set_title("Ride Demand Throughout the Day")
    st.pyplot(fig)

    # Correlation Heatmap
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Feature Correlation Heatmap")

        # Select only numerical columns
        numeric_cols = df.select_dtypes(include=["number"]).copy()

        # Remove outliers using percentiles (99th percentile)
        for col in numeric_cols.columns:
            upper_limit = numeric_cols[col].quantile(0.99)  # 99th percentile
            numeric_cols = numeric_cols[numeric_cols[col] <= upper_limit]

        # Drop rows with NaN values (if any)
        numeric_cols.dropna(inplace=True)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")

        st.pyplot(fig)

# if st.sidebar.checkbox("Show Model Performance"):
#     st.subheader("Model Performance")

#     # Evaluate Surge Model
#     surge_pred_test = rf_model.predict(X_test)
#     surge_accuracy = accuracy_score(y_test, surge_pred_test)

#     # Evaluate Price Model
#     price_score = price_model.score(X_test_price, y_test_price)

#     # Evaluate Demand Model
#     demand_score = demand_model.score(X_test_d, y_test_d)

#     st.write(f"🔹 **Surge Model Accuracy:** {surge_accuracy:.2%}")
#     st.write(f"🔹 **Price Model Score (R²):** {price_score:.2f}")
#     st.write(f"🔹 **Demand Model Score (R²):** {demand_score:.2f}")


