import pandas as pd
import numpy as np
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# --- Initialize Page ---
st.set_page_config(page_title="Taxi Analytics Suite", layout="wide")

# --- Data Loading & Feature Engineering ---
@st.cache_data
def load_data():
    df = pd.read_parquet("yellow_tripdata_2025-01.parquet")
    
    # Convert datetime features
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day'] = df['tpep_pickup_datetime'].dt.day
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Calculate derived metrics
    df['fare_per_mile'] = df['fare_amount'] / df['trip_distance']
    df['speed_mph'] = df['trip_distance'] / (df['trip_duration']/60)
    
    # --- Anomaly Detection ---
    # Price anomalies using IQR
    Q1 = df['fare_amount'].quantile(0.25)
    Q3 = df['fare_amount'].quantile(0.75)
    df['price_anomaly'] = (df['fare_amount'] < (Q1 - 1.5*(Q3-Q1))) | (df['fare_amount'] > (Q3 + 1.5*(Q3-Q1)))
    
    # Distance anomalies using Z-score
    df['distance_zscore'] = zscore(df['trip_distance'])
    df['distance_anomaly'] = abs(df['distance_zscore']) > 3
    
    return df

df = load_data()

# --- Model Training & Saving ---
# Train and save duration model
X_duration = df[['trip_distance', 'hour']]
y_duration = df['trip_duration']
duration_model = LinearRegression().fit(X_duration, y_duration)
joblib.dump(duration_model, "duration_model.pkl")

# --- Model Loading ---
@st.cache_resource
def load_models():
    return {
        "duration_model": joblib.load("duration_model.pkl"),
        "price_model": RandomForestRegressor().fit(df[['trip_distance', 'hour']], df['fare_amount'])
    }

models = load_models()

# --- UI Pages ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Anomaly Detection", "Predictions"])

if page == "Dashboard":
    st.title("📊 Taxi Metrics Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fare Distribution")
        fig, ax = plt.subplots()
        sns.boxplot(x=df['fare_amount'])
        st.pyplot(fig)
        
        st.subheader("Hourly Demand Heatmap")
        demand_pivot = df.pivot_table(index='hour', columns='day', values='VendorID', aggfunc='count')
        plt.figure(figsize=(12,6))
        sns.heatmap(demand_pivot, cmap='viridis')
        st.pyplot(plt.gcf())
    
    with col2:
        st.subheader("Distance vs Fare")
        plt.figure(figsize=(8,5))
        sns.scatterplot(x='trip_distance', y='fare_amount', data=df)
        st.pyplot(plt.gcf())
        
        st.subheader("Anomaly Summary")
        anomalies = pd.DataFrame({
            "Type": ["Price", "Distance"],
            "Count": [df['price_anomaly'].sum(), df['distance_anomaly'].sum()]
        })
        st.bar_chart(anomalies.set_index('Type'))

elif page == "Anomaly Detection":
    st.title("🚨 Anomaly Detection Center")
    
    anomaly_type = st.selectbox("Select Anomaly Type", ["Price", "Distance"])
    
    if anomaly_type == "Price":
        st.subheader("Price Anomalies")
        threshold = st.slider("IQR Threshold", 1.0, 3.0, 1.5)
        Q1 = df['fare_amount'].quantile(0.25)
        Q3 = df['fare_amount'].quantile(0.75)
        anomalies = df[(df['fare_amount'] < (Q1 - threshold*(Q3-Q1))) | 
                      (df['fare_amount'] > (Q3 + threshold*(Q3-Q1)))]
        st.dataframe(anomalies[['fare_amount', 'trip_distance', 'hour']].head(10))
    
    elif anomaly_type == "Distance":
        st.subheader("Distance Anomalies")
        anomalies = df[df['distance_anomaly']]
        st.dataframe(anomalies[['trip_distance', 'fare_amount', 'hour']].head(10))

elif page == "Predictions":
    st.title("🔮 Ride Predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Fare Prediction")
        distance = st.number_input("Distance (miles)", min_value=0.1, value=2.5)
        hour = st.slider("Hour", 0, 23, 12)
        if st.button("Predict Fare"):
            prediction = models['price_model'].predict([[distance, hour]])
            st.success(f"Predicted Fare: ${prediction[0]:.2f}")
    
    with col2:
        st.subheader("Duration Prediction")
        if st.button("Predict Duration"):
            prediction = models['duration_model'].predict([[distance, hour]])
            st.success(f"Predicted Duration: {prediction[0]:.1f} minutes")
