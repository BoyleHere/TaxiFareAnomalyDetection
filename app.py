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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pydeck as pdk

# Set Streamlit page config
st.set_page_config(page_title="Ride Prediction App", layout="wide")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_parquet("yellow_tripdata_2025-01.parquet", engine="pyarrow")
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day'] = df['tpep_pickup_datetime'].dt.day
    df['month'] = df['tpep_pickup_datetime'].dt.month
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
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

st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Surge & Price Prediction", "Data Visualization", "Model Accuracy & Map"])

if page == "Surge & Price Prediction":
    st.title("🚖 Surge Prediction & Ride Price Estimator")

    fare_amount = st.number_input("Fare Amount", min_value=0.0)
    trip_distance = st.number_input("Trip Distance (miles)", min_value=0.0)
    hour = st.slider("Hour of the day", 0, 23, 12)
    day = st.slider("Day of the month", 1, 31, 15)
    month = st.slider("Month", 1, 12, 6)

    if st.button("Predict Surge"):
        input_data = np.array([[fare_amount, trip_distance, hour, day, month]])
        input_data_scaled = scaler.transform(input_data)
        y_probs = rf_model.predict_proba(input_data_scaled)[:, 1]
        prediction = (y_probs >= 0.5).astype(int)
        surge = "Yes" if prediction[0] == 1 else "No"
        st.write(f"Surge Pricing? {surge}")
        st.write(f"Probability of Surge: {y_probs[0]:.2f}")

    if st.button("Predict Price"):
        user_input_price = np.array([[trip_distance, hour, month]])
        predicted_price = price_model.predict(user_input_price)
        st.success(f"Estimated Ride Price: ${predicted_price[0]:.2f} 💰")

if page == "Data Visualization":
    st.title("📊 Data Visualization")
    valid_data = df[(df["trip_distance"] > 0) & (df["fare_amount"] > 0)]
    fare_upper_limit = valid_data["fare_amount"].quantile(0.99)
    distance_upper_limit = valid_data["trip_distance"].quantile(0.99)
    filtered_data = valid_data[(valid_data["fare_amount"] <= fare_upper_limit) & 
                               (valid_data["trip_distance"] <= distance_upper_limit)]
    median_fare = filtered_data["fare_amount"].median()

    st.subheader("Trip Distance vs. Fare Amount")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x=filtered_data["trip_distance"], y=filtered_data["fare_amount"], alpha=0.5, ax=ax)
    ax.axhline(median_fare, color="red", linestyle="dashed", label=f"Median Fare: ${median_fare:.2f}")
    ax.set_xlim(0, filtered_data["trip_distance"].max())
    ax.set_ylim(0, filtered_data["fare_amount"].max())
    ax.set_xlabel("Trip Distance (miles)")
    ax.set_ylabel("Fare Amount ($)")
    ax.set_title("Trip Distance vs. Fare Amount")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Ride Demand Over Time")
    ride_counts = df["hour"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(x=ride_counts.index, y=ride_counts.values, marker="o", ax=ax)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Rides")
    ax.set_title("Ride Demand Throughout the Day")
    st.pyplot(fig)

    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Feature Correlation Heatmap")
        numeric_cols = df.select_dtypes(include=["number"]).copy()
        for col in numeric_cols.columns:
            upper_limit = numeric_cols[col].quantile(0.99)
            numeric_cols = numeric_cols[numeric_cols[col] <= upper_limit]
        numeric_cols.dropna(inplace=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        st.pyplot(fig)

    st.subheader("Additional Insights")
    col1, col2 = st.columns(2)
    with col1:
        avg_duration = df["trip_duration"].mean()
        st.metric(label="Average Trip Duration (min)", value=f"{avg_duration:.2f}")
    with col2:
        avg_fare = df["fare_amount"].mean()
        st.metric(label="Average Fare Amount ($)", value=f"{avg_fare:.2f}")

if page == "Model Accuracy & Map":
    st.title("📍 Interactive Map & Model Accuracy Checker")

    st.subheader("Model Accuracy")
    features = df[["fare_amount", "trip_distance", "hour", "day", "month"]]
    labels = (df["fare_amount"] > df["fare_amount"].median()).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    X_train_scaled = scaler.transform(X_train)
    y_pred = rf_model.predict(X_train_scaled)

    acc = accuracy_score(y_train, y_pred)
    st.metric("Training Accuracy", f"{acc:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_train, y_pred))

    st.subheader("Pickup Locations Map")
    if "pickup_longitude" in df.columns and "pickup_latitude" in df.columns:
        map_df = df[["pickup_longitude", "pickup_latitude"]].dropna().sample(n=1000)
        st.pydeck_chart(pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=map_df["pickup_latitude"].mean(),
                longitude=map_df["pickup_longitude"].mean(),
                zoom=10,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    "HexagonLayer",
                    data=map_df,
                    get_position='[pickup_longitude, pickup_latitude]',
                    radius=100,
                    elevation_scale=4,
                    elevation_range=[0, 1000],
                    pickable=True,
                    extruded=True,
                )
            ],
        ))
    else:
        st.warning("Pickup location columns not available in dataset.")
