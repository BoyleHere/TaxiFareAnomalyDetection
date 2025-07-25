# 🚕 FareMiner: Fare Prediction & Anomaly Detection in NYC Taxi Data

FareMiner is a machine learning pipeline designed to **predict taxi fares** and **detect anomalies** in ride data using supervised and unsupervised learning techniques. Built on the **NYC Taxi Fare Prediction dataset**, this project enhances fare transparency, identifies suspicious pricing patterns, and supports real-time analytics through an interactive **Streamlit dashboard**.

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Objectives](#objectives)
- [Methodology](#methodology)
- [Technologies Used](#technologies-used)
- [Experimental Results](#experimental-results)
- [Future Work](#future-work)
- [Authors](#authors)
- [License](#license)

---

## 📖 Introduction

With the surge of GPS-tracked ride data, urban transportation systems have the opportunity to utilize machine learning for **fare modeling** and **fraud detection**. FareMiner combines **regression models** for price prediction and **unsupervised anomaly detection** methods to build a smart, real-time fare analytics platform.

---

## 🌟 Features

- 🔍 Predict taxi fares using features like time, location, and demand.
- 🚨 Detect abnormal rides based on distance, fare, and time.
- 🧠 Models: XGBoost, Random Forest, Linear Regression, DBSCAN, Isolation Forest.
- 📈 Interactive visualizations: heatmaps, anomaly clusters, fare trends.
- 🖥️ Real-time dashboard via **Streamlit**.

---

## 🎯 Objectives

- Analyze and preprocess NYC Taxi Fare data from Kaggle.
- Engineer temporal, spatial, and demand-supply features.
- Train and evaluate regression models for fare prediction.
- Use **Z-Score**, **IQR**, **DBSCAN**, and **Isolation Forest** for anomaly detection.
- Deploy a real-time **Streamlit UI** for end-user interaction and insights.

---

## ⚙️ Methodology

### 1. Data Gathering & Cleaning
- Imported from Kaggle.
- Removed invalid entries (e.g., negative fares, nulls).
- Normalized time and location values.

### 2. Feature Engineering
- **Temporal**: Hour, weekday, holiday, peak times.
- **Spatial**: Geohashed pickup/dropoff clusters.
- **Derived Metrics**: Fare/mile, surge ratio, idle time.
- **Demand-Supply**: Location-based ride frequency, idle gaps.

### 3. Fare Prediction
- 📉 Baseline: Linear Regression.
- 🌲 Advanced: XGBoost & Random Forest.
- 📊 Evaluation: RMSE, MAE.

### 4. Anomaly Detection
- 📈 Residual-based: Prediction errors → Isolation Forest.
- 🔍 Feature-based: Fare/mile, duration → Z-Score, DBSCAN.
- ✅ Hybrid Flagging: Mark rides as anomalous if flagged by both methods.

### 5. Streamlit Deployment
- Filters by cab type, location, time.
- Visuals for residual plots, anomaly maps, demand trends.

---

## 🧪 Experimental Results

- 📊 Dataset: NYC Taxi Fare (~1M cleaned rides from 55M+ entries).
- 📌 Anomaly Detection Accuracy: **~95.6% weighted F1-score**.
- 🧾 High precision (98.4%) for normal rides, 50% recall for anomalies.
- ✅ Strong correlation between distance and fare.
- 🚗 Demand peaks around 5–6 PM.

---

## 🚀 Future Work

- Integrate real-time traffic, weather, or event data.
- Deploy on live GPS-enabled ride streams.
- Incorporate **LSTM Autoencoders** for deep anomaly detection.
- Expand to Uber/Lyft and other cities.

---

## 👨‍💻 Authors

- **Archit Singh** – 220911464 – Dept. of ICT, MIT Manipal  
- **Prashast Saxena** – 220911536 – Dept. of ICT, MIT Manipal  
- **Arnav Aradhya** – 220911612 – Dept. of ICT, MIT Manipal  

---

## 📄 License

This project is part of an academic submission and is open for educational and research use only. Please cite appropriately.

---

## 📂 Dataset Source

[NYC Taxi Fare Prediction – Kaggle](https://www.kaggle.com/c/nyc-taxi-fare-prediction)
