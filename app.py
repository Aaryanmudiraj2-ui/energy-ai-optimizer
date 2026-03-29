
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import detect_anomalies
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="AI Energy Optimizer", layout="wide")

st.title("⚡ AI Energy Optimizer")

# Load data
df, anomalies = detect_anomalies()

# ---------------- DATA ----------------
st.subheader("📊 Full Data")
st.write(df)

st.subheader("🚨 Energy Waste / Anomalies")
st.write(anomalies)

if not anomalies.empty:
    st.warning("⚠ High energy usage detected!")
else:
    st.success("✅ System running efficiently!")

# ---------------- GRAPH ----------------
st.subheader("📈 Power Usage Graph")

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(df['timestamp'], df['power_usage'], marker='o', label="Normal")

if not anomalies.empty:
    ax.scatter(anomalies['timestamp'], anomalies['power_usage'], label="Anomaly")

ax.set_title("Power Usage vs Time")
ax.set_xlabel("Time")
ax.set_ylabel("Power Usage")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# ---------------- SAVINGS ----------------
st.subheader("💰 Estimated Savings")

if not anomalies.empty:
    waste = anomalies['power_usage'].sum()
    savings = waste * 0.1
    st.write(f"Estimated Energy Waste: {waste}")
    st.write(f"Potential Savings: ₹{savings}")

# ---------------- PREDICTION ----------------
st.subheader("🔮 Future Prediction")

X = df[['server_load', 'temperature']]
y = df['power_usage']

model = LinearRegression()
model.fit(X, y)

future = pd.DataFrame([[85, 30]], columns=['server_load','temperature'])
prediction = model.predict(future)

st.metric("Predicted Power Usage", f"{prediction[0]:.2f} kW")

# ---------------- COST ----------------
st.subheader("💰 Cost Analysis")

cost_per_unit = 10
total_cost = df['power_usage'].sum() * cost_per_unit

st.write(f"Total Energy Cost: ₹{total_cost}")