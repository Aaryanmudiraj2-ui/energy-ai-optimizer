import streamlit as st

import pandas as pd

from model import detect_anomalies

st.set_page_config(page_title="AI Energy Optimizer", layout="wide")

st.subheader("💰 Estimated Savings")

st.title("⚡ AI Energy Optimizer")

df, anomalies = detect_anomalies()

st.subheader("📊 Full Data")
st.write(df)

st.subheader("🚨 Energy Waste / Anomalies")
st.write(anomalies)

if not anomalies.empty:
    st.warning("⚠ High energy usage detected! Optimize cooling or load balancing.")
else:
    st.success("✅ System running efficiently!")

st.subheader("💰 Estimated Savings")

if not anomalies.empty:
    waste = anomalies['power_usage'].sum()
    savings = waste * 0.1  # assume 10% saving
    st.write(f"Estimated Energy Waste: {waste}")
    st.write(f"Potential Savings: ₹{savings}")

import matplotlib.pyplot as plt

st.subheader("📈 Power Usage Graph")

# --- Yeh wala section rakho aur update karo ---
fig, ax = plt.subplots(figsize=(10,5))

# Normal Data (Blue Line)
ax.plot(df['timestamp'], df['power_usage'], marker='o', label="Normal", color='#1f77b4')

# Highlight Anomalies (Red Dots) - Yeh line dhyan se dekhna
ax.scatter(anomalies['timestamp'], anomalies['power_usage'], color='red', label="Anomaly", s=150, zorder=5)

ax.set_title("AI Energy Analysis: Power Usage vs Time")
ax.set_xlabel("Time (Hours)")
ax.set_ylabel("Power Usage (kW)")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

st.pyplot(fig)

st.subheader("💰 Estimated Savings")

if not anomalies.empty:
    waste = anomalies['power_usage'].sum()
    savings = waste * 0.1  # assume 10% saving
    st.write(f"Estimated Energy Waste: {waste}")
    st.write(f"Potential Savings: ₹{savings}")

import time
import numpy as np

st.subheader("⏱ Real-Time Simulation")

placeholder = st.empty()

for i in range(20):
    new_load = np.random.randint(50, 100)
    new_temp = np.random.randint(20, 40)
    new_power = new_load * 2 + new_temp

    new_data = {
        "timestamp": len(df) + i,
        "server_load": new_load,
        "temperature": new_temp,
        "power_usage": new_power
    }

    df.loc[len(df)] = new_data

    placeholder.line_chart(df['power_usage'])

    time.sleep(1)

    # --- ML Prediction Section ---
from sklearn.linear_model import LinearRegression
import numpy as np

st.subheader("🔮 Future Power Prediction")

# Model ko train karne ke liye data taiyar kar rahe hain
X = df[['server_load', 'temperature']] 
y = df['power_usage']

# Linear Regression model banao aur train karo
model = LinearRegression()
model.fit(X, y)

# Maan lo future mein server load 85% aur temperature 30°C hoga
future_data = pd.DataFrame([[85, 30]],
columns=['server_load','temperature'])
prediction = model.predict(future_data)

# Result ko screen par dikhao
st.metric(label="Predicted Future Power Usage", value=f"{prediction[0]:.2f} kW")
st.write(f"💡 AI Suggestion: If load reaches 85%, expect {prediction[0]:.2f} kW usage.")

cost_per_unit = 10  # ₹ per unit

st.subheader("💰 Cost Analysis")

total_cost = df['power_usage'].sum() * cost_per_unit

