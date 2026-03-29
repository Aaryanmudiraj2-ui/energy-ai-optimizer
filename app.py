import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# 🔥 1. TITLE (SABSE UPAR)
st.title("⚡ AI Energy Optimizer")
st.caption("AI-powered energy optimization for data centers & industries")

# 🔥 2. FILE UPLOAD (TITLE KE BAAD)
uploaded_file = st.file_uploader("📂 Upload Your Energy Data (CSV)", type=["csv", "txt"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload your data to continue")
    st.stop()

# 🔥 3. DATA SHOW (OPTIONAL)
st.subheader("📊 Data Preview")
st.write(df.head())

# 🔥 4. FAKE ANOMALY LOGIC (YA TERA EXISTING LOGIC)
anomalies = df[df['power_usage'] > df['power_usage'].mean()]

# 🔥 5. KPI DASHBOARD
col1, col2, col3 = st.columns(3)

col1.metric("Total Usage", f"{df['power_usage'].sum():.2f} kW")
col2.metric("Anomalies", len(anomalies))
col3.metric("Estimated Savings", f"₹{anomalies['power_usage'].sum()*0.1:.2f}")

# 🔥 6. GRAPH
st.subheader("📈 Energy Usage Graph")
st.line_chart(df['power_usage'])

# 🔥 7. AI INSIGHTS
st.subheader("🧠 AI Insights")

for index, row in anomalies.iterrows():
    st.write(f"⚠ High usage at {row['timestamp']} → Possible inefficiency")

# 🔥 8. ALERT
if not anomalies.empty:
    st.error("🚨 Immediate Action Required!")

# 🔥 9. DOWNLOAD REPORT
csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Report",
    data=csv,
    file_name='energy_report.csv',
    mime='text/csv',
)


# ---------------- SAVINGS ----------------
st.subheader("💰 Estimated Savings")

if not anomalies.empty:
    waste = anomalies['power_usage'].sum()
    savings = waste * 0.1
    st.write(f"Estimated Energy Waste: {waste}")
    st.write(f"Potential Savings: ₹{savings}")

# ---------------- PREDICTION ----------------
st.subheader("🔮 Future Prediction")

 # --- PREDICTION LOGIC ---
if 'voltage' in df.columns and 'temperature' in df.columns:
    # Humne server_load ki jagah voltage use kiya hai
    X = df[['voltage', 'temperature']] 
    y = df['power_usage']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Prediction (Next prediction ke liye voltage 230 aur temp 30 maana hai)
    future = pd.DataFrame([[230, 30]], columns=['voltage', 'temperature'])
    prediction = model.predict(future)
    st.write(f"Predicted Power Usage: {prediction[0]:.2f} units")
else:
    st.error("❌ Error: CSV file mein columns match nahi ho rahe!")
    st.info(f"Aapki file mein ye columns hain: {list(df.columns)}")

# ---------------- COST ----------------
st.subheader("💰 Cost Analysis")

cost_per_unit = 10
total_cost = df['power_usage'].sum() * cost_per_unit

st.write(f"Total Energy Cost: ₹{total_cost}")
