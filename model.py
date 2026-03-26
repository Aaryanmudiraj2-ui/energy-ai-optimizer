import pandas as pd
from sklearn.ensemble import IsolationForest

def load_data():
    df = pd.read_csv("data.csv")
    return df

def train_model(df):
    model = IsolationForest(contamination=0.2)
    df['anomaly'] = model.fit_predict(df[['server_load','temperature','power_usage']])
    return df

def detect_anomalies():
    df = load_data()
    df = train_model(df)
    anomalies = df[df['anomaly'] == -1]
    return df, anomalies