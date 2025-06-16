import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="📈 Stock Analyzer India", layout="wide")
st.title("📊 Indian Stock Analyzer using ML (Random Forest + KMeans)")
st.write("🔍 Enter a stock symbol (e.g., RELIANCE.NS, TCS.NS) to analyze 6-month price trends, predict price movement, and view clustering.")

# ---- Input ----
stock = st.text_input("Enter NSE Stock Symbol (e.g., RELIANCE.NS)", value="RELIANCE.NS")

# ---- Load Data ----
try:
    data = yf.download(stock, period="6mo", interval="1d")
    if data.empty:
        st.error("❌ Invalid stock symbol or no data found.")
    else:
        st.success("✅ Data loaded successfully!")
        
        # ---- Feature Engineering ----
        df = data.copy()
        df['Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Return'].rolling(window=5).std()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)

        st.subheader("📌 Latest Stock Data")
        st.dataframe(df.tail())

        # ---- Model Training ----
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Volatility']
        X = df[features]
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        st.subheader("🎯 Price Movement Prediction (Random Forest)")
        st.write(f"**Model Accuracy**: `{acc*100:.2f}%`")
        st.write("✅ 1 = Price Up, ❌ 0 = Price Down")

        # ---- Tomorrow's Prediction ----
        latest_input = X.tail(1)
        prediction = model.predict(latest_input)[0]
        st.subheader("📈 Tomorrow's Predicted Movement")
        st.info("🔼 Upward" if prediction == 1 else "🔽 Downward")

        # ---- Confusion Matrix Plot ----
        st.subheader("📊 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        # ---- KMeans Clustering ----
        st.subheader("🔍 Clustering Based on Stock Features")
        kmeans = KMeans(n_clusters=3, random_state=0)
        df['Cluster'] = kmeans.fit_predict(X)

        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x='Return', y='Volatility', hue='Cluster', palette='Set2', ax=ax2)
        plt.title("KMeans Clustering of Return vs Volatility")
        st.pyplot(fig2)

        st.success("✅ Clustered Stock Data")
        st.dataframe(df[['Close', 'Return', 'Volatility', 'Cluster']].tail())

except Exception as e:
    st.error(f"⚠️ Error: {e}")
