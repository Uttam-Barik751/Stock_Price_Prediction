import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

st.title("📈 Stock Price Prediction Bot")

st.subheader("Enter a stock ticker and get a prediction for the next closing price.")


ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, GOOG):", "AAPL")

if st.button("Predict"):
    st.write(f"Fetching data for **{ticker.upper()}**...")


    data = yf.download(ticker, period="60d")
    
    if data.empty:
        st.error("Failed to load data. Please check the ticker symbol.")
    else:
        
        st.subheader("Recent Closing Prices")
        st.line_chart(data['Close'])

        df = data[['Close']].copy()
        df['Target'] = df['Close'].shift(-1)
        df.dropna(inplace=True)

        X = df[['Close']]
        y = df['Target']

        model = LinearRegression()
        model.fit(X, y)

        last_price = df[['Close']].iloc[-1]
        predicted_price = model.predict(last_price.values.reshape(1, -1))[0]

        st.success(f"Predicted next closing price: ${predicted_price:.2f}")

       
        age = st.slider("Select your age", 1, 100, 25)
        st.write("Your age is:", age)

        if st.checkbox("Show secret tip"):
            st.info("Tip: Never trade based on bots. Use them to learn, not to earn.")


st.caption("Built with computer by Streamlit")
