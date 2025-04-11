import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load saved model
rf_model = joblib.load("random_forest_model.pkl")  # Ensure the model file exists

# Load test data from CSV
X_test = pd.read_csv("x_test.csv")  # Load features
Y_test = pd.read_csv("y_test.csv")  # Load target values

# Convert to numpy arrays
X_test = X_test.values
Y_test = Y_test.values.flatten()  # Ensure it's a 1D array

# Predict using the model
Y_pred = rf_model.predict(X_test)

# Streamlit App
st.title("📈 XRP Crypto Futures Price Prediction")

## **1️⃣ Introduction**
st.markdown("""
### **Project Overview**
This project aims to predict the **future prices of XRP cryptocurrency** based on past trends as seen on CoinGecko, sentiment scores of the reddit posts, and moving averages. Using **Random Forest Regression**, we analyze market data and attempt to forecast prices to help traders make informed decisions for Futures Trading.
""")

## **2️⃣ About XRP & Futures Trading**
st.markdown("""
### **What is XRP?**
XRP is a cryptocurrency designed for fast and low-cost cross-border transactions, developed by Ripple Labs. It is widely used in remittances and institutional payments.

### **What is Futures Trading?**
Futures trading allows investors to speculate on the future price of an asset. Unlike spot trading, futures contracts let traders **buy or sell at a predetermined price**, which can lead to higher profits or losses.
""")

## **3️⃣ Methodology**
st.markdown("""
### **🔍 Methodology**
1. **Data Collection**: Gather historical XRP prices from CoinGecko, and sentiment scores of Reddit posts.
2. **Data Preprocessing**: Scale data and generate sequence data.
3. **Model Training**: Train a **Random Forest Regression** model on historical trends.
4. **Evaluation**: Compare predictions with actual prices and analyze model performance.
""")

## **4️⃣ Predicted vs Actual Prices**
st.subheader("📉 Predicted vs. Actual Prices")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(Y_test, label="Actual Prices", marker="o", linestyle="-")
ax.plot(Y_pred, label="Predicted Prices", marker="x", linestyle="--")
ax.set_title("📈 Actual vs Predicted XRP Prices")
ax.set_xlabel("Time")
ax.set_ylabel("Price (USD)")
ax.legend()
ax.grid()
st.pyplot(fig)

## **5️⃣ Model Evaluation**
st.title("📊 Model Performance Evaluation")

# Compute model evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_pred)

# Optimal Scores (Benchmarking)
optimal_mae = 0  # Should be closer to 0
optimal_mse = 0   # Should be closer to 0
optimal_rmse = 0  # Should be closer to 0
optimal_r2 = 1    # Should be closer to 1

## **1️⃣ Explanation of Evaluation Metrics**
st.markdown("""
### **🔹 Understanding Model Evaluation Metrics**
1. **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values. Lower is better.
2. **Mean Squared Error (MSE)**: Similar to MAE but squares the differences, penalizing larger errors.
3. **Root Mean Squared Error (RMSE)**: The square root of MSE, keeping units consistent with the target variable.
4. **R-Squared (R²) Score**: Measures how well the model explains the variance in data. Higher is better.
""")

## **2️⃣ Model Performance Comparison**
st.subheader("📈 Model vs. Optimal Performance")

st.markdown(f"""
- **Mean Absolute Error (MAE)**: `{mae:.4f}`
- **Mean Squared Error (MSE)**: `{mse:.4f}` 
- **Root Mean Squared Error (RMSE)**: `{rmse:.4f}` 
- **R-Squared (R²) Score**: `{r2:.4f}`
""")

## **6️⃣ Conclusion**
st.markdown("""
### **🏆 Conclusion**
While the model can predict XRP price trends, its accuracy is currently limited. The **negative R² score** suggests that the model does not explain variance well, and improvements in **feature engineering** or using an **LSTM model** might yield better results.
""")