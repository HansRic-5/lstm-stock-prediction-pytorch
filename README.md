# Stock Price Prediction using LSTM (PyTorch)

A deep learning project implementing Long Short-Term Memory (LSTM) neural networks to forecast future closing prices for Advanced Micro Devices (AMD) and Apple Inc. (AAPL) stocks.

---

## 📌 Project Overview

Stock market forecasting is a challenging task due to high volatility and non-linear temporal dependencies. This project builds a robust time-series forecasting pipeline using PyTorch, comparing a standard baseline LSTM against an optimized LSTM architecture enhanced with regularization and learning rate scheduling.

### Key Objectives
- **Data Preprocessing & Sliding Windows**: Load daily historical stock data, handle chronological splitting, scale close prices via `MinMaxScaler`, and structure sequences into 5-day sliding windows (`window_size=5`, `horizon=1`).
- **Baseline vs. Optimized Architectures**: Design, train, and compare a 50-unit baseline LSTM with a regularized 32-unit optimized LSTM model.
- **Robust Training Pipeline**: Implement `ReduceLROnPlateau` scheduling, gradient clipping, and Early Stopping to prevent overfitting.
- **Comprehensive Evaluation**: Assess prediction accuracy on a 1-year test set using **RMSE**, **MAE**, and **MAPE (%)**.

---

## 🛠️ Tech Stack & Libraries

- **Language**: Python
- **Deep Learning**: PyTorch (`torch.nn`, `torch.optim`)
- **Data Manipulation**: Pandas, NumPy
- **Scaling & Metrics**: Scikit-Learn (`MinMaxScaler`, Mean Squared Error, Mean Absolute Error)
- **Visualization**: Matplotlib, Seaborn
