# U.S. Oil Production Forecasting

This repository contains a complete forecasting pipeline to predict monthly U.S. crude oil production. The project combines classical time series models, machine learning, hybrid techniques, and monitoring tools to provide accurate short- and long-term forecasts.

## Project Objectives

- Generate accurate forecasts of U.S. crude oil production
- Support both short-term operational planning and long-term strategy
- Monitor model performance over time using drift detection and retrain the model
- Deliver insights via an interactive web application

## Key Features

- **Data Source**: EIA monthly crude oil production data - ([**Dataset Link**](https://www.eia.gov/dnav/pet/hist/LeafHandler.ashx?n=PET&s=MCRFPUS1&f=M)) 
- **Models Implemented**:
  - Short-Term: ETS, ARIMA + STL, SARIMA, Prophet, STL + LSTM
  - Long-Term: XGBoost, Hybrid STL + XGBoost, Prophet (log), STL + LSTM
- **Best Models**:
  - ETS for short-term forecasting (RMSE: 1.89%, MAPE: 1.43%)
  - Hybrid STL + XGBoost for long-term (RMSE: 5.14%, MAPE: 4.12%)
- **Evaluation Metrics**: RMSE, MAE, MAPE with weighted scoring
- **Drift Detection**:
  - Concept drift: Rolling RMSE and MAPE
  - Data drift: KS test on input feature distributions
  - Retrain the model the best model whenever needed
- **Monitoring & Logging**: Weights & Biases (W&B)
- **Deployment**: Streamlit app for forecast visualization

## Project Structure

- `data/` – Raw and processed data
- `notebooks/` – Jupyter notebooks for EDA, modeling, evaluation
- `models/` – Saved model artifacts
- `app/` – Streamlit app source code
- `requirements.txt` – Python dependencies
