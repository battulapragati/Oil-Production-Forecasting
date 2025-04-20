import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from datetime import datetime, date, timedelta
import os
import json
import pickle
import base64
import joblib
import altair as alt
from scipy.stats import norm
from PIL import Image
import xgboost as xgb

st.set_page_config(
    page_title = "Oil Production Forecasting",
    page_icon="Oil Production.png",
    layout="wide",
    initial_sidebar_state="auto"
)

current_date = date.today()

st.sidebar.title("Page Options")
options_button = st.sidebar.radio("Select a page from below", options=("Business Understanding", "Short-Term Forecast", "Long-Term Forecast", "About"), index=1)

if options_button == "Business Understanding":
    img = Image.open('Oil Prod.png')
    st.image(img, caption='Oil Production Forecasting - USA', use_column_width=True)
    st.markdown("""
        ### **Business Problem:**
            Uncontrolled production changes can lead to price volatility, affecting economies worldwide. To stabilize oil prices and curb inflation, 
            policymakers and oil producers need accurate monthly forecasts of U.S. oil production.
        """, unsafe_allow_html=True)

def calculate_conf_intervals(forecast, residuals, confidence=0.95):
    z_score = norm.ppf(0.5 + confidence / 2)  # Z-score for confidence level
    std_err = np.std(residuals)
    lower_bound = round(forecast - z_score * std_err)
    upper_bound = round(forecast + z_score * std_err)
    return lower_bound, upper_bound

# Function to generate forecast
def generate_forecast(model, periods):
    forecast = model.forecast(periods)
    conf_int = model.conf_int(alpha=0.05)  # Assuming 95% confidence interval
    lower_bound = conf_int[:, 0]
    upper_bound = conf_int[:, 1]

    return forecast, lower_bound, upper_bound

# --------------------------------Short-Term Forecast----------------------------
if options_button == "Short-Term Forecast":
    # Load the short-term model
    model = joblib.load('best_shortterm_model_ets.pkl')
    st.title("Oil Production Short-Term Forecasting (in Thousand barrels)")
    months = st.radio("Select months to forecast", [1, 3, 6, 12], horizontal=True)
    confidence = st.radio("Select confidence level", [90, 95], horizontal=True)
  
    if st.button("Apply"):
        # Generate forecasts
        forecast = model.forecast(months)
        
        # Calculate confidence intervals
        residuals = model.resid  # Access model residuals
        confidence_level = confidence / 100
        lower, upper = calculate_conf_intervals(forecast, residuals, confidence=confidence_level)
        
        # Round forecast and bounds to whole numbers
        forecast = np.round(forecast).astype(int)
        lower = np.round(lower).astype(int)
        upper = np.round(upper).astype(int)
        
        # Generate future dates starting from today
        today = datetime.now()
        future_dates = [today + timedelta(days=30 * i) for i in range(1, months + 1)]
        future_dates = [date.strftime('%Y-%m-%d') for date in future_dates]
        
        st.write(f"Forecast for {months} months with {confidence}% confidence level in thousand barrels")
        
        # Display forecast data
        forecast_table = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast,
            "Lower Bound": lower,
            "Upper Bound": upper
        })
        # Reset the index and drop it, display the table without the index
        forecast_table_reset = forecast_table.reset_index(drop=True)

        # Display the table without the index
        st.table(forecast_table_reset)
        
        # Visualize the forecast using Altair
        st.subheader("Short-term forecast chart with lower and upper bounds")
        forecast_chart = alt.Chart(forecast_table).mark_line(point=True).encode(
            x=alt.X('Date:T', 
            title='Date',
            axis=alt.Axis(
                format='%Y-%m-%d', 
                labelAngle=45,  
                tickCount='day',  
                tickMinStep=1,  
            )
        ),
            y=alt.Y('Forecast:Q', title='Production Forecast'),
            tooltip=[
                alt.Tooltip('Date:T', title='Date'),
                alt.Tooltip('Forecast:Q', title='Forecast'),
                alt.Tooltip('Lower Bound:Q', title='Lower Bound'),
                alt.Tooltip('Upper Bound:Q', title='Upper Bound')
            ]
        ).properties(
            width=700,
            height=400
        )
        
        # Add confidence interval as an area chart
        confidence_interval = alt.Chart(forecast_table).mark_area(opacity=0.3).encode(
            x='Date:T',
            y='Lower Bound:Q',
            y2='Upper Bound:Q',
            tooltip=[
                alt.Tooltip('Date:T', title='Date'),
                alt.Tooltip('Lower Bound:Q', title='Lower Bound'),
                alt.Tooltip('Upper Bound:Q', title='Upper Bound')
            ]
        )

        # Combine the line chart and confidence interval area
        combined_chart = confidence_interval + forecast_chart
        st.altair_chart(combined_chart, use_container_width=True)


# --------------------------------Long-Term Forecast----------------------------
elif options_button == "Long-Term Forecast":
    # Load the long-term model
    st.title("Oil Production Long-Term Forecasting")
    model = joblib.load('best_longterm_model_hybrid_xgb.pkl')
    years = st.radio("Select years to forecast", [1, 2, 3, 4, 5], horizontal=True)
    confidence = st.radio("Select confidence level", [90, 95], horizontal=True)
    if st.button("Apply"):
        # Generate forecasts
        forecast = 1.2 * model.forecast(years)
        # Calculate confidence intervals
        residuals = model.resid 
        confidence_level = confidence / 100
        lower, upper = calculate_conf_intervals(forecast, residuals, confidence=confidence_level)
        
        # Round forecast and bounds to whole numbers
        forecast = np.round(forecast).astype(int)
        lower = np.round(lower).astype(int)
        upper = np.round(upper).astype(int)
        
        # Generate future dates starting from today
        today = datetime.now()
        future_dates = [today + timedelta(days=365 * i) for i in range(1, years + 1)]
        future_dates = [date.strftime('%Y-%m-%d') for date in future_dates]
        
        st.write(f"Forecast for {years} years with {confidence}% confidence level")
        
        # Display forecast data
        forecast_table = pd.DataFrame({
            "Date": future_dates,
            "Forecast": forecast,
            "Lower Bound": lower,
            "Upper Bound": upper
        })
        # Reset the index and drop it, then display the table without the index
        forecast_table_reset = forecast_table.reset_index(drop=True)
         
        # Now display the table without the index
        st.table(forecast_table_reset)
        
        # Visualize the forecast using Altair
        st.subheader("Long-term forecast chart with lower and upper bounds")
        forecast_chart = alt.Chart(forecast_table).mark_line(point=True).encode(
            x=alt.X('Date:T', 
            title='Date',
            axis=alt.Axis(
                format='%Y-%m-%d',  
                labelAngle=45,  
                tickCount='day', 
                tickMinStep=1,  
            )
        ),
            y=alt.Y('Forecast:Q', title='Production Forecast'),
            tooltip=[
                alt.Tooltip('Date:T', title='Date'),
                alt.Tooltip('Forecast:Q', title='Forecast'),
                alt.Tooltip('Lower Bound:Q', title='Lower Bound'),
                alt.Tooltip('Upper Bound:Q', title='Upper Bound')
            ]
        ).properties(
            width=700,
            height=400
        )
        
        # Add confidence interval as an area chart
        confidence_interval = alt.Chart(forecast_table).mark_area(opacity=0.3).encode(
            x='Date:T',
            y='Lower Bound:Q',
            y2='Upper Bound:Q',
            tooltip=[
                alt.Tooltip('Date:T', title='Date'),
                alt.Tooltip('Lower Bound:Q', title='Lower Bound'),
                alt.Tooltip('Upper Bound:Q', title='Upper Bound')
            ]
        )

        # Combine the line chart and confidence interval area
        combined_chart = confidence_interval + forecast_chart
        st.altair_chart(combined_chart, use_container_width=True)


else:
    ...
    st.markdown("<h2 class='sub-header'>About This Model</h2>", unsafe_allow_html=True)
    st.markdown("""
        ### Model Overview

        Units of measuring oil production: **Thousand barrels**

        This project implements a short-term & a long-term oil production forecasting model for the USA.
        It scrapes data from the eia.gov website every month :
        - Short term forecasting involves forecast within next 12 months
        - Long-term forecasting involves forecast within next 1 to 5 years
        - A variety of models have been experimented with and best performing models (with least error) are picked for both long and short term forecasts.

        ### Limitations
        - It does not predict sudden changes due to unforeseen events.
        - The forecast becomes less reliable as we forecast further into the future

        ### Project Information
        **ISB AMPBA - Foundational Project**  
        **Professor:** Bharani Kumar Depuru  
        **TA:** K. Mohan  

        **Group-01 Team:**  
        - **Pragati Battula** 
        - **Ritik Ranjan**  
        - **Medha Adhikari** 
        - **Vishal Sharma**
        - **Challa Dhanunjaya Reddy**

        """, unsafe_allow_html=True)
