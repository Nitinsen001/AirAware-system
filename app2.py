import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================================================
#  Data Preprocesing pipelines
# ==========================================================================

def preprocess_data(df):
    
    df = df.copy()
    
    # data conversion
    df["Date"] = pd.to_datetime(df["Date"])
    
    #sorting
    df = df.sort_values("Date")
    
    # remove duplicates
    df = df.drop_duplicates()
    
    # missing values handling
    df = df.fillna(method="ffill")
    df = df.fillna(method="bfill")
    
    return df

# ==============================================================================
#  feature engineering
# ==============================================================================

def feature_engineering(df):
    
    df = df.copy()
    
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["day_of_week"] = df["Date"].dt.isocalendar().week
    
    df["is_weekend"] = df["day_of_week"].apply(lambda x:1 if x>=5 else 0)
    
    # polution interaction features
    df["PM_ratio"] = df["PM2.5"] / (df["PM10"] + 1)
    
    df["gas_polution"] = df["NO2"] + df["SO2"] + df["CO"]
    
    return df

# ================================================================================
#  outlier removal
# ================================================================================

def remove_outliers(df):
    
    q1 = df["AQI"].quantile(0.25)
    q3 = df["AQI"].quantile(0.75)
    IQR = q3 - q1
    
    df = df[(df["AQI"] >= q1 - 1.5 * IQR) & (df["AQI"] <= q3 + 1.5 * IQR)]
    return df

# =====================================
# ALERT SYSTEM
# =====================================

def generate_aqi_alert(aqi):

    if aqi <= 50:
        return {
            "status": "Good",
            "color": "green",
            "alert": "Air quality is good.",
            "advice": "You can safely perform outdoor activities."
        }

    elif aqi <= 100:
        return {
            "status": "Moderate",
            "color": "blue",
            "alert": "Air quality is acceptable but not ideal.",
            "advice": "Sensitive individuals should limit prolonged outdoor exposure."
        }

    elif aqi <= 150:
        return {
            "status": "Unhealthy for Sensitive Groups",
            "color": "orange",
            "alert": "Sensitive groups may experience health effects.",
            "advice": "Children, elderly, and asthma patients should avoid outdoor activities."
        }

    elif aqi <= 200:
        return {
            "status": "Unhealthy",
            "color": "red",
            "alert": "Everyone may start experiencing health effects.",
            "advice": "Limit outdoor activities and consider wearing a mask."
        }

    elif aqi <= 300:
        return {
            "status": "Very Unhealthy",
            "color": "purple",
            "alert": "Health alert: serious health effects possible.",
            "advice": "Avoid outdoor exposure and stay indoors."
        }

    else:
        return {
            "status": "Hazardous",
            "color": "darkred",
            "alert": "Emergency conditions! Severe health risk.",
            "advice": "Avoid going outside. Use air purifiers if possible."
        }

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("air_quality.csv")

df = preprocess_data(df)
df = feature_engineering(df)
df = remove_outliers(df)

# -------------------------
# City Selection
# -------------------------
st.sidebar.title(" AirAware Dashboard")
st.sidebar.markdown("---")

city = st.sidebar.selectbox(" Select City", df["City"].unique(), help="Choose a city to view air quality data")

st.sidebar.markdown("---")

# Add some info about the selected city
if city:
    city_data_count = len(df[df["City"] == city])
    st.sidebar.metric("Data Points", city_data_count)
    
    # Show last AQI if available
    if city_data_count > 0:
        last_aqi = df[df["City"] == city]["AQI"].iloc[-1]
        st.sidebar.metric("Latest AQI", f"{round(last_aqi, 1)}")

st.sidebar.markdown("---")
if st.sidebar.button(" Refresh Data", help="Reload the data and predictions"):
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("###  AQI Categories")
st.sidebar.markdown("- **Good**: 0-50")
st.sidebar.markdown("- **Moderate**: 51-100")
st.sidebar.markdown("- **Unhealthy**: 101-200")
st.sidebar.markdown("- **Hazardous**: >200")

# -------------------------
# Data Processing
# -------------------------
city_df = df[df["City"] == city][["Date","AQI","PM2.5","PM10","NO2","SO2","CO"]].dropna()

city_df = city_df.rename(columns={
    "Date":"ds",
    "AQI":"y"
})

# -------------------------
# Train Model
# -------------------------
def Train_prphet(city_df):
    prophet_df = city_df[["ds","y","PM2.5","PM10","NO2","SO2","CO"]]
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    model.add_regressor("PM2.5")
    model.add_regressor("PM10")
    model.add_regressor("NO2")
    model.add_regressor("SO2")
    model.add_regressor("CO")
    
    model.fit(prophet_df)
    
    return model,prophet_df

model,prophet_df = Train_prphet(city_df)

# -------------------------
# Future Prediction
# -------------------------
future = model.make_future_dataframe(periods=7)

future["PM2.5"] = prophet_df["PM2.5"].iloc[-1]
future["PM10"] = prophet_df["PM10"].iloc[-1]
future["SO2"] = prophet_df["SO2"].iloc[-1]
future["NO2"] = prophet_df["NO2"].iloc[-1]
future["CO"] = prophet_df["CO"].iloc[-1]

forecast = model.predict(future)

pred = forecast[["ds","yhat"]].tail(7)

pred = pred.rename(columns={
    "ds":"Date",
    "yhat":"Predicted_AQI"
})

# -------------------------
# AQI Category
# -------------------------
def category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Moderate"

    elif aqi <= 200:
        return "Unhealthy"

    else:
        return "Hazardous"


pred["AQI_Category"] = pred["Predicted_AQI"].apply(category)
pred["Alert"] = pred["Predicted_AQI"].apply(lambda x: generate_aqi_alert(x)["alert"])

# -------------------------
# Today AQI
# -------------------------
today_aqi = city_df["y"].iloc[-1]
today_cat = category(today_aqi)

# Main Page Layout
st.title(" AirAware - AQI Prediction Dashboard")
st.markdown("### Real-time Air Quality Monitoring & Forecasting")

st.divider()

# Today's Overview Section
st.header(" Today's Air Quality Overview")

# Key Metrics Row
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Current AQI", f"{round(today_aqi, 1)}", help="Air Quality Index value")

with col2:
    st.metric("Category", today_cat, help="Air quality category")

with col3:
    # Calculate trend (compare to yesterday if available)
    if len(city_df) > 1:
        yesterday_aqi = city_df["y"].iloc[-2]
        trend = round(today_aqi - yesterday_aqi, 1)
        st.metric("Change from Yesterday", f"{trend:+.1f}", help="AQI change from previous day")
    else:
        st.metric("Data Points", len(city_df), help="Number of data points available")

st.divider()

# Today's AQI and Alert in columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("<> Current AQI Status")
    if today_cat == "Good":
        st.success(f"**{round(today_aqi,2)}** - {today_cat}")
    elif today_cat == "Moderate":
        st.warning(f"**{round(today_aqi,2)}** - {today_cat}")
    elif today_cat == "Unhealthy":
        st.error(f"**{round(today_aqi,2)}** - {today_cat}")
    else:
        st.markdown(f'<div style="background-color: red; color: white; padding: 10px; border-radius: 5px; animation: blink 1s infinite;">⚠️ **{round(today_aqi,2)}** - {today_cat}</div>', unsafe_allow_html=True)

with col2:
    st.subheader("� Air Quality Alert")
    
    alert = generate_aqi_alert(today_aqi)
    
    # Status with color
    color_map = {
        "green": "#28a745",
        "blue": "#007bff",
        "orange": "#fd7e14",
        "red": "#dc3545",
        "purple": "#6f42c1",
        "darkred": "#8b0000"
    }
    
    status_color = color_map.get(alert["color"], "#6c757d")
    
    st.markdown(f"""
    <div style="background-color: {status_color}; color: white; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
        <strong>Status: {alert['status']}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    st.write(f"! **Alert:** {alert['alert']}")
    st.write(f" **Recommendation:** {alert['advice']}")
    
    # Add blinking for hazardous
    if alert["status"] == "Hazardous":
        st.markdown("""
        <style>
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0; }
            100% { opacity: 1; }
        }
        .blink {
            animation: blink 1s infinite;
        }
        </style>
        <div class="blink" style="background-color: darkred; color: white; padding: 10px; border-radius: 5px; margin-top: 10px;">
            🚨 Emergency Alert!
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# Prediction Table
# -------------------------
st.header(" 7-Day AQI Forecast")

# Create tabs for different views
tab1, tab2 = st.tabs([" Data Table", " Trend Chart"])

with tab1:
    st.dataframe(pred, use_container_width=True)

with tab2:
    with st.expander("View Forecast Trend", expanded=True):
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

# Add CSS for blinking
st.markdown("""
<style>
@keyframes blink {
    0% { opacity: 1; }
    50% { opacity: 0; }
    100% { opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

st.divider()

# Footer with additional information
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("** City:** " + city)
    st.markdown(f"** Data Points:** {len(city_df)}")

with col2:
    st.markdown("** Last Updated:** Today")
    st.markdown("** Model:** Facebook Prophet")

with col3:
    st.markdown("** Tip:** Check alerts regularly")
    st.markdown("** Emergency:** Follow local guidelines")

st.caption("AirAware Dashboard - Powered by Streamlit & Machine Learning")
def developer_info():
    st.info("👨‍💻 Developed by Nitin Sen - python Intern")
    
st.button(" About Developer", on_click=developer_info)
