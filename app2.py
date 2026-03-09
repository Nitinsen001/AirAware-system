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
st.sidebar.title("🌍 AirAware Dashboard")

city = st.sidebar.selectbox("Select City", df["City"].unique())

if st.sidebar.button("🔄 Refresh Data"):
    st.rerun()

st.title("🌍 AirAware - AQI Prediction Dashboard")

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

# -------------------------
# Alert
# -------------------------
def alert(aqi):

    if aqi <= 50:
        return "Air is Safe"

    elif aqi <= 100:
        return "Sensitive people be careful"

    elif aqi <= 200:
        return "Wear Mask"

    else:
        return "Avoid Outdoor Activity"


pred["Alert"] = pred["Predicted_AQI"].apply(alert)

# -------------------------
# Today AQI
# -------------------------
today_aqi = city_df["y"].iloc[-1]
today_cat = category(today_aqi)

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("📅 Today's AQI")
    if today_cat == "Good":
        st.success(f"**{round(today_aqi,2)}** - {today_cat}")
    elif today_cat == "Moderate":
        st.warning(f"**{round(today_aqi,2)}** - {today_cat}")
    elif today_cat == "Unhealthy":
        st.error(f"**{round(today_aqi,2)}** - {today_cat}")
    else:
        st.markdown(f'<div style="background-color: red; color: white; padding: 10px; border-radius: 5px; animation: blink 1s infinite;">⚠️ **{round(today_aqi,2)}** - {today_cat}</div>', unsafe_allow_html=True)

with col2:
    st.subheader("💡 Alert")
    alert_msg = alert(today_aqi)
    if "Safe" in alert_msg:
        st.success(alert_msg)
    elif "careful" in alert_msg:
        st.warning(alert_msg)
    elif "Mask" in alert_msg:
        st.error(alert_msg)
    else:
        st.markdown(f'<div style="background-color: red; color: white; padding: 10px; border-radius: 5px; animation: blink 1s infinite;">🚨 {alert_msg}</div>', unsafe_allow_html=True)

# -------------------------
# Prediction Table
# -------------------------
st.subheader("📊 Next 7 Days AQI Prediction")

st.dataframe(pred)

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

# -------------------------
# Forecast Graph
# -------------------------
st.subheader("📈 AQI Forecast Trend")

with st.expander("Show Forecast Graph"):
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

st.markdown("---")
st.markdown("**Built with ❤️ using Streamlit and Prophet**")
st.markdown(f"**The {city} are ❤️ AQI values predicted **")