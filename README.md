# AirAware-system
AirAware is a Machine Learning-based system that predicts Air Quality Index (AQI) using pollutant and temporal features.
The system analyzes historical air quality data and provides intelligent insights into pollution trends and health risk levels.

This project goes beyond basic prediction by including:

AQI value prediction (Regression)

Seasonal & temporal analysis

Feature importance analysis

Health risk recommendation system

Data visualization and trend analysis

🎯 Problem Statement

Air pollution is a major environmental and health concern.
Accurate AQI prediction helps in:

Early pollution alerts

Public health awareness

Smart city planning

Environmental monitoring

🧠 Machine Learning Approach

Problem Type: Regression

Target Variable: AQI

Algorithm Used: Random Forest Regressor

Evaluation Metrics:

Mean Absolute Error (MAE)

R² Score

📊 Dataset

The dataset contains air pollutant measurements including:

PM2.5

PM10

NO

NO2

NOx

NH3

CO

SO2

O3

Benzene

Toluene

Xylene

Date

City

Feature engineering was performed to extract:

Year

Month

Day

Day of Week

Season

Rolling averages

Pollutant ratios

⚙️ Project Workflow

Data Cleaning

Missing Value Handling

Feature Engineering

Encoding Categorical Variables

Exploratory Data Analysis (EDA)

Model Training

Model Evaluation

Feature Importance Analysis

📈 Exploratory Data Analysis

AQI distribution analysis

Correlation heatmap

Pollutant vs AQI relationship

Seasonal trends

Monthly AQI patterns

City-wise AQI comparison

🚀 Enhancements Implemented

Feature Importance Visualization

Seasonal Pollution Analysis

Health Risk Recommendation System

Temporal Feature Extraction

Pollutant Impact Analysis

🛠️ Tech Stack

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn
