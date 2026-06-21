# bohachenko-air-raid-analysis
Project created with AI assistance for Stage 2 of the selection process for KSE AI Agentic Summer School 


# Time Series Analysis of Air Raid Alerts in Ukraine (2022)

## 📌 Project Overview

This project presents an end-to-end data science pipeline for analyzing and forecasting air raid alert events in Ukraine using time series methods. The objective is not only to build a predictive model, but to critically evaluate the applicability of classical time series techniques (ARIMA) to event-driven, externally influenced data.

The analysis follows a structured workflow: from raw event data preprocessing, through exploratory data analysis and statistical diagnostics, to baseline forecasting and model evaluation.

---

## 🎯 Problem Statement

Air raid alerts represent event-based, irregularly spaced signals driven by external geopolitical factors. The goal of this project is to:

- Transform raw event data into a structured time series representation
- Analyze temporal patterns and statistical properties
- Evaluate whether classical ARIMA models can meaningfully capture the dynamics of such a process
- Build a baseline forecasting model and assess its limitations

---

## 📊 Dataset Description

The dataset is sourced from a public Kaggle repository: *Air raid sirens in Ukraine (2022)*.

It consists of event-based records, where each row represents a single air raid alert interval.

**Columns:**

- `oblast` — region in Ukraine
- `raion` — district (partially missing)
- `hromada` — local community (partially missing)
- `level` — alert level/category
- `started_at` — start timestamp of alert (UTC)
- `finished_at` — end timestamp of alert (UTC)

Key characteristics:
- ~14,000+ records
- Full coverage of 2022
- Event-based (interval data), not pre-aggregated time series
- Converted into daily aggregated counts for analysis

---

## 🧠 Methodology

The project follows a structured time series analysis pipeline:

### 1. Data Preprocessing
- Parsing datetime fields (`started_at`, `finished_at`)
- Handling missing categorical values
- Ensuring temporal consistency

### 2. Time Series Construction
- Aggregation of events into daily counts of air raid alerts
- Transformation from interval-based events to time series representation

### 3. Exploratory Data Analysis (EDA)
- Trend visualization over time
- Weekly and monthly seasonality analysis
- Rolling statistics (mean and standard deviation)

### 4. Stationarity Diagnostics
- Augmented Dickey-Fuller (ADF) test
- Rolling mean and variance inspection
- Autocorrelation analysis (ACF/PACF)

### 5. Forecasting (Baseline Models)
- ARIMA(1,0,1)
- ARIMA(0,0,1)
- Time-based train/test split (80/20)
- Evaluation using AIC and RMSE

---

## 📈 Key Findings

- The time series exhibits **weak but present autocorrelation structure**
- ADF test suggests the series is **statistically stationary at the 5% significance level**
- Rolling statistics indicate **regime shifts and non-uniform variance over time**
- The process is **noise-dominant with externally driven shocks**

---

## 🤖 Forecasting Results

Two baseline ARIMA models were evaluated:

- ARIMA(1,0,1): lower AIC, slightly worse generalization (higher RMSE)
- ARIMA(0,0,1): better RMSE, slightly worse in-sample fit

### Key insight:
The models primarily capture the **overall level of the series**, but fail to reproduce short-term volatility and spikes.

This indicates that ARIMA acts more as a **smoothing approximation** rather than a precise forecasting tool for this type of data.

---

## 📌 Conclusion

Classical ARIMA models demonstrate **limited explanatory and predictive power** for event-based air raid alert data.

The main reasons include:
- External (non-time-dependent) drivers of events
- High noise-to-signal ratio
- Regime changes over time
- Weak and inconsistent autocorrelation structure

As a result, the forecasting problem is better described as **a stochastic event intensity estimation problem** rather than a classical time series forecasting task.

---

## 🚀 How to Run the Project

### 1. Clone repository
```bash
git clone https://github.com/your-username/air-raid-time-series-analysis.git
cd air-raid-time-series-analysis

2. Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
3. Install dependencies
pip install -r requirements.txt
4. Run analysis

You can run the project via Jupyter notebooks:

jupyter notebook notebooks/01_eda.ipynb
jupyter notebook notebooks/02_forecasting.ipynb
📁 Project Structure
project/
│
├── data/                  # Raw dataset
├── notebooks/            # EDA and forecasting notebooks
├── src/                  # Python modules (preprocessing, modeling)
├── outputs/              # Generated plots and results
├── requirements.txt      # Dependencies
└── README.md             # Project documentation


📌 Author Notes

This project was developed as an end-to-end data science exercise focusing on real-world event-driven time series data and the limitations of classical forecasting methods in such contexts.