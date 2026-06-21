#1st graph
import pandas as pd
import matplotlib.pyplot as plt

# 1. load data
df = pd.read_csv("data/raw/full_data.csv")

# 2. ensure datetime
df["started_at"] = pd.to_datetime(df["started_at"])

# 3. extract date (baseline assumption: event belongs to start day)
df["date"] = df["started_at"].dt.date

# 4. aggregate to daily counts
daily_alerts = df.groupby("date").size().reset_index(name="alerts")

# 5. convert back to datetime index
daily_alerts["date"] = pd.to_datetime(daily_alerts["date"])
daily_alerts = daily_alerts.sort_values("date")
daily_alerts = daily_alerts.set_index("date")

# 6. create full date range (IMPORTANT for time series correctness)
full_range = pd.date_range(start=daily_alerts.index.min(),
                           end=daily_alerts.index.max(),
                           freq="D")

daily_alerts = daily_alerts.reindex(full_range, fill_value=0)
daily_alerts.index.name = "date"

# 7. plot
plt.figure(figsize=(14,5))
plt.plot(daily_alerts.index, daily_alerts["alerts"])
plt.title("Daily Air Raid Alerts in Ukraine (2022)")
plt.xlabel("Date")
plt.ylabel("Number of alerts")
plt.tight_layout()
plt.show()

# 8. rolling statistics (7-day window)
daily_alerts["rolling_mean_7"] = daily_alerts["alerts"].rolling(window=7).mean()

# 9. plot raw vs smoothed


#2nd graph
import matplotlib.pyplot as plt

plt.figure(figsize=(14,5))
plt.plot(daily_alerts.index, daily_alerts["alerts"], alpha=0.4, label="Raw")
plt.plot(daily_alerts.index, daily_alerts["rolling_mean_7"], linewidth=2, label="7-day rolling mean")

plt.title("Ukraine Air Raid Alerts: Raw vs Smoothed Trend")
plt.xlabel("Date")
plt.ylabel("Alerts per day")
plt.legend()
plt.tight_layout()

plt.show()
plt.close()




import pandas as pd
import matplotlib.pyplot as plt

# load data (якщо ще не в пам’яті)
df = pd.read_csv("data/raw/full_data.csv")

# datetime conversion (safe)
df["started_at"] = pd.to_datetime(df["started_at"])

# extract weekday
df["weekday"] = df["started_at"].dt.day_name()

# order weekdays correctly (важливо для графіка)
weekday_order = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

weekday_counts = (
    df["weekday"]
    .value_counts()
    .reindex(weekday_order)
)

# plot
plt.figure(figsize=(10,5))
plt.bar(weekday_counts.index, weekday_counts.values)

plt.title("Air Raid Alerts by Weekday (Ukraine, 2022)")
plt.xlabel("Weekday")
plt.ylabel("Number of alerts")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()
plt.close()



#3rd graph
import pandas as pd
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("data/raw/full_data.csv")

# datetime conversion
df["started_at"] = pd.to_datetime(df["started_at"])

# extract month
df["month"] = df["started_at"].dt.to_period("M")

# aggregate
monthly_counts = df.groupby("month").size()

# convert index to timestamp for plotting
monthly_counts.index = monthly_counts.index.to_timestamp()

# plot
plt.figure(figsize=(10,5))
plt.plot(monthly_counts.index, monthly_counts.values, marker="o")

plt.title("Monthly Air Raid Alerts in Ukraine (2022)")
plt.xlabel("Month")
plt.ylabel("Number of alerts")

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.close()

#Step 4
# rolling statistics
daily_alerts["rolling_mean_7"] = (
    daily_alerts["alerts"]
    .rolling(window=7)
    .mean()
)

daily_alerts["rolling_std_7"] = (
    daily_alerts["alerts"]
    .rolling(window=7)
    .std()
)

# visualization
plt.figure(figsize=(14, 6))

plt.plot(
    daily_alerts.index,
    daily_alerts["alerts"],
    alpha=0.4,
    label="Original series"
)

plt.plot(
    daily_alerts.index,
    daily_alerts["rolling_mean_7"],
    linewidth=2,
    label="Rolling mean (7 days)"
)

plt.plot(
    daily_alerts.index,
    daily_alerts["rolling_std_7"],
    linewidth=2,
    label="Rolling std (7 days)"
)

plt.title("Rolling Mean and Rolling Standard Deviation")
plt.xlabel("Date")
plt.ylabel("Alerts")

plt.legend()
plt.tight_layout()

plt.show()
plt.close()


#STep 4.2

from statsmodels.tsa.stattools import adfuller

result = adfuller(daily_alerts["alerts"])

print("ADF statistic:", result[0])
print("p-value:", result[1])

print("\nCritical values:")
for key, value in result[4].items():
    print(f"{key}: {value}")

# STep 4.3

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ensure no NaN
series = daily_alerts["alerts"].dropna()

plt.figure(figsize=(12, 5))

# ACF
plt.subplot(1, 2, 1)
plot_acf(series, lags=30, ax=plt.gca())
plt.title("ACF (Autocorrelation)")

# PACF
plt.subplot(1, 2, 2)
plot_pacf(series, lags=30, ax=plt.gca(), method="ywm")

plt.tight_layout()
plt.show()
plt.close()



import pandas as pd

# якщо daily_alerts вже існує — пропускаємо load
# daily_alerts має індекс datetime

series = daily_alerts["alerts"]

# гарантуємо порядок
series = series.sort_index()

# split point (80% train / 20% test)
split_idx = int(len(series) * 0.8)

train = series.iloc[:split_idx]
test = series.iloc[split_idx:]

print("Train size:", len(train))
print("Test size:", len(test))



from statsmodels.tsa.arima.model import ARIMA

model_101 = ARIMA(train, order=(1,0,1))
model_101_fit = model_101.fit()

pred_101 = model_101_fit.forecast(steps=len(test))

model_001 = ARIMA(train, order=(0,0,1))
model_001_fit = model_001.fit()

pred_001 = model_001_fit.forecast(steps=len(test))

print("ARIMA(1,0,1) AIC:", model_101_fit.aic)
print("ARIMA(0,0,1) AIC:", model_001_fit.aic)

import numpy as np

rmse_101 = np.sqrt(((test - pred_101) ** 2).mean())
rmse_001 = np.sqrt(((test - pred_001) ** 2).mean())

print("RMSE ARIMA(1,0,1):", rmse_101)
print("RMSE ARIMA(0,0,1):", rmse_001)




#final step visual result of the forecast

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

# actual values
plt.plot(test.index, test.values, label="Actual", linewidth=2)

# predictions
plt.plot(test.index, pred_101, label="ARIMA(1,0,1)", linestyle="--")
plt.plot(test.index, pred_001, label="ARIMA(0,0,1)", linestyle="--")

plt.title("Actual vs Predicted Air Raid Alerts (Test Period)")
plt.xlabel("Date")
plt.ylabel("Number of Alerts")
plt.legend()
plt.tight_layout()
plt.show()
plt.close()