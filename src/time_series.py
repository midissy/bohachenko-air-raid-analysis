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