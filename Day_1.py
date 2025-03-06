import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Set a default style that is built in
plt.style.use('ggplot')

# --- Input Parameters ---
API_KEY = "wM8ugoMZGzMm4sBNdAb64ndgTEygMn0x79w31jEQ"
series_id = "PET.RWTC.D"  # WTI crude oil spot price
url = f"https://api.eia.gov/v2/seriesid/{series_id}?api_key={API_KEY}"

# --- Data Fetching ---
response = requests.get(url)
if response.status_code != 200:
    raise Exception("Error fetching data from EIA API. Status Code: " + str(response.status_code))
data = response.json()

# --- Data Processing ---
# Expecting data structure: data["response"]["data"] is a list of dictionaries with keys ("period" and "value")
raw_data = data["response"]["data"]
# Extract period and value; assume period format is 'YYYY-MM-DD'
dates_str = [entry["period"] for entry in raw_data]
values = np.array([float(entry["value"]) for entry in raw_data])

# Reverse the lists for chronological order (oldest first)
dates_str = dates_str[::-1]
values = values[::-1]

# Convert the date strings into datetime objects
dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates_str]

# --- Basic Statistics ---
overall_mean = np.mean(values)
overall_std = np.std(values)
overall_max = np.max(values)
overall_min = np.min(values)
print(f"Mean: {overall_mean:.2f}, Std: {overall_std:.2f}, Max: {overall_max:.2f}, Min: {overall_min:.2f}")

# --- Kalman Filter Smoothing ---
def kalman_filter(zs, Q=1e-5, R=0.1**2):
    n = len(zs)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhat[0] = zs[0]
    P[0] = 1.0
    for k in range(1, n):
        # Prediction calcuations 
        xhat_minus = xhat[k-1]
        P_minus = P[k-1] + Q
        # Update
        K = P_minus / (P_minus + R)
        xhat[k] = xhat_minus + K * (zs[k] - xhat_minus)
        P[k] = (1 - K) * P_minus
    return xhat

kalman_estimates = kalman_filter(values)

# --- Rolling High and Low (Window = 10 days) ---
window_size = 10
rolling_high = np.full(len(values), np.nan)
rolling_low = np.full(len(values), np.nan)
for i in range(window_size - 1, len(values)):
    rolling_high[i] = np.max(values[i - window_size + 1 : i + 1])
    rolling_low[i] = np.min(values[i - window_size + 1 : i + 1])

# --- Linear Trendline (via Linear Regression) ---
# Convert dates to numbers for regression
dates_num = mdates.date2num(dates)
coeffs = np.polyfit(dates_num, values, 1)
trendline_func = np.poly1d(coeffs)
trend_values = trendline_func(dates_num)

# --- Forecasting using Trend Extrapolation ---
forecast_days = 10
last_date = dates[-1]
forecast_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
forecast_dates_num = mdates.date2num(forecast_dates)
forecast_values = trendline_func(forecast_dates_num)

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Top subplot: Price analysis & markings
ax1.plot(dates, values, label='Original Data', marker='o', markersize=3, linestyle='-')
ax1.plot(dates, kalman_estimates, label='Kalman Estimate', marker='x', markersize=3, linestyle='--')
ax1.plot(dates, rolling_high, label=f'{window_size}-Day Rolling High', linestyle=':', linewidth=2)
ax1.plot(dates, rolling_low, label=f'{window_size}-Day Rolling Low', linestyle=':', linewidth=2)
ax1.plot(dates, trend_values, label='Trendline', linestyle='-.', linewidth=2)
ax1.plot(forecast_dates, forecast_values, label='Forecast', marker='D', linestyle='--', color='magenta')
ax1.set_title('WTI Crude Oil Spot Price Analysis (APIv2)')
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True)

# Annotate max and min values on the original data
max_index = np.argmax(values)
min_index = np.argmin(values)
ax1.annotate(f'Max: {values[max_index]:.2f}', xy=(dates[max_index], values[max_index]),
             xytext=(dates[max_index], values[max_index] + 5),
             arrowprops=dict(facecolor='green', shrink=0.05))
ax1.annotate(f'Min: {values[min_index]:.2f}', xy=(dates[min_index], values[min_index]),
             xytext=(dates[min_index], values[min_index] - 5),
             arrowprops=dict(facecolor='red', shrink=0.05))

# Bottom subplot: Error analysis (absolute error between original and Kalman estimates)
error = np.abs(values - kalman_estimates)
ax2.plot(dates, error, label='Absolute Error', marker='s', markersize=3, linestyle='-', color='orange')
ax2.set_title('Error Analysis')
ax2.set_xlabel('Date')
ax2.set_ylabel('Error')
ax2.legend()
ax2.grid(True)

# Format the x-axis to show dates nicely
ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate()

plt.tight_layout()
plt.show()
