# Monthly Sales Forecast using ARIMA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# 1️⃣ Load Processed Dataset
# =========================
df = pd.read_csv("processed_data.csv")

# Create proper datetime index
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
monthly_sales = df.groupby('Date')['Amount'].sum().sort_index()

# =========================
# 2️⃣ Train ARIMA Model
# =========================
# (p,d,q) = (1,1,1) is a good starting point
model = ARIMA(monthly_sales, order=(1, 1, 1))
model_fit = model.fit()

# In-sample predictions
predictions = model_fit.predict(start=0, end=len(monthly_sales)-1)

# =========================
# 3️⃣ Evaluation
# =========================
mse = mean_squared_error(monthly_sales, predictions)
r2 = r2_score(monthly_sales, predictions)

# =========================
# 4️⃣ Forecast Next 6 Months
# =========================
forecast_steps = 6
forecast = model_fit.forecast(steps=forecast_steps)

# Create future dates
last_date = monthly_sales.index[-1]
future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=forecast_steps,
    freq='MS'
)

# =========================
# 5️⃣ Plot Results
# =========================
plt.figure(figsize=(12,6))
plt.plot(monthly_sales.index, monthly_sales, label="Actual", marker='o')
plt.plot(monthly_sales.index, predictions, label="Fitted", linestyle='--')
plt.plot(future_dates, forecast, label="Forecast", marker='o')
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Monthly Sales Forecast (ARIMA)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 6️⃣ Print Results
# =========================
print("Next 6 Months Forecast:")
for date, value in zip(future_dates, forecast):
    print(date.strftime("%Y-%m"), ":", round(value, 2))

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", round(mse, 2))
print("R2 Score:", round(r2, 4))