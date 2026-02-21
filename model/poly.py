# Monthly Sales Forecast using Polynomial Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# 1️⃣ Load Processed Data
# =========================
df = pd.read_csv("processed_data.csv")

# Create proper date column
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str))
monthly_sales = df.groupby('Date')['Amount'].sum().sort_index()

# Convert to DataFrame
monthly_sales = monthly_sales.reset_index()

# Create numeric time index
monthly_sales['MonthIndex'] = np.arange(len(monthly_sales))

X = monthly_sales[['MonthIndex']]
y = monthly_sales['Amount']

# =========================
# 2️⃣ Polynomial Transformation
# =========================
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# =========================
# 3️⃣ Train Model
# =========================
model = LinearRegression()
model.fit(X_poly, y)

# Predictions on training data
y_pred = model.predict(X_poly)

# Evaluation
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# =========================
# 4️⃣ Forecast Next 6 Months
# =========================
future_index = np.arange(len(monthly_sales), len(monthly_sales) + 6).reshape(-1,1)
future_poly = poly.transform(future_index)
future_predictions = model.predict(future_poly)

# Create future dates
last_date = monthly_sales['Date'].iloc[-1]
future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=6,
    freq='MS'
)

# =========================
# 5️⃣ Plot
# =========================
plt.figure(figsize=(12,6))

plt.plot(monthly_sales['Date'], y, marker='o', label="Actual")
plt.plot(monthly_sales['Date'], y_pred, linestyle='--', label="Fitted")
plt.plot(future_dates, future_predictions, marker='o', label="Forecast")

plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Monthly Sales Forecast (Polynomial Regression)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 6️⃣ Print Results
# =========================
print("Next 6 Months Forecast:")
for date, value in zip(future_dates, future_predictions):
    print(date.strftime("%Y-%m"), ":", round(value, 2))

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", round(mse, 2))
print("R2 Score:", round(r2, 4))