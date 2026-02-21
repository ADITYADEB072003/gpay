# Monthly Sales Forecast (Using Year-Month Dataset)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# 1️⃣ Load Dataset
# =========================
df = pd.read_csv("processed_clean.csv")

print("Dataset Loaded:", df.shape)

# =========================
# 2️⃣ Convert Year-Month to Period
# =========================
df['YearMonth'] = pd.to_datetime(df['Year-Month']).dt.to_period('M')

# =========================
# 3️⃣ Group Monthly Sales
# =========================
monthly_sales = df.groupby('YearMonth')['Amount'].sum().reset_index()

# Sort properly
monthly_sales = monthly_sales.sort_values('YearMonth').reset_index(drop=True)

# Convert Period to string for plotting
monthly_sales['YearMonth_str'] = monthly_sales['YearMonth'].astype(str)

# Create Month Index
monthly_sales['MonthIndex'] = np.arange(len(monthly_sales))

X = monthly_sales[['MonthIndex']]
y = monthly_sales['Amount']

# =========================
# 4️⃣ Train Model
# =========================
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# =========================
# 5️⃣ Predict Next 6 Months
# =========================
future_index = np.arange(len(monthly_sales), len(monthly_sales) + 6).reshape(-1, 1)
future_predictions = model.predict(future_index)

# Generate future month labels
last_period = monthly_sales['YearMonth'].iloc[-1]
future_months = [(last_period + i).strftime('%Y-%m') for i in range(1, 7)]

# Combine historical + forecast
all_months = monthly_sales['YearMonth_str'].tolist() + future_months
all_sales = list(monthly_sales['Amount']) + list(future_predictions)

# =========================
# 6️⃣ Plot Actual vs Predicted
# =========================

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales['YearMonth_str'], y, marker='o', label='Actual Sales')
plt.plot(monthly_sales['YearMonth_str'], y_pred, marker='o', linestyle='--', label='Predicted Sales')
plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Actual vs Predicted Monthly Sales")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 7️⃣ Plot Forecast (Historical + Next 6 Months)
# =========================

plt.figure(figsize=(12, 6))
plt.plot(all_months, all_sales, marker='o')
plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Monthly Sales Forecast (Next 6 Months)")
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================
# 7️⃣ Print Results
# =========================
print("Next 6 Months Forecast:")
for m, p in zip(future_months, future_predictions):
    print(m, ":", round(p, 2))

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", round(mse, 2))
print("R2 Score:", round(r2, 4))