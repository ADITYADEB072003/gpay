# Monthly Sales Forecast (Using New Dataset)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =========================
# 1️⃣ Load Dataset
# =========================
df = pd.read_csv("processed_data3.csv")

print("Dataset Loaded:", df.shape)

# =========================
# 2️⃣ Convert Creation time to Datetime
# =========================
df['Creation time'] = pd.to_datetime(df['Creation time'])

# Extract Year & Month from datetime (safer method)
df['Year'] = df['Creation time'].dt.year
df['Month'] = df['Creation time'].dt.month

# =========================
# 3️⃣ Group Monthly Sales
# =========================
monthly_sales = df.groupby(
    [df['Creation time'].dt.to_period("M")]
)['Amount'].sum().reset_index()

monthly_sales.columns = ['YearMonth', 'Amount']

# Convert Period to string
monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)

# Sort properly
monthly_sales = monthly_sales.sort_values('YearMonth').reset_index(drop=True)

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
last_period = pd.Period(monthly_sales['YearMonth'].iloc[-1], freq='M')
future_months = [(last_period + i).strftime('%Y-%m') for i in range(1, 7)]

# Combine historical + forecast
all_months = monthly_sales['YearMonth'].tolist() + future_months
all_sales = list(monthly_sales['Amount']) + list(future_predictions)

# =========================
# 6️⃣ Plot
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