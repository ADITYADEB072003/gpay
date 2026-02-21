# Monthly Sales Forecast (Next 6 Months)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("processed_data2.csv")

# Parse dates safely
df['Creation time'] = pd.to_datetime(
    df['Creation time'],
    format='mixed',
    dayfirst=True,
    errors='coerce'
)

df = df.dropna(subset=['Creation time'])

# Create Year-Month column
df['YearMonth'] = df['Creation time'].dt.to_period('M')

# Group monthly sales
monthly_sales = df.groupby('YearMonth')['Amount'].sum().reset_index()

# Convert to numeric index
monthly_sales['MonthIndex'] = np.arange(len(monthly_sales))

X = monthly_sales[['MonthIndex']]
y = monthly_sales['Amount']

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predictions on historical data for evaluation
y_pred = model.predict(X)

# Calculate MSE and R2 Score
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Predict next 6 months
future_index = np.arange(len(monthly_sales), len(monthly_sales) + 6).reshape(-1, 1)
future_predictions = model.predict(future_index)

# Create future month labels
last_period = monthly_sales['YearMonth'].iloc[-1]
future_months = [str(last_period + i + 1) for i in range(6)]

# Combine historical + forecast for plotting
all_months = list(monthly_sales['YearMonth'].astype(str)) + future_months
all_sales = list(monthly_sales['Amount']) + list(future_predictions)

# Plot with improved scaling
plt.figure(figsize=(12, 6))
plt.plot(all_months, all_sales, marker='o')
plt.xticks(rotation=45)
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Monthly Sales Forecast (Next 6 Months)")
plt.grid(True)
plt.tight_layout()
plt.show()

print("Next 6 Months Forecast:")
for m, p in zip(future_months, future_predictions):
    print(m, ":", round(p, 2))

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", round(mse, 2))
print("R2 Score (Accuracy):", round(r2, 4))