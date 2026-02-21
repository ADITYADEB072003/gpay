import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
file_path = "report.csv"   # Change path if needed
df = pd.read_csv(file_path)

# Convert 'Creation time' column to datetime
df['Creation time'] = pd.to_datetime(
    df['Creation time'],
    dayfirst=True,
    errors='coerce'
)

# Remove invalid date rows (if any)
df = df.dropna(subset=['Creation time'])

# Create Month-Year column
df['Month'] = df['Creation time'].dt.to_period('M')

# Group by Month
monthly_report = df.groupby('Month').agg(
    Total_Transactions=('Transaction ID', 'count'),
    Total_Sales=('Amount', 'sum'),
    Total_Net_Amount=('Net Amount', 'sum')
).reset_index()

# Convert Month to string
monthly_report['Month'] = monthly_report['Month'].astype(str)

# Print Report
print("\nMonthly Sales Report\n")
print(monthly_report)

# Save report to CSV (optional)
monthly_report.to_csv("monthly_sales_report.csv", index=False)

# Plot Monthly Total Sales
plt.figure()
plt.bar(monthly_report['Month'], monthly_report['Total_Sales'])
plt.xticks(rotation=45)
plt.title("Monthly Total Sales")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.show()