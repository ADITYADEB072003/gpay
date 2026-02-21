import pandas as pd

# Load both CSV files
file1 = "GPay_Business_Transactions_20260101-20260221_1771692241301.csv"
file2 = "report.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Combine datasets (append rows)
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save combined file
combined_df.to_csv("combined_report.csv", index=False)

# Display first few rows
print("Combined Data Preview:")
print(combined_df.head())

print("\nTotal Rows:", len(combined_df))