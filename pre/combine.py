import pandas as pd

# Load both CSV files
file1 = "/Users/adityadebchowdhury/Desktop/gpay/processed_clean.csv"
file2 = "/Users/adityadebchowdhury/Desktop/gpay/processed_clean3.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Combine datasets (append rows)
combined_df = pd.concat([df1, df2], ignore_index=True)

# Save combined file
combined_df.to_csv("combined_report2.csv", index=False)

# Display first few rows
print("Combined Data Preview:")
print(combined_df.head())

print("\nTotal Rows:", len(combined_df))