import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_path, output_path="processed_data3.csv"):
    
    # =========================
    # 1️⃣ Load Dataset
    # =========================
    df = pd.read_csv(input_path)
    print("Original Shape:", df.shape)

    # =========================
    # 2️⃣ Handle Missing Values
    # =========================
    df = df.dropna(subset=['Amount', 'Creation time'])

    # =========================
    # 3️⃣ Clean Numeric Columns
    # =========================
    df['Amount'] = df['Amount'].astype(str).str.replace(',', '')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    # Optional columns
    if 'Net Amount' in df.columns:
        df['Net Amount'] = pd.to_numeric(df['Net Amount'], errors='coerce')

    df = df.dropna(subset=['Amount'])

    # =========================
    # 4️⃣ Date Feature Engineering
    # =========================
    df['Creation time'] = pd.to_datetime(
        df['Creation time'],
        format='mixed',
        dayfirst=True,
        errors='coerce'
    )

    df = df.dropna(subset=['Creation time'])

    df['Year'] = df['Creation time'].dt.year
    df['Month'] = df['Creation time'].dt.month
    df['Day'] = df['Creation time'].dt.day
    df['Weekday'] = df['Creation time'].dt.weekday

    # =========================
    # 5️⃣ Encode Categorical Columns
    # =========================
    categorical_cols = ['Paid via', 'Status', 'Type']

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # =========================
    # 6️⃣ Save Processed Dataset
    # =========================
    df.to_csv(output_path, index=False)

    print("Processed Shape:", df.shape)
    print(f"✅ Processed dataset saved as '{output_path}'")

    return df


# Run directly
if __name__ == "__main__":
    preprocess_data("report.csv")