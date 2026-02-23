import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(input_path, output_path="processed_clean2.csv"):

    df = pd.read_csv(input_path)
    print("Original Columns:")
    print(df.columns.tolist())

    # =========================
    # 1️⃣ Standardize Column Names
    # =========================
    df.columns = df.columns.str.strip()

    # =========================
    # 2️⃣ Merge Duplicate Meaning Columns
    # =========================

    if 'Payer/Receiver' in df.columns and 'Payer' in df.columns:
        df['Payer'] = df['Payer'].fillna(df['Payer/Receiver'])
        df.drop(columns=['Payer/Receiver'], inplace=True)

    if 'Processing fee' in df.columns and 'Processing Fee' in df.columns:
        df['Processing Fee'] = df['Processing Fee'].fillna(df['Processing fee'])
        df.drop(columns=['Processing fee'], inplace=True)

    if 'Net amount' in df.columns and 'Net Amount' in df.columns:
        df['Net Amount'] = df['Net Amount'].fillna(df['Net amount'])
        df.drop(columns=['Net amount'], inplace=True)

    if 'Type' in df.columns and 'Type (UPI / UPI CC)' in df.columns:
        df['Type'] = df['Type'].fillna(df['Type (UPI / UPI CC)'])
        df.drop(columns=['Type (UPI / UPI CC)'], inplace=True)

    # =========================
    # 3️⃣ Clean Numeric Columns
    # =========================
    df['Amount'] = (
        df['Amount']
        .astype(str)
        .str.replace(',', '', regex=False)
    )
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')

    if 'Net Amount' in df.columns:
        df['Net Amount'] = pd.to_numeric(df['Net Amount'], errors='coerce')

    if 'Processing Fee' in df.columns:
        df['Processing Fee'] = pd.to_numeric(df['Processing Fee'], errors='coerce')

    # =========================
    # 4️⃣ SAFE DATE PARSING (Mixed Formats Handled)
    # =========================

    original_dates = df['Creation time']

    # ISO format
    parsed_iso = pd.to_datetime(
        original_dates,
        format='%Y-%m-%d %H:%M:%S',
        errors='coerce'
    )

    # Text format
    parsed_text = pd.to_datetime(
        original_dates,
        format='%d %b %Y, %H:%M',
        errors='coerce'
    )

    df['Creation time'] = parsed_iso.fillna(parsed_text)

    df = df.dropna(subset=['Amount', 'Creation time'])

    # =========================
    # 5️⃣ Convert to COMMON FORMAT (YYYY-MM ONLY)
    # =========================
    df['Year-Month'] = df['Creation time'].dt.to_period('M').astype(str)

    # Remove full datetime + day + weekday
    df.drop(columns=['Creation time'], inplace=True)

    # =========================
    # 6️⃣ Encode Categorical Columns
    # =========================
    categorical_cols = ['Paid via', 'Status', 'Type']

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # =========================
    # 7️⃣ Reorder Columns
    # =========================
    final_columns = [
        'Transaction ID',
        'Payer',
        'Amount',
        'Processing Fee',
        'Net Amount',
        'Paid via',
        'Status',
        'Type',
        'Year-Month'
    ]

    existing_columns = [col for col in final_columns if col in df.columns]
    df = df[existing_columns]

    df.to_csv(output_path, index=False)

    print("\nCleaned Columns:")
    print(df.columns.tolist())
    print("Unique Year-Month values:", df['Year-Month'].unique())
    print("Final Shape:", df.shape)

    return df


if __name__ == "__main__":
    preprocess_data("/Users/adityadebchowdhury/Desktop/gpay/combined_report2.csv")