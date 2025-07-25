# preprocessing.py
import pandas as pd

def load_monthly_case_data(file_path: str, sheet_name="GABUNGAN"):
    # Load Excel
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()

    # Convert date column
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'], errors='coerce')
    df = df.dropna(subset=['TANGGAL'])

    # Extract monthly period
    df['BULAN'] = df['TANGGAL'].dt.to_period('M').astype(str)

    # Group by month
    monthly_data = df.groupby('BULAN').size().reset_index(name='JUMLAH_KASUS')
    monthly_data['BULAN'] = pd.to_datetime(monthly_data['BULAN'])
    monthly_data = monthly_data.sort_values(by='BULAN').reset_index(drop=True)

    return monthly_data

def clean_profile_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df['USIA'] = pd.to_numeric(df['USIA'], errors='coerce')

    df['PEKERJAAN'] = df['PEKERJAAN'].astype(str).str.strip().str.upper()
    df['PENDIDIKAN TERAKHIR'] = df['PENDIDIKAN TERAKHIR'].astype(str).str.strip().str.upper()

    # Drop invalid values
    df = df[~df['PEKERJAAN'].isin(["", "NAN", "NA", "TIDAK DIISI", None])]
    df = df[~df['PENDIDIKAN TERAKHIR'].isin(["", "NAN", "NA", "TIDAK DIISI", None])]

    # Group ages into bins
    age_bins = [10, 18, 25, 35, 50, 100]
    age_labels = ['≤18', '19–25', '26–35', '36–50', '50+']
    df['AGE_GROUP'] = pd.cut(df['USIA'], bins=age_bins, labels=age_labels, right=False)

    return df