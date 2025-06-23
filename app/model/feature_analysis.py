from sklearn.preprocessing import LabelEncoder
import logging
import pandas as pd
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)

def prepare_features(
    df: pd.DataFrame,
    features: List[str]
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    logger.info(f"Starting feature preparation with {len(df)} rows")

    try:
        df = df.copy()

        # Validate features
        missing = [col for col in features if col not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        feature_data = df[features].dropna()
        if feature_data.empty:
            raise ValueError("No valid data remaining after dropping NA.")

        encoders = {}
        processed = pd.DataFrame(index=feature_data.index)

        for col in features:
            if feature_data[col].dtype == 'object':
                enc = LabelEncoder()
                processed[col] = enc.fit_transform(feature_data[col].astype(str))
                encoders[col] = enc
            else:
                processed[col] = feature_data[col].values

        return processed, encoders

    except Exception as e:
        logger.exception(f"Error in prepare_features: {e}")
        raise

def profile_summary(df: pd.DataFrame) -> dict:
    df = df.copy()
    df.columns = df.columns.str.strip()
    df['USIA'] = pd.to_numeric(df['USIA'], errors='coerce')

    df['PEKERJAAN'] = df['PEKERJAAN'].astype(str).str.strip().str.upper()
    df['PENDIDIKAN TERAKHIR'] = df['PENDIDIKAN TERAKHIR'].astype(str).str.strip().str.upper()

    age_bins = [10, 18, 25, 35, 50, 100]
    age_labels = ['≤18', '19–25', '26–35', '36–50', '50+']
    df['AGE_GROUP'] = pd.cut(df['USIA'], bins=age_bins, labels=age_labels, right=False)

    df = df[~df['PEKERJAAN'].isin(["", "NAN", "NA", "TIDAK DIISI", None])]
    df = df[~df['PENDIDIKAN TERAKHIR'].isin(["", "NAN", "NA", "TIDAK DIISI", None])]

    valid_edu = df[df['PENDIDIKAN TERAKHIR'].notna()]
    valid_job = df[df['PEKERJAAN'].notna()]

    result = {
        "top_education": valid_edu['PENDIDIKAN TERAKHIR'].mode().iloc[0] if not valid_edu.empty else "N/A",
        "education_distribution_percent": df['PENDIDIKAN TERAKHIR'].value_counts(normalize=True).round(4).mul(100).to_dict(),
        "most_common_age_group": str(df['AGE_GROUP'].mode().iloc[0]) if not df['AGE_GROUP'].isna().all() else "N/A",
        "age_group_distribution_percent": df['AGE_GROUP'].value_counts(normalize=True).round(4).mul(100).to_dict(),
        "most_common_job": valid_job['PEKERJAAN'].mode().iloc[0] if not valid_job.empty else "N/A",
        "job_distribution_percent": df['PEKERJAAN'].value_counts(normalize=True).round(4).mul(100).to_dict()
    }

    return result
