from app.config import SessionLocal
from app.model.database import DatasetUpload
import pandas as pd

def get_dataset_by_id(dataset_id: int, sheet_name="GABUNGAN"):
    db = SessionLocal()
    dataset = db.query(DatasetUpload).filter(DatasetUpload.id == dataset_id).first()
    db.close()

    if not dataset:
        raise FileNotFoundError(f"No dataset found with ID {dataset_id}")

    df = pd.read_excel(dataset.path, sheet_name=sheet_name)
    df.columns = df.columns.str.strip()
    return df