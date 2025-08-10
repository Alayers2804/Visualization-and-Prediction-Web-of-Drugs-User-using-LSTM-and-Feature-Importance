from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
import pandas as pd
from typing import List, Dict
from app.config import get_db
from app.model.database import DatasetUpload, SuspectRecord
from app.util.dataset_loader import get_dataset_by_id

router = APIRouter()

@router.get("/suspects/list", response_model=List[Dict])
def get_merged_suspects(dataset_upload_id: int = Query(...), db: Session = Depends(get_db)):
    suspects = db.query(SuspectRecord).all()

    suspects_data = []
    for s in suspects:
        suspects_data.append({
            "id": s.id,
            "tanggal": s.tanggal.isoformat() if s.tanggal else None,
            "no_register": s.no_register,
            "nama_tersangka": s.nama_tersangka,
            "no_lkn": s.no_lkn,
            "domisili": s.domisili,
            "pasal_disangkakan": s.pasal_disangkakan,
            "jenis_kelamin": s.jenis_kelamin,
            "tempat_lahir": s.tempat_lahir,
            "tanggal_lahir": s.tanggal_lahir.isoformat() if s.tanggal_lahir else None,
            "agama": s.agama,
            "usia": s.usia,
            "pendidikan_terakhir": s.pendidikan_terakhir,
            "pekerjaan": s.pekerjaan,
            "rekomendasi_tat": s.rekomendasi_tat,
        })

    try:
        df_dataset = get_dataset_by_id(dataset_upload_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset file: {e}")

    dataset_data = df_dataset.to_dict(orient='records')

    merged_data = suspects_data + dataset_data

    return merged_data
