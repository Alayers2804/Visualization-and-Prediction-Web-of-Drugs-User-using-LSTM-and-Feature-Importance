from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil, os
from app.config import SessionLocal
from app.model.database import DatasetUpload, SuspectRecord
import pandas as pd
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import SessionLocal

router = APIRouter()

UPLOAD_DIR = "data/raw_uploads"

class SuspectRecordCreate(BaseModel):
    tanggal: datetime
    no_register: str
    nama_tersangka: str
    no_lkn: str
    domisili: str
    pasal_disangkakan: str
    jenis_kelamin: str
    tempat_lahir: str
    tanggal_lahir: datetime
    agama: str
    usia: int
    pendidikan_terakhir: str
    pekerjaan: str
    rekomendasi_tat: str

@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        # Validate file extension
        if not file.filename.endswith(".xlsx"):
            raise HTTPException(status_code=400, detail="Hanya file .xlsx yang diperbolehkan")

        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Save metadata to DB
        db = SessionLocal()
        dataset = DatasetUpload(filename=file.filename, path=file_path)
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        db.close()

        return {
            "message": "Berhasil mengunggah dan memverifikasi dataset.",
            "filename": dataset.filename,
            "uploaded_at": dataset.uploaded_at,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list-datasets")
def list_uploaded_datasets():
    db = SessionLocal()
    datasets = db.query(DatasetUpload).all()
    db.close()
    return datasets

@router.post("/suspects/")
def create_suspect(record: SuspectRecordCreate, db: Session = Depends(SessionLocal)):
    db_record = SuspectRecord(**record.dict())
    db.add(db_record)
    db.commit()
    db.refresh(db_record)
    return {"message": "Record created", "id": db_record.id}