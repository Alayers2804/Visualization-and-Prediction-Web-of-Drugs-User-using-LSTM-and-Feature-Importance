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
def create_suspect(record: SuspectRecordCreate):

    def validate_date(date_value, field_name):
        if date_value is None:
            raise HTTPException(
                status_code=400,
                detail={field_name: "This field is required"}
            )

        # Accept both string & datetime
        if isinstance(date_value, datetime):
            return date_value

        if isinstance(date_value, str):
            formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d",
                "%d-%m-%Y",
                "%d/%m/%Y",
                "%Y/%m/%d",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(date_value, fmt)
                except ValueError:
                    continue

        raise HTTPException(
            status_code=400,
            detail={field_name: f"Invalid date format: '{date_value}'. Expected one of: YYYY-MM-DD, DD-MM-YYYY, or ISO datetime"}
        )

    # Run validation
    record.tanggal = validate_date(record.tanggal, "tanggal")
    record.tanggal_lahir = validate_date(record.tanggal_lahir, "tanggal_lahir")

    db = SessionLocal()

    # Save to DB
    db_record = SuspectRecord(**record.dict())
    db.add(db_record)
    db.commit()
    db.refresh(db_record)

    return {"message": "Record created", "id": db_record.id}