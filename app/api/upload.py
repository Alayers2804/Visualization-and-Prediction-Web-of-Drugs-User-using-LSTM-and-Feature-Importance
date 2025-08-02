from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil, os
from app.config import SessionLocal
from app.model.database import DatasetUpload

router = APIRouter()

UPLOAD_DIR = "data/raw_uploads"

@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Save file info to DB
        db = SessionLocal()
        dataset = DatasetUpload(filename=file.filename, path=file_path)
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        db.close()

        return {
            "message": "Dataset uploaded and recorded successfully",
            "filename": dataset.filename,
            "path": dataset.path,
            "uploaded_at": dataset.uploaded_at,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list-datasets")
def list_uploaded_datasets():
    db = SessionLocal()
    datasets = db.query(DatasetUpload).all()
    db.close()
    return datasets
