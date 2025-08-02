from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class DatasetUpload(Base):
    __tablename__ = "dataset_uploads"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, nullable=False)
    path = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)

class SuspectRecord(Base):
    __tablename__ = "suspect_records"

    id = Column(Integer, primary_key=True, index=True)
    tanggal = Column(DateTime)
    no_register = Column(String)
    nama_tersangka = Column(String)
    no_lkn = Column(String)
    domisili = Column(String)
    pasal_disangkakan = Column(String)
    jenis_kelamin = Column(String)
    tempat_lahir = Column(String)
    tanggal_lahir = Column(DateTime)
    agama = Column(String)
    usia = Column(Integer)
    pendidikan_terakhir = Column(String)
    pekerjaan = Column(String)
    rekomendasi_tat = Column(String)
