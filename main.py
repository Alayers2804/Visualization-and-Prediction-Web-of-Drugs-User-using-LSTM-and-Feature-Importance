# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import predict, feature, profile, upload
from app.config import init_db

init_db()
app = FastAPI(title="TAT & Rehab ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, tags=["Prediction"])
app.include_router(feature.router, tags=["Feature Analysis"])
app.include_router(profile.router, tags=["Profile Summary"])
app.include_router(upload.router, tags=["Dataset Upload"])
