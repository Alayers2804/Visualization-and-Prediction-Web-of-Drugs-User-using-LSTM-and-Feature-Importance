# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import predict, feature, profile

app = FastAPI(title="TAT & Rehab ML Service")

# Allow frontend (e.g., React on localhost:3000)
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

