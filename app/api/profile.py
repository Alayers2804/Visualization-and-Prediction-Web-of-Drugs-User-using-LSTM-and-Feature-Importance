# app/api.py
from fastapi import APIRouter, Query
from typing import List
import pandas as pd
from app.model.preprocessing import clean_profile_data
from app.model.feature_analysis import profile_summary
from fastapi.responses import FileResponse, JSONResponse
from app.config import FILE_PATH as file_path
from app.util.dataset_loader import get_dataset_by_id
from app.config import get_timestamp, visualizer

router = APIRouter()


@router.get("/profile-summary")
def drug_user_profile_summary(
    dataset_id: int = Query(..., description="ID of the dataset to use")
):
    df = get_dataset_by_id(dataset_id)

    try:
        cleaned_df = clean_profile_data(df)
        result = profile_summary(cleaned_df)
        return result
    except Exception as e:
        return {"error": str(e)}


@router.get("/profile-summary/plot")
def profile_plot(
    features: List[str] = Query(default=[]),
    dataset_id: int = Query(..., description="ID of the dataset to use"),
    return_base64: bool = Query(default=False),
):
    df = get_dataset_by_id(dataset_id)
    cleaned_df = clean_profile_data(df)

    if not features:
        features = ["PENDIDIKAN TERAKHIR", "PEKERJAAN", "USIA", "AGE_GROUP"]

    missing = [f for f in features if f not in cleaned_df.columns]
    if missing:
        return {"error": f"Features not found after cleaning: {missing}"}

    if return_base64:
        image_map = {}
        for feature in features:
            img_b64 = visualizer.plot_profile_distribution(
                cleaned_df, feature, return_base64=True
            )
            image_map[feature] = img_b64
        return JSONResponse(content={"images_base64": image_map})
    else:
        filenames = []
        for feature in features:
            filename = visualizer.plot_profile_distribution(
                cleaned_df,
                feature,
                filename=f"profile_{feature.lower()}_{get_timestamp}.png",
            )
            filenames.append(filename)
        # For now, just return the first image (or zip later)
        return FileResponse(filenames[0], media_type="image/png")
