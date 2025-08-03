# app/api.py
from fastapi import APIRouter, Query
from typing import List
import pandas as pd
from app.model.feature_analysis import prepare_features
from fastapi.responses import FileResponse, JSONResponse
from scipy.stats import entropy
import numpy as np
from app.util.dataset_loader import get_dataset_by_id
from app.config import get_timestamp, visualizer

router = APIRouter()


@router.get("/features")
def important_features(
    features: List[str] = Query(default=["USIA", "PEKERJAAN", "PENDIDIKAN TERAKHIR"]),
    dataset_id: int = Query(..., description="ID of the dataset to use")
):
    
    df = get_dataset_by_id(dataset_id)
    df.columns = df.columns.str.strip()

    for col in features:
        if col not in df.columns:
            return {"error": f"Column '{col}' not found in dataset."}

    Xf, _ = prepare_features(df, features)

    result = {}
    for col in features:
        values = Xf[col]
        var_score = np.var(values) / (np.mean(values) + 1e-6)
        value_counts = pd.Series(values).value_counts(normalize=True)
        entropy_score = entropy(value_counts)

        result[col] = {
            "variance_score": round(float(var_score), 4),
            "entropy_score": round(float(entropy_score), 4)
        }

    return {
        "feature_relevance": result,
        "used_features": features
    }

@router.get("/features/plot")
def feature_plot(
    features: List[str] = Query(default=["USIA", "PEKERJAAN", "PENDIDIKAN TERAKHIR"]),
    dataset_id: int = Query(..., description="ID of the dataset to use"),
    return_base64: bool = Query(False)
):
    df = get_dataset_by_id(dataset_id)
    Xf, _ = prepare_features(df, features)

    importance = {}
    for col in features:
        values = Xf[col]
        var_score = np.var(values) / (np.mean(values) + 1e-6)
        importance[col] = var_score

    if return_base64:
        image_base64 = visualizer.plot_feature_importance(
            importance, return_base64=True
        )
        return JSONResponse(content={
            "image_base64": image_base64,
            "feature_importance": importance,
            "used_features": features
        })
    else:
        feature_str = "_".join(features).lower()
        filename = visualizer.plot_feature_importance(
            importance, filename=f"feature_importance_{feature_str}_{get_timestamp}.png"
        )
        return FileResponse(filename, media_type="image/png")
    
