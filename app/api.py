# app/api.py
from fastapi import APIRouter, Query
from typing import List
import pandas as pd
from app.model.preprocessing import load_monthly_case_data, clean_profile_data
from app.model.lstm_model import prepare_lstm_data, build_lstm_model, forecast_next
from app.model.feature_analysis import prepare_features, profile_summary
from fastapi.responses import FileResponse, JSONResponse
from app.util.visualization import Visualizer
from scipy.stats import entropy
import numpy as np
from datetime import datetime, timedelta

router = APIRouter()
visualizer = Visualizer()
file_path = "data/dataset_tat_rehab.xlsx"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


@router.get("/predict")
def predict(
    n_months: int = 2,
    target_column: str = "JUMLAH_KASUS"
):
    """
    Predict next n_months for the selected target column with future month labels.
    """
    monthly_data = load_monthly_case_data(file_path)

    if target_column not in monthly_data.columns:
        return {"error": f"Target column '{target_column}' not found in dataset."}

    X, y, scaler = prepare_lstm_data(monthly_data, target_column=target_column, sequence_length=3)
    model = build_lstm_model(X.shape[1:])
    model.fit(X, y, epochs=200, verbose=0)

    prediction = forecast_next(model, X[-1], scaler, n_steps=n_months).ravel().tolist()

    last_date = monthly_data["BULAN"].max() 
    future_months = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_months, freq='MS')
    future_months_str = future_months.strftime("%Y-%m")

    forecast_result = [
        {"month": month, "predicted_cases": round(value, 2)}
        for month, value in zip(future_months_str, prediction)
    ]

    return JSONResponse(content={
        "forecast": forecast_result,
        "target_column": target_column
    })

@router.get("/predict/plot")
def predict_plot(
    n_months: int = 2,
    target_column: str = "JUMLAH_KASUS",
    return_base64: bool = Query(False)
):
    monthly_data = load_monthly_case_data(file_path)

    if target_column not in monthly_data.columns:
        return {"error": f"Target column '{target_column}' not found."}

    X, y, scaler = prepare_lstm_data(
        monthly_data, target_column=target_column, sequence_length=3
    )

    model = build_lstm_model(X.shape[1:])
    model.fit(X, y, epochs=200, verbose=0)

    forecast = forecast_next(
        model, X[-1], scaler, n_steps=n_months
    ).ravel().tolist()

    if return_base64:
        img_base64 = visualizer.plot_forecast(
            df=monthly_data,
            predictions=forecast,
            date_column="BULAN",
            value_column=target_column,
            forecast_months=n_months,
            return_base64=True
        )
        return JSONResponse(content={
            "image_base64": img_base64,
            "forecast_values": forecast,
            "target_column": target_column
        })

    else:
        filename = visualizer.plot_forecast(
            df=monthly_data,
            predictions=forecast,
            date_column="BULAN",
            value_column=target_column,
            forecast_months=n_months,
            filename=f"forecast_{target_column}_{timestamp}.png"
        )
        return FileResponse(filename, media_type="image/png")

@router.get("/features")
def important_features(
    features: List[str] = Query(default=["USIA", "PEKERJAAN", "PENDIDIKAN TERAKHIR"])
):
    file_path = "data/dataset_tat_rehab.xlsx"
    df = pd.read_excel(file_path, sheet_name="GABUNGAN")
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
    return_base64: bool = Query(False)
):
    df = pd.read_excel(file_path, sheet_name="GABUNGAN")
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
            importance, filename=f"feature_importance_{feature_str}_{timestamp}.png"
        )
        return FileResponse(filename, media_type="image/png")
    
@router.get("/profile-summary")
def drug_user_profile_summary():
    df = pd.read_excel(file_path, sheet_name="GABUNGAN")

    try:
        cleaned_df = clean_profile_data(df)
        result = profile_summary(cleaned_df)
        return result
    except Exception as e:
        return {"error": str(e)}


@router.get("/profile-summary/plot")
def profile_plot(
    features: List[str] = Query(default=[]),
    return_base64: bool = Query(default=False)
):
    df = pd.read_excel(file_path, sheet_name="GABUNGAN")
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
                cleaned_df, feature, filename=f"profile_{feature.lower()}_{timestamp}.png"
            )
            filenames.append(filename)
        # For now, just return the first image (or zip later)
        return FileResponse(filenames[0], media_type="image/png")