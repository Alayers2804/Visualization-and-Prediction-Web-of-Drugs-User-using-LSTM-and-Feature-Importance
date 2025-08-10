# app/api.py
from fastapi import APIRouter, Query
import pandas as pd
from app.model.preprocessing import load_monthly_case_data
from app.model.lstm_model import prepare_lstm_data, build_lstm_model, forecast_next
from sklearn.metrics import mean_absolute_error, mean_squared_error
from io import BytesIO
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from fastapi.responses import FileResponse, JSONResponse
import numpy as np
from app.config import FILE_PATH as file_path
from app.util.dataset_loader import get_dataset_by_id
from app.config import get_timestamp, visualizer

router = APIRouter()


@router.get("/predict")
def predict(
    n_months: int = 2,
    target_column: str = "JUMLAH_KASUS",
    dataset_id: int = Query(..., description="ID of the dataset to use"),
):
    """
    Predict next n_months for the selected target column with future month labels.
    """
    monthly_data = load_monthly_case_data(dataset_id)

    if target_column not in monthly_data.columns:
        return {"error": f"Target column '{target_column}' not found in dataset."}

    X, y, scaler = prepare_lstm_data(
        monthly_data, target_column=target_column, sequence_length=3
    )
    model = build_lstm_model(X.shape[1:])
    model.fit(X, y, epochs=200, verbose=0)

    prediction = forecast_next(model, X[-1], scaler, n_steps=n_months).ravel().tolist()

    last_date = monthly_data["BULAN"].max()
    future_months = pd.date_range(
        start=last_date + pd.DateOffset(months=1), periods=n_months, freq="MS"
    )
    future_months_str = future_months.strftime("%Y-%m")

    forecast_result = [
        {"month": month, "predicted_cases": round(value, 2)}
        for month, value in zip(future_months_str, prediction)
    ]

    return JSONResponse(
        content={"forecast": forecast_result, "target_column": target_column}
    )

@router.get("/predict/test")
def test_prediction_metrics(
    test_size: int = 3,
    target_column: str = "JUMLAH_KASUS",
    dataset_id: int = Query(..., description="ID of the dataset to use"),
):
    
    monthly_data = load_monthly_case_data(dataset_id)

    if target_column not in monthly_data.columns:
        return {"error": f"Target column '{target_column}' not found in dataset."}

    X, y, scaler = prepare_lstm_data(
        monthly_data, target_column=target_column, sequence_length=3
    )

    # Validate test size
    if test_size >= len(X):
        return {"error": f"Test size too large. Must be < {len(X)}"}

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    model = build_lstm_model(X.shape[1:])
    model.fit(X_train, y_train, epochs=200, verbose=0)

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler.inverse_transform(y_test).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return JSONResponse(
        content={
            "evaluation": {
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mape": f"{round(mape, 2)}%",
            },
            "target_column": target_column,
            "actual_vs_predicted": [
                {"actual": round(a, 2), "predicted": round(p, 2)}
                for a, p in zip(y_true.tolist(), y_pred.tolist())
            ],
        }
    )


@router.get("/predict/plot")
def predict_plot(
    n_months: int = 2,
    target_column: str = "JUMLAH_KASUS",
    dataset_id: int = Query(..., description="ID of the dataset to use"),
    return_base64: bool = Query(False),
):
    monthly_data = load_monthly_case_data(dataset_id)

    if target_column not in monthly_data.columns:
        return {"error": f"Target column '{target_column}' not found."}

    X, y, scaler = prepare_lstm_data(
        monthly_data, target_column=target_column, sequence_length=3
    )

    model = build_lstm_model(X.shape[1:])
    model.fit(X, y, epochs=200, verbose=0)

    forecast = forecast_next(model, X[-1], scaler, n_steps=n_months).ravel().tolist()

    if return_base64:
        img_base64 = visualizer.plot_forecast(
            df=monthly_data,
            predictions=forecast,
            date_column="BULAN",
            value_column=target_column,
            forecast_months=n_months,
            return_base64=True,
        )
        return JSONResponse(
            content={
                "image_base64": img_base64,
                "forecast_values": forecast,
                "target_column": target_column,
            }
        )

    else:
        filename = visualizer.plot_forecast(
            df=monthly_data,
            predictions=forecast,
            date_column="BULAN",
            value_column=target_column,
            forecast_months=n_months,
            filename=f"forecast_{target_column}_{get_timestamp}.png",
        )
        return FileResponse(filename, media_type="image/png")


@router.get("/predict/available-areas")
def list_areas(dataset_id: int = Query(..., description="ID of the dataset to use")):
    df = get_dataset_by_id(dataset_id)
    areas = df["DOMISILI"].dropna().unique().tolist()
    return {"areas": sorted(areas)}


@router.get("/predict/heatmap")
def generate_heatmap(
    month: str = None,
    dataset_id: int = Query(..., description="ID of the dataset to use"),
    n_months: int = 1,
    target_column: str = "JUMLAH_KASUS",
):
    df = get_dataset_by_id(dataset_id)
    areas = df["DOMISILI"].dropna().unique()

    predictions = []

    for area in areas:
        monthly_data = load_monthly_case_data(dataset_id, area=area)

        if monthly_data.empty or target_column not in monthly_data.columns:
            predictions.append({
                "DOMISILI": area.upper().strip(),
                "JUMLAH_PREDIKSI": 0
            })
            continue

        X, y, scaler = prepare_lstm_data(
            monthly_data, target_column=target_column, sequence_length=3
        )

        # Skip if not enough data to form sequences
        if X.ndim != 3 or X.shape[0] == 0:
            predictions.append({
                "DOMISILI": area.upper().strip(),
                "JUMLAH_PREDIKSI": 0
            })
            continue

        model = build_lstm_model(X.shape[1:])
        model.fit(X, y, epochs=200, verbose=0)

        prediction = forecast_next(model, X[-1], scaler, n_steps=n_months).ravel().tolist()
        first_pred = round(prediction[0], 2)
        predictions.append({
            "DOMISILI": area.upper().strip(),
            "JUMLAH_PREDIKSI": first_pred
        })

    predictions_df = pd.DataFrame(predictions)

    return visualizer.plot_heatmap_dual_interactive(
        df=df,
        map_path="data/kota-kabupaten.json",
        predictions_df=predictions_df,
        month=month
    )
