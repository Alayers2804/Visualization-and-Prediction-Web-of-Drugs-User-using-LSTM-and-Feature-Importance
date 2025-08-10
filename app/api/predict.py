# app/api.py
from fastapi import APIRouter, Query, HTTPException
import pandas as pd
from app.model.preprocessing import load_monthly_case_data
from app.model.lstm_model import prepare_lstm_data, build_lstm_model, forecast_next
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

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
    test_size_ratio: float = 0.2,
    val_size_ratio: float = 0.1,
    target_column: str = "JUMLAH_KASUS",
    dataset_id: int = Query(...),
):
    monthly_data = load_monthly_case_data(dataset_id)

    if target_column not in monthly_data.columns:
        return {"error": f"Target column '{target_column}' not found."}

    seq_len = 6
    X, y, scaler = prepare_lstm_data(monthly_data, target_column=target_column, sequence_length=seq_len)

    n_samples = len(X)
    total_ratio = test_size_ratio + val_size_ratio

    SMALL_DATASET_THRESHOLD = 300  

    if n_samples < SMALL_DATASET_THRESHOLD:
        if test_size_ratio >= 1 or test_size_ratio <= 0:
            raise HTTPException(status_code=400, detail="test_size_ratio must be between 0 and 1")

        if n_samples * (1 - test_size_ratio) < 1:
            raise HTTPException(status_code=400, detail="Dataset too small for the requested test size.")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_ratio, shuffle=False)
        X_val, y_val = None, None
    else:
        if n_samples * (1 - total_ratio) < 1:
            raise HTTPException(status_code=400, detail="Dataset too small for the requested train/val/test split.")

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=total_ratio, shuffle=False)
        val_ratio_adjusted = val_size_ratio / total_ratio
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - val_ratio_adjusted, shuffle=False)

    model = build_lstm_model(X.shape[1:])

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    if X_val is not None:
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
    else:
        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler.inverse_transform(y_test).ravel()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
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
