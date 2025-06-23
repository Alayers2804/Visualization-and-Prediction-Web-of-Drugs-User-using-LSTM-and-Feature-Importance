# lstm_model.py
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional

def prepare_lstm_data(
    df: pd.DataFrame,
    target_column: str,
    sequence_length: int = 3,
    scaler: Optional[MinMaxScaler] = None
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Converts a univariate time series column into supervised sequences for LSTM.

    Parameters:
        df: DataFrame containing the time series
        target_column: Column to forecast
        sequence_length: Number of previous steps used for prediction
        scaler: Optional external MinMaxScaler (for consistency in pipeline)

    Returns:
        X: input sequences
        y: output values
        scaler: fitted scaler
    """
    values = df[target_column].values.reshape(-1, 1)

    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)
    else:
        scaled = scaler.transform(values)

    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i - sequence_length:i])
        y.append(scaled[i])

    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    """
    Builds and compiles an LSTM model for sequence forecasting.
    """
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def forecast_next(
    model: Sequential,
    last_sequence: np.ndarray,
    scaler: MinMaxScaler,
    n_steps: int = 1
) -> np.ndarray:
    """
    Forecast future values using trained LSTM model.

    Parameters:
        model: Trained LSTM model
        last_sequence: Last known sequence (shape = [seq_len, 1])
        scaler: Fitted MinMaxScaler
        n_steps: Number of future steps to forecast

    Returns:
        Array of inverse-transformed predicted values
    """
    predictions = []
    seq = last_sequence.copy()

    for _ in range(n_steps):
        pred = model.predict(seq[np.newaxis, :, :], verbose=0)[0]
        predictions.append(pred)
        seq = np.vstack([seq[1:], pred])

    return scaler.inverse_transform(np.array(predictions))
