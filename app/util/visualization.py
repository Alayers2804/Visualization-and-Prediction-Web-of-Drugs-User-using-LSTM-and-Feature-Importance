import matplotlib.pyplot as plt
import pandas as pd
import os
import base64
import io

class Visualizer:
    def __init__(self, output_dir="static"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def fig_to_base64(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return encoded

    def save_plot(self, fig, filename: str):
        full_path = os.path.join(self.output_dir, filename)
        fig.savefig(full_path)
        plt.close(fig)
        return full_path

    def plot_forecast(self, df, predictions, date_column, value_column,
                      forecast_months, filename=None, return_base64=False):
        last_date = df[date_column].iloc[-1]
        future_dates = pd.date_range(last_date + pd.DateOffset(months=1), periods=forecast_months, freq='MS')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df[date_column], df[value_column], label="Actual")
        ax.plot(future_dates, predictions, linestyle='--', marker='o', label="Forecast")
        ax.set_title(f"Forecast of {value_column}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        if return_base64:
            return self.fig_to_base64(fig)
        else:
            return self.save_plot(fig, filename)

    def plot_feature_importance(self, importance: dict, filename=None, return_base64=False):
        features = list(importance.keys())
        values = list(importance.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(features, values)
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance")
        ax.invert_yaxis()

        if return_base64:
            return self.fig_to_base64(fig)
        else:
            return self.save_plot(fig, filename)

    def plot_profile_distribution(self, df, feature, filename=None, return_base64=False):
        value_counts = df[feature].value_counts().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        value_counts.plot(kind="bar", ax=ax)
        ax.set_title(f"Distribution of {feature}")
        ax.set_ylabel("Count")

        if return_base64:
            return self.fig_to_base64(fig)
        else:
            return self.save_plot(fig, filename)
