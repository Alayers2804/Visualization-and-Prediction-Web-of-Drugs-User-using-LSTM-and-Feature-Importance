import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os
import base64
import io
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from folium.features import GeoJsonTooltip

class Visualizer:
    def __init__(self, output_dir="static"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def fig_to_base64(self, fig, bbox_inches='tight'):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches=bbox_inches)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return encoded

    def save_plot(self, fig, filename: str):
        full_path = os.path.join(self.output_dir, filename)
        fig.savefig(full_path)
        plt.close(fig)
        return full_path
    
    def plot_forecast(
        self, df, predictions, date_column, value_column,
        forecast_months, filename=None, return_base64=False
    ):
        # Ensure date_column is datetime
        df[date_column] = pd.to_datetime(df[date_column])

        last_date = df[date_column].iloc[-1]
        future_dates = pd.date_range(
            last_date + pd.DateOffset(months=1),
            periods=forecast_months,
            freq='MS'
        )

        if len(predictions) > forecast_months:
            step = len(predictions) // forecast_months
            monthly_preds = [predictions[i] for i in range(0, len(predictions), step)]
            monthly_preds = monthly_preds[:forecast_months]
        else:
            monthly_preds = predictions

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10), sharex=False)
        plt.subplots_adjust(hspace=0.5)

        # Actual data plot
        ax1.plot(df[date_column], df[value_column], marker='o', color='blue')
        ax1.set_title(f"Actual {value_column} Data")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Value")
        ax1.grid(True)
        ax1.tick_params(axis='x', labelrotation=45)
        for label in ax1.get_xticklabels():
            label.set_ha('right')

        ax1.set_xlim(df[date_column].min() - pd.Timedelta(days=10), df[date_column].max() + pd.Timedelta(days=10))

        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())

        # Prediction plot
        ax2.plot(
            future_dates, monthly_preds,
            linestyle='--', marker='o', color='orange', markersize=8
        )
        ax2.set_title(f"Predicted {value_column} for Next {forecast_months} Months")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Value")
        ax2.grid(True)

        ax2.tick_params(axis='x', labelrotation=45)
        for label in ax2.get_xticklabels():
            label.set_ha('right')

        ax2.set_xlim(future_dates[0] - pd.Timedelta(days=10), future_dates[-1] + pd.Timedelta(days=10))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())

        fig.tight_layout()

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

    def plot_heatmap_dual_interactive(self, df, map_path: str, predictions_df: pd.DataFrame, month: str = None) -> dict:
        df.columns = df.columns.str.strip()
        df["TANGGAL"] = pd.to_datetime(df["TANGGAL"], errors="coerce")
        df["BULAN"] = df["TANGGAL"].dt.to_period("M").astype(str)

        if month:
            df = df[df["BULAN"] == month]

        df["DOMISILI"] = df["DOMISILI"].str.upper().str.strip()
        case_counts = (
            df.groupby("DOMISILI", as_index=False)
            .size()
            .rename(columns={"size": "TOTAL_KASUS"})
        )

        map_df = gpd.read_file(map_path)
        map_df["DOMISILI"] = map_df["NAME_2"].str.upper().str.strip()

        merged = map_df.merge(case_counts, on="DOMISILI", how="left")
        merged["TOTAL_KASUS"] = merged["TOTAL_KASUS"].fillna(0)

        predictions_df["DOMISILI"] = predictions_df["DOMISILI"].str.upper().str.strip()
        merged = merged.merge(predictions_df, on="DOMISILI", how="left")
        merged["JUMLAH_PREDIKSI"] = merged["JUMLAH_PREDIKSI"].fillna(0)

        if merged["TOTAL_KASUS"].sum() == 0 and merged["JUMLAH_PREDIKSI"].sum() == 0:
            return {"error": f"Tidak ada data untuk bulan {month or 'yang dipilih'}."}

        center = [merged.geometry.centroid.y.mean(), merged.geometry.centroid.x.mean()]
        m = folium.Map(location=center, zoom_start=8, tiles="cartodbpositron")

        folium.Choropleth(
            geo_data=merged.to_json(),
            data=merged,
            columns=["DOMISILI", "JUMLAH_PREDIKSI"],
            key_on="feature.properties.DOMISILI",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Jumlah Prediksi"
        ).add_to(m)

        # Tooltip with both
        folium.GeoJson(
            merged,
            name="Kasus & Prediksi",
            style_function=lambda x: {
                "fillColor": "#transparent",
                "color": "black",
                "weight": 0.5
            },
            tooltip=GeoJsonTooltip(
                fields=["DOMISILI", "TOTAL_KASUS", "JUMLAH_PREDIKSI"],
                aliases=["Domisili:", "Jumlah Kasus Aktual:", "Jumlah Prediksi:"],
                localize=True
            )
        ).add_to(m)

        return {"map_html": m._repr_html_()}
