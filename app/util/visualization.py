import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os
import base64
import io
import geopandas as gpd
import matplotlib.pyplot as plt

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

    def plot_heatmap_dual(self, df, map_path: str, month: str = None) -> dict:
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

        if merged["TOTAL_KASUS"].sum() == 0:
            return {
                "error": f"Tidak ada data kasus untuk bulan {month or 'yang dipilih'}."
            }
        
        plot_df = merged.copy()
        plot_df.loc[plot_df["TOTAL_KASUS"] == 0, "TOTAL_KASUS"] = None 
        # Plot full map
        fig_full, ax_full = plt.subplots(1, 1, figsize=(10, 12))

        plot_df.plot(
            column="TOTAL_KASUS",
            cmap="Reds",
            linewidth=0.5,
            ax=ax_full,
            edgecolor="0.8",
            legend=True,
            scheme="quantiles",  # Or "equal_interval" if you prefer
            missing_kwds={
                "color": "lightgrey",
                "label": "No data",
            },
        )

        ax_full.set_title(f"Peta Kasus per DOMISILI {'(' + month + ')' if month else ''}")
        ax_full.axis("off")
        leg = ax_full.get_legend()
        if leg:
            leg.set_title("Jumlah Kasus")

        plt.tight_layout()
        full_base64 = self.fig_to_base64(fig_full, bbox_inches='tight')

        # Plot zoomed (focused) map
        non_zero = merged[merged["TOTAL_KASUS"] > 0]
        fig_zoom, ax_zoom = plt.subplots(1, 1, figsize=(14, 16))

        plot_df.plot(
            column="TOTAL_KASUS",
            cmap="Reds",
            linewidth=0.5,
            ax=ax_zoom,
            edgecolor="0.8",
            legend=True,
            scheme="quantiles",
            missing_kwds={
                "color": "lightgrey",
                "label": "No data",
            },
        )

        ax_zoom.set_title(f"Zoomed Peta Kasus {'(' + month + ')' if month else ''}")
        ax_zoom.axis("off")

        if not non_zero.empty:
            minx, miny, maxx, maxy = non_zero.total_bounds
            ax_zoom.set_xlim(minx, maxx)
            ax_zoom.set_ylim(miny, maxy)

        leg = ax_zoom.get_legend()
        if leg:
            leg.set_title("Jumlah Kasus")

        for idx, row in merged[merged["TOTAL_KASUS"] > 0].iterrows():
            label = f"{row['DOMISILI']}\n{int(row['TOTAL_KASUS'])}"
            ax_zoom.annotate(
                text=label,
                xy=(row["geometry"].centroid.x, row["geometry"].centroid.y),
                ha="center",
                fontsize=6,
                color="black",
                weight="bold",
            )

        plt.tight_layout()
        zoom_base64 = self.fig_to_base64(fig_zoom, bbox_inches='tight')

        return {
            "full_map_base64": full_base64,
            "focused_map_base64": zoom_base64,
        }
