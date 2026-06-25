from __future__ import annotations

import os
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import plotly.express as px

ROOT = Path(__file__).resolve().parents[3]
ASSETS_DIR = ROOT / "reports" / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def save_plotly_figure(fig, name: str) -> Path | None:
    """Save a Plotly figure to a png file and return the path."""
    out = ASSETS_DIR / f"{name}.png"
    # Use kaleido via plotly.io
    try:
        pio.write_image(fig, str(out), format="png", scale=2)
        return out
    except Exception:
        # If the image renderer fails, do not return a non-image fallback.
        return None

def save_matplotlib_figure(fig, name: str) -> Path:
    out = ASSETS_DIR / f"{name}.png"
    fig.savefig(str(out), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out

def render_tradeoff_scatter(history_df: pd.DataFrame, name: str = "tradeoff") -> Path | None:
    if history_df.empty:
        return None
    fig = px.scatter(
        history_df,
        x="accuracy",
        y="spd",
        color="model",
        symbol="mitigation_method",
        size="overall_fairness",
        hover_data=["model", "mitigation_method", "overall_fairness"],
    )
    return save_plotly_figure(fig, name)

def render_leaderboard_table(leaderboard_df: pd.DataFrame, name: str = "leaderboard") -> Path | None:
    if leaderboard_df.empty:
        return None
    # Create a simple bar chart of Overall Fairness Score
    df = leaderboard_df.copy()
    if "Overall Fairness Score" not in df.columns:
        return None
    fig = px.bar(df, x="Model", y="Overall Fairness Score", color="Model")
    return save_plotly_figure(fig, name)

def render_shap_global(importance_df: pd.DataFrame, name: str = "shap_global") -> Path | None:
    if importance_df is None or importance_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    top = importance_df.head(10).sort_values(by="Importance")
    ax.barh(top["Feature"], top["Importance"], color="#1e3d59")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top 10 Feature Importances (SHAP)")
    return save_matplotlib_figure(fig, name)
