import pandas as pd
import numpy as np
from pathlib import Path

from trustlens.io.visuals import render_tradeoff_scatter, render_leaderboard_table, render_shap_global


def make_sample_history():
    df = pd.DataFrame([
        {"accuracy": 0.85, "spd": -0.12, "overall_fairness": 72.5, "model": "RF", "mitigation_method": "Baseline"},
        {"accuracy": 0.80, "spd": -0.05, "overall_fairness": 92.5, "model": "LR", "mitigation_method": "Baseline"}
    ])
    return df


def test_render_tradeoff_and_leaderboard(tmp_path):
    df = make_sample_history()
    img = render_tradeoff_scatter(df, name="test_tradeoff")
    assert img is not None
    assert Path(img).exists()

    # Build leaderboard-like DF
    lb = pd.DataFrame({
        "Rank": [1, 2],
        "Model": ["LR", "RF"],
        "Mitigation": ["Baseline", "Baseline"],
        "Accuracy": [0.8, 0.85],
        "SPD": [-0.05, -0.12],
        "DIR": [0.85, 0.65],
        "F1 Score": [0.785, 0.835],
        "Overall Fairness Score": [92.5, 72.5]
    })
    img2 = render_leaderboard_table(lb, name="test_leaderboard")
    assert img2 is not None
    assert Path(img2).exists()


def test_render_shap_global(tmp_path):
    df = pd.DataFrame({"Feature": [f"f{i}" for i in range(10)], "Importance": np.linspace(0.1, 1.0, 10)})
    img = render_shap_global(df, name="test_shap")
    assert img is not None
    assert Path(img).exists()
