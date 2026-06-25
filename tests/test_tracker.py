import pytest
import pandas as pd
from pathlib import Path
from trustlens.bias.experiment_tracker import ExperimentTracker

def test_experiment_tracker_logging(tmp_path):
    history_file = tmp_path / "test_history.csv"
    tracker = ExperimentTracker(history_path=history_file)
    
    # Empty tracker check
    df_empty = tracker.get_history()
    assert df_empty.empty
    
    # Log run
    tracker.log_experiment(
        dataset="Adult Census",
        protected_attribute="sex",
        model="Random Forest",
        accuracy=0.85,
        precision=0.84,
        recall=0.83,
        f1_score=0.835,
        roc_auc=0.91,
        spd=-0.15,
        dir_val=0.65,
        eod=-0.10,
        equalized_odds=0.12,
        overall_fairness=72.5,
        mitigation_method="Baseline"
    )
    
    df = tracker.get_history()
    assert len(df) == 1
    assert df.iloc[0]["model"] == "Random Forest"
    assert df.iloc[0]["overall_fairness"] == pytest.approx(72.5)
    
    # Log duplicated run key with different accuracy -> should overwrite to keep leaderboard neat
    tracker.log_experiment(
        dataset="Adult Census",
        protected_attribute="sex",
        model="Random Forest",
        accuracy=0.87,
        precision=0.84,
        recall=0.83,
        f1_score=0.835,
        roc_auc=0.91,
        spd=-0.15,
        dir_val=0.65,
        eod=-0.10,
        equalized_odds=0.12,
        overall_fairness=72.5,
        mitigation_method="Baseline"
    )
    df_updated = tracker.get_history()
    assert len(df_updated) == 1
    assert df_updated.iloc[0]["accuracy"] == pytest.approx(0.87)
    
    # Leaderboard ranking test
    tracker.log_experiment(
        dataset="Adult Census",
        protected_attribute="sex",
        model="Logistic Regression",
        accuracy=0.80,
        precision=0.79,
        recall=0.78,
        f1_score=0.785,
        roc_auc=0.85,
        spd=-0.05,
        dir_val=0.85,
        eod=-0.02,
        equalized_odds=0.04,
        overall_fairness=92.5,
        mitigation_method="Baseline"
    )
    
    leaderboard = tracker.get_leaderboard("Adult Census", "sex")
    assert len(leaderboard) == 2
    # Logistic Regression should be Rank 1 due to higher overall fairness score (92.5 vs 72.5)
    assert leaderboard.iloc[0]["Model"] == "Logistic Regression"
    assert leaderboard.iloc[0]["Rank"] == 1
    assert leaderboard.iloc[1]["Model"] == "Random Forest"
    assert leaderboard.iloc[1]["Rank"] == 2
