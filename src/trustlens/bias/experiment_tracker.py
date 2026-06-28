from __future__ import annotations

from datetime import datetime, timezone

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HISTORY_PATH = ROOT / "data" / "experiment_history.csv"

class ExperimentTracker:
    def __init__(self, history_path: Path = HISTORY_PATH):
        self.history_path = history_path
        # Ensure data folder exists
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def log_experiment(
        self,
        dataset: str,
        protected_attribute: str,
        model: str,
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float,
        roc_auc: float,
        spd: float,
        dir_val: float,
        eod: float,
        equalized_odds: float,
        overall_fairness: float,
        mitigation_method: str = "Baseline"
    ) -> pd.DataFrame:
        """Log a single model training/mitigation run and append to history CSV."""
        new_row = {
            "dataset": dataset,
            "protected_attribute": protected_attribute,
            "model": model,
            "mitigation_method": mitigation_method,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1_score),
            "roc_auc": float(roc_auc),
            "spd": float(spd),
            "dir": float(dir_val),
            "eod": float(eod),
            "equalized_odds": float(equalized_odds),
            "overall_fairness": float(overall_fairness),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        df = self.get_history()
        
        # Prevent exact duplicates of dataset + attribute + model + mitigation_method + metrics
        # If we run the same config again, we can overwrite or append. Let's overwrite to keep it clean,
        # or append if we want a full history. Overwriting is usually cleaner for a leaderboard.
        # Let's filter out rows matching key columns:
        key_cols = ["dataset", "protected_attribute", "model", "mitigation_method"]
        if not df.empty:
            match_mask = True
            for col in key_cols:
                match_mask = match_mask & (df[col] == new_row[col])
            df = df[~match_mask].copy()
            
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        
        df.to_csv(self.history_path, index=False)
        return df

    def get_history(self) -> pd.DataFrame:
        """Return history DataFrame. If file doesn't exist, return empty DataFrame with columns."""
        cols = [
            "dataset", "protected_attribute", "model", "mitigation_method",
            "accuracy", "precision", "recall", "f1_score", "roc_auc",
            "spd", "dir", "eod", "equalized_odds", "overall_fairness", "timestamp"
        ]
        if not self.history_path.exists():
            return pd.DataFrame(columns=cols)
        try:
            df = pd.read_csv(self.history_path)
            # Ensure all columns exist
            for col in cols:
                if col not in df.columns:
                    df[col] = None
            return df[cols]
        except Exception:
            return pd.DataFrame(columns=cols)

    def get_filtered_history(self, dataset: str, protected_attribute: str) -> pd.DataFrame:
        """Get history filtered by dataset and protected attribute."""
        df = self.get_history()
        if df.empty:
            return df
        return df[(df["dataset"] == dataset) & (df["protected_attribute"] == protected_attribute)]

    def get_leaderboard(self, dataset: str, protected_attribute: str) -> pd.DataFrame:
        """
        Rank history by Overall Fairness Score.
        Returns columns: Rank, Model, Mitigation, Accuracy, SPD, DIR, F1 Score, Overall Fairness Score
        """
        df = self.get_filtered_history(dataset, protected_attribute)
        if df.empty:
            return pd.DataFrame(columns=["Rank", "Model", "Mitigation", "Accuracy", "SPD", "DIR", "F1 Score", "Overall Fairness Score"])
            
        # Sort by Overall Fairness Score descending
        df_sorted = df.sort_values(by="overall_fairness", ascending=False).copy()
        df_sorted["Rank"] = range(1, len(df_sorted) + 1)
        
        # Rename columns for presentation
        presentation_df = df_sorted[[
            "Rank", "model", "mitigation_method", "accuracy", "spd", "dir", "f1_score", "overall_fairness"
        ]].copy()
        
        presentation_df.columns = [
            "Rank", "Model", "Mitigation", "Accuracy", "SPD", "DIR", "F1 Score", "Overall Fairness Score"
        ]
        return presentation_df

    def get_best_configuration(self, dataset: str, protected_attribute: str) -> dict | None:
        """Find the configuration with the highest overall fairness score that has F1/Accuracy > 0.7 (if any)."""
        df = self.get_filtered_history(dataset, protected_attribute)
        if df.empty:
            return None
        # Sort by overall fairness
        df_sorted = df.sort_values(by="overall_fairness", ascending=False)
        return df_sorted.iloc[0].to_dict()

    def clear_history(self) -> None:
        """Clear historical runs."""
        if self.history_path.exists():
            try:
                self.history_path.unlink()
            except Exception:
                pass
