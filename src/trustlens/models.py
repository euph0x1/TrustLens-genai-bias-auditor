from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class ModelRegistry:
    """Manages training and evaluation of baseline and mitigated models."""
    
    @staticmethod
    def get_estimator(model_type: str, random_state: int = 42):
        """Instantiate base ML models."""
        if model_type == "Logistic Regression":
            return LogisticRegression(max_iter=1000, random_state=random_state)
        elif model_type == "Random Forest":
            return RandomForestClassifier(n_estimators=100, random_state=random_state)
        elif model_type == "XGBoost":
            # Lazy import since it is in requirements
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @classmethod
    def train_model(
        cls,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: pd.Series | None = None
    ):
        """Train a selected model with optional sample weights."""
        model = cls.get_estimator(model_type)
        if sample_weight is not None:
            # Align sample weights index with y_train
            weights = np.asarray(sample_weight)
            model.fit(X_train, y_train, sample_weight=weights)
        else:
            model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Compute standard classification evaluation metrics on test set."""
        y_pred = model.predict(X_test)
        
        # In case predict returns continuous probabilities (e.g. some regressors/wrappers), threshold at 0.5
        if y_pred.dtype.kind in "fc":
            y_pred = (y_pred >= 0.5).astype(int)
            
        y_true = np.asarray(y_test)
        
        # Probability for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
        else:
            y_prob = y_pred
            
        # Handle single class case in test data for ROC-AUC
        if len(np.unique(y_true)) < 2:
            roc_auc = 0.5
        else:
            roc_auc = float(roc_auc_score(y_true, y_prob))

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": roc_auc,
            "y_pred": y_pred
        }
