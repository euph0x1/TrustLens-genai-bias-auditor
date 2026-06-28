from __future__ import annotations

import numpy as np
import pandas as pd
from trustlens.fairness.metrics import compute_fairness_metrics

class BiasDetector:
    def __init__(self):
        pass

    def analyze_model_bias(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_features: pd.Series,
        privileged_label: str = "Privileged",
        unprivileged_label: str = "Unprivileged"
    ) -> dict:
        """
        Evaluate model predictions and calculate group-wise prediction rates and fairness metrics.
        """
        y_pred = model.predict(X_test)
        if y_pred.dtype.kind in "fc":
            y_pred = (y_pred >= 0.5).astype(int)
            
        y_true = np.asarray(y_test)
        sf = np.asarray(sensitive_features)
        
        # Calculate fairness metrics
        fairness_results = compute_fairness_metrics(y_true, y_pred, sf)
        
        # Calculate group-wise rates
        group_results = {}
        for g_val, g_label in [(0, unprivileged_label), (1, privileged_label)]:
            mask = (sf == g_val)
            if not np.any(mask):
                continue
                
            y_t_g = y_true[mask]
            y_p_g = y_pred[mask]
            
            # Selection rate
            sel_rate = float(np.mean(y_p_g == 1))
            
            # Accuracy
            acc = float(np.mean(y_t_g == y_p_g))
            
            # TPR, FPR, FNR, TNR
            pos_mask = (y_t_g == 1)
            neg_mask = (y_t_g == 0)
            
            tpr = 1.0 if not np.any(pos_mask) else float(np.mean(y_p_g[pos_mask] == 1))
            fnr = 1.0 - tpr
            
            fpr = 0.0 if not np.any(neg_mask) else float(np.mean(y_p_g[neg_mask] == 1))
            tnr = 1.0 - fpr
            
            group_results[g_label] = {
                "selection_rate": sel_rate,
                "accuracy": acc,
                "tpr": tpr,
                "fpr": fpr,
                "fnr": fnr,
                "tnr": tnr,
                "sample_count": int(np.sum(mask))
            }
            
        return {
            "fairness_metrics": fairness_results,
            "group_metrics": group_results
        }
