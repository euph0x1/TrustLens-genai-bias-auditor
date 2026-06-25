from __future__ import annotations

import numpy as np

def statistical_parity_difference(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    sensitive_features: np.ndarray | list
) -> float:
    """
    Compute Statistical Parity Difference.
    Difference in positive prediction rates between unprivileged (0) and privileged (1) groups.
    Formula: P(y_pred = 1 | sens = 0) - P(y_pred = 1 | sens = 1)
    """
    y_p = np.asarray(y_pred)
    sf = np.asarray(sensitive_features)
    
    mask_unpriv = (sf == 0)
    mask_priv = (sf == 1)
    
    if not np.any(mask_unpriv) or not np.any(mask_priv):
        return 0.0
        
    rate_unpriv = np.mean(y_p[mask_unpriv] == 1)
    rate_priv = np.mean(y_p[mask_priv] == 1)
    
    return float(rate_unpriv - rate_priv)


def disparate_impact_ratio(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    sensitive_features: np.ndarray | list
) -> float:
    """
    Compute Disparate Impact Ratio.
    Ratio of selection rates of unprivileged (0) to privileged (1) groups.
    Formula: P(y_pred = 1 | sens = 0) / P(y_pred = 1 | sens = 1)
    """
    y_p = np.asarray(y_pred)
    sf = np.asarray(sensitive_features)
    
    mask_unpriv = (sf == 0)
    mask_priv = (sf == 1)
    
    if not np.any(mask_unpriv) or not np.any(mask_priv):
        return 1.0
        
    rate_unpriv = np.mean(y_p[mask_unpriv] == 1)
    rate_priv = np.mean(y_p[mask_priv] == 1)
    
    if rate_priv == 0:
        return 1.0 if rate_unpriv == 0 else 999.0
        
    return float(rate_unpriv / rate_priv)


def equal_opportunity_difference(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    sensitive_features: np.ndarray | list
) -> float:
    """
    Compute Equal Opportunity Difference.
    Difference in True Positive Rates (TPR) between unprivileged (0) and privileged (1) groups.
    Formula: TPR_unpriv - TPR_priv
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    sf = np.asarray(sensitive_features)
    
    mask_unpriv_pos = (sf == 0) & (y_t == 1)
    mask_priv_pos = (sf == 1) & (y_t == 1)
    
    if not np.any(mask_unpriv_pos) or not np.any(mask_priv_pos):
        return 0.0
        
    tpr_unpriv = np.mean(y_p[mask_unpriv_pos] == 1)
    tpr_priv = np.mean(y_p[mask_priv_pos] == 1)
    
    return float(tpr_unpriv - tpr_priv)


def equalized_odds_difference(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    sensitive_features: np.ndarray | list
) -> float:
    """
    Compute Equalized Odds Difference.
    The maximum of absolute differences in True Positive Rates (TPR) and False Positive Rates (FPR).
    Formula: max(|TPR_unpriv - TPR_priv|, |FPR_unpriv - FPR_priv|)
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    sf = np.asarray(sensitive_features)
    
    # TPR calculation
    mask_unpriv_pos = (sf == 0) & (y_t == 1)
    mask_priv_pos = (sf == 1) & (y_t == 1)
    
    tpr_diff = 0.0
    if np.any(mask_unpriv_pos) and np.any(mask_priv_pos):
        tpr_unpriv = np.mean(y_p[mask_unpriv_pos] == 1)
        tpr_priv = np.mean(y_p[mask_priv_pos] == 1)
        tpr_diff = abs(tpr_unpriv - tpr_priv)
        
    # FPR calculation
    mask_unpriv_neg = (sf == 0) & (y_t == 0)
    mask_priv_neg = (sf == 1) & (y_t == 0)
    
    fpr_diff = 0.0
    if np.any(mask_unpriv_neg) and np.any(mask_priv_neg):
        fpr_unpriv = np.mean(y_p[mask_unpriv_neg] == 1)
        fpr_priv = np.mean(y_p[mask_priv_neg] == 1)
        fpr_diff = abs(fpr_unpriv - fpr_priv)
        
    return float(max(tpr_diff, fpr_diff))


def overall_fairness_score(
    spd: float,
    dir_ratio: float,
    eod: float,
    eq_odds: float
) -> float:
    """
    Compute Overall Fairness Score (0-100%).
    Normalized average of SPD, DIR, EOD, and Equalized Odds.
    """
    # 1. DIR normalization
    if dir_ratio <= 0:
        score_dir = 0.0
    else:
        score_dir = min(dir_ratio, 1.0 / dir_ratio)
    score_dir = min(score_dir, 1.0)
    
    # 2. SPD normalization (perfect is 0, worst is 1)
    score_spd = 1.0 - min(abs(spd), 1.0)
    
    # 3. EOD normalization (perfect is 0, worst is 1)
    score_eod = 1.0 - min(abs(eod), 1.0)
    
    # 4. EqOdds normalization (perfect is 0, worst is 1)
    score_eq_odds = 1.0 - min(abs(eq_odds), 1.0)
    
    overall = (score_dir + score_spd + score_eod + score_eq_odds) / 4.0
    return float(overall * 100.0)


def get_certification_status(
    spd: float,
    dir_ratio: float
) -> tuple[str, str, int]:
    """
    Returns (status_text, badge_text, status_code).
    Codes: 0 = Certified Fair, 1 = Needs Mitigation, 2 = High Disparity Risk.
    """
    abs_spd = abs(spd)
    
    # Certified Fair: DIR >= 0.8 and SPD <= 0.1
    if dir_ratio >= 0.8 and abs_spd <= 0.1:
        return "Certified Fair", "✅ Certified Fair", 0
        
    # High Disparity Risk: DIR < 0.6 or SPD > 0.2
    if dir_ratio < 0.6 or abs_spd > 0.2:
        return "High Disparity Risk", "❌ High Disparity Risk", 2
        
    # Default: Needs Mitigation
    return "Needs Mitigation", "⚠️ Needs Mitigation", 1


def compute_fairness_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    sensitive_features: np.ndarray | list
) -> dict[str, float]:
    """Helper to return all metrics as a dictionary."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    sf = np.asarray(sensitive_features)
    
    spd = statistical_parity_difference(y_t, y_p, sf)
    dir_ratio = disparate_impact_ratio(y_t, y_p, sf)
    eod = equal_opportunity_difference(y_t, y_p, sf)
    eq_odds = equalized_odds_difference(y_t, y_p, sf)
    
    overall = overall_fairness_score(spd, dir_ratio, eod, eq_odds)
    status_text, badge, code = get_certification_status(spd, dir_ratio)
    
    return {
        "statistical_parity_difference": spd,
        "disparate_impact_ratio": dir_ratio,
        "equal_opportunity_difference": eod,
        "equalized_odds_difference": eq_odds,
        "overall_fairness_score": overall,
        "certification_code": float(code)
    }
