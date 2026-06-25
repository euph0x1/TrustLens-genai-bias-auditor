import pytest
import numpy as np
from trustlens.fairness.metrics import (
    statistical_parity_difference,
    disparate_impact_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
    overall_fairness_score,
    get_certification_status
)

def test_statistical_parity_difference():
    # Group 0 (unpriv): 3 positive out of 10 -> 0.3 selection rate
    # Group 1 (priv): 8 positive out of 10 -> 0.8 selection rate
    y_true = np.ones(20)
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    sf = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,      1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    assert statistical_parity_difference(y_true, y_pred, sf) == pytest.approx(-0.5)


def test_disparate_impact_ratio():
    # Group 0 rate: 0.3. Group 1 rate: 0.6. DIR = 0.3 / 0.6 = 0.5
    y_true = np.ones(20)
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    sf = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,      1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    assert disparate_impact_ratio(y_true, y_pred, sf) == pytest.approx(0.5)


def test_equal_opportunity_difference():
    # Group 0 TPR: 2 / 4 = 0.5
    # Group 1 TPR: 4 / 5 = 0.8
    # EOD = 0.5 - 0.8 = -0.3
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0,  1, 1, 1, 1, 0, 1, 1, 0, 0, 0])
    sf = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,      1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    assert equal_opportunity_difference(y_true, y_pred, sf) == pytest.approx(-0.3)


def test_equalized_odds_difference():
    # Group 0 TPR: 0.5, FPR: 2 / 6 = 0.333
    # Group 1 TPR: 0.8, FPR: 2 / 5 = 0.4
    # TPR diff = 0.3, FPR diff = 0.067
    # EqOdds = max(0.3, 0.067) = 0.3
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0,  1, 1, 1, 1, 0, 1, 1, 0, 0, 0])
    sf = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,      1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    assert equalized_odds_difference(y_true, y_pred, sf) == pytest.approx(0.3)


def test_overall_fairness_score():
    # DIR = 1.0, SPD = 0.0, EOD = 0.0, EqOdds = 0.0 -> 100%
    assert overall_fairness_score(0.0, 1.0, 0.0, 0.0) == 100.0
    
    # DIR = 0.5 (score = 0.5), SPD = -0.5 (score = 0.5), EOD = -0.5 (score = 0.5), EqOdds = 0.5 (score = 0.5) -> 50%
    assert overall_fairness_score(-0.5, 0.5, -0.5, 0.5) == 50.0


def test_get_certification_status():
    # Perfect
    status, badge, code = get_certification_status(0.0, 1.0)
    assert status == "Certified Fair"
    assert code == 0
    
    # High Disparity
    status, badge, code = get_certification_status(0.25, 0.5)
    assert status == "High Disparity Risk"
    assert code == 2
    
    # Moderate / Warning
    status, badge, code = get_certification_status(0.15, 0.75)
    assert status == "Needs Mitigation"
    assert code == 1
