import pytest
import pandas as pd
import numpy as np
from trustlens.bias.bias_mitigation import calculate_reweighing_weights

def test_reweighing_weights_calculation():
    # Simple data with severe bias:
    # Group 0: all positive (3/3 positive)
    # Group 1: all negative (3/3 negative)
    y_train = pd.Series([1, 1, 1, 0, 0, 0])
    sens_train = pd.Series([0, 0, 0, 1, 1, 1])
    
    weights = calculate_reweighing_weights(y_train, sens_train)
    
    assert len(weights) == 6
    # For unprivileged (0) positive: P(A=0)=0.5, P(Y=1)=0.5, P(A=0,Y=1)=0.5 -> W = 0.5 * 0.5 / 0.5 = 0.5
    # For privileged (1) negative: P(A=1)=0.5, P(Y=0)=0.5, P(A=1,Y=0)=0.5 -> W = 0.5 * 0.5 / 0.5 = 0.5
    # The weights should balance out
    for w in weights:
        assert w == pytest.approx(0.5)
