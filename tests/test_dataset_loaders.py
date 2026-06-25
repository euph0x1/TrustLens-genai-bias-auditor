import pytest
import pandas as pd
from trustlens.bias.dataset_loader import DatasetRegistry

def test_dataset_registry_listing():
    datasets = DatasetRegistry.list_datasets()
    assert len(datasets) == 3
    ids = [d["id"] for d in datasets]
    assert "adult" in ids
    assert "ibm_hr" in ids
    assert "hiring" in ids


def test_adult_census_loader():
    loader = DatasetRegistry.get_loader("adult")
    df = loader.load_raw_data()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1000
    assert loader.target_column in df.columns
    assert "sex" in df.columns
    assert "race" in df.columns
    
    # Preprocess test
    X_tr, X_te, y_tr, y_te, s_tr, s_te, feat_names = loader.preprocess_data(df, "sex")
    assert len(X_tr) > 0
    assert len(X_te) > 0
    assert len(feat_names) > 0
    # Check that sex_Male or similar dummy exists
    assert any(f.startswith("sex_") for f in feat_names)


def test_ibm_hr_loader():
    loader = DatasetRegistry.get_loader("ibm_hr")
    df = loader.load_raw_data()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1000
    assert loader.target_column in df.columns
    
    X_tr, X_te, y_tr, y_te, s_tr, s_te, feat_names = loader.preprocess_data(df, "Gender")
    assert len(X_tr) > 0
    assert len(X_te) > 0


def test_hiring_loader():
    loader = DatasetRegistry.get_loader("hiring")
    df = loader.load_raw_data()
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1000
    assert loader.target_column in df.columns
    
    X_tr, X_te, y_tr, y_te, s_tr, s_te, feat_names = loader.preprocess_data(df, "Gender")
    assert len(X_tr) > 0
    assert len(X_te) > 0
