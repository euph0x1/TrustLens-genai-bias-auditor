from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"

class DatasetLoader:
    def __init__(self, name: str, csv_path: Path):
        self.name = name
        self.csv_path = csv_path
        self.target_column = ""
        self.protected_attributes = {}  # attr_name -> {privileged_val, unprivileged_val}
        self.categorical_features = []
        self.numerical_features = []

    def exists(self) -> bool:
        return self.csv_path.exists()

    def generate_fallback(self) -> pd.DataFrame:
        raise NotImplementedError

    def load_raw_data(self) -> pd.DataFrame:
        if self.exists():
            try:
                df = pd.read_csv(self.csv_path)
                # Verify schema has target column, else generate fallback
                if self.target_column in df.columns:
                    return df
            except Exception:
                pass
        
        # Fallback to generating synthetic data
        df = self.generate_fallback()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.csv_path, index=False)
        return df

    def preprocess_data(self, df: pd.DataFrame, protected_attribute: str) -> tuple[
        pd.DataFrame,  # X_train
        pd.DataFrame,  # X_test
        pd.Series,     # y_train
        pd.Series,     # y_test
        pd.Series,     # sens_train
        pd.Series,     # sens_test
        list[str]      # feature_names
    ]:
        # Drop columns not used
        y = df[self.target_column].copy()
        
        # Determine sensitive attribute groups
        # Derived age group if age_group is requested but doesn't exist
        sens_col = protected_attribute
        if sens_col == "age_group" and "age_group" not in df.columns:
            # Look for numerical age
            age_col = next((c for c in df.columns if c.lower() == "age"), None)
            if age_col:
                df = df.copy()
                df["age_group"] = np.where(df[age_col] <= 35, "Young", "Older")
            else:
                df = df.copy()
                df["age_group"] = "Older" # Default fallback
        
        sens_series = df[sens_col].copy()
        
        # Keep features
        all_features = self.categorical_features + self.numerical_features
        # Ensure we don't include target
        all_features = [f for f in all_features if f != self.target_column]
        
        # We also might want to drop target and protected attribute from the feature set 
        # to avoid direct leak, or keep them depending on the experimental setup.
        # PRD specifies: Target: Hire/Reject, Features: Gender, Age, Education... 
        # and Protected Attributes: Gender, Age Group.
        # Typically, fairness frameworks train with protected attributes and then evaluate,
        # but let's keep all features in X to allow SHAP explanation of protected attributes!
        X = df[all_features].copy()
        
        # Split train/test (80/20) with stratification if possible
        X_train_raw, X_test_raw, y_train, y_test, sens_train_raw, sens_test_raw = train_test_split(
            X, y, sens_series, test_size=0.2, random_state=42, stratify=y
        )
        
        # Preprocessing: Manual mapping of categories to numeric/one-hot encoding
        # This keeps feature representation clean for SHAP and constraints.
        X_train = pd.DataFrame(index=X_train_raw.index)
        X_test = pd.DataFrame(index=X_test_raw.index)
        
        # Scale numerical
        for col in self.numerical_features:
            if col in X_train_raw.columns:
                mean = X_train_raw[col].mean()
                std = X_train_raw[col].std() + 1e-9
                X_train[col] = (X_train_raw[col] - mean) / std
                X_test[col] = (X_test_raw[col] - mean) / std
                
        # One-hot encode categorical features
        # Keep track of categories to prevent mismatch
        feature_names = list(self.numerical_features)
        
        for col in self.categorical_features:
            if col in X_train_raw.columns:
                # Find unique categories from raw df to ensure identical encoding
                cats = sorted(df[col].dropna().unique())
                for cat in cats:
                    dummy_name = f"{col}_{cat}"
                    X_train[dummy_name] = (X_train_raw[col] == cat).astype(float)
                    X_test[dummy_name] = (X_test_raw[col] == cat).astype(float)
                    feature_names.append(dummy_name)
                    
        # Encode sensitive attribute as binary: 1 for privileged, 0 for unprivileged
        priv_vals = self.protected_attributes[protected_attribute]["privileged"]
        if not isinstance(priv_vals, list):
            priv_vals = [priv_vals]
            
        sens_train = sens_train_raw.isin(priv_vals).astype(int)
        sens_test = sens_test_raw.isin(priv_vals).astype(int)
        
        return X_train, X_test, y_train, y_test, sens_train, sens_test, feature_names


class AdultCensusLoader(DatasetLoader):
    def __init__(self):
        super().__init__("Adult Census Income", DATA_DIR / "adult.csv")
        self.target_column = "income"
        self.protected_attributes = {
            "sex": {"privileged": "Male", "unprivileged": "Female", "label": "Gender (Sex)"},
            "race": {"privileged": "White", "unprivileged": "Non-White", "label": "Race"},
            "age_group": {"privileged": "Older", "unprivileged": "Young", "label": "Age Group"}
        }
        self.numerical_features = ["age", "capital-gain", "capital-loss", "hours-per-week"]
        self.categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex"]

    def generate_fallback(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 1000
        
        age = np.random.randint(18, 75, n)
        sex = np.random.choice(["Male", "Female"], n, p=[0.67, 0.33])
        race = np.random.choice(["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"], n, p=[0.85, 0.09, 0.03, 0.01, 0.02])
        workclass = np.random.choice(["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov"], n)
        education = np.random.choice(["Bachelors", "Some-college", "HS-grad", "Masters", "Doctorate", "Assoc-voc"], n, p=[0.25, 0.3, 0.3, 0.1, 0.02, 0.03])
        marital_status = np.random.choice(["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"], n)
        occupation = np.random.choice(["Exec-managerial", "Prof-specialty", "Craft-repair", "Sales", "Adm-clerical", "Tech-support"], n)
        relationship = np.random.choice(["Husband", "Wife", "Own-child", "Unmarried", "Not-in-family"], n)
        hours_per_week = np.random.randint(20, 60, n)
        capital_gain = np.random.choice([0, 5000, 10000, 20000], n, p=[0.92, 0.04, 0.03, 0.01])
        capital_loss = np.random.choice([0, 1000, 2000], n, p=[0.96, 0.03, 0.01])
        
        # Target Income with intentional bias
        # Base probability depends on education and capital gain
        edu_weight = {"Bachelors": 0.2, "Some-college": 0.1, "HS-grad": 0.05, "Masters": 0.4, "Doctorate": 0.6, "Assoc-voc": 0.1}
        prob = np.array([edu_weight[e] for e in education])
        prob += np.where(capital_gain > 0, 0.3, 0.0)
        prob += np.where(hours_per_week > 40, 0.1, 0.0)
        
        # Inject bias: Female and Black and Young have lower likelihoods
        prob += np.where(sex == "Male", 0.2, -0.15)
        prob += np.where(race == "White", 0.1, -0.1)
        prob += np.where(age > 35, 0.15, -0.1)
        
        # Keep in bounds
        prob = np.clip(prob, 0.01, 0.99)
        income = np.random.binomial(1, prob)
        
        df = pd.DataFrame({
            "age": age,
            "workclass": workclass,
            "education": education,
            "marital-status": marital_status,
            "occupation": occupation,
            "relationship": relationship,
            "race": race,
            "sex": sex,
            "capital-gain": capital_gain,
            "capital-loss": capital_loss,
            "hours-per-week": hours_per_week,
            "income": income
        })
        # Add derived column to raw fallback
        df["age_group"] = np.where(df["age"] <= 35, "Young", "Older")
        return df


class IBMHRAnalyticsLoader(DatasetLoader):
    def __init__(self):
        super().__init__("IBM HR Analytics", DATA_DIR / "ibm_hr.csv")
        self.target_column = "Attrition"
        self.protected_attributes = {
            "Gender": {"privileged": "Male", "unprivileged": "Female", "label": "Gender"},
            "MaritalStatus": {"privileged": "Married", "unprivileged": "Single", "label": "Marital Status"},
            "age_group": {"privileged": "Older", "unprivileged": "Young", "label": "Age Group"}
        }
        self.numerical_features = ["Age", "MonthlyIncome", "YearsAtCompany", "TotalWorkingYears", "WorkLifeBalance"]
        self.categorical_features = ["Gender", "MaritalStatus", "JobRole", "OverTime", "BusinessTravel"]

    def generate_fallback(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 1000
        
        age = np.random.randint(18, 60, n)
        gender = np.random.choice(["Male", "Female"], n, p=[0.6, 0.4])
        marital_status = np.random.choice(["Single", "Married", "Divorced"], n, p=[0.32, 0.46, 0.22])
        job_role = np.random.choice([
            "Sales Executive", "Research Scientist", "Laboratory Technician", 
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"
        ], n)
        overtime = np.random.choice(["Yes", "No"], n, p=[0.28, 0.72])
        business_travel = np.random.choice(["Travel_Rarely", "Travel_Frequently", "Non-Travel"], n, p=[0.7, 0.2, 0.1])
        
        total_working_years = np.clip(age - np.random.randint(18, 22, n), 0, None)
        years_at_company = np.array([np.random.randint(0, max(1, t + 1)) for t in total_working_years])
        work_life_balance = np.random.randint(1, 5, n)
        
        # Monthly Income correlated with age & working years
        monthly_income = 2500 + total_working_years * 450 + np.random.randint(-1000, 1500, n)
        monthly_income = np.clip(monthly_income, 1900, 20000)
        
        # Attrition logic (1 for Yes/Attrition, 0 for No)
        # Younger, single, high overtime, low income, low work-life balance are more likely to leave
        prob = 0.2
        prob += np.where(age <= 30, 0.2, -0.1)
        prob += np.where(marital_status == "Single", 0.15, -0.05)
        prob += np.where(overtime == "Yes", 0.25, -0.1)
        prob += np.where(monthly_income < 4000, 0.15, -0.05)
        prob += np.where(work_life_balance == 1, 0.2, -0.05)
        
        # Inject bias: Female employees have lower attrition base but we skew to demonstrate gender disparity
        prob += np.where(gender == "Female", 0.1, -0.05)
        
        prob = np.clip(prob, 0.02, 0.98)
        attrition = np.random.binomial(1, prob)
        
        df = pd.DataFrame({
            "Age": age,
            "Gender": gender,
            "MaritalStatus": marital_status,
            "JobRole": job_role,
            "OverTime": overtime,
            "BusinessTravel": business_travel,
            "TotalWorkingYears": total_working_years,
            "YearsAtCompany": years_at_company,
            "WorkLifeBalance": work_life_balance,
            "MonthlyIncome": monthly_income,
            "Attrition": attrition
        })
        df["age_group"] = np.where(df["Age"] <= 35, "Young", "Older")
        return df


class SyntheticHiringLoader(DatasetLoader):
    def __init__(self):
        super().__init__("Synthetic Hiring Dataset", DATA_DIR / "hiring_dataset.csv")
        self.target_column = "Hire"
        self.protected_attributes = {
            "Gender": {"privileged": "Male", "unprivileged": "Female", "label": "Gender"},
            "age_group": {"privileged": "Older", "unprivileged": "Young", "label": "Age Group"}
        }
        self.numerical_features = ["Age", "Years of Experience", "Skills Score"]
        self.categorical_features = ["Gender", "Education", "University Tier", "Employment Gap"]

    def generate_fallback(self) -> pd.DataFrame:
        np.random.seed(42)
        n = 1000
        
        gender = np.random.choice(["Male", "Female"], n, p=[0.55, 0.45])
        age = np.random.randint(21, 65, n)
        education = np.random.choice(["Bachelors", "Masters", "PhD"], n, p=[0.55, 0.35, 0.1])
        uni_tier = np.random.choice(["Tier 1", "Tier 2", "Tier 3"], n, p=[0.25, 0.5, 0.25])
        gap = np.random.choice(["Yes", "No"], n, p=[0.2, 0.8])
        
        skills_score = np.random.randint(40, 101, n)
        # Years of experience correlates with age
        years_exp = np.clip(age - 22 - np.random.randint(0, 5, n), 0, None)
        
        # Hiring decision (1 for Hire, 0 for Reject)
        # Higher skills score, education, and experience improve hire probability
        edu_map = {"Bachelors": 0.1, "Masters": 0.25, "PhD": 0.4}
        uni_map = {"Tier 1": 0.2, "Tier 2": 0.1, "Tier 3": 0.0}
        
        prob = 0.1 + (skills_score - 40) / 120.0
        prob += np.array([edu_map[e] for e in education])
        prob += np.array([uni_map[u] for u in uni_tier])
        prob += np.where(gap == "No", 0.1, -0.1)
        prob += np.clip(years_exp * 0.015, 0, 0.2)
        
        # Introduce intentional biases
        # 1. Gender bias: Females are hired less often for similar skills/experience
        prob += np.where(gender == "Male", 0.12, -0.12)
        # 2. Age bias: Younger candidates (<=35) are hired slightly more than older candidates (>35)
        prob += np.where(age <= 35, 0.08, -0.08)
        
        prob = np.clip(prob, 0.02, 0.98)
        hire = np.random.binomial(1, prob)
        
        df = pd.DataFrame({
            "Age": age,
            "Gender": gender,
            "Education": education,
            "University Tier": uni_tier,
            "Employment Gap": gap,
            "Years of Experience": years_exp,
            "Skills Score": skills_score,
            "Hire": hire
        })
        df["age_group"] = np.where(df["Age"] <= 35, "Young", "Older")
        return df


class DatasetRegistry:
    _registry = {
        "adult": AdultCensusLoader,
        "ibm_hr": IBMHRAnalyticsLoader,
        "hiring": SyntheticHiringLoader
    }

    @classmethod
    def get_loader(cls, dataset_id: str) -> DatasetLoader:
        if dataset_id not in cls._registry:
            raise ValueError(f"Unknown dataset: {dataset_id}")
        return cls._registry[dataset_id]()

    @classmethod
    def list_datasets(cls) -> list[dict[str, str]]:
        return [
            {"id": "adult", "name": "Adult Census Income Dataset"},
            {"id": "ibm_hr", "name": "IBM HR Analytics Dataset"},
            {"id": "hiring", "name": "Synthetic Hiring Dataset (Demo)"}
        ]
