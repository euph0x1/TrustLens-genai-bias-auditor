from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

from trustlens.bias.dataset_loader import DatasetRegistry
from trustlens.models import ModelRegistry
from trustlens.bias.detector import BiasDetector
from trustlens.bias.bias_mitigation import (
    calculate_reweighing_weights,
    train_fairlearn_reduction,
    train_fairlearn_postprocessing
)
from trustlens.bias.experiment_tracker import ExperimentTracker
from trustlens.bias.research_findings import generate_research_findings
from trustlens.explain.shap_explainer import TrustLensSHAPExplainer
from trustlens.fairness.counterfactual import CounterfactualEvaluator
from trustlens.io.audit_report_generator import generate_pdf_report

class TrustLensPipeline:
    def __init__(self):
        self.tracker = ExperimentTracker()
        self.detector = BiasDetector()
        self.cf_evaluator = CounterfactualEvaluator()
        
        # State
        self.dataset_id = ""
        self.protected_attribute = ""
        self.df_raw = None
        self.loader = None
        
        # Data splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.sens_train = None
        self.sens_test = None
        self.feature_names = None
        
        # Models cache: model_type -> fitted_estimator
        self.baseline_models = {}
        self.mitigated_models = {} # key format: f"{model_type}_{mitigation_method}_{constraint}"
        
        # Evaluation caches
        self.baseline_metrics = {} # model_type -> metrics dict
        self.mitigated_metrics = {} # key -> metrics dict

    def load_dataset(self, dataset_id: str, protected_attribute: str):
        """Set active dataset and protected attribute, load and preprocess splits."""
        self.dataset_id = dataset_id
        self.protected_attribute = protected_attribute
        
        self.loader = DatasetRegistry.get_loader(dataset_id)
        self.df_raw = self.loader.load_raw_data()
        
        # Check if attribute exists in the registry definition
        if protected_attribute not in self.loader.protected_attributes:
            # Pick the first available
            self.protected_attribute = list(self.loader.protected_attributes.keys())[0]
        else:
            self.protected_attribute = protected_attribute
            
        # Get splits
        splits = self.loader.preprocess_data(self.df_raw, self.protected_attribute)
        (self.X_train, self.X_test, self.y_train, self.y_test, 
         self.sens_train, self.sens_test, self.feature_names) = splits
         
        # Reset caches
        self.baseline_models = {}
        self.mitigated_models = {}
        self.baseline_metrics = {}
        self.mitigated_metrics = {}

    def train_baseline_models(self) -> dict:
        """Train and evaluate baseline models (Logistic Regression, Random Forest, XGBoost)."""
        results = {}
        for m_type in ["Logistic Regression", "Random Forest", "XGBoost"]:
            model = ModelRegistry.train_model(m_type, self.X_train, self.y_train)
            self.baseline_models[m_type] = model
            
            # Evaluate performance & bias
            eval_perf = ModelRegistry.evaluate_model(model, self.X_test, self.y_test)
            bias_res = self.detector.analyze_model_bias(
                model, self.X_test, self.y_test, self.sens_test
            )
            
            # Merge metrics
            metrics = {**eval_perf, **bias_res["fairness_metrics"]}
            self.baseline_metrics[m_type] = metrics
            
            # Log run
            self.tracker.log_experiment(
                dataset=self.loader.name,
                protected_attribute=self.protected_attribute,
                model=m_type,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                roc_auc=metrics["roc_auc"],
                spd=metrics["statistical_parity_difference"],
                dir_val=metrics["disparate_impact_ratio"],
                eod=metrics["equal_opportunity_difference"],
                equalized_odds=metrics["equalized_odds_difference"],
                overall_fairness=metrics["overall_fairness_score"],
                mitigation_method="Baseline"
            )
            results[m_type] = metrics
            
        return results

    def run_mitigation(self, model_type: str, method: str, constraint: str = "Demographic Parity") -> dict:
        """
        Run bias mitigation pipeline:
        - method: "Reweighing", "Fairlearn Reduction", "Threshold Optimizer"
        """
        mit_key = f"{model_type}_{method}_{constraint}"
        
        # Ensure baseline is trained first
        if model_type not in self.baseline_models:
            self.train_baseline_models()
            
        base_model = self.baseline_models[model_type]
        
        mit_model = None
        if method == "Reweighing":
            # 1. Preprocessing: compute weights
            weights = calculate_reweighing_weights(self.y_train, self.sens_train)
            # 2. Retrain model with weights
            mit_model = ModelRegistry.train_model(model_type, self.X_train, self.y_train, sample_weight=weights)
            
        elif method == "Fairlearn Reduction":
            # In-processing: Exponentiated Gradient
            mit_model = train_fairlearn_reduction(
                model_type, self.X_train, self.y_train, self.sens_train, constraint
            )
            
        elif method == "Threshold Optimizer":
            # Post-processing
            mit_model = train_fairlearn_postprocessing(
                base_model, self.X_train, self.y_train, self.sens_train, constraint
            )
            
        if mit_model is None:
            raise ValueError(f"Unknown mitigation method: {method}")
            
        self.mitigated_models[mit_key] = mit_model
        
        # Evaluate mitigated model
        eval_perf = ModelRegistry.evaluate_model(mit_model, self.X_test, self.y_test)
        bias_res = self.detector.analyze_model_bias(
            mit_model, self.X_test, self.y_test, self.sens_test
        )
        
        metrics = {**eval_perf, **bias_res["fairness_metrics"]}
        self.mitigated_metrics[mit_key] = metrics
        
        # Log run
        self.tracker.log_experiment(
            dataset=self.loader.name,
            protected_attribute=self.protected_attribute,
            model=model_type,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1_score=metrics["f1_score"],
            roc_auc=metrics["roc_auc"],
            spd=metrics["statistical_parity_difference"],
            dir_val=metrics["disparate_impact_ratio"],
            eod=metrics["equal_opportunity_difference"],
            equalized_odds=metrics["equalized_odds_difference"],
            overall_fairness=metrics["overall_fairness_score"],
            mitigation_method=f"{method} ({constraint})" if method != "Reweighing" else "Reweighing"
        )
        
        return metrics

    def explain_model(self, model_type: str, mit_key: str | None = None) -> TrustLensSHAPExplainer:
        """Instantiate and fit a SHAP explainer on baseline or mitigated models."""
        model = self.mitigated_models[mit_key] if mit_key else self.baseline_models[model_type]
        return TrustLensSHAPExplainer(model, self.X_train, model_type)

    def evaluate_counterfactual(self, model_type: str, candidate: dict, mit_key: str | None = None) -> list[dict]:
        """Compute counterfactual predictions for a candidate row."""
        model = self.mitigated_models[mit_key] if mit_key else self.baseline_models[model_type]
        return self.cf_evaluator.evaluate_candidate(
            model, self.loader, self.df_raw, candidate, self.protected_attribute, self.feature_names
        )

    def generate_findings(self) -> list[str]:
        """Run research findings engine on current dataset runs."""
        hist = self.tracker.get_history()
        return generate_research_findings(hist, self.loader.name)

    def generate_pdf_report(self, model_type: str, mit_key: str | None = None) -> Path:
        """Generate PDF audit report from current state."""
        base_metrics = self.baseline_metrics.get(model_type)
        if base_metrics is None:
            self.train_baseline_models()
            base_metrics = self.baseline_metrics[model_type]
            
        mit_metrics = self.mitigated_metrics.get(mit_key) if mit_key else None
        mit_method = mit_key.split("_")[1] if mit_key else "None"
        
        leaderboard = self.tracker.get_leaderboard(self.loader.name, self.protected_attribute)
        findings = self.generate_findings()
        
        return generate_pdf_report(
            dataset_name=self.loader.name,
            protected_attribute=self.protected_attribute,
            model_name=model_type,
            baseline_metrics=base_metrics,
            mitigation_method=mit_method,
            mitigated_metrics=mit_metrics,
            leaderboard_df=leaderboard,
            findings=findings
        )
