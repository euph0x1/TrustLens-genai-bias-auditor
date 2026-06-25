# Implementation Plan - TrustLens: Explainable Fairness Auditing Framework (Certification & Leaderboard Edition)

This document outlines the revised plan to refactor TrustLens.

## Goal Description

We will refactor TrustLens to support:
1. **Real Dataset Strategy**: Load real datasets (Adult Census Income, IBM HR Analytics) as primary data sources, falling back to high-fidelity synthetic generators if files are missing.
2. **Fairness vs. Accuracy Tradeoff Page**: Dedicated tab visualizing predictive performance vs. fairness using scatter plots and Pareto frontier charts.
3. **Research Findings Engine (`research_findings.py`)**: Automatically analyze model evaluations to summarize key findings.
4. **Experiment Tracker (`experiment_tracker.py`)**: Log experiments (Dataset, Protected Attribute, Model, Accuracy, Precision, Recall, F1, SPD, DIR, EOD, Equalized Odds, Mitigation Method, Timestamp) to compile model comparison history and find the best configuration.
5. **Fairness Leaderboard**: Dynamically rank configurations on the Tradeoff page using an **Overall Fairness Score** (derived from normalized SPD, DIR, EOD, and Equalized Odds).
6. **Fairness Certification**: A structured validation badge at the end of audits certifying whether a configuration is `Certified Fair`, `Needs Mitigation`, or presents a `High Disparity Risk`.

---

## User Review Required

> [!IMPORTANT]
> - **Overall Fairness Score Formulation**: We will compute the Overall Fairness Score as a percentage (0-100%):
>   - $Score_{DIR} = \min(DIR, 1/DIR)$ if $DIR > 0$ else 0, capped at 1.0.
>   - $Score_{SPD} = 1.0 - \min(|SPD|, 1.0)$.
>   - $Score_{EOD} = 1.0 - \min(|EOD|, 1.0)$.
>   - $Score_{EqOdds} = 1.0 - \min(|EqOdds|, 1.0)$.
>   - $Overall = \text{Average}(Score_{DIR}, Score_{SPD}, Score_{EOD}, Score_{EqOdds}) \times 100$.
> - **Certification Thresholds**:
>   - **Certified Fair** (Green): $DIR \ge 0.8$ AND $|SPD| \le 0.1$.
>   - **Needs Mitigation** (Yellow): $DIR$ in $[0.6, 0.8)$ OR $|SPD|$ in $(0.1, 0.2]$.
>   - **High Disparity Risk** (Red): $DIR < 0.6$ OR $|SPD| > 0.2$.

---

## Open Questions

- None. All requirements have been incorporated.

---

## Proposed Changes

We will group our modifications into several components.

### 1. Core Modules

#### [NEW] [experiment_tracker.py](file:///c:/Users/ahmed/ws/ML_bias_detection/src/trustlens/bias/experiment_tracker.py)
A class `ExperimentTracker` to log and save run metrics:
- Logs model performance and fairness metrics to a local database/file (`data/experiment_history.csv` or session state).
- Exposes:
  - `get_history()`: return all logged runs as a DataFrame.
  - `get_best_configuration(dataset, protected_attribute)`: return the configuration with the highest combination of accuracy and fairness (e.g. highest Overall Fairness Score or a balanced metric).
  - `clear_history()`: clear previous runs.

#### [NEW] [research_findings.py](file:///c:/Users/ahmed/ws/ML_bias_detection/src/trustlens/bias/research_findings.py)
Analyzes logged experiments and current models to write summaries for display on the dashboard and report.

#### [MODIFY] [metrics.py](file:///c:/Users/ahmed/ws/ML_bias_detection/src/trustlens/fairness/metrics.py)
Update to:
- Compute standard fairness metrics.
- Implement the `overall_fairness_score` and `get_certification_status` logic.

#### [NEW] [dataset_loader.py](file:///c:/Users/ahmed/ws/ML_bias_detection/src/trustlens/bias/dataset_loader.py)
Handles data loading (real/fallback generation) for Adult Census, IBM HR, and Synthetic Hiring.

#### [NEW] [bias_mitigation.py](file:///c:/Users/ahmed/ws/ML_bias_detection/src/trustlens/bias/bias_mitigation.py)
Mitigation module supporting Reweighing, Fairlearn reductions (under Demographic Parity, Equal Opportunity, Equalized Odds), and Threshold Optimizer.

#### [NEW] [shap_explainer.py](file:///c:/Users/ahmed/ws/ML_bias_detection/src/trustlens/explain/shap_explainer.py)
Generates SHAP importances.

---

### 2. Model & Pipeline

#### [MODIFY] [models.py](file:///c:/Users/ahmed/ws/ML_bias_detection/src/trustlens/models.py)
Encapsulates estimator training for LR, RF, XGBoost.

#### [MODIFY] [pipeline.py](file:///c:/Users/ahmed/ws/ML_bias_detection/src/trustlens/pipeline.py)
Manages workflow state, integrates with the `ExperimentTracker`, and produces evaluation metrics.

---

### 3. Dashboard UI & Reports

#### [NEW] [audit_report_generator.py](file:///c:/Users/ahmed/ws/ML_bias_detection/src/trustlens/io/audit_report_generator.py)
PDF report builder that outputs tables, charts, experiment leaderboard comparison, SHAP values, research findings, and the **Fairness Certification** badge.

#### [MODIFY] [streamlit_app.py](file:///c:/Users/ahmed/ws/ML_bias_detection/app/streamlit_app.py)
Updated page structure:
1. **Dataset Analysis**: View profiling & correlations.
2. **Model Training**: Train models and log them to the `ExperimentTracker`.
3. **Fairness Evaluation**: Multi-metric scorecard and dynamic **Fairness Certification** display.
4. **Fairness Tradeoff Analysis**: Dedicated tab showing the **Fairness Leaderboard** (ranked by Overall Fairness Score), model history plots, and tradeoff scatter charts.
5. **Bias Detection & Counterfactual Lab**: Candidate interactive playground.
6. **Bias Mitigation**: Apply Reweighing, reductions, or thresholding, then log to the `ExperimentTracker` for comparison.
7. **Explainability**: SHAP value global and local plots.
8. **Audit Reports**: Read research findings, view certification status, and download the PDF.

---

### 4. Configs & Dependencies

#### [MODIFY] [requirements.txt](file:///c:/Users/ahmed/ws/ML_bias_detection/requirements.txt)
Remove unused GenAI models and install:
- `xgboost>=1.7.0`
- `fairlearn>=0.10.0`
- `shap>=0.42.0`
- `reportlab>=4.0.0`
- `matplotlib>=3.7.0`
- `plotly>=5.15.0`

---

## Verification Plan

### Automated Tests
Run:
- `pytest tests/test_metrics.py` (fairness & certification math)
- `pytest tests/test_tracker.py` (verify experiment tracking state and leaderboard ranking)
- `pytest tests/test_dataset_loaders.py` (real/synthetic loading logic)
- `pytest tests/test_mitigation.py` (verify reductions)
- `pytest tests/test_findings.py` (verify findings output strings)

### Manual Verification
- Start dashboard: `streamlit run app/streamlit_app.py`
- Verify pages function as expected. Run different models/mitigations, view their entries on the leaderboard, check the certification badge, and download the PDF.

## Progress Update (Automated)

The following components have been implemented and verified via unit tests and manual inspection of the codebase:

- **Core Modules Implemented:** `experiment_tracker.py`, `research_findings.py`, `dataset_loader.py`, `bias_mitigation.py`, `shap_explainer.py`, `models.py`, `pipeline.py`, `audit_report_generator.py`, `streamlit_app.py`.
- **Fairness Metrics:** `overall_fairness_score` and `get_certification_status` implemented in `src/trustlens/fairness/metrics.py` and covered by unit tests.
- **Leaderboards & Tracking:** Experiment tracking, leaderboard sorting, and history persistence implemented in `src/trustlens/bias/experiment_tracker.py` and validated by `tests/test_tracker.py`.

## Next Steps

1. Expand `audit_report_generator.py` visuals (add chart snapshots) and attach SHAP plots to the PDF.
2. Add additional unit tests for mitigation algorithms and end-to-end pipeline flows (`tests/test_mitigation.py`, `tests/test_findings.py`).
3. Implement CI workflow to run the test suite in a clean environment.

If you'd like, I can start on step 1 (embed SHAP and leaderboard charts into the PDF) now.
