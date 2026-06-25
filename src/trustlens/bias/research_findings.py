from __future__ import annotations

import pandas as pd
import numpy as np

def generate_research_findings(history_df: pd.DataFrame, current_dataset: str) -> list[str]:
    """
    Given the experiment history, generate a list of key insights.
    """
    findings = []
    
    if history_df.empty:
        return ["No experiment history found. Train models and run mitigations to generate insights."]

    # Filter to current dataset
    df = history_df[history_df["dataset"] == current_dataset].copy()
    if df.empty:
        return ["No experiments ran yet for this dataset. Train models to see insights."]

    # 1. Best Predictive Model (Baseline)
    baselines = df[df["mitigation_method"] == "Baseline"]
    if not baselines.empty:
        best_acc_row = baselines.loc[baselines["accuracy"].idxmax()]
        findings.append(
            f"**{best_acc_row['model']}** achieved the highest baseline predictive accuracy "
            f"({best_acc_row['accuracy']:.1%}) on the test set."
        )
        
        # 2. Largest Disparity (Baseline)
        worst_spd_row = baselines.loc[baselines["spd"].abs().idxmax()]
        findings.append(
            f"**{worst_spd_row['model']}** exhibited the largest baseline demographic disparity, "
            f"with a Statistical Parity Difference (SPD) of **{worst_spd_row['spd']:.3f}** "
            f"(Disparate Impact Ratio of **{worst_spd_row['dir']:.3f}**)."
        )
    else:
        # If no baselines exist but models were trained
        best_acc_row = df.loc[df["accuracy"].idxmax()]
        findings.append(
            f"**{best_acc_row['model']} ({best_acc_row['mitigation_method']})** achieved the highest predictive accuracy "
            f"({best_acc_row['accuracy']:.1%}) among all evaluated configurations."
        )

    # 3. Mitigation Impact Analysis
    mitigated = df[df["mitigation_method"] != "Baseline"]
    if not mitigated.empty and not baselines.empty:
        # Check if we have matching model pairings
        reductions = []
        for index, mit_row in mitigated.iterrows():
            model_name = mit_row["model"]
            mit_method = mit_row["mitigation_method"]
            base_match = baselines[baselines["model"] == model_name]
            if not base_match.empty:
                base_spd = abs(base_match.iloc[0]["spd"])
                mit_spd = abs(mit_row["spd"])
                if base_spd > 0:
                    pct_reduction = (base_spd - mit_spd) / base_spd * 100.0
                    reductions.append((pct_reduction, model_name, mit_method, mit_spd))
                    
        if reductions:
            # Get largest reduction
            reductions.sort(reverse=True, key=lambda x: x[0])
            best_reduction = reductions[0]
            findings.append(
                f"Mitigation using **{best_reduction[2]}** on **{best_reduction[1]}** successfully "
                f"reduced Statistical Parity Difference by **{best_reduction[0]:.1f}%** "
                f"(down to SPD of **{best_reduction[3]:.3f}**)."
            )

    # 4. Best Fairness configuration
    best_fair_row = df.loc[df["overall_fairness"].idxmax()]
    findings.append(
        f"The fairest configuration was **{best_fair_row['model']}** with **{best_fair_row['mitigation_method']}** mitigation, "
        f"achieving an Overall Fairness Score of **{best_fair_row['overall_fairness']:.1f}%** "
        f"while maintaining **{best_fair_row['accuracy']:.1%}** accuracy."
    )

    # 5. Dataset specific or overall attribute analysis
    # If there are multiple protected attributes in history for this dataset, compare them
    dataset_all_runs = history_df[history_df["dataset"] == current_dataset]
    if len(dataset_all_runs["protected_attribute"].unique()) > 1:
        # Compare base disparities for different attributes
        attr_spds = {}
        for attr in dataset_all_runs["protected_attribute"].unique():
            attr_baselines = dataset_all_runs[(dataset_all_runs["protected_attribute"] == attr) & (dataset_all_runs["mitigation_method"] == "Baseline")]
            if not attr_baselines.empty:
                attr_spds[attr] = attr_baselines["spd"].abs().max()
                
        if len(attr_spds) >= 2:
            max_attr = max(attr_spds, key=attr_spds.get)
            min_attr = min(attr_spds, key=attr_spds.get)
            
            # Map clean names if possible
            clean_names = {
                "sex": "Gender (Sex)", "Gender": "Gender", 
                "race": "Race", "MaritalStatus": "Marital Status",
                "age_group": "Age Group"
            }
            max_name = clean_names.get(max_attr, max_attr)
            min_name = clean_names.get(min_attr, min_attr)
            
            if attr_spds[max_attr] > attr_spds[min_attr] + 0.05:
                findings.append(
                    f"**{max_name}** disparities were larger than **{min_name}** disparities "
                    f"in the {current_dataset} dataset, indicating stronger historical bias along the {max_name} axis."
                )

    return findings
