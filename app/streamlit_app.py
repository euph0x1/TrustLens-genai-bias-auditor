import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Setup Python paths
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trustlens.pipeline import TrustLensPipeline
from trustlens.bias.dataset_loader import DatasetRegistry

# Page Config
st.set_page_config(
    page_title="TrustLens | Responsible AI Auditing Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Slate / Blue Theme)
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f8fafc;
    }
    .metric-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 1.2rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3d59;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .cert-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid;
    }
    .cert-fair {
        background-color: #f0fdf4;
        border-color: #16a34a;
        color: #14532d;
    }
    .cert-warn {
        background-color: #fefce8;
        border-color: #ca8a04;
        color: #713f12;
    }
    .cert-risk {
        background-color: #fef2f2;
        border-color: #dc2626;
        color: #7f1d1d;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Session State Pipeline Initialization
if "pipeline" not in st.session_state:
    st.session_state.pipeline = TrustLensPipeline()
pipeline = st.session_state.pipeline

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.title("🛡️ TrustLens Control")
st.sidebar.caption("Explainable Fairness Auditing Framework")

# Dataset Selector
datasets = DatasetRegistry.list_datasets()
dataset_options = {d["id"]: d["name"] for d in datasets}
selected_dataset_id = st.sidebar.selectbox(
    "Select Dataset",
    options=list(dataset_options.keys()),
    format_func=lambda x: dataset_options[x]
)

# Protected Attribute Selector
# Instantiate loader briefly to query attributes
temp_loader = DatasetRegistry.get_loader(selected_dataset_id)
available_attributes = list(temp_loader.protected_attributes.keys())
clean_attr_names = {k: temp_loader.protected_attributes[k].get("label", k) for k in available_attributes}

selected_attribute = st.sidebar.selectbox(
    "Protected Attribute",
    options=available_attributes,
    format_func=lambda x: clean_attr_names[x]
)

# Re-load pipeline splits if selections changed
if (pipeline.dataset_id != selected_dataset_id or 
    pipeline.protected_attribute != selected_attribute):
    with st.spinner("Loading dataset splits..."):
        pipeline.load_dataset(selected_dataset_id, selected_attribute)

# Sidebar metadata summary
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Target Class:** `{pipeline.loader.target_column}`")
st.sidebar.markdown(f"**Train size:** {len(pipeline.X_train)} | **Test size:** {len(pipeline.X_test)}")
st.sidebar.markdown(f"**Features count:** {len(pipeline.feature_names)}")

# Define Pages / Tabs
tab_analysis, tab_train, tab_fairness, tab_tradeoffs, tab_detection, tab_mitigation, tab_explain, tab_reports = st.tabs([
    " Dataset Analysis",
    " Model Training",
    " Fairness Scorecard",
    " Tradeoff Analysis",
    " Counterfactual Lab",
    " Bias Mitigation",
    " SHAP Explanations",
    " Audit Reports"
])

# ==========================================
# PAGE 1: DATASET ANALYSIS
# ==========================================
with tab_analysis:
    st.header(" Dataset Profiling & Representation Analysis")
    st.markdown("Profile the dataset's distribution, protected attributes balance, and correlations before model training.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Protected Attribute Representation")
        # Counts
        attr_counts = pipeline.df_raw[selected_attribute].value_counts()
        fig_attr = px.pie(
            values=attr_counts.values,
            names=attr_counts.index,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_attr.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig_attr, use_container_width=True)
        
        # Details
        st.markdown(f"**Imbalance Ratio (Min/Max group size):** {attr_counts.min() / attr_counts.max():.2f}")
        
    with col2:
        st.subheader("Selection Rate per Protected Group")
        target_col = pipeline.loader.target_column
        rates = pipeline.df_raw.groupby(selected_attribute)[target_col].mean().reset_index()
        rates.columns = [selected_attribute, "Selection Rate"]
        rates["Selection Rate"] = rates["Selection Rate"] * 100.0
        
        fig_rate = px.bar(
            rates,
            x=selected_attribute,
            y="Selection Rate",
            color=selected_attribute,
            labels={"Selection Rate": "Selection Rate (%)"},
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_rate.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=300)
        st.plotly_chart(fig_rate, use_container_width=True)
        
    # Feature correlation heatmap
    st.subheader("Numerical Features Correlation Heatmap")
    num_df = pipeline.df_raw[pipeline.loader.numerical_features + [target_col]].corr()
    fig_corr = px.imshow(
        num_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1]
    )
    fig_corr.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# PAGE 2: MODEL TRAINING
# ==========================================
with tab_train:
    st.header(" Baseline Model Training & Performance Comparison")
    st.markdown("Train standard predictive classifiers (Logistic Regression, Random Forest, XGBoost) and compare performance.")
    
    if st.button("Train Models", type="primary"):
        with st.spinner("Training baseline classifiers..."):
            pipeline.train_baseline_models()
            st.success("Models trained successfully!")
            
    if not pipeline.baseline_models:
        st.info("Click **Train Models** to begin training estimators.")
    else:
        # Display side-by-side table
        st.subheader("Classification Performance Metrics (Test Split)")
        perf_data = []
        for model_name, metrics in pipeline.baseline_metrics.items():
            perf_data.append({
                "Model": model_name,
                "Accuracy": f"{metrics['accuracy']:.1%}",
                "Precision": f"{metrics['precision']:.1%}",
                "Recall": f"{metrics['recall']:.1%}",
                "F1 Score": f"{metrics['f1_score']:.1%}",
                "ROC-AUC": f"{metrics['roc_auc']:.2f}"
            })
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
        
        # Performance comparison bar chart
        df_plot = pd.DataFrame(pipeline.baseline_metrics).T[["accuracy", "f1_score", "roc_auc"]].reset_index()
        df_plot.columns = ["Model", "Accuracy", "F1 Score", "ROC-AUC"]
        df_melt = df_plot.melt(id_vars="Model", var_name="Metric", value_name="Score")
        
        fig_bar = px.bar(
            df_melt,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.D3
        )
        fig_bar.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

# ==========================================
# PAGE 3: FAIRNESS SCORECARD
# ==========================================
with tab_fairness:
    st.header(" Algorithmic Fairness Scorecard")
    
    if not pipeline.baseline_models:
        st.info("Train baseline models first to compute fairness scorecards.")
    else:
        st.markdown("Evaluates demographic disparities against mathematical fairness metrics.")
        
        # Tab selector for model metrics
        sel_model = st.selectbox("Select Model to Evaluate", list(pipeline.baseline_models.keys()))
        metrics = pipeline.baseline_metrics[sel_model]
        
        # Certification Panel
        st.subheader("TrustLens Fairness Certification Badge")
        spd = metrics["statistical_parity_difference"]
        dir_val = metrics["disparate_impact_ratio"]
        code = int(metrics["certification_code"])
        
        if code == 0:
            st.markdown(
                f"""
                <div class="cert-card cert-fair">
                    <h4> CERTIFIED FAIR</h4>
                    <p><b>{sel_model}</b> satisfies traditional legal and mathematical fairness standards. 
                    No severe disparate impact or selection bias detected.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif code == 1:
            st.markdown(
                f"""
                <div class="cert-card cert-warn">
                    <h4>⚠️ NEEDS MITIGATION</h4>
                    <p><b>{sel_model}</b> exhibits moderate demographic bias. Mitigation is recommended before production deployment.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="cert-card cert-risk">
                    <h4>❌ HIGH DISPARITY RISK</h4>
                    <p><b>{sel_model}</b> exhibits severe demographic bias. Predictability is strongly biased towards privileged groups. 
                    Mitigation must be applied to comply with ethical guidelines.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Scorecard Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['overall_fairness_score']:.1f}%</div>
                    <div class="metric-label">Overall Fairness Score</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{spd:.3f}</div>
                    <div class="metric-label">Statistical Parity Diff (SPD)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{dir_val:.3f}</div>
                    <div class="metric-label">Disparate Impact Ratio (DIR)</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-value">{metrics['equalized_odds_difference']:.3f}</div>
                    <div class="metric-label">Equalized Odds Diff</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Bar comparison
        st.subheader("Selection Rates by Demographic Group")
        res_g = pipeline.detector.analyze_model_bias(
            pipeline.baseline_models[sel_model], pipeline.X_test, pipeline.y_test, pipeline.sens_test
        )
        
        rates_df = pd.DataFrame(res_g["group_metrics"]).T.reset_index()
        rates_df.columns = ["Group", "Selection Rate", "Accuracy", "TPR", "FPR", "FNR", "TNR", "Sample Count"]
        rates_df["Selection Rate"] = rates_df["Selection Rate"] * 100.0
        
        fig_rates = px.bar(
            rates_df,
            x="Group",
            y="Selection Rate",
            color="Group",
            labels={"Selection Rate": "Selection Rate (%)"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_rates.update_layout(height=300)
        st.plotly_chart(fig_rates, use_container_width=True)

# ==========================================
# PAGE 4: TRADEOFF ANALYSIS
# ==========================================
with tab_tradeoffs:
    st.header("📉 Fairness vs. Accuracy Tradeoff Analysis")
    st.markdown("Examine the relationship between model predictive accuracy and demographic fairness across all configurations.")
    
    history_df = pipeline.tracker.get_filtered_history(pipeline.loader.name, selected_attribute)
    
    if history_df.empty:
        st.info("Log runs by training models and applying mitigations to inspect tradeoffs.")
    else:
        # Plot Scatter
        fig_scatter = px.scatter(
            history_df,
            x="accuracy",
            y="spd",
            color="model",
            symbol="mitigation_method",
            size="overall_fairness",
            hover_data=["model", "mitigation_method", "overall_fairness"],
            labels={"accuracy": "Predictive Accuracy", "spd": "Statistical Parity Diff (SPD)"},
            title="Accuracy vs. Statistical Parity Difference (Tradeoff Map)",
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Leaderboard
        st.subheader("🏆 TrustLens Algorithmic Fairness Leaderboard")
        leaderboard = pipeline.tracker.get_leaderboard(pipeline.loader.name, selected_attribute)
        st.dataframe(leaderboard, use_container_width=True)
        
        if st.button("Clear Run History"):
            pipeline.tracker.clear_history()
            st.success("History cleared!")
            st.rerun()

# ==========================================
# PAGE 5: COUNTERFACTUAL LAB
# ==========================================
with tab_detection:
    st.header("🔬 Counterfactual Evaluation Lab")
    st.markdown("Test single candidates by dynamically modifying their features and swapping protected attributes to observe changes in prediction outcome.")
    
    if not pipeline.baseline_models:
        st.info("Train baseline models first to enable candidate counterfactual testing.")
    else:
        # Dropdowns to load candidate
        st.subheader("Select Sample Candidate")
        sample_idx = st.selectbox("Sample Candidates", list(range(min(20, len(pipeline.df_raw)))))
        sample_cand = pipeline.df_raw.iloc[sample_idx].drop([pipeline.loader.target_column, "age_group"], errors="ignore").to_dict()
        
        # Editing form
        st.subheader("Interactive Feature Inputs")
        candidate_input = {}
        
        # Draw inputs dynamically based on features
        cols = st.columns(3)
        for idx, col_name in enumerate(pipeline.loader.numerical_features):
            col_target = cols[idx % 3]
            min_val = float(pipeline.df_raw[col_name].min())
            max_val = float(pipeline.df_raw[col_name].max())
            mean_val = float(sample_cand.get(col_name, pipeline.df_raw[col_name].mean()))
            candidate_input[col_name] = col_target.number_input(
                col_name, min_value=min_val, max_value=max_val, value=mean_val
            )
            
        for idx, col_name in enumerate(pipeline.loader.categorical_features):
            col_target = cols[(idx + len(pipeline.loader.numerical_features)) % 3]
            cats = sorted(pipeline.df_raw[col_name].dropna().unique())
            default_cat = sample_cand.get(col_name, cats[0])
            if default_cat not in cats:
                default_cat = cats[0]
            candidate_input[col_name] = col_target.selectbox(
                col_name, options=cats, index=cats.index(default_cat)
            )
            
        # Select active model to test candidate
        sel_cf_model = st.selectbox("Active Classifier Model", list(pipeline.baseline_models.keys()), key="cf_model")
        
        # Save in session state for explanation tab
        st.session_state["active_candidate"] = candidate_input
        st.session_state["active_cf_model"] = sel_cf_model
        
        if st.button("Run Counterfactual Test", type="primary"):
            cf_results = pipeline.evaluate_counterfactual(
                sel_cf_model, candidate_input
            )
            
            # Show side-by-side results
            st.subheader("Outcome Swap Disparity Results")
            for res in cf_results:
                badge = "✅ Accept/Positive" if res["prediction"] == 1 else "❌ Reject/Negative"
                color = "green" if res["prediction"] == 1 else "red"
                st.markdown(
                    f"- **{res['label']}**: {badge} (Probability: **{res['probability']:.1%}**) | "
                    f"Features: `{res['data']}`"
                )

# ==========================================
# PAGE 6: BIAS MITIGATION
# ==========================================
with tab_mitigation:
    st.header("🩹 Demographic Bias Mitigation Studio")
    st.markdown("Apply Responsible AI algorithms across the model pipeline (pre, in, or post-processing) to mitigate disparities.")
    
    if not pipeline.baseline_models:
        st.info("Train baseline models first to evaluate mitigation methods.")
    else:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            mit_model = st.selectbox("Base Estimator Model", list(pipeline.baseline_models.keys()), key="mit_model_sel")
            mit_method = st.selectbox(
                "Mitigation Strategy Method",
                ["Reweighing", "Fairlearn Reduction", "Threshold Optimizer"]
            )
        with col_m2:
            mit_constraint = st.selectbox(
                "Fairness Reduction / Threshold Constraint",
                ["Demographic Parity", "Equal Opportunity", "Equalized Odds"],
                disabled=(mit_method == "Reweighing")
            )
            
        if st.button("Apply Mitigation", type="primary"):
            with st.spinner(f"Running {mit_method} mitigation..."):
                metrics = pipeline.run_mitigation(mit_model, mit_method, mit_constraint)
                st.success("Mitigation completed and logged!")
                
        # Compare before vs after if current model has a mitigation run
        mit_key = f"{mit_model}_{mit_method}_{mit_constraint}"
        if mit_key in pipeline.mitigated_metrics:
            st.subheader("Before vs. After Mitigation Comparison")
            base_m = pipeline.baseline_metrics[mit_model]
            mit_m = pipeline.mitigated_metrics[mit_key]
            
            comparison_data = [
                {"Metric": "Accuracy", "Baseline": f"{base_m['accuracy']:.1%}", "Mitigated": f"{mit_m['accuracy']:.1%}"},
                {"Metric": "F1 Score", "Baseline": f"{base_m['f1_score']:.1%}", "Mitigated": f"{mit_m['f1_score']:.1%}"},
                {"Metric": "Statistical Parity Difference", "Baseline": f"{base_m['statistical_parity_difference']:.3f}", "Mitigated": f"{mit_m['statistical_parity_difference']:.3f}"},
                {"Metric": "Disparate Impact Ratio", "Baseline": f"{base_m['disparate_impact_ratio']:.3f}", "Mitigated": f"{mit_m['disparate_impact_ratio']:.3f}"},
                {"Metric": "Overall Fairness Score", "Baseline": f"{base_m['overall_fairness_score']:.1f}%", "Mitigated": f"{mit_m['overall_fairness_score']:.1f}%"}
            ]
            st.table(pd.DataFrame(comparison_data))

# ==========================================
# PAGE 7: SHAP EXPLANATIONS
# ==========================================
with tab_explain:
    st.header("💡 Model Explainability & Feature Contribution (SHAP)")
    st.markdown("Interrogate model classification drivers and protected attribute influence using the SHAP framework.")
    
    if not pipeline.baseline_models:
        st.info("Train baseline models first to compute SHAP values.")
    else:
        # Model selector
        exp_model = st.selectbox("Select Model to Explain", list(pipeline.baseline_models.keys()), key="exp_model_sel")
        
        # Check if mitigated is available
        exp_mit = st.checkbox("Explain Mitigated Version (if trained)")
        active_mit_key = None
        if exp_mit:
            # Look for available mitigated keys matching exp_model
            keys = [k for k in pipeline.mitigated_models.keys() if k.startswith(exp_model)]
            if keys:
                active_mit_key = st.selectbox("Mitigated Configuration", keys)
            else:
                st.warning("No mitigated run logged for this model yet. Showing baseline.")
                
        # Fit SHAP explainer
        with st.spinner("Computing SHAP values (this may take a few seconds)..."):
            explainer = pipeline.explain_model(exp_model, active_mit_key)
            
        if explainer.explainer is None:
            st.error("SHAP Explainer could not be fitted.")
        else:
            col_sh1, col_sh2 = st.columns(2)
            
            with col_sh1:
                st.subheader("Global Feature Importance (SHAP)")
                global_df = explainer.get_global_importance(pipeline.X_test)
                
                fig_glob = px.bar(
                    global_df.head(10),
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Top 10 Feature Contributions",
                    color_discrete_sequence=["#1e3d59"]
                )
                fig_glob.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_glob, use_container_width=True)
                
            with col_sh2:
                st.subheader("Demographic Influence on Predictions")
                # Influence
                influence = explainer.get_protected_attribute_influence(
                    pipeline.X_test, [selected_attribute]
                )
                st.markdown(
                    f"**Protected Attribute Influence ({clean_attr_names[selected_attribute]}):** "
                    f"{influence.get(selected_attribute, 0.0):.2f}% of model variance is driven by this feature."
                )
                
            # Local Candidate Explanation
            st.subheader("Local Prediction Explanation")
            candidate_data = st.session_state.get("active_candidate")
            if candidate_data is None:
                st.info("To see a local prediction explanation, select and run a candidate in the **Counterfactual Lab** page.")
            else:
                # Preprocess candidate into aligned training shape
                var_df = pd.DataFrame([candidate_data])
                cand_processed = {}
                for col in pipeline.loader.numerical_features:
                    train_col = pipeline.df_raw[col]
                    mean = train_col.mean()
                    std = train_col.std() + 1e-9
                    cand_processed[col] = (float(var_df.iloc[0][col]) - mean) / std
                for col in pipeline.loader.categorical_features:
                    cats = sorted(pipeline.df_raw[col].dropna().unique())
                    cand_val = var_df.iloc[0][col]
                    for cat in cats:
                        cand_processed[f"{col}_{cat}"] = 1.0 if cand_val == cat else 0.0
                        
                cand_df = pd.DataFrame(index=[0])
                for feat in pipeline.feature_names:
                    cand_df[feat] = cand_processed.get(feat, 0.0)
                    
                local_df = explainer.get_local_explanation(cand_df)
                
                fig_loc = px.bar(
                    local_df.head(10),
                    x="SHAP Value",
                    y="Feature",
                    orientation="h",
                    title="Feature Contribution for Selected Candidate",
                    color="SHAP Value",
                    color_continuous_scale="RdBu"
                )
                fig_loc.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_loc, use_container_width=True)

# ==========================================
# PAGE 8: REPORTS & CONCLUSIONS
# ==========================================
with tab_reports:
    st.header("📝 Audit Report Generation & Insights Summary")
    st.markdown("Review dynamic findings generated by the Research Findings Engine, and download a detailed PDF report.")
    
    if not pipeline.baseline_models:
        st.info("Train baseline models first to produce findings and audit reports.")
    else:
        st.subheader("TrustLens Automatic Research Findings")
        findings = pipeline.generate_findings()
        for f in findings:
            st.markdown(f"- {f}")
            
        st.subheader("Generate Audit Report PDF")
        rep_model = st.selectbox("Audited Model Configuration", list(pipeline.baseline_models.keys()), key="rep_model_sel")
        
        rep_mit = st.checkbox("Include Mitigated Configuration in PDF", key="rep_mit_sel")
        rep_mit_key = None
        if rep_mit:
            keys = [k for k in pipeline.mitigated_models.keys() if k.startswith(rep_model)]
            if keys:
                rep_mit_key = st.selectbox("Configuration", keys, key="rep_mit_key_sel")
            else:
                st.warning("No mitigated run logged for this model yet. Generating report for baseline only.")
                
        if st.button("Generate Audit Report", type="primary"):
            with st.spinner("Compiling PDF report..."):
                pdf_path = pipeline.generate_pdf_report(rep_model, rep_mit_key)
                
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                    
                st.success("PDF generated successfully!")
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=pdf_path.name,
                    mime="application/pdf"
                )
