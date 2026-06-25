from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Image
from trustlens.io.visuals import render_tradeoff_scatter, render_leaderboard_table, render_shap_global

ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = ROOT / "reports"

def generate_pdf_report(
    dataset_name: str,
    protected_attribute: str,
    model_name: str,
    baseline_metrics: dict,
    mitigation_method: str,
    mitigated_metrics: dict | None,
    leaderboard_df: pd.DataFrame,
    findings: list[str],
    output_path: Path | None = None
) -> Path:
    """
    Generate a professional PDF audit report summarizing the fairness evaluation.
    """
    if output_path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f"fairness_audit_{ts}.pdf"
        
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    primary_color = colors.HexColor("#1e3d59")
    secondary_color = colors.HexColor("#17b978")
    accent_color = colors.HexColor("#ff6f3c")
    neutral_dark = colors.HexColor("#222831")
    neutral_light = colors.HexColor("#f5f5f5")
    
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=primary_color,
        alignment=0,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=primary_color,
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "BodyText_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=neutral_dark,
        spaceAfter=8
    )
    
    bullet_style = ParagraphStyle(
        "BulletText_Custom",
        parent=styles["Bullet"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=neutral_dark,
        spaceAfter=6,
        bulletIndent=10,
        leftIndent=25
    )
    
    meta_style = ParagraphStyle(
        "MetaText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#555555")
    )
    
    story = []
    
    # --- HEADER SECTION ---
    story.append(Paragraph("TrustLens Fairness Audit Report", title_style))
    
    meta_data = [
        [Paragraph(f"<b>Dataset Audited:</b> {dataset_name}", meta_style),
         Paragraph(f"<b>Generated On:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}", meta_style)],
        [Paragraph(f"<b>Protected Attribute:</b> {protected_attribute}", meta_style),
         Paragraph("<b>Framework:</b> TrustLens Responsible AI Engine v0.2.0", meta_style)]
    ]
    meta_table = Table(meta_data, colWidths=[250, 250])
    meta_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 15))
    
    # --- SECTION 1: CERTIFICATION STATUS ---
    story.append(Paragraph("1. Executive Summary & Certification", h1_style))
    
    # Determine certification badge
    spd = baseline_metrics["statistical_parity_difference"]
    dir_val = baseline_metrics["disparate_impact_ratio"]
    
    from trustlens.fairness.metrics import get_certification_status
    status_text, badge_text, code = get_certification_status(spd, dir_val)
    
    # Badge formatting
    if code == 0:
        bg_color = colors.HexColor("#d4edda")
        text_color = colors.HexColor("#155724")
        desc = "The model meets the standard algorithmic fairness criteria. Disparities are within acceptable legal and academic thresholds (80% rule)."
    elif code == 1:
        bg_color = colors.HexColor("#fff3cd")
        text_color = colors.HexColor("#856404")
        desc = "The model exhibits moderate demographic disparities. Mitigation techniques are recommended to balance selection rates before deployment."
    else:
        bg_color = colors.HexColor("#f8d7da")
        text_color = colors.HexColor("#721c24")
        desc = "Severe demographic disparities detected. Immediate bias mitigation required. The model violates traditional regulatory thresholds."
        
    badge_style = ParagraphStyle(
        "BadgeText",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        textColor=text_color,
        alignment=1
    )
    
    cert_data = [
        [Table([[Paragraph(f"<b>{badge_text.upper()}</b>", badge_style)]],
               colWidths=[180],
               style=[
                   ("BACKGROUND", (0, 0), (-1, -1), bg_color),
                   ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                   ("TOPPADDING", (0, 0), (-1, -1), 10),
                   ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                   ("BOX", (0, 0), (-1, -1), 1, text_color)
               ]),
         Paragraph(f"<b>Status Explanation:</b> {desc}<br/><br/>"
                   f"<b>Statistical Parity Difference:</b> {spd:.3f}<br/>"
                   f"<b>Disparate Impact Ratio:</b> {dir_val:.3f}", body_style)]
    ]
    cert_table = Table(cert_data, colWidths=[200, 300])
    cert_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("RIGHTPADDING", (0, 0), (0, 0), 15),
    ]))
    story.append(cert_table)
    story.append(Spacer(1, 15))
    
    # --- SECTION 2: AUDITED CONFIGURATION ---
    story.append(Paragraph("2. Audited Configuration & Performance", h1_style))
    story.append(Paragraph(
        f"This audit evaluates the <b>{model_name}</b> model on <b>{dataset_name}</b>, using "
        f"<b>{protected_attribute}</b> as the protected demographic variable.", body_style
    ))
    
    # Performance metrics table
    perf_headers = ["Metric", "Baseline", "Mitigated Status"]
    mit_val = "N/A" if mitigated_metrics is None else f"{mitigated_metrics['accuracy']:.1%}"
    
    metrics_data = [
        ["Accuracy", f"{baseline_metrics['accuracy']:.1%}", "N/A" if mitigated_metrics is None else f"{mitigated_metrics['accuracy']:.1%}"],
        ["Precision", f"{baseline_metrics['precision']:.1%}", "N/A" if mitigated_metrics is None else f"{mitigated_metrics['precision']:.1%}"],
        ["Recall", f"{baseline_metrics['recall']:.1%}", "N/A" if mitigated_metrics is None else f"{mitigated_metrics['recall']:.1%}"],
        ["F1 Score", f"{baseline_metrics['f1_score']:.1%}", "N/A" if mitigated_metrics is None else f"{mitigated_metrics['f1_score']:.1%}"],
        ["ROC-AUC", f"{baseline_metrics['roc_auc']:.2f}", "N/A" if mitigated_metrics is None else f"{mitigated_metrics['roc_auc']:.2f}"]
    ]
    
    table_content = [[Paragraph(f"<b>{h}</b>", body_style) for h in perf_headers]]
    for row in metrics_data:
        table_content.append([Paragraph(cell, body_style) for cell in row])
        
    perf_table = Table(table_content, colWidths=[180, 160, 160])
    perf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, neutral_light])
    ]))
    # Quick styling adjustment for headers in code:
    for col_idx in range(len(perf_headers)):
        perf_table.setStyle(TableStyle([
            ("TEXTCOLOR", (col_idx, 0), (col_idx, 0), colors.white)
        ]))
    story.append(perf_table)
    story.append(Spacer(1, 15))
    
    # --- SECTION 3: FAIRNESS SCORECARD ---
    story.append(Paragraph("3. Fairness Metrics Comparison", h1_style))
    
    fairness_headers = ["Fairness Metric", "Baseline", "Mitigated", "Ideal Value", "Status"]
    
    baseline_spd = baseline_metrics["statistical_parity_difference"]
    baseline_dir = baseline_metrics["disparate_impact_ratio"]
    baseline_eod = baseline_metrics["equal_opportunity_difference"]
    baseline_eq_odds = baseline_metrics["equalized_odds_difference"]
    baseline_overall = baseline_metrics["overall_fairness_score"]
    
    if mitigated_metrics is not None:
        mit_spd = mitigated_metrics["statistical_parity_difference"]
        mit_dir = mitigated_metrics["disparate_impact_ratio"]
        mit_eod = mitigated_metrics["equal_opportunity_difference"]
        mit_eq_odds = mitigated_metrics["equalized_odds_difference"]
        mit_overall = mitigated_metrics["overall_fairness_score"]
        
        status_txt = "Improved" if mit_overall > baseline_overall else "No Change"
    else:
        mit_spd = 0.0
        mit_dir = 1.0
        mit_eod = 0.0
        mit_eq_odds = 0.0
        mit_overall = 0.0
        status_txt = "N/A"
        
    fair_rows = [
        ["Statistical Parity Diff (SPD)", f"{baseline_spd:.3f}", f"{mit_spd:.3f}" if mitigated_metrics else "N/A", "0.000", status_txt],
        ["Disparate Impact Ratio (DIR)", f"{baseline_dir:.3f}", f"{mit_dir:.3f}" if mitigated_metrics else "N/A", "1.000", status_txt],
        ["Equal Opportunity Diff (EOD)", f"{baseline_eod:.3f}", f"{mit_eod:.3f}" if mitigated_metrics else "N/A", "0.000", status_txt],
        ["Equalized Odds Difference", f"{baseline_eq_odds:.3f}", f"{mit_eq_odds:.3f}" if mitigated_metrics else "N/A", "0.000", status_txt],
        ["Overall Fairness Score", f"{baseline_overall:.1f}%", f"{mit_overall:.1f}%" if mitigated_metrics else "N/A", "100.0%", status_txt]
    ]
    
    fair_table_content = [[Paragraph(f"<b>{h}</b>", body_style) for h in fairness_headers]]
    for row in fair_rows:
        fair_table_content.append([Paragraph(cell, body_style) for cell in row])
        
    fair_table = Table(fair_table_content, colWidths=[180, 80, 80, 80, 80])
    fair_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), primary_color),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, neutral_light])
    ]))
    story.append(fair_table)
    story.append(Spacer(1, 15))
    
    # --- SECTION 4: LEADERBOARD ---
    story.append(Paragraph("4. Dataset Algorithmic Leaderboard", h1_style))
    story.append(Paragraph(
        "Ranked experiment configurations evaluated on the active dataset, sorted by Overall Fairness Score:", body_style
    ))
    
    lead_headers = ["Rank", "Model", "Mitigation", "Accuracy", "SPD", "DIR", "F1", "Fairness Score"]
    lead_table_content = [[Paragraph(f"<b>{h}</b>", body_style) for h in lead_headers]]
    
    # Limit to top 5 for PDF spacing
    for idx, row in leaderboard_df.head(5).iterrows():
        lead_table_content.append([
            Paragraph(str(row["Rank"]), body_style),
            Paragraph(str(row["Model"]), body_style),
            Paragraph(str(row["Mitigation"]), body_style),
            Paragraph(f"{row['Accuracy']:.1%}" if isinstance(row['Accuracy'], float) else str(row['Accuracy']), body_style),
            Paragraph(f"{row['SPD']:.3f}" if isinstance(row['SPD'], float) else str(row['SPD']), body_style),
            Paragraph(f"{row['DIR']:.3f}" if isinstance(row['DIR'], float) else str(row['DIR']), body_style),
            Paragraph(f"{row['F1 Score']:.1%}" if isinstance(row['F1 Score'], float) else str(row['F1 Score']), body_style),
            Paragraph(f"{row['Overall Fairness Score']:.1f}%" if isinstance(row['Overall Fairness Score'], float) else str(row['Overall Fairness Score']), body_style)
        ])
        
    lead_table = Table(lead_table_content, colWidths=[40, 100, 100, 60, 50, 50, 50, 70])
    lead_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b5c8f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
        ("TOPPADDING", (0, 0), (-1, 0), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, neutral_light])
    ]))
    story.append(lead_table)
    story.append(Spacer(1, 15))

    # Embed leaderboard chart image if available
    try:
        img_path = render_leaderboard_table(leaderboard_df, name="leaderboard")
        if img_path is not None and img_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            story.append(Paragraph("Leaderboard Visualization", h2_style))
            story.append(Image(str(img_path), width=420, height=180))
            story.append(Spacer(1, 12))
    except Exception:
        # Fail gracefully if visualization cannot be rendered or embedded
        pass
    
    # --- SECTION 5: FINDINGS ---
    story.append(Paragraph("5. Key Research Findings", h1_style))
    for f in findings:
        story.append(Paragraph(f, bullet_style))
    story.append(Spacer(1, 15))

    # Add tradeoff scatter visualization if history is provided via leaderboard_df copy
    try:
        trade_img = render_tradeoff_scatter(leaderboard_df.rename(columns={
            'Accuracy': 'accuracy', 'SPD': 'spd', 'Overall Fairness Score': 'overall_fairness', 'Model': 'model', 'Mitigation': 'mitigation_method'
        }), name="tradeoff")
        if trade_img is not None and trade_img.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            story.append(Paragraph("Accuracy vs. SPD Tradeoff", h2_style))
            story.append(Image(str(trade_img), width=420, height=240))
            story.append(Spacer(1, 12))
    except Exception:
        pass
    
    # --- SECTION 6: RECOMMENDATIONS ---
    story.append(Paragraph("6. Audit Recommendations", h1_style))
    
    recs = [
        "<b>Model Selection:</b> Random Forest or XGBoost achieve higher accuracies but learn and amplify demographic biases. Mitigated XGBoost / RF is preferred over baseline models.",
        "<b>Mitigation Strategy:</b> Fairlearn Exponentiated Gradient achieves high fairness (SPD close to 0) but can incur a slight accuracy penalty. Reweighing is suggested for moderate disparities.",
        "<b>Governance Policy:</b> Keep thresholds adjusted dynamically in production environments to maintain Equal Opportunity parity."
    ]
    for r in recs:
        story.append(Paragraph(r, bullet_style))
        
    # Build Document
    doc.build(story)
    return output_path
