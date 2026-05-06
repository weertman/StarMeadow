"""
ShoreZone-Based Sunflower Star Recovery Analysis

Uses the ShoreZone spatial join data to investigate habitat characteristics
associated with Pycnopodia helianthoides population recovery following the
Sea Star Wasting Disease (SSWD) epidemic.

Research Questions:
1. What habitat characteristics predict higher encounter rates?
2. Are there identifiable refugia where populations persisted?
3. Does ShoreZone eelgrass data corroborate dive survey findings?
4. How does wave exposure relate to recovery?
5. Does prey availability (bivalves) predict sunflower star presence?
6. Which unsurveyed sites should be prioritized for future monitoring?

Analyses:
- Random Forest habitat suitability model
- Refugia site characterization
- Dual eelgrass indicator comparison
- Wave exposure analysis
- Prey availability analysis
- Priority site identification

Inputs:
- outputs/15_shorezone_site_analysis/site_shorezone_pycno_summary.csv
- outputs/15_shorezone_site_analysis/shorezone_*_proportions.csv

Outputs:
- Model performance metrics and feature importances
- Refugia habitat profiles
- Statistical comparisons
- Priority site rankings
- Publication-ready figures

Author: Star Meadow Project
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr, pearsonr, chi2_contingency
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
import joblib
from datetime import datetime
from typing import Optional, List
import argparse

# Machine learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error

from utils import get_output_dir, load_data, set_style, save_figure, DATA_DIR

OUTPUT_DIR = get_output_dir(__file__)

# Markdown summary written for embedding in the site coverage diagnostic map
RESULTS_MD = OUTPUT_DIR / "results.md"


def _escape_md(value) -> str:
    """Escape values for safe insertion into a markdown table cell."""
    if value is None:
        return ""
    s = str(value)
    # Keep tables well-formed
    return s.replace("|", "\\|").replace("\n", " ")


def df_to_md_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    """
    Convert a dataframe to a simple markdown table without requiring external deps.
    """
    if df is None or len(df) == 0:
        return "_(no rows)_"

    df_show = df.head(max_rows).copy()
    cols: List[str] = [str(c) for c in df_show.columns]

    header = "| " + " | ".join(_escape_md(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    rows = []
    for _, row in df_show.iterrows():
        rows.append("| " + " | ".join(_escape_md(row[c]) for c in cols) + " |")

    out = "\n".join([header, sep] + rows)
    if len(df) > max_rows:
        out += f"\n\n_(showing first {max_rows} of {len(df)} rows)_"
    return out


def write_results_md(output_dir: Path = OUTPUT_DIR) -> Optional[Path]:
    """
    Write a compact markdown summary of the analysis to results.md.

    This is used by `code/15b_site_coverage_map.py` to populate the slide-out
    “Analysis Results” panel.
    """
    lines: List[str] = []
    lines.append("# ShoreZone Recovery Analysis Results")
    lines.append("")
    lines.append(f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    lines.append("")

    # 1) Model summary + feature importance
    lines.append("## Habitat suitability model")
    perm_path = output_dir / "feature_importance_permutation.csv"
    mdi_path = output_dir / "feature_importance_mdi.csv"

    if perm_path.exists():
        perm = pd.read_csv(perm_path)
        perm = perm.sort_values("Importance_Mean", ascending=False)
        lines.append("")
        lines.append("### Top predictors (permutation importance)")
        lines.append(df_to_md_table(perm[["Feature", "Importance_Mean", "Importance_Std"]], max_rows=12))
    elif mdi_path.exists():
        mdi = pd.read_csv(mdi_path).sort_values("Importance", ascending=False)
        lines.append("")
        lines.append("### Top predictors (MDI importance)")
        lines.append(df_to_md_table(mdi[["Feature", "Importance"]], max_rows=12))
    else:
        lines.append("")
        lines.append("_Feature importance outputs not found._")

    # 2) Validation summary
    val_path = output_dir / "validation_analysis.csv"
    lines.append("")
    lines.append("## Validation (observed vs predicted)")
    if val_path.exists():
        val = pd.read_csv(val_path)
        within = None
        if "Within_CI" in val.columns:
            # Within_CI is saved as True/False strings in CSV; normalize
            within_bool = val["Within_CI"].astype(str).str.lower().isin(["true", "1", "yes"])
            within = float(within_bool.mean()) if len(within_bool) else None
        if within is not None:
            lines.append(f"- Within 95% CI: {within*100:.0f}% ({int((within_bool).sum())}/{len(val)})")
        lines.append("")
        show_cols = [c for c in ["SiteName", "MeanEncounterRate", "Predicted_Rate", "CI_95_Low", "CI_95_High", "Within_CI"] if c in val.columns]
        if show_cols:
            lines.append("### Validation table (sample)")
            lines.append(df_to_md_table(val[show_cols], max_rows=12))
    else:
        lines.append("")
        lines.append("_validation_analysis.csv not found._")

    # 3) Refugia comparison
    ref_path = output_dir / "refugia_comparison.csv"
    lines.append("")
    lines.append("## Refugia comparison")
    if ref_path.exists():
        ref = pd.read_csv(ref_path)
        # Show strongest contrasts first
        if "Difference" in ref.columns:
            ref = ref.reindex(ref["Difference"].abs().sort_values(ascending=False).index)
        show_cols = [c for c in ["Feature", "Refugia_Mean", "NonRefugia_Mean", "Difference", "P_Value", "Significant"] if c in ref.columns]
        lines.append(df_to_md_table(ref[show_cols], max_rows=12))
    else:
        lines.append("")
        lines.append("_refugia_comparison.csv not found._")

    # 4) Priority sites summary
    prio_path = output_dir / "priority_sites_with_uncertainty.csv"
    lines.append("")
    lines.append("## Priority sites (undersampled)")
    if prio_path.exists():
        prio = pd.read_csv(prio_path)
        if "Priority_Tier" in prio.columns:
            tier_counts = prio["Priority_Tier"].fillna("Unknown").value_counts()
            lines.append("")
            lines.append("### Tier counts")
            tier_df = tier_counts.rename_axis("Priority_Tier").reset_index(name="Count")
            lines.append(df_to_md_table(tier_df, max_rows=10))

        # Show top candidates by Priority_Score if present
        if "Priority_Score" in prio.columns:
            prio_sorted = prio.sort_values("Priority_Score", ascending=False)
        else:
            prio_sorted = prio

        show_cols = [c for c in ["SiteName", "Predicted_Rate", "CI_95_Low", "CI_95_High", "Priority_Score", "Priority_Tier"] if c in prio_sorted.columns]
        if show_cols:
            lines.append("")
            lines.append("### Top candidates (sample)")
            lines.append(df_to_md_table(prio_sorted[show_cols], max_rows=15))
    else:
        lines.append("")
        lines.append("_priority_sites_with_uncertainty.csv not found._")

    content = "\n".join(lines).strip() + "\n"
    out_path = output_dir / "results.md"
    out_path.write_text(content, encoding="utf-8")
    return out_path

# Input data from previous analysis
SHOREZONE_OUTPUT = Path(__file__).parent.parent / "outputs" / "15_shorezone_site_analysis"


# =============================================================================
# Data Loading
# =============================================================================

def load_site_shorezone_summary() -> pd.DataFrame:
    """
    Load the merged site-ShoreZone-Pycnopodia summary with habitat features.
    
    For nearest-neighbor join, also loads one-hot encoded ShoreZone features
    from the proportion files.
    """
    summary_path = SHOREZONE_OUTPUT / "site_shorezone_pycno_summary.csv"
    
    if not summary_path.exists():
        raise FileNotFoundError(
            f"ShoreZone summary not found at {summary_path}\n"
            "Please run 15_shorezone_site_analysis.py first."
        )
    
    df = pd.read_csv(summary_path)
    print(f"Loaded site summary: {len(df)} sites")
    
    # Load and merge one-hot encoded habitat features
    habitat_features = load_habitat_features()
    if habitat_features is not None:
        df = df.merge(habitat_features, on="SiteName", how="left")
        print(f"After merging habitat features: {len(df.columns)} columns")
    
    return df


def load_shorezone_proportions(category: str) -> pd.DataFrame:
    """Load ShoreZone category proportions for a specific attribute."""
    prop_path = SHOREZONE_OUTPUT / f"shorezone_{category}_proportions.csv"
    
    if prop_path.exists():
        return pd.read_csv(prop_path)
    else:
        print(f"  Warning: {prop_path} not found")
        return None


def prepare_analysis_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for analysis by filtering and creating derived variables.
    """
    # Filter to sites with both ShoreZone and Pycnopodia data
    df_analysis = df.dropna(subset=["MeanEncounterRate"]).copy()
    
    print(f"Sites with Pycnopodia data: {len(df_analysis)}")
    
    # Create categorical variables
    df_analysis["HasDetections"] = df_analysis["TotalPycnoCount"] > 0
    df_analysis["HighEncounterRate"] = df_analysis["MeanEncounterRate"] > df_analysis["MeanEncounterRate"].median()
    
    # Detection rate categories
    df_analysis["DetectionCategory"] = pd.cut(
        df_analysis["DetectionRate"],
        bins=[-0.01, 0, 0.25, 0.5, 1.01],
        labels=["Never", "Rare", "Occasional", "Frequent"]
    )
    
    return df_analysis


# =============================================================================
# 1. Habitat Suitability Model
# =============================================================================

# ShoreZone categories to use as features (one-hot encoded)
SHOREZONE_CATEGORIES = [
    "EXP_CLASS",    # Wave exposure (stable)
    "SED_SOURCE",   # Sediment source (stable)
    "SED_ABUND",    # Sediment abundance (stable)
    "ZONECOMP",     # Zone composition (stable)
    "BC_CLASS",     # Biotic class
    "NRDA_CLASS",   # NRDA classification
    "HAB_CALC",     # Habitat calculation
    "ZOS_UNIT",     # Zostera (eelgrass)
    "FUC_UNIT",     # Fucus (rockweed)
    "ULV_UNIT",     # Ulva (sea lettuce)
    "NER_UNIT",     # Nereocystis (bull kelp)
    "OYS_UNIT",     # Oyster (prey species)
    "MUS_UNIT",     # Mussel (prey species)
]


def load_habitat_features() -> pd.DataFrame:
    """
    Load one-hot encoded ShoreZone features from proportion files.
    
    With nearest-neighbor join, each site has 1 ShoreZone segment,
    so we use the direct categorical values (one-hot encoded) instead
    of diversity metrics.
    """
    print("\nLoading habitat features...")
    
    all_features = None
    
    for category in SHOREZONE_CATEGORIES:
        prop_path = SHOREZONE_OUTPUT / f"shorezone_{category}_proportions.csv"
        
        if not prop_path.exists():
            print(f"  Warning: {category} proportions not found")
            continue
        
        props = pd.read_csv(prop_path)
        
        # Convert boolean columns to int for modeling
        for col in props.columns:
            if col != "SiteName" and props[col].dtype == bool:
                props[col] = props[col].astype(int)
        
        if all_features is None:
            all_features = props
        else:
            all_features = all_features.merge(props, on="SiteName", how="outer")
        
        n_cols = len([c for c in props.columns if c != "SiteName"])
        print(f"  {category}: {n_cols} features")
    
    if all_features is not None:
        feature_cols = [c for c in all_features.columns if c != "SiteName"]
        print(f"\nTotal habitat features: {len(feature_cols)}")
    
    return all_features


def get_habitat_feature_cols(df: pd.DataFrame) -> list:
    """Get list of habitat feature columns (one-hot encoded ShoreZone attributes)."""
    # Look for one-hot encoded columns (e.g., EXP_CLASS_SP, BC_CLASS_25)
    # Exclude diversity metrics (_Richness, _Shannon, etc.) - only use one-hot class codes
    diversity_suffixes = ("_Richness", "_Shannon", "_Simpson", "_Evenness")
    feature_cols = [c for c in df.columns if any(
        c.startswith(f"{cat}_") and c != f"{cat}_value" 
        for cat in SHOREZONE_CATEGORIES
    ) and not c.endswith(diversity_suffixes)]
    return feature_cols


def get_diversity_features(df: pd.DataFrame) -> list:
    """
    DEPRECATED: Get diversity metric columns.
    
    With nearest-neighbor join, diversity metrics are all 1 (single segment per site).
    Use get_habitat_feature_cols() instead.
    """
    # First try one-hot encoded features (new method)
    habitat_cols = get_habitat_feature_cols(df)
    if habitat_cols:
        return habitat_cols
    
    # Fallback to old diversity metrics (for backwards compatibility)
    diversity_cols = [c for c in df.columns if any(x in c for x in ["_Richness", "_Shannon", "_Simpson"])]
    diversity_cols = [c for c in diversity_cols if "_Evenness" not in c]
    return diversity_cols


def build_habitat_suitability_model(df: pd.DataFrame):
    """
    Build Random Forest model predicting encounter rate from ShoreZone features.
    """
    print("\n" + "=" * 60)
    print("HABITAT SUITABILITY MODEL")
    print("=" * 60)
    
    # Get feature columns
    feature_cols = get_diversity_features(df)
    
    # Filter to complete cases
    df_model = df.dropna(subset=feature_cols + ["MeanEncounterRate"])
    
    print(f"\nSites with complete data: {len(df_model)}")
    print(f"Features: {len(feature_cols)}")
    
    if len(df_model) < 20:
        print("Insufficient data for modeling")
        return None, None, None
    
    X = df_model[feature_cols].values
    y = df_model["MeanEncounterRate"].values
    
    # Handle any remaining NaN/inf
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    # Random Forest model
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation
    print("\nCross-validation (5-fold)...")
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    cv_predictions = cross_val_predict(rf, X, y, cv=5)
    
    print(f"  R² scores: {cv_scores}")
    print(f"  Mean R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Correlation between predicted and actual
    r_pred, p_pred = pearsonr(y, cv_predictions)
    print(f"  Predicted vs Actual: r = {r_pred:.3f}, p = {p_pred:.4f}")
    
    # Fit final model on all data
    rf.fit(X, y)
    
    # Feature importances
    importances = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(importances.head(10).to_string(index=False))
    
    # Permutation importance (more reliable)
    print("\nCalculating permutation importance...")
    perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    
    perm_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance_Mean": perm_importance.importances_mean,
        "Importance_Std": perm_importance.importances_std
    }).sort_values("Importance_Mean", ascending=False)
    
    print("\nTop 10 Permutation Importances:")
    print(perm_df.head(10).to_string(index=False))
    
    # Store results
    results = {
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
        "pred_actual_r": r_pred,
        "pred_actual_p": p_pred,
        "feature_importance": importances,
        "perm_importance": perm_df,
        "cv_predictions": cv_predictions,
        "actual": y,
        "sites": df_model["SiteName"].values
    }
    
    # Save model and feature columns for reuse by scripts 18 and 19
    model_path = OUTPUT_DIR / "trained_model.joblib"
    model_data = {
        "model": rf,
        "feature_cols": feature_cols,
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
    }
    joblib.dump(model_data, model_path)
    print(f"\n  Saved model to: {model_path}")
    
    return rf, feature_cols, results


def plot_model_results(results: dict, df_model: pd.DataFrame):
    """Plot habitat suitability model results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Predicted vs Actual
    ax1 = axes[0, 0]
    ax1.scatter(results["actual"], results["cv_predictions"], 
                alpha=0.6, edgecolor="black", linewidth=0.5)
    
    # Add 1:1 line
    max_val = max(results["actual"].max(), results["cv_predictions"].max())
    ax1.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="1:1 line")
    
    # Add regression line
    z = np.polyfit(results["actual"], results["cv_predictions"], 1)
    p = np.poly1d(z)
    ax1.plot(sorted(results["actual"]), p(sorted(results["actual"])), 
             "b-", linewidth=2, label=f"Fit (r={results['pred_actual_r']:.2f})")
    
    ax1.set_xlabel("Actual Encounter Rate (Pycno/hr)")
    ax1.set_ylabel("Predicted Encounter Rate (CV)")
    ax1.set_title(f"Model Performance\nCV R² = {results['cv_r2_mean']:.3f} ± {results['cv_r2_std']:.3f}")
    ax1.legend()
    
    # 2. Feature Importance (MDI)
    ax2 = axes[0, 1]
    top_features = results["feature_importance"].head(15)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(top_features)))
    
    bars = ax2.barh(range(len(top_features)), top_features["Importance"].values, color=colors)
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features["Feature"].values, fontsize=8)
    ax2.set_xlabel("Mean Decrease in Impurity")
    ax2.set_title("Feature Importance (MDI)")
    ax2.invert_yaxis()
    
    # 3. Permutation Importance
    ax3 = axes[1, 0]
    top_perm = results["perm_importance"].head(15)
    
    ax3.barh(range(len(top_perm)), top_perm["Importance_Mean"].values,
             xerr=top_perm["Importance_Std"].values, color="steelblue", capsize=3)
    ax3.set_yticks(range(len(top_perm)))
    ax3.set_yticklabels(top_perm["Feature"].values, fontsize=8)
    ax3.set_xlabel("Mean Accuracy Decrease")
    ax3.set_title("Permutation Importance")
    ax3.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax3.invert_yaxis()
    
    # 4. Residuals
    ax4 = axes[1, 1]
    residuals = results["actual"] - results["cv_predictions"]
    ax4.scatter(results["cv_predictions"], residuals, alpha=0.6, edgecolor="black", linewidth=0.5)
    ax4.axhline(0, color="red", linestyle="--", linewidth=2)
    ax4.set_xlabel("Predicted Encounter Rate")
    ax4.set_ylabel("Residual (Actual - Predicted)")
    ax4.set_title("Residual Plot")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "habitat_suitability_model")
    plt.close()


# =============================================================================
# 2. Refugia Characterization
# =============================================================================

def analyze_refugia(df: pd.DataFrame):
    """
    Identify and characterize potential refugia sites.
    
    Refugia = sites with consistent Pycnopodia detections,
    potentially representing populations that persisted through SSWD.
    """
    print("\n" + "=" * 60)
    print("REFUGIA ANALYSIS")
    print("=" * 60)
    
    # Define refugia as sites with detection rate > 50%
    refugia = df[df["DetectionRate"] >= 0.5].copy()
    non_refugia = df[(df["DetectionRate"] < 0.5) & (df["DetectionRate"] > 0)].copy()
    never_detected = df[df["DetectionRate"] == 0].copy()
    
    print(f"\nSite Classification:")
    print(f"  Refugia (detection ≥50%): {len(refugia)} sites")
    print(f"  Non-refugia (0 < detection < 50%): {len(non_refugia)} sites")
    print(f"  Never detected: {len(never_detected)} sites")
    
    if len(refugia) < 3:
        print("Insufficient refugia sites for analysis")
        return None
    
    # Compare ShoreZone characteristics
    diversity_cols = get_diversity_features(df)
    
    results = []
    
    print("\n--- ShoreZone Characteristics: Refugia vs Non-Refugia ---")
    
    for col in diversity_cols[:20]:  # Top 20 features
        if col not in df.columns:
            continue
            
        ref_vals = refugia[col].dropna()
        non_ref_vals = non_refugia[col].dropna()
        
        if len(ref_vals) < 3 or len(non_ref_vals) < 3:
            continue
        
        # Mann-Whitney U test
        try:
            u_stat, p_val = mannwhitneyu(ref_vals, non_ref_vals, alternative="two-sided")
        except:
            continue
        
        ref_mean = ref_vals.mean()
        non_ref_mean = non_ref_vals.mean()
        diff = ref_mean - non_ref_mean
        
        results.append({
            "Feature": col,
            "Refugia_Mean": ref_mean,
            "NonRefugia_Mean": non_ref_mean,
            "Difference": diff,
            "U_Statistic": u_stat,
            "P_Value": p_val,
            "Significant": p_val < 0.05
        })
        
        if p_val < 0.1:
            direction = "higher" if diff > 0 else "lower"
            sig = "**" if p_val < 0.05 else "*"
            print(f"  {col}: Refugia {direction} ({ref_mean:.3f} vs {non_ref_mean:.3f}), p={p_val:.4f} {sig}")
    
    results_df = pd.DataFrame(results).sort_values("P_Value")
    
    # Save results
    results_path = OUTPUT_DIR / "refugia_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Saved: {results_path}")
    
    # List refugia sites
    print("\n--- Refugia Sites ---")
    refugia_info = refugia[["SiteName", "DetectionRate", "MeanEncounterRate", "Basin", "HasEelgrass"]].sort_values(
        "MeanEncounterRate", ascending=False
    )
    print(refugia_info.head(20).to_string(index=False))
    
    return results_df, refugia, non_refugia


def plot_refugia_comparison(df: pd.DataFrame, refugia_results: pd.DataFrame):
    """Plot refugia vs non-refugia comparisons."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Define groups
    refugia = df[df["DetectionRate"] >= 0.5]
    non_refugia = df[(df["DetectionRate"] < 0.5) & (df["DetectionRate"] > 0)]
    never = df[df["DetectionRate"] == 0]
    
    # 1. Encounter rate by detection category
    ax1 = axes[0, 0]
    
    categories = ["Refugia\n(≥50% detection)", "Occasional\n(<50% detection)", "Never\nDetected"]
    means = [refugia["MeanEncounterRate"].mean(), 
             non_refugia["MeanEncounterRate"].mean(),
             0]  # Never detected have 0 rate
    stds = [refugia["MeanEncounterRate"].std(),
            non_refugia["MeanEncounterRate"].std(),
            0]
    counts = [len(refugia), len(non_refugia), len(never)]
    
    colors = ["#27ae60", "#f39c12", "#e74c3c"]
    bars = ax1.bar(categories, means, yerr=stds, capsize=5, color=colors, edgecolor="black")
    
    for i, (bar, n) in enumerate(zip(bars, counts)):
        ax1.annotate(f"n={n}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 5), textcoords="offset points", ha="center", fontsize=10)
    
    ax1.set_ylabel("Mean Encounter Rate (Pycno/hr)")
    ax1.set_title("Encounter Rate by Site Category")
    
    # 2. Top differentiating features
    ax2 = axes[0, 1]
    
    if refugia_results is not None and len(refugia_results) > 0:
        # Get top 10 features by significance
        top_features = refugia_results.head(10)
        
        y_pos = range(len(top_features))
        colors = ["#27ae60" if d > 0 else "#e74c3c" for d in top_features["Difference"]]
        
        ax2.barh(y_pos, top_features["Difference"], color=colors, edgecolor="black")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_features["Feature"], fontsize=8)
        ax2.axvline(0, color="black", linestyle="-", linewidth=1)
        ax2.set_xlabel("Difference (Refugia - Non-Refugia)")
        ax2.set_title("Top Differentiating Features\n(Green = higher in refugia)")
        
        # Add significance markers
        for i, (_, row) in enumerate(top_features.iterrows()):
            if row["P_Value"] < 0.05:
                ax2.annotate("**", xy=(row["Difference"], i), fontsize=12, fontweight="bold")
            elif row["P_Value"] < 0.1:
                ax2.annotate("*", xy=(row["Difference"], i), fontsize=12)
    
    # 3. Basin distribution
    ax3 = axes[1, 0]
    
    basin_refugia = refugia.groupby("Basin").size()
    basin_all = df.groupby("Basin").size()
    basin_pct = (basin_refugia / basin_all * 100).fillna(0).sort_values(ascending=True)
    
    ax3.barh(basin_pct.index, basin_pct.values, color="steelblue", edgecolor="black")
    ax3.set_xlabel("% of Sites that are Refugia")
    ax3.set_title("Refugia Distribution by Basin")
    ax3.set_xlim(0, 100)
    
    # 4. Eelgrass in refugia vs non-refugia
    ax4 = axes[1, 1]
    
    eelgrass_refugia = refugia["HasEelgrass"].mean() * 100
    eelgrass_non = non_refugia["HasEelgrass"].mean() * 100 if len(non_refugia) > 0 else 0
    eelgrass_never = never["HasEelgrass"].mean() * 100 if len(never) > 0 else 0
    
    x = ["Refugia", "Occasional", "Never"]
    y = [eelgrass_refugia, eelgrass_non, eelgrass_never]
    
    ax4.bar(x, y, color=["#27ae60", "#f39c12", "#e74c3c"], edgecolor="black")
    ax4.set_ylabel("% Sites with Eelgrass")
    ax4.set_title("Eelgrass Presence by Site Category")
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "refugia_analysis")
    plt.close()


# =============================================================================
# 3. Eelgrass Indicator Comparison
# =============================================================================

def analyze_eelgrass_indicators(df: pd.DataFrame):
    """
    Compare two independent eelgrass indicators:
    1. Dive survey HabitatType (HasEelgrass)
    2. ShoreZone ZOS_UNIT (Zostera presence in shoreline inventory)
    """
    print("\n" + "=" * 60)
    print("EELGRASS INDICATOR COMPARISON")
    print("=" * 60)
    
    # Create binary ShoreZone eelgrass indicator
    df = df.copy()
    
    # ShoreZone eelgrass: Check for ZOS_UNIT presence (P=patchy or C=continuous)
    # With nearest-neighbor join, use one-hot encoded features
    if "ZOS_UNIT_P" in df.columns or "ZOS_UNIT_C" in df.columns:
        zos_p = df.get("ZOS_UNIT_P", 0).fillna(0).astype(bool)
        zos_c = df.get("ZOS_UNIT_C", 0).fillna(0).astype(bool)
        df["ShoreZone_Eelgrass"] = zos_p | zos_c
    elif "ZOS_UNIT_Richness" in df.columns:
        # Fallback for old buffer-join data
        df["ShoreZone_Eelgrass"] = df["ZOS_UNIT_Richness"] > 1
    else:
        print("No ZOS_UNIT features found in data")
        return None
    
    # Dive survey eelgrass
    df["DiveSurvey_Eelgrass"] = df["HasEelgrass"].fillna(False)
    
    # Filter to sites with both indicators
    df_valid = df.dropna(subset=["ShoreZone_Eelgrass", "DiveSurvey_Eelgrass", "MeanEncounterRate"])
    
    print(f"\nSites with both indicators: {len(df_valid)}")
    
    # Cross-tabulation
    print("\n--- Cross-tabulation: Dive Survey vs ShoreZone Eelgrass ---")
    crosstab = pd.crosstab(
        df_valid["DiveSurvey_Eelgrass"], 
        df_valid["ShoreZone_Eelgrass"],
        margins=True
    )
    
    # Rename indices/columns safely based on actual values
    new_index = []
    for idx in crosstab.index:
        if idx == False:
            new_index.append("No Eelgrass (Dive)")
        elif idx == True:
            new_index.append("Eelgrass (Dive)")
        else:
            new_index.append("Total")
    crosstab.index = new_index
    
    new_cols = []
    for col in crosstab.columns:
        if col == False:
            new_cols.append("No Eelgrass (ShoreZone)")
        elif col == True:
            new_cols.append("Eelgrass (ShoreZone)")
        else:
            new_cols.append("Total")
    crosstab.columns = new_cols
    
    print(crosstab)
    
    # Agreement rate
    agreement = ((df_valid["DiveSurvey_Eelgrass"] == df_valid["ShoreZone_Eelgrass"]).mean() * 100)
    print(f"\nAgreement rate: {agreement:.1f}%")
    
    # Chi-square test
    contingency = pd.crosstab(df_valid["DiveSurvey_Eelgrass"], df_valid["ShoreZone_Eelgrass"])
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"Chi-square test: χ² = {chi2:.2f}, p = {p:.4f}")
    
    # Test each indicator's relationship with encounter rate
    print("\n--- Encounter Rate by Eelgrass Indicator ---")
    
    results = []
    
    for indicator in ["DiveSurvey_Eelgrass", "ShoreZone_Eelgrass"]:
        present = df_valid[df_valid[indicator] == True]["MeanEncounterRate"]
        absent = df_valid[df_valid[indicator] == False]["MeanEncounterRate"]
        
        u_stat, p_val = mannwhitneyu(present, absent, alternative="two-sided")
        
        print(f"\n{indicator}:")
        print(f"  Present: n={len(present)}, mean={present.mean():.3f}")
        print(f"  Absent: n={len(absent)}, mean={absent.mean():.3f}")
        print(f"  Mann-Whitney U: p = {p_val:.4f}")
        
        results.append({
            "Indicator": indicator,
            "N_Present": len(present),
            "N_Absent": len(absent),
            "Mean_Present": present.mean(),
            "Mean_Absent": absent.mean(),
            "U_Statistic": u_stat,
            "P_Value": p_val
        })
    
    # Combined indicator (both agree eelgrass present)
    df_valid["Both_Eelgrass"] = df_valid["DiveSurvey_Eelgrass"] & df_valid["ShoreZone_Eelgrass"]
    
    print("\n--- Combined Indicator (Both Agree) ---")
    both_present = df_valid[df_valid["Both_Eelgrass"] == True]["MeanEncounterRate"]
    not_both = df_valid[df_valid["Both_Eelgrass"] == False]["MeanEncounterRate"]
    
    if len(both_present) >= 3 and len(not_both) >= 3:
        u_stat, p_val = mannwhitneyu(both_present, not_both, alternative="two-sided")
        print(f"  Both indicators agree eelgrass: n={len(both_present)}, mean={both_present.mean():.3f}")
        print(f"  Not both: n={len(not_both)}, mean={not_both.mean():.3f}")
        print(f"  Mann-Whitney U: p = {p_val:.4f}")
    
    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "eelgrass_indicator_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Saved: {results_path}")
    
    return df_valid, results_df


def plot_eelgrass_comparison(df: pd.DataFrame):
    """Plot eelgrass indicator comparison."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    df = df.copy()
    # Use one-hot encoded ZOS_UNIT features
    if "ZOS_UNIT_P" in df.columns or "ZOS_UNIT_C" in df.columns:
        zos_p = df.get("ZOS_UNIT_P", 0).fillna(0).astype(bool)
        zos_c = df.get("ZOS_UNIT_C", 0).fillna(0).astype(bool)
        df["ShoreZone_Eelgrass"] = zos_p | zos_c
    else:
        df["ShoreZone_Eelgrass"] = df["ZOS_UNIT_Richness"] > 1
    df["DiveSurvey_Eelgrass"] = df["HasEelgrass"].fillna(False).astype(bool)
    df["ShoreZone_Eelgrass"] = df["ShoreZone_Eelgrass"].fillna(False).astype(bool)
    df_valid = df.dropna(subset=["MeanEncounterRate"])
    
    # 1. Agreement matrix heatmap
    ax1 = axes[0, 0]
    
    # Create 4 categories
    df_valid = df_valid.copy()
    df_valid["Agreement"] = "Neither"
    df_valid.loc[df_valid["DiveSurvey_Eelgrass"] & df_valid["ShoreZone_Eelgrass"], "Agreement"] = "Both Detect"
    df_valid.loc[df_valid["DiveSurvey_Eelgrass"] & ~df_valid["ShoreZone_Eelgrass"], "Agreement"] = "Dive Only"
    df_valid.loc[~df_valid["DiveSurvey_Eelgrass"] & df_valid["ShoreZone_Eelgrass"], "Agreement"] = "ShoreZone Only"
    
    agreement_counts = df_valid["Agreement"].value_counts()
    colors = {"Both Detect": "#27ae60", "Dive Only": "#3498db", 
              "ShoreZone Only": "#9b59b6", "Neither": "#e74c3c"}
    
    bars = ax1.bar(agreement_counts.index, agreement_counts.values,
                   color=[colors.get(x, "gray") for x in agreement_counts.index],
                   edgecolor="black")
    ax1.set_ylabel("Number of Sites")
    ax1.set_title("Eelgrass Indicator Agreement")
    ax1.tick_params(axis="x", rotation=15)
    
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center")
    
    # 2. Encounter rate by indicator
    ax2 = axes[0, 1]
    
    categories = ["Dive: Yes", "Dive: No", "ShoreZone: Yes", "ShoreZone: No"]
    means = [
        df_valid[df_valid["DiveSurvey_Eelgrass"]]["MeanEncounterRate"].mean(),
        df_valid[~df_valid["DiveSurvey_Eelgrass"]]["MeanEncounterRate"].mean(),
        df_valid[df_valid["ShoreZone_Eelgrass"]]["MeanEncounterRate"].mean(),
        df_valid[~df_valid["ShoreZone_Eelgrass"]]["MeanEncounterRate"].mean()
    ]
    
    bar_colors = ["#27ae60", "#e74c3c", "#27ae60", "#e74c3c"]
    bars = ax2.bar(categories, means, color=bar_colors, edgecolor="black")
    ax2.set_ylabel("Mean Encounter Rate (Pycno/hr)")
    ax2.set_title("Encounter Rate by Eelgrass Indicator")
    ax2.tick_params(axis="x", rotation=15)
    
    # 3. Encounter rate by agreement category
    ax3 = axes[1, 0]
    
    order = ["Both Detect", "Dive Only", "ShoreZone Only", "Neither"]
    order = [o for o in order if o in df_valid["Agreement"].unique()]
    
    if len(order) > 0:
        sns.boxplot(data=df_valid, x="Agreement", y="MeanEncounterRate", 
                    order=order,
                    palette=colors, ax=ax3)
        ax3.set_xlabel("")
        ax3.set_ylabel("Mean Encounter Rate (Pycno/hr)")
        ax3.set_title("Encounter Rate by Indicator Agreement")
    
    # 4. Scatter plot
    ax4 = axes[1, 1]
    
    colors_scatter = df_valid["Agreement"].map(colors)
    ax4.scatter(df_valid["ZOS_UNIT_Richness"], df_valid["MeanEncounterRate"],
                c=colors_scatter, s=60, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax4.set_xlabel("ShoreZone Zostera Richness")
    ax4.set_ylabel("Mean Encounter Rate (Pycno/hr)")
    ax4.set_title("Encounter Rate vs ShoreZone Eelgrass Diversity")
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
    ax4.legend(handles=legend_elements, loc="upper right", fontsize=8)
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "eelgrass_indicator_comparison")
    plt.close()


# =============================================================================
# 4. Wave Exposure Analysis
# =============================================================================

def analyze_wave_exposure(df: pd.DataFrame):
    """
    Analyze relationship between wave exposure and Pycnopodia presence.
    """
    print("\n" + "=" * 60)
    print("WAVE EXPOSURE ANALYSIS")
    print("=" * 60)
    
    # Get exposure columns that are already in df (from load_habitat_features)
    # Exclude diversity metrics (_Richness, _Shannon, etc.) - only use one-hot class codes
    diversity_suffixes = ("_Richness", "_Shannon", "_Simpson", "_Evenness")
    exp_cols = [c for c in df.columns 
                if c.startswith("EXP_CLASS_") and not c.endswith(diversity_suffixes)]
    
    if not exp_cols:
        print("No exposure columns found in data")
        return None
    
    # Filter to sites with exposure data and encounter rate
    df_exp = df.dropna(subset=["MeanEncounterRate"]).copy()
    # Check that at least some exposure data exists
    has_exposure = df_exp[exp_cols].notna().any(axis=1)
    df_exp = df_exp[has_exposure]
    
    print(f"\nSites with exposure data: {len(df_exp)}")
    print(f"Exposure categories: {exp_cols}")
    
    # Correlation with encounter rate
    print("\n--- Correlation: Exposure vs Encounter Rate ---")
    
    results = []
    for col in exp_cols:
        if col in df_exp.columns:
            valid = df_exp[[col, "MeanEncounterRate"]].dropna()
            if len(valid) >= 10:
                r, p = spearmanr(valid[col], valid["MeanEncounterRate"])
                print(f"  {col}: ρ = {r:.3f}, p = {p:.4f}")
                results.append({"Exposure_Class": col, "Spearman_rho": r, "P_Value": p})
    
    # Dominant exposure analysis
    print("\n--- Sites by Dominant Exposure ---")
    
    # Find dominant exposure for each site
    df_exp["Dominant_Exposure"] = df_exp[exp_cols].idxmax(axis=1)
    
    for exp in df_exp["Dominant_Exposure"].unique():
        subset = df_exp[df_exp["Dominant_Exposure"] == exp]
        print(f"  {exp}: n={len(subset)}, mean rate={subset['MeanEncounterRate'].mean():.3f}")
    
    # Kruskal-Wallis test
    groups = [df_exp[df_exp["Dominant_Exposure"] == exp]["MeanEncounterRate"].values 
              for exp in df_exp["Dominant_Exposure"].unique()]
    groups = [g for g in groups if len(g) >= 3]
    
    if len(groups) >= 2:
        h_stat, kw_p = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis test: H = {h_stat:.2f}, p = {kw_p:.4f}")
    
    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "exposure_analysis.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Saved: {results_path}")
    
    return df_exp, results_df


def plot_exposure_analysis(df_exp: pd.DataFrame):
    """Plot wave exposure analysis."""
    
    # Get exposure columns that are already in df_exp
    # Exclude diversity metrics - only use one-hot class codes
    diversity_suffixes = ("_Richness", "_Shannon", "_Simpson", "_Evenness")
    exp_cols = [c for c in df_exp.columns 
                if c.startswith("EXP_CLASS_") and not c.endswith(diversity_suffixes)]
    
    if not exp_cols:
        print("No exposure columns found for plotting")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Encounter rate by dominant exposure
    ax1 = axes[0]
    
    df_exp = df_exp.copy()
    df_exp["Dominant_Exposure"] = df_exp[exp_cols].idxmax(axis=1)
    
    # Order by mean encounter rate
    order = df_exp.groupby("Dominant_Exposure")["MeanEncounterRate"].mean().sort_values(ascending=False).index
    
    sns.boxplot(data=df_exp, x="Dominant_Exposure", y="MeanEncounterRate",
                order=order, palette="viridis", ax=ax1)
    ax1.set_xlabel("Dominant Exposure Class")
    ax1.set_ylabel("Mean Encounter Rate (Pycno/hr)")
    ax1.set_title("Encounter Rate by Dominant Wave Exposure")
    ax1.tick_params(axis="x", rotation=45)
    
    # Add sample sizes
    for i, exp in enumerate(order):
        n = len(df_exp[df_exp["Dominant_Exposure"] == exp])
        ax1.annotate(f"n={n}", xy=(i, df_exp["MeanEncounterRate"].max() * 0.95),
                     ha="center", fontsize=9)
    
    # 2. Box plot: Encounter rate by protected water presence
    ax2 = axes[1]
    
    # Look for "P" or "Protected" column (EXP_CLASS_P = protected waters)
    protected_cols = [c for c in exp_cols if c == "EXP_CLASS_P" or ("_P" in c and "SEMI" not in c.upper())]
    
    if protected_cols:
        protected_col = protected_cols[0]
        
        # For binary data (one-hot encoded), use box plot instead of scatter
        df_plot = df_exp.copy()
        df_plot["Protected_Bool"] = df_plot[protected_col].astype(bool)
        df_plot["Protected"] = df_plot["Protected_Bool"].map({False: "Not Protected", True: "Protected"})
        
        colors = {"Protected": "#27ae60", "Not Protected": "#3498db"}
        sns.boxplot(
            data=df_plot,
            x="Protected",
            y="MeanEncounterRate",
            order=["Not Protected", "Protected"],
            palette=colors,
            ax=ax2,
        )
        ax2.set_xlabel("Wave Exposure")
        ax2.set_ylabel("Mean Encounter Rate (Pycno/hr)")
        ax2.set_title("Encounter Rate: Protected vs Exposed Waters")
        
        # Add sample sizes and statistical test
        protected = df_plot[df_plot["Protected_Bool"]]["MeanEncounterRate"]
        not_protected = df_plot[~df_plot["Protected_Bool"]]["MeanEncounterRate"]
        
        ax2.annotate(f"n={len(not_protected)}", xy=(0, df_plot["MeanEncounterRate"].max() * 0.9),
                     ha="center", fontsize=9)
        ax2.annotate(f"n={len(protected)}", xy=(1, df_plot["MeanEncounterRate"].max() * 0.9),
                     ha="center", fontsize=9)
        
        if len(protected) >= 3 and len(not_protected) >= 3:
            u_stat, p_val = mannwhitneyu(protected, not_protected, alternative="two-sided")
            ax2.annotate(f"Mann-Whitney p = {p_val:.4f}", xy=(0.5, 0.95), xycoords="axes fraction",
                         ha="center", fontsize=10)
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "exposure_analysis")
    plt.close()


# =============================================================================
# 5. Prey Availability Analysis
# =============================================================================

def analyze_prey_availability(df: pd.DataFrame):
    """
    Analyze relationship between potential prey (bivalves) and Pycnopodia presence.
    
    ShoreZone prey indicators (one-hot encoded):
    - OYS_UNIT_P/C: Oyster presence (patchy/continuous)
    - MUS_UNIT_P/C: Mussel presence (patchy/continuous)
    """
    print("\n" + "=" * 60)
    print("PREY AVAILABILITY ANALYSIS")
    print("=" * 60)
    
    df_valid = df.dropna(subset=["MeanEncounterRate"]).copy()
    
    # Prey indicator mappings: name -> list of columns indicating presence (P=patchy, C=continuous)
    # Exclude empty indicators and diversity metrics
    diversity_suffixes = ("_Richness", "_Shannon", "_Simpson", "_Evenness")
    prey_indicators = {
        "Oyster": [c for c in df_valid.columns 
                   if c.startswith("OYS_UNIT_") 
                   and not c.endswith(diversity_suffixes)
                   and c not in ["OYS_UNIT_ ", "OYS_UNIT_"]],
        "Mussel": [c for c in df_valid.columns 
                   if c.startswith("MUS_UNIT_") 
                   and not c.endswith(diversity_suffixes)
                   and c not in ["MUS_UNIT_ ", "MUS_UNIT_"]]
    }
    
    # Filter to indicators that have columns
    prey_indicators = {k: v for k, v in prey_indicators.items() if v}
    
    if not prey_indicators:
        print("No prey indicator columns found")
        return None
    
    print(f"\nPrey indicators found:")
    for name, cols in prey_indicators.items():
        print(f"  {name}: {cols}")
    
    results = []
    
    for prey_name, presence_cols in prey_indicators.items():
        # Prey is present if any of the presence columns (P or C) is True
        has_prey = df_valid[presence_cols].any(axis=1)
        
        valid = df_valid[["MeanEncounterRate"]].copy()
        valid["has_prey"] = has_prey
        valid = valid.dropna()
        
        if len(valid) < 10:
            continue
        
        present = valid[valid["has_prey"]]["MeanEncounterRate"]
        absent = valid[~valid["has_prey"]]["MeanEncounterRate"]
        
        print(f"\n--- {prey_name} ---")
        print(f"  Present: n={len(present)}, mean={present.mean():.3f}" if len(present) > 0 else f"  Present: n=0")
        print(f"  Absent: n={len(absent)}, mean={absent.mean():.3f}" if len(absent) > 0 else f"  Absent: n=0")
        
        if len(present) >= 3 and len(absent) >= 3:
            u_stat, p_val = mannwhitneyu(present, absent, alternative="two-sided")
            print(f"  Mann-Whitney U: p = {p_val:.4f}")
            
            results.append({
                "Prey_Indicator": prey_name,
                "N_Present": len(present),
                "N_Absent": len(absent),
                "Mean_Present": present.mean(),
                "Mean_Absent": absent.mean(),
                "Difference": present.mean() - absent.mean(),
                "U_Statistic": u_stat,
                "P_Value": p_val
            })
        else:
            print(f"  Insufficient samples for statistical test")
    
    # Combined prey presence
    if len(prey_indicators) >= 2:
        # Count how many prey types are present
        prey_presence_cols = []
        for prey_name, presence_cols in prey_indicators.items():
            col_name = f"has_{prey_name.lower()}"
            df_valid[col_name] = df_valid[presence_cols].any(axis=1)
            prey_presence_cols.append(col_name)
        
        df_valid["Total_Prey_Types"] = df_valid[prey_presence_cols].sum(axis=1)
        
        valid_combined = df_valid[["Total_Prey_Types", "MeanEncounterRate"]].dropna()
        if len(valid_combined) >= 10:
            r, p = spearmanr(valid_combined["Total_Prey_Types"], valid_combined["MeanEncounterRate"])
            print(f"\nCombined Prey Types (0-{len(prey_indicators)}):")
            print(f"  Spearman ρ = {r:.3f}, p = {p:.4f}")
    
    if results:
        results_df = pd.DataFrame(results)
        results_path = OUTPUT_DIR / "prey_availability_analysis.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\n  Saved: {results_path}")
        return results_df
    
    return None


def plot_prey_analysis(df: pd.DataFrame):
    """Plot prey availability analysis."""
    
    df_valid = df.dropna(subset=["MeanEncounterRate"]).copy()
    
    # Prey indicator mappings: name -> list of columns indicating presence (P=patchy, C=continuous)
    # Exclude empty indicators and diversity metrics
    diversity_suffixes = ("_Richness", "_Shannon", "_Simpson", "_Evenness")
    prey_indicators = {
        "Oyster": [c for c in df_valid.columns 
                   if c.startswith("OYS_UNIT_") 
                   and not c.endswith(diversity_suffixes)
                   and c not in ["OYS_UNIT_ ", "OYS_UNIT_"]],
        "Mussel": [c for c in df_valid.columns 
                   if c.startswith("MUS_UNIT_") 
                   and not c.endswith(diversity_suffixes)
                   and c not in ["MUS_UNIT_ ", "MUS_UNIT_"]]
    }
    
    # Filter to indicators that have columns
    prey_indicators = {k: v for k, v in prey_indicators.items() if v}
    
    if not prey_indicators:
        return
    
    n_plots = len(prey_indicators) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    for i, (prey_name, presence_cols) in enumerate(prey_indicators.items()):
        ax = axes[i]
        
        # Prey is present if any presence column (P or C) is True
        present_col = f"{prey_name}_Present"
        label_col = f"{prey_name}_Presence"
        df_valid[present_col] = df_valid[presence_cols].any(axis=1).astype(bool)
        df_valid[label_col] = df_valid[present_col].map({False: "Absent", True: "Present"})
        
        colors = {"Present": "#27ae60", "Absent": "#e74c3c"}
        
        sns.boxplot(
            data=df_valid,
            x=label_col,
            y="MeanEncounterRate",
            order=["Absent", "Present"],
            palette=colors,
            ax=ax,
        )
        
        ax.set_xlabel(prey_name)
        ax.set_ylabel("Mean Encounter Rate (Pycno/hr)")
        ax.set_title(f"Encounter Rate by {prey_name} Presence")
    
    # Combined prey plot
    ax = axes[-1]
    if len(prey_indicators) >= 2:
        # Count how many prey types are present
        prey_presence_cols = [f"{name}_Present" for name in prey_indicators.keys()]
        df_valid["Total_Prey_Types"] = df_valid[prey_presence_cols].sum(axis=1)
        
        ax.scatter(df_valid["Total_Prey_Types"], df_valid["MeanEncounterRate"],
                   alpha=0.6, edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Number of Prey Types Present")
        ax.set_ylabel("Mean Encounter Rate (Pycno/hr)")
        ax.set_title("Encounter Rate vs Prey Type Diversity")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "prey_availability_analysis")
    plt.close()


# =============================================================================
# 6. Priority Site Identification
# =============================================================================

def identify_priority_sites_with_uncertainty(
    df: pd.DataFrame, 
    model, 
    feature_cols: list,
    n_bootstrap: int = 100
):
    """
    Identify priority sites with bootstrap uncertainty quantification.
    
    Combines:
    - Prediction of encounter rates for undersampled sites
    - Bootstrap confidence intervals for uncertainty
    - Spatial clustering of priority sites
    - Validation against sites with limited observations
    """
    print("\n" + "=" * 60)
    print("PRIORITY SITE IDENTIFICATION WITH UNCERTAINTY")
    print("=" * 60)
    
    if model is None:
        print("No model available for prediction")
        return None
    
    # Prepare training data for bootstrap
    df_train = df[
        (df["TotalSurveyTime"].fillna(0) >= 60) &
        (df["MeanEncounterRate"].notna())
    ].dropna(subset=feature_cols).copy()
    
    X_train = df_train[feature_cols].values
    y_train = df_train["MeanEncounterRate"].values
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    
    # Identify undersampled sites
    df_priority = df.copy()
    df_priority["SurveyEffort"] = df_priority["TotalSurveyTime"].fillna(0)
    undersampled = df_priority[df_priority["SurveyEffort"] < 60].copy()
    undersampled_complete = undersampled.dropna(subset=feature_cols)
    
    print(f"\nTraining sites: {len(df_train)}")
    print(f"Undersampled sites (< 60 min): {len(undersampled)}")
    print(f"Undersampled with complete features: {len(undersampled_complete)}")
    
    if len(undersampled_complete) == 0:
        print("No undersampled sites with complete data")
        return None
    
    X_predict = undersampled_complete[feature_cols].values
    X_predict = np.nan_to_num(X_predict, nan=0, posinf=0, neginf=0)
    
    # Bootstrap predictions for uncertainty
    print(f"\nRunning {n_bootstrap} bootstrap iterations...")
    bootstrap_predictions = np.zeros((n_bootstrap, len(X_predict)))
    
    for i in range(n_bootstrap):
        if (i + 1) % 25 == 0:
            print(f"  Iteration {i + 1}/{n_bootstrap}")
        
        # Bootstrap sample
        indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]
        
        # Fit model and predict
        boot_model = RandomForestRegressor(**model.get_params())
        boot_model.fit(X_boot, y_boot)
        bootstrap_predictions[i] = boot_model.predict(X_predict)
    
    # Calculate statistics
    pred_mean = bootstrap_predictions.mean(axis=0)
    pred_std = bootstrap_predictions.std(axis=0)
    pred_ci_low = np.percentile(bootstrap_predictions, 2.5, axis=0)
    pred_ci_high = np.percentile(bootstrap_predictions, 97.5, axis=0)
    
    # Create results dataframe
    results = undersampled_complete[["SiteName", "Lat", "Long", "TotalSurveyTime", "MeanEncounterRate"]].copy()
    results["Predicted_Rate"] = pred_mean
    results["Prediction_Std"] = pred_std
    results["CI_95_Low"] = pred_ci_low
    results["CI_95_High"] = pred_ci_high
    results["CI_Width"] = pred_ci_high - pred_ci_low
    results["Relative_Uncertainty"] = pred_std / (pred_mean + 0.01)
    
    # Priority score (penalize high uncertainty)
    results["Priority_Score"] = results["Predicted_Rate"] * (1 - results["Relative_Uncertainty"].clip(0, 1))
    
    # Priority tiers
    results["Priority_Tier"] = pd.cut(
        results["Priority_Score"],
        bins=[-np.inf, results["Priority_Score"].quantile(0.5),
              results["Priority_Score"].quantile(0.75),
              results["Priority_Score"].quantile(0.9), np.inf],
        labels=["Low", "Medium", "High", "Very High"]
    )
    
    results = results.sort_values("Priority_Score", ascending=False)
    
    # Summary
    print("\n--- Prediction Summary ---")
    print(f"Mean predicted rate: {pred_mean.mean():.3f} Pycno/hr")
    print(f"Mean uncertainty (std): {pred_std.mean():.3f} Pycno/hr")
    print(f"Sites with predicted rate > 5/hr: {(pred_mean > 5).sum()}")
    
    print("\n--- Priority Tier Distribution ---")
    for tier in ["Very High", "High", "Medium", "Low"]:
        count = (results["Priority_Tier"] == tier).sum()
        print(f"  {tier}: {count} sites")
    
    print("\n--- Top 15 Priority Sites ---")
    display_cols = ["SiteName", "Predicted_Rate", "CI_95_Low", "CI_95_High", "Priority_Tier"]
    print(results.head(15)[display_cols].to_string(index=False))
    
    # Save
    results_path = OUTPUT_DIR / "priority_sites_with_uncertainty.csv"
    results.to_csv(results_path, index=False)
    print(f"\n  Saved: {results_path}")
    
    return results, bootstrap_predictions


def analyze_spatial_clustering(results: pd.DataFrame, n_clusters: int = 5):
    """Identify spatial clusters of priority sites."""
    print("\n" + "=" * 60)
    print("SPATIAL CLUSTERING ANALYSIS")
    print("=" * 60)
    
    df_spatial = results.dropna(subset=["Lat", "Long"]).copy()
    
    if len(df_spatial) < n_clusters:
        print(f"Insufficient sites for clustering (need >{n_clusters})")
        return None, None
    
    coords = df_spatial[["Lat", "Long"]].values
    linkage_matrix = linkage(coords, method="ward")
    clusters = fcluster(linkage_matrix, n_clusters, criterion="maxclust")
    df_spatial["Cluster"] = clusters
    
    print("\n--- Cluster Summary ---")
    cluster_summary = []
    for cluster_id in range(1, n_clusters + 1):
        cluster_sites = df_spatial[df_spatial["Cluster"] == cluster_id]
        if len(cluster_sites) == 0:
            continue
        
        summary = {
            "Cluster": cluster_id,
            "N_Sites": len(cluster_sites),
            "Mean_Predicted_Rate": cluster_sites["Predicted_Rate"].mean(),
            "Top_Site": cluster_sites.iloc[0]["SiteName"]
        }
        cluster_summary.append(summary)
        
        print(f"\nCluster {cluster_id}: {len(cluster_sites)} sites")
        print(f"  Mean rate: {summary['Mean_Predicted_Rate']:.3f} Pycno/hr")
        print(f"  Top sites: {', '.join(cluster_sites.head(3)['SiteName'].tolist())}")
    
    summary_df = pd.DataFrame(cluster_summary).sort_values("Mean_Predicted_Rate", ascending=False)
    
    cluster_path = OUTPUT_DIR / "priority_site_clusters.csv"
    df_spatial.to_csv(cluster_path, index=False)
    print(f"\n  Saved: {cluster_path}")
    
    return df_spatial, summary_df


def validate_predictions(results: pd.DataFrame):
    """Validate predictions against sites with limited but non-zero observations."""
    print("\n" + "=" * 60)
    print("VALIDATION ANALYSIS")
    print("=" * 60)
    
    sites_with_obs = results[results["MeanEncounterRate"].notna()].copy()
    
    print(f"\nPrediction sites with some observations: {len(sites_with_obs)}")
    
    if len(sites_with_obs) < 5:
        print("Insufficient sites for validation")
        return None
    
    comparison = sites_with_obs[["SiteName", "TotalSurveyTime", "MeanEncounterRate", 
                                  "Predicted_Rate", "CI_95_Low", "CI_95_High"]].copy()
    comparison["Difference"] = comparison["Predicted_Rate"] - comparison["MeanEncounterRate"]
    comparison["Within_CI"] = (
        (comparison["MeanEncounterRate"] >= comparison["CI_95_Low"]) & 
        (comparison["MeanEncounterRate"] <= comparison["CI_95_High"])
    )
    
    r_corr, p_corr = pearsonr(comparison["MeanEncounterRate"], comparison["Predicted_Rate"])
    rmse = np.sqrt(mean_squared_error(comparison["MeanEncounterRate"], comparison["Predicted_Rate"]))
    within_ci_pct = comparison["Within_CI"].mean() * 100
    
    print(f"\n--- Validation Statistics ---")
    print(f"Correlation (r): {r_corr:.3f} (p = {p_corr:.4f})")
    print(f"RMSE: {rmse:.3f} Pycno/hr")
    print(f"Observations within 95% CI: {within_ci_pct:.1f}%")
    
    validation_path = OUTPUT_DIR / "validation_analysis.csv"
    comparison.to_csv(validation_path, index=False)
    print(f"\n  Saved: {validation_path}")
    
    return comparison


def plot_priority_sites_with_uncertainty(results: pd.DataFrame, df_surveyed: pd.DataFrame, 
                                         validation: pd.DataFrame = None):
    """Plot priority site identification with uncertainty."""
    
    if results is None or len(results) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Top sites with error bars
    ax1 = axes[0, 0]
    top_sites = results.head(25)
    
    colors = {"Very High": "#27ae60", "High": "#f1c40f", 
              "Medium": "#e67e22", "Low": "#e74c3c"}
    bar_colors = [colors.get(tier, "gray") for tier in top_sites["Priority_Tier"]]
    
    ax1.barh(range(len(top_sites)), top_sites["Predicted_Rate"], 
             xerr=[top_sites["Predicted_Rate"] - top_sites["CI_95_Low"],
                   top_sites["CI_95_High"] - top_sites["Predicted_Rate"]],
             capsize=3, color=bar_colors, edgecolor="black", alpha=0.7)
    ax1.set_yticks(range(len(top_sites)))
    ax1.set_yticklabels(top_sites["SiteName"], fontsize=7)
    ax1.set_xlabel("Predicted Encounter Rate (Pycno/hr)")
    ax1.set_title("Top 25 Priority Sites with 95% CI")
    ax1.invert_yaxis()
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
    ax1.legend(handles=legend_elements, loc="lower right")
    
    # 2. Priority tier pie chart
    ax2 = axes[0, 1]
    tier_counts = results["Priority_Tier"].value_counts()
    tier_order = ["Very High", "High", "Medium", "Low"]
    tier_counts = tier_counts.reindex(tier_order).dropna()
    
    ax2.pie(tier_counts.values, labels=tier_counts.index, autopct="%1.1f%%",
            colors=[colors[t] for t in tier_counts.index],
            explode=[0.05 if t == "Very High" else 0 for t in tier_counts.index])
    ax2.set_title("Priority Tier Distribution")
    
    # 3. Geographic distribution
    ax3 = axes[1, 0]
    results_with_coords = results.dropna(subset=["Lat", "Long"])
    tier_colors = [colors.get(tier, "gray") for tier in results_with_coords["Priority_Tier"]]
    
    ax3.scatter(results_with_coords["Long"], results_with_coords["Lat"],
                c=tier_colors, s=results_with_coords["Priority_Score"] * 5 + 10,
                alpha=0.7, edgecolor="black", linewidth=0.3)
    
    # Highlight top 10
    top_10 = results_with_coords.head(10)
    ax3.scatter(top_10["Long"], top_10["Lat"],
                facecolors="none", edgecolors="black", s=200, linewidth=2)
    
    for _, row in top_10.head(5).iterrows():
        ax3.annotate(row["SiteName"][:15], xy=(row["Long"], row["Lat"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=8)
    
    ax3.set_xlabel("Longitude")
    ax3.set_ylabel("Latitude")
    ax3.set_title("Geographic Distribution of Priority Sites")
    ax3.set_aspect("equal")
    
    # 4. Validation plot (if available)
    ax4 = axes[1, 1]
    
    if validation is not None and len(validation) >= 5:
        val_colors = ["#27ae60" if w else "#e74c3c" for w in validation["Within_CI"]]
        
        ax4.scatter(validation["MeanEncounterRate"], validation["Predicted_Rate"],
                    c=val_colors, s=80, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax4.errorbar(validation["MeanEncounterRate"], validation["Predicted_Rate"],
                     yerr=[validation["Predicted_Rate"] - validation["CI_95_Low"],
                           validation["CI_95_High"] - validation["Predicted_Rate"]],
                     fmt="none", ecolor="gray", alpha=0.5, capsize=3)
        
        max_val = max(validation["MeanEncounterRate"].max(), validation["Predicted_Rate"].max())
        ax4.plot([0, max_val], [0, max_val], "r--", linewidth=2, label="1:1 line")
        
        r_corr, _ = pearsonr(validation["MeanEncounterRate"], validation["Predicted_Rate"])
        within_ci = validation["Within_CI"].mean() * 100
        ax4.set_xlabel("Observed Encounter Rate (Pycno/hr)")
        ax4.set_ylabel("Predicted Encounter Rate (Pycno/hr)")
        ax4.set_title(f"Validation: r = {r_corr:.3f}, {within_ci:.0f}% within CI\n(Green = within 95% CI)")
        ax4.legend()
    else:
        # Show distribution instead
        observed = df_surveyed.dropna(subset=["MeanEncounterRate"])["MeanEncounterRate"]
        predicted = results["Predicted_Rate"]
        
        ax4.hist(observed, bins=20, alpha=0.5, label="Observed", color="gray", edgecolor="black")
        ax4.hist(predicted, bins=20, alpha=0.5, label="Predicted", color="blue", edgecolor="black")
        ax4.axvline(observed.mean(), color="gray", linestyle="--", linewidth=2)
        ax4.axvline(predicted.mean(), color="blue", linestyle="--", linewidth=2)
        ax4.set_xlabel("Encounter Rate (Pycno/hr)")
        ax4.set_ylabel("Number of Sites")
        ax4.set_title("Distribution of Observed vs Predicted Rates")
        ax4.legend()
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "priority_sites_with_uncertainty")
    plt.close()


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def main():
    """Run the complete ShoreZone recovery analysis pipeline."""
    
    print("\n" + "=" * 70)
    print("SHOREZONE-BASED SUNFLOWER STAR RECOVERY ANALYSIS")
    print("=" * 70)
    
    set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Load data
    print("\n" + "-" * 60)
    print("LOADING DATA")
    print("-" * 60)
    
    try:
        df = load_site_shorezone_summary()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return
    
    df_analysis = prepare_analysis_data(df)
    
    # 1. Habitat Suitability Model
    print("\n" + "-" * 60)
    print("1. HABITAT SUITABILITY MODEL")
    print("-" * 60)
    
    model, feature_cols, model_results = build_habitat_suitability_model(df_analysis)
    
    if model_results:
        plot_model_results(model_results, df_analysis)
        
        # Save model results
        model_results["feature_importance"].to_csv(OUTPUT_DIR / "feature_importance_mdi.csv", index=False)
        model_results["perm_importance"].to_csv(OUTPUT_DIR / "feature_importance_permutation.csv", index=False)
    
    # 2. Refugia Analysis
    print("\n" + "-" * 60)
    print("2. REFUGIA ANALYSIS")
    print("-" * 60)
    
    refugia_output = analyze_refugia(df_analysis)
    if refugia_output:
        refugia_results, refugia_sites, non_refugia_sites = refugia_output
        plot_refugia_comparison(df_analysis, refugia_results)
    else:
        refugia_results = None
    
    # 3. Eelgrass Indicator Comparison
    print("\n" + "-" * 60)
    print("3. EELGRASS INDICATOR COMPARISON")
    print("-" * 60)
    
    eelgrass_output = analyze_eelgrass_indicators(df_analysis)
    if eelgrass_output:
        plot_eelgrass_comparison(df_analysis)
    
    # 4. Wave Exposure Analysis
    print("\n" + "-" * 60)
    print("4. WAVE EXPOSURE ANALYSIS")
    print("-" * 60)
    
    exp_output = analyze_wave_exposure(df_analysis)
    if exp_output:
        df_exp, exp_results = exp_output
        plot_exposure_analysis(df_exp)
    
    # 5. Prey Availability Analysis
    print("\n" + "-" * 60)
    print("5. PREY AVAILABILITY ANALYSIS")
    print("-" * 60)
    
    prey_results = analyze_prey_availability(df_analysis)
    if prey_results is not None:
        plot_prey_analysis(df_analysis)
    
    # 6. Priority Site Identification with Uncertainty
    print("\n" + "-" * 60)
    print("6. PRIORITY SITE IDENTIFICATION")
    print("-" * 60)
    
    priority_result = identify_priority_sites_with_uncertainty(df, model, feature_cols, n_bootstrap=100)
    
    if priority_result is not None:
        priority_sites, bootstrap_preds = priority_result
        
        # Spatial clustering
        print("\n" + "-" * 60)
        print("7. SPATIAL CLUSTERING")
        print("-" * 60)
        
        df_spatial, cluster_summary = analyze_spatial_clustering(priority_sites, n_clusters=5)
        
        # Validation
        print("\n" + "-" * 60)
        print("8. VALIDATION ANALYSIS")
        print("-" * 60)
        
        validation = validate_predictions(priority_sites)
        
        # Plot with uncertainty
        plot_priority_sites_with_uncertainty(priority_sites, df_analysis, validation)
    else:
        priority_sites = None
        validation = None
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    n_priority = len(priority_sites) if priority_sites is not None else 0
    n_very_high = (priority_sites["Priority_Tier"] == "Very High").sum() if priority_sites is not None else 0
    
    print(f"""
Sites analyzed: {len(df_analysis)}

1. HABITAT SUITABILITY MODEL
   - Cross-validation R²: {model_results['cv_r2_mean']:.3f} ± {model_results['cv_r2_std']:.3f}
   - Top predictor: {model_results['perm_importance'].iloc[0]['Feature']}

2. REFUGIA ANALYSIS
   - Refugia sites identified: {len(df_analysis[df_analysis['DetectionRate'] >= 0.5])}
   - Key differentiating features saved to refugia_comparison.csv

3. EELGRASS INDICATORS
   - Dive survey vs ShoreZone comparison completed
   - Results saved to eelgrass_indicator_comparison.csv

4. WAVE EXPOSURE
   - Exposure-encounter rate relationships analyzed
   - Results saved to exposure_analysis.csv

5. PREY AVAILABILITY
   - Bivalve presence analysis completed
   - Results saved to prey_availability_analysis.csv

6. PRIORITY SITES (with uncertainty quantification)
   - {n_priority} undersampled sites ranked with 95% confidence intervals
   - {n_very_high} sites classified as "Very High" priority
   - Spatial clustering identifies regional hotspots
   - Results saved to priority_sites_with_uncertainty.csv

Output files saved to: {OUTPUT_DIR}
""")
    
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70 + "\n")

    # Write markdown summary for embedding in the site coverage map
    try:
        md_path = write_results_md(OUTPUT_DIR)
        print(f"Wrote results summary: {md_path}")
    except Exception as e:
        print(f"Warning: could not write results.md summary: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShoreZone recovery analysis")
    parser.add_argument(
        "--write-results-only",
        action="store_true",
        help="Write outputs/16_shorezone_recovery_analysis/results.md from existing CSV outputs and exit.",
    )
    args = parser.parse_args()

    if args.write_results_only:
        md_path = write_results_md(OUTPUT_DIR)
        print(f"Wrote results summary: {md_path}")
    else:
        main()

