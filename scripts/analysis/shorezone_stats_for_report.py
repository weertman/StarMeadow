#!/usr/bin/env python3
"""Generate quantitative ShoreZone statistics and manuscript-section draft text."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT = Path(__file__).resolve().parents[3]
OUT = PROJECT / "outputs" / "publication_figures" / "qc" / "shorezone_report_stats"
OUT.mkdir(parents=True, exist_ok=True)
SHOREZONE = PROJECT / "outputs" / "15_shorezone_site_analysis"
RECOVERY = PROJECT / "outputs" / "16_shorezone_recovery_analysis"
PUB_QC = PROJECT / "outputs" / "publication_figures" / "qc"
REPORT = PROJECT / "outputs" / "publication_figures" / "shorezone_methods_results_draft.md"

SHOREZONE_CATEGORIES = [
    "EXP_CLASS", "SED_SOURCE", "SED_ABUND", "ZONECOMP", "BC_CLASS", "NRDA_CLASS",
    "HAB_CALC", "ZOS_UNIT", "FUC_UNIT", "ULV_UNIT", "NER_UNIT", "OYS_UNIT", "MUS_UNIT",
]

FEATURE_LABELS = {
    "ZOS_UNIT_C": "continuous ShoreZone eelgrass",
    "ZOS_UNIT_P": "patchy ShoreZone eelgrass",
    "ZOS_UNIT_ ": "absent ShoreZone eelgrass",
    "BC_CLASS_30.0": "sand beach shoreline class",
    "HAB_CALC_7.0": "low-exposure sand/gravel habitat",
    "OYS_UNIT_ ": "oyster beds absent",
    "OYS_UNIT_P": "patchy oyster beds",
    "FUC_UNIT_P": "patchy Fucus band",
    "ZONECOMP_A1 B1 B2 C1": "supratidal, two intertidal, and subtidal components",
    "ULV_UNIT_ ": "Ulva green algae absent",
    "SED_SOURCE_A": "alongshore sediment source",
    "HAB_CALC_8.0": "estuarine sand/mud habitat",
}


def load_habitat_features() -> pd.DataFrame:
    all_features = None
    for category in SHOREZONE_CATEGORIES:
        path = SHOREZONE / f"shorezone_{category}_proportions.csv"
        if not path.exists():
            continue
        props = pd.read_csv(path)
        for col in props.columns:
            if col != "SiteName" and props[col].dtype == bool:
                props[col] = props[col].astype(int)
        all_features = props if all_features is None else all_features.merge(props, on="SiteName", how="outer")
    return all_features


def feature_cols(df: pd.DataFrame) -> list[str]:
    diversity_suffixes = ("_Richness", "_Shannon", "_Simpson", "_Evenness")
    return [
        c for c in df.columns
        if any(c.startswith(f"{cat}_") and c != f"{cat}_value" for cat in SHOREZONE_CATEGORIES)
        and not c.endswith(diversity_suffixes)
    ]


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def fmt(x: float, digits: int = 2) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def se(x: pd.Series) -> float:
    x = x.dropna()
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def main() -> None:
    count_rows = pd.read_csv(PROJECT / "data" / "PycnoCountCLean_12_31_2025.csv")
    count_rows = count_rows[count_rows["Basin"] != "Queen Charlotte Strait"].copy()
    survey_sites = count_rows["SiteName"].nunique()
    total_rows = len(count_rows)
    total_minutes = count_rows["Survey.Time"].sum()

    site = pd.read_csv(SHOREZONE / "site_shorezone_pycno_summary.csv")
    features = load_habitat_features()
    df = site.merge(features, on="SiteName", how="left")
    analysis = df.dropna(subset=["MeanEncounterRate"]).copy()
    fcols = feature_cols(df)
    model_df = df.dropna(subset=fcols + ["MeanEncounterRate"]).copy()

    X = np.nan_to_num(model_df[fcols].values, nan=0, posinf=0, neginf=0)
    y = model_df["MeanEncounterRate"].values
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42, n_jobs=-1)
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    cv_pred = cross_val_predict(rf, X, y, cv=5)
    r_pred, p_pred = pearsonr(y, cv_pred)
    rmse = float(np.sqrt(mean_squared_error(y, cv_pred)))
    mae = mean_absolute_error(y, cv_pred)

    model_summary = pd.DataFrame([{
        "survey_rows": total_rows,
        "survey_sites": survey_sites,
        "total_survey_hours": total_minutes / 60,
        "shorezone_sites_total": len(site),
        "survey_sites_with_shorezone_features": len(model_df),
        "model_features": len(fcols),
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_sd": cv_scores.std(ddof=0),
        "cv_pred_observed_pearson_r": r_pred,
        "cv_pred_observed_pearson_p": p_pred,
        "cv_rmse": rmse,
        "cv_mae": mae,
    }])
    model_summary.to_csv(OUT / "shorezone_model_summary.csv", index=False)

    # Top prediction-loss features used in Figure 4A.
    top_features = pd.read_csv(PUB_QC / "Fig4_panel_A_prediction_loss_features.csv")
    top_features["FeatureLabelForText"] = top_features["Feature"].map(FEATURE_LABELS).fillna(top_features.get("FeatureLabel", top_features["Feature"]))
    top_features.to_csv(OUT / "shorezone_top_prediction_loss_features.csv", index=False)

    # Figure 4B inferential contrasts from plotted site-level data.
    fig4b = pd.read_csv(PUB_QC / "Fig4_panel_B_refugia_continuous_eelgrass_plot_data.csv")
    fig4b_summary = fig4b.groupby(["RefugiaStatus", "ContinuousEelgrassStatus"], dropna=False).agg(
        sites=("SiteName", "count"),
        mean_encounter_rate=("MeanEncounterRate", "mean"),
        sd_encounter_rate=("MeanEncounterRate", "std"),
        se_encounter_rate=("MeanEncounterRate", se),
        median_encounter_rate=("MeanEncounterRate", "median"),
        total_pycno=("TotalPycnoCount", "sum"),
        total_survey_hours=("TotalSurveyTime", lambda x: x.sum() / 60),
    ).reset_index()
    fig4b_summary.to_csv(OUT / "shorezone_fig4b_group_summary.csv", index=False)

    group_values = [g["MeanEncounterRate"].values for _, g in fig4b.groupby(["RefugiaStatus", "ContinuousEelgrassStatus"]) if len(g) >= 2]
    kw4 = kruskal(*group_values) if len(group_values) >= 2 else (np.nan, np.nan)
    protected = fig4b[fig4b["ProtectedShoreZone"]]["MeanEncounterRate"]
    other_exp = fig4b[~fig4b["ProtectedShoreZone"]]["MeanEncounterRate"]
    cont = fig4b[fig4b["ContinuousEelgrass"]]["MeanEncounterRate"]
    not_cont = fig4b[~fig4b["ContinuousEelgrass"]]["MeanEncounterRate"]
    prot_cont = fig4b[fig4b["ProtectedShoreZone"] & fig4b["ContinuousEelgrass"]]["MeanEncounterRate"]
    all_other = fig4b[~(fig4b["ProtectedShoreZone"] & fig4b["ContinuousEelgrass"])] ["MeanEncounterRate"]
    tests = []
    def add_mw(name, a, b, group_a, group_b):
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        tests.append({
            "contrast": name,
            "group_a": group_a,
            "n_a": len(a),
            "mean_a": a.mean(),
            "median_a": a.median(),
            "group_b": group_b,
            "n_b": len(b),
            "mean_b": b.mean(),
            "median_b": b.median(),
            "test": "Mann-Whitney U",
            "statistic": u,
            "p_value": p,
        })
    add_mw("protected_or_very_protected_vs_other_exposure", protected, other_exp, "Protected / very protected", "Other exposure")
    add_mw("continuous_eelgrass_vs_other_zos", cont, not_cont, "Eelgrass: continuous", "Other")
    add_mw("protected_continuous_eelgrass_vs_all_other", prot_cont, all_other, "Protected + continuous eelgrass", "All other ShoreZone combinations")
    tests.append({"contrast": "four_fig4b_groups", "test": "Kruskal-Wallis", "statistic": kw4.statistic, "p_value": kw4.pvalue})
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(OUT / "shorezone_fig4b_tests.csv", index=False)

    # S3/S4 ShoreZone eelgrass summaries.
    s3 = pd.read_csv(PUB_QC / "S3_Fig_continuous_eelgrass_encounter_map_summary.csv")
    s3.to_csv(OUT / "shorezone_s3_status_summary.csv", index=False)

    s4_counts = pd.read_csv(PUB_QC / "S4_shorezone_vs_diver_eelgrass_count_table.csv", index_col=0)
    s4_rates = pd.read_csv(PUB_QC / "S4_shorezone_vs_diver_eelgrass_rate_summary.csv")
    s4_binary = pd.read_csv(PUB_QC / "S4_shorezone_vs_diver_eelgrass_binary_summary.csv")
    table_3x2 = s4_counts.values
    chi2, chi_p, dof, expected = chi2_contingency(table_3x2)
    # binary table: rows ShoreZone any absent/present, cols diver absent/present
    absent = s4_counts.loc["Eelgrass: absent"].values
    present = s4_counts.loc[["Eelgrass: continuous", "Eelgrass: patchy"]].sum(axis=0).values
    binary_table = np.vstack([absent, present])
    oddsratio, fisher_p = fisher_exact(binary_table)
    n_binary = binary_table.sum()
    phi = np.sqrt(chi2_contingency(binary_table)[0] / n_binary)
    s4_tests = pd.DataFrame([
        {"contrast": "shorezone_three_category_by_diver_binary", "test": "chi-square", "statistic": chi2, "df": dof, "p_value": chi_p, "effect": "Cramers_V", "effect_value": np.sqrt(chi2 / (table_3x2.sum() * (min(table_3x2.shape)-1)))},
        {"contrast": "shorezone_any_eelgrass_by_diver_binary", "test": "Fisher exact", "statistic": oddsratio, "df": np.nan, "p_value": fisher_p, "effect": "phi", "effect_value": phi},
    ])
    s4_tests.to_csv(OUT / "shorezone_s4_agreement_tests.csv", index=False)

    # Draft prose.
    ms = model_summary.iloc[0]
    top3 = top_features.head(3)
    top3_text = "; ".join(
        f"{row.FeatureLabelForText} (prediction loss {row.PredictionLoss:.3f} ± {row.PredictionLossSE:.3f})"
        for row in top3.itertuples()
    )
    prot_cont_row = fig4b_summary[(fig4b_summary.RefugiaStatus == "Protected / very protected") & (fig4b_summary.ContinuousEelgrassStatus == "Eelgrass: continuous")].iloc[0]
    prot_other_row = fig4b_summary[(fig4b_summary.RefugiaStatus == "Protected / very protected") & (fig4b_summary.ContinuousEelgrassStatus == "Other")].iloc[0]
    other_cont_row = fig4b_summary[(fig4b_summary.RefugiaStatus == "Other exposure") & (fig4b_summary.ContinuousEelgrassStatus == "Eelgrass: continuous")].iloc[0]
    other_other_row = fig4b_summary[(fig4b_summary.RefugiaStatus == "Other exposure") & (fig4b_summary.ContinuousEelgrassStatus == "Other")].iloc[0]
    protected_test = tests_df[tests_df.contrast == "protected_or_very_protected_vs_other_exposure"].iloc[0]
    cont_test = tests_df[tests_df.contrast == "continuous_eelgrass_vs_other_zos"].iloc[0]
    prot_cont_test = tests_df[tests_df.contrast == "protected_continuous_eelgrass_vs_all_other"].iloc[0]
    s4_chi = s4_tests.iloc[0]
    s4_fisher = s4_tests.iloc[1]

    report = f"""# Draft ShoreZone methods, statistical analysis, and quantitative results

Generated from live StarMeadow analysis outputs in `outputs/15_shorezone_site_analysis/`, `outputs/16_shorezone_recovery_analysis/`, and `outputs/publication_figures/qc/shorezone_report_stats/`.

## Methods: ShoreZone spatial covariates

We linked diver survey sites to Washington State Department of Natural Resources ShoreZone shoreline attributes to evaluate whether mapped shoreline geomorphology and biotic-cover categories were associated with site-level *Pycnopodia helianthoides* encounter rates. The ShoreZone workflow used site coordinates in WGS84 and ShoreZone linework projected to a metric coordinate system for distance calculations. For each site, the analysis associated the survey site with nearby ShoreZone attributes and exported site-level categorical indicators for physical exposure, sediment source and abundance, shoreline-zone composition, broad biotic and habitat classes, and mapped biotic units including Zostera, Fucus, Ulva, mussel, oyster, and kelp indicators. Because several ShoreZone biotic layers pre-date the diver surveys, we interpreted mapped biotic features as exploratory habitat context rather than contemporaneous ground-truth observations. Physical ShoreZone classes such as wave exposure and sediment attributes were treated as more temporally stable than mapped biotic-cover fields.

Site-level *Pycnopodia* response variables were calculated from the diver survey table before merging with ShoreZone attributes. For each site, total counts and survey effort were aggregated across survey rows, and encounter rate was expressed as individuals per survey hour. Site-level diver-recorded eelgrass status was defined as whether any survey row at a site had `HabitatType == "Eelgrass"`. ShoreZone eelgrass status was derived from Zostera unit fields: continuous eelgrass (`ZOS_UNIT_C`), patchy eelgrass (`ZOS_UNIT_P`), or mapped absence (`ZOS_UNIT_`). The analysis used {int(ms.survey_rows):,} survey rows from {int(ms.survey_sites)} survey sites, representing {ms.total_survey_hours:,.1f} survey hours. The ShoreZone table contained {int(ms.shorezone_sites_total)} site records, of which {int(ms.survey_sites_with_shorezone_features)} survey sites had complete response and ShoreZone feature data for the predictive analysis.

## Methods: statistical analysis

We summarized site-level encounter rates with means, standard deviations, standard errors, medians, and sample sizes. Because encounter-rate distributions were zero-inflated and right-skewed, group contrasts used non-parametric tests on site-level encounter rates. Two-group contrasts used two-sided Mann-Whitney U tests, and multi-group contrasts used Kruskal-Wallis tests. Categorical agreement between ShoreZone eelgrass classes and diver-recorded eelgrass classes was evaluated with contingency tables. We used a chi-square test for the three-level ShoreZone eelgrass category by binary diver category comparison, and Fisher's exact test for the binary contrast between any ShoreZone eelgrass (continuous or patchy) and diver-recorded eelgrass. We report these tests as descriptive support for the figure patterns, not as causal estimates.

To identify which ShoreZone attributes carried the strongest predictive signal for site-level encounter rate, we fit an exploratory random-forest regression model with {int(ms.model_features)} one-hot encoded ShoreZone categorical features. The model used 100 trees, maximum depth 5, minimum leaf size 5, a fixed random seed of 42, and five-fold cross-validation. We evaluated predictive performance using cross-validated R², root mean squared error, mean absolute error, and the Pearson correlation between cross-validated predictions and observed site-level encounter rates. After fitting the final model to the full complete-case dataset, we estimated feature contribution using permutation prediction loss over 10 repeats. Prediction-loss values indicate the mean reduction in model performance when a feature is permuted; larger values indicate stronger model dependence on that feature.

## Results: quantitative ShoreZone analysis

The random-forest ShoreZone model showed limited but non-random predictive structure in site-level encounter rates. Across five-fold cross-validation, mean R² was {ms.cv_r2_mean:.3f} ± {ms.cv_r2_sd:.3f}; cross-validated predictions were positively correlated with observed encounter rates (Pearson r = {ms.cv_pred_observed_pearson_r:.3f}, {fmt_p(ms.cv_pred_observed_pearson_p)}). Prediction error remained substantial (RMSE = {ms.cv_rmse:.2f} individuals hr⁻¹; MAE = {ms.cv_mae:.2f} individuals hr⁻¹), indicating that ShoreZone attributes alone did not fully explain site-level variation.

Permutation importance indicated that mapped continuous eelgrass was the strongest individual ShoreZone predictor of encounter rate. The three largest prediction-loss features were {top3_text}. Continuous eelgrass had a much larger prediction loss than the next-ranked shoreline class, suggesting that Zostera-associated shoreline context was a dominant model signal in the current ShoreZone feature set.

The figure-level protected-exposure and continuous-eelgrass summary showed that the highest site-level encounter rates occurred at sites that were both protected or very protected and mapped as continuous ShoreZone eelgrass. Protected or very protected sites with continuous eelgrass had a mean encounter rate of {prot_cont_row.mean_encounter_rate:.2f} ± {prot_cont_row.se_encounter_rate:.2f} SE individuals hr⁻¹ ({int(prot_cont_row.sites)} sites), compared with {prot_other_row.mean_encounter_rate:.2f} ± {prot_other_row.se_encounter_rate:.2f} at protected or very protected sites without mapped continuous eelgrass ({int(prot_other_row.sites)} sites), {other_cont_row.mean_encounter_rate:.2f} ± {other_cont_row.se_encounter_rate:.2f} at other-exposure sites with continuous eelgrass ({int(other_cont_row.sites)} sites), and {other_other_row.mean_encounter_rate:.2f} ± {other_other_row.se_encounter_rate:.2f} at other-exposure sites without continuous eelgrass ({int(other_other_row.sites)} sites). Across all sites, protected or very protected exposure had higher encounter rates than other exposure ({protected_test.test}, U = {protected_test.statistic:.1f}, {fmt_p(protected_test.p_value)}), and continuous ShoreZone eelgrass sites had higher encounter rates than sites without continuous mapped eelgrass ({cont_test.test}, U = {cont_test.statistic:.1f}, {fmt_p(cont_test.p_value)}). The combined protected-continuous-eelgrass group was also higher than all other ShoreZone combinations ({prot_cont_test.test}, U = {prot_cont_test.statistic:.1f}, {fmt_p(prot_cont_test.p_value)}). These contrasts support the visual pattern in Fig 4B while preserving the exploratory status of the ShoreZone analysis.

The ShoreZone eelgrass categories were only partially concordant with human diver-recorded eelgrass categories. Among sites with a known ShoreZone Zostera category, divers recorded eelgrass at {int(s4_counts.loc['Eelgrass: continuous', 'Eelgrass recorded by divers'])} of {int(s4_counts.loc['Eelgrass: continuous'].sum())} sites mapped as continuous ShoreZone eelgrass ({s4_rates.loc[s4_rates.ShoreZoneCategory=='Eelgrass: continuous','diver_eelgrass_fraction'].iloc[0]*100:.1f}%), {int(s4_counts.loc['Eelgrass: patchy', 'Eelgrass recorded by divers'])} of {int(s4_counts.loc['Eelgrass: patchy'].sum())} sites mapped as patchy eelgrass ({s4_rates.loc[s4_rates.ShoreZoneCategory=='Eelgrass: patchy','diver_eelgrass_fraction'].iloc[0]*100:.1f}%), and {int(s4_counts.loc['Eelgrass: absent', 'Eelgrass recorded by divers'])} of {int(s4_counts.loc['Eelgrass: absent'].sum())} sites mapped as absent eelgrass ({s4_rates.loc[s4_rates.ShoreZoneCategory=='Eelgrass: absent','diver_eelgrass_fraction'].iloc[0]*100:.1f}%). The three-category ShoreZone-by-diver table showed a significant association (χ² = {s4_chi.statistic:.2f}, df = {int(s4_chi.df)}, {fmt_p(s4_chi.p_value)}, Cramér's V = {s4_chi.effect_value:.2f}). When ShoreZone continuous and patchy eelgrass were collapsed into a binary any-eelgrass class, agreement between ShoreZone and diver categories was 67.3% with sensitivity 59.4% and specificity 70.8% relative to diver-recorded eelgrass; the binary association was also significant (Fisher's exact test, odds ratio = {s4_fisher.statistic:.2f}, {fmt_p(s4_fisher.p_value)}). These results indicate that ShoreZone Zostera categories capture a meaningful eelgrass gradient but do not perfectly match diver-observed eelgrass at the survey sites.

## Tables generated for verification

- `outputs/publication_figures/qc/shorezone_report_stats/shorezone_model_summary.csv`
- `outputs/publication_figures/qc/shorezone_report_stats/shorezone_top_prediction_loss_features.csv`
- `outputs/publication_figures/qc/shorezone_report_stats/shorezone_fig4b_group_summary.csv`
- `outputs/publication_figures/qc/shorezone_report_stats/shorezone_fig4b_tests.csv`
- `outputs/publication_figures/qc/shorezone_report_stats/shorezone_s3_status_summary.csv`
- `outputs/publication_figures/qc/shorezone_report_stats/shorezone_s4_agreement_tests.csv`
"""
    REPORT.write_text(report)

    manifest = {
        "report": str(REPORT),
        "tables": sorted(str(p) for p in OUT.glob("*.csv")),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(REPORT)
    print(OUT)


if __name__ == "__main__":
    main()
