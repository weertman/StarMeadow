#!/usr/bin/env python3
"""Diver-only manuscript statistics for StarMeadow Figs 1-3.

Outputs descriptive audits, site-aware count-model tables, and a site-cluster
bootstrap for individual lengths. The script intentionally excludes ShoreZone
analyses.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import mannwhitneyu, ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PUB_QC_DIR = PROJECT_ROOT / "outputs" / "publication_figures" / "qc"
STATS_DIR = PROJECT_ROOT / "outputs" / "publication_figures" / "stats"
STATS_DIR.mkdir(parents=True, exist_ok=True)

COUNT_FILE = DATA_DIR / "PycnoCountCLean_12_31_2025.csv"
LENGTH_FILE = DATA_DIR / "PycnoLengthCLean_12_31_2025.csv"
FIG1_PLOTTED_FILE = PUB_QC_DIR / "Fig1_site_map_plotted_sites.csv"
FIG1_MISSING_FILE = PUB_QC_DIR / "Fig1_site_map_survey_sites_missing_coordinates.csv"
FIG1_EXCLUDED_FILE = PUB_QC_DIR / "Fig1_site_map_excluded_north_vancouver_island_sites.csv"

BASIN_ORDER = [
    "Howe Sound",
    "Strait of Georgia",
    "San Juan",
    "Strait of Juan de Fuca",
    "Admiralty Inlet",
    "Whidbey",
    "Central",
    "South",
    "Hood Canal",
]
HABITAT_ORDER = ["Soft Bottom", "Eelgrass", "Artificial Reef", "Kelp Forest", "Natural Reef", "Sponge Garden"]
EELGRASS_ORDER = ["Eelgrass at site", "No eelgrass at site"]
RNG = np.random.default_rng(42)


@dataclass
class ModelSpec:
    name: str
    formula: str
    family: str
    note: str


def bool_status(value: object) -> str:
    return "Eelgrass at site" if bool(value) else "No eelgrass at site"


def se(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def q(series: pd.Series, prob: float) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) == 0:
        return np.nan
    return float(np.quantile(x, prob))


def load_count_data() -> pd.DataFrame:
    df = pd.read_csv(COUNT_FILE)
    df = df[df["Basin"] != "Queen Charlotte Strait"].copy()
    df["SurveyTimeMinutes"] = pd.to_numeric(df["Survey.Time"], errors="coerce")
    df["SurveyHours"] = df["SurveyTimeMinutes"] / 60.0
    df["Pycnopodia_count"] = pd.to_numeric(df["Pycnopodia_count"], errors="coerce").fillna(0)
    df["EncounterRateHr"] = np.where(
        df["SurveyHours"] > 0,
        df["Pycnopodia_count"] / df["SurveyHours"],
        np.nan,
    )
    df["SiteHasEelgrass"] = df.groupby("SiteName")["HabitatType"].transform(lambda x: (x == "Eelgrass").any())
    df["SiteHasEelgrassInt"] = df["SiteHasEelgrass"].astype(int)
    df["EelgrassStatus"] = df["SiteHasEelgrass"].map(bool_status)
    df["Detected"] = df["Pycnopodia_count"] > 0
    df = df.dropna(subset=["SiteName", "Basin", "HabitatType", "SurveyHours", "EncounterRateHr"])
    df = df[df["SurveyHours"] > 0].copy()
    df["LogSurveyHours"] = np.log(df["SurveyHours"])
    return df


def load_length_data() -> pd.DataFrame:
    df = pd.read_csv(LENGTH_FILE)
    df = df[df["Basin"] != "Queen Charlotte Strait"].copy()
    df["SurveyTimeMinutes"] = pd.to_numeric(df["Survey.Time"], errors="coerce")
    df["Length_cm"] = pd.to_numeric(df["Length(cm)"].astype(str).str.strip(), errors="coerce")
    df["SiteHasEelgrass"] = df.groupby("SiteName")["HabitatType"].transform(lambda x: (x == "Eelgrass").any())
    df["SiteHasEelgrassInt"] = df["SiteHasEelgrass"].astype(int)
    df["EelgrassStatus"] = df["SiteHasEelgrass"].map(bool_status)
    df = df[df["Length_cm"] > 0].dropna(subset=["SiteName", "Basin", "HabitatType", "Length_cm"]).copy()
    df["LogLength_cm"] = np.log(df["Length_cm"])
    return df


def write_input_audit(count_df: pd.DataFrame, length_df: pd.DataFrame) -> pd.DataFrame:
    audit = pd.DataFrame([
        {
            "source": str(COUNT_FILE.relative_to(PROJECT_ROOT)),
            "table": "count_rows",
            "rows": len(count_df),
            "sites": count_df["SiteName"].nunique(),
            "basins": count_df["Basin"].nunique(),
            "habitats": count_df["HabitatType"].nunique(),
            "total_survey_hours": count_df["SurveyHours"].sum(),
            "total_pycno_count": count_df["Pycnopodia_count"].sum(),
            "zero_fraction": float((count_df["Pycnopodia_count"] == 0).mean()),
            "survey_hours_min": count_df["SurveyHours"].min(),
            "survey_hours_median": count_df["SurveyHours"].median(),
            "survey_hours_max": count_df["SurveyHours"].max(),
            "encounter_rate_min": count_df["EncounterRateHr"].min(),
            "encounter_rate_median": count_df["EncounterRateHr"].median(),
            "encounter_rate_max": count_df["EncounterRateHr"].max(),
        },
        {
            "source": str(LENGTH_FILE.relative_to(PROJECT_ROOT)),
            "table": "positive_length_rows",
            "rows": len(length_df),
            "sites": length_df["SiteName"].nunique(),
            "basins": length_df["Basin"].nunique(),
            "habitats": length_df["HabitatType"].nunique(),
            "total_survey_hours": np.nan,
            "total_pycno_count": np.nan,
            "zero_fraction": np.nan,
            "survey_hours_min": np.nan,
            "survey_hours_median": np.nan,
            "survey_hours_max": np.nan,
            "encounter_rate_min": np.nan,
            "encounter_rate_median": np.nan,
            "encounter_rate_max": np.nan,
        },
    ])
    audit.to_csv(STATS_DIR / "diver_stats_input_audit.csv", index=False)
    return audit


def write_fig1_site_summary(count_df: pd.DataFrame) -> pd.DataFrame:
    site = count_df.groupby("SiteName", dropna=False).agg(
        Basin=("Basin", "first"),
        HasEelgrass=("SiteHasEelgrass", "first"),
        EelgrassStatus=("EelgrassStatus", "first"),
        NTransects=("SiteName", "size"),
        TotalSurveyHours=("SurveyHours", "sum"),
        TotalPycno=("Pycnopodia_count", "sum"),
        MeanEncounterRateHr=("EncounterRateHr", "mean"),
        MedianEncounterRateHr=("EncounterRateHr", "median"),
        TransectDetectionFraction=("Detected", "mean"),
    ).reset_index()
    site["TotalEncounterRateHr"] = np.where(site["TotalSurveyHours"] > 0, site["TotalPycno"] / site["TotalSurveyHours"], np.nan)

    if FIG1_PLOTTED_FILE.exists():
        plotted = pd.read_csv(FIG1_PLOTTED_FILE)
        coord_cols = [c for c in ["SiteName", "Lat", "Long", "CoordinateSource", "CoordinateConfidence", "CoordinateNote"] if c in plotted.columns]
        coords = plotted[coord_cols].drop_duplicates("SiteName")
        site = site.merge(coords, on="SiteName", how="left")
    site["HasCoordinatesInFig1QC"] = site[["Lat", "Long"]].notna().all(axis=1) if {"Lat", "Long"}.issubset(site.columns) else False
    site.to_csv(STATS_DIR / "diver_fig1_site_summary.csv", index=False)
    return site


def sparse_flag(rows: int, sites: int, total_count: float) -> bool:
    return bool((sites < 3) or (rows < 20) or (total_count <= 0))


def write_group_summaries(count_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    habitat = count_df.groupby(["HabitatType", "EelgrassStatus"], observed=False).agg(
        rows=("EncounterRateHr", "count"),
        sites=("SiteName", "nunique"),
        total_survey_hours=("SurveyHours", "sum"),
        total_pycno=("Pycnopodia_count", "sum"),
        zero_fraction=("Pycnopodia_count", lambda x: float((x == 0).mean())),
        mean_encounter_rate_hr=("EncounterRateHr", "mean"),
        median_encounter_rate_hr=("EncounterRateHr", "median"),
        sd_encounter_rate_hr=("EncounterRateHr", "std"),
        se_encounter_rate_hr=("EncounterRateHr", se),
        max_encounter_rate_hr=("EncounterRateHr", "max"),
    ).reset_index()
    habitat["effort_standardized_rate_hr"] = np.where(
        habitat["total_survey_hours"] > 0,
        habitat["total_pycno"] / habitat["total_survey_hours"],
        np.nan,
    )
    habitat["sparse_cell_flag"] = [sparse_flag(r, s, c) for r, s, c in zip(habitat.rows, habitat.sites, habitat.total_pycno)]
    habitat["HabitatType"] = pd.Categorical(habitat["HabitatType"], HABITAT_ORDER, ordered=True)
    habitat["EelgrassStatus"] = pd.Categorical(habitat["EelgrassStatus"], EELGRASS_ORDER, ordered=True)
    habitat = habitat.sort_values(["HabitatType", "EelgrassStatus"])
    habitat.to_csv(STATS_DIR / "diver_fig2_habitat_eelgrass_summary.csv", index=False)

    basin = count_df.groupby(["Basin", "EelgrassStatus"], observed=False).agg(
        rows=("EncounterRateHr", "count"),
        sites=("SiteName", "nunique"),
        total_survey_hours=("SurveyHours", "sum"),
        total_pycno=("Pycnopodia_count", "sum"),
        zero_fraction=("Pycnopodia_count", lambda x: float((x == 0).mean())),
        mean_encounter_rate_hr=("EncounterRateHr", "mean"),
        median_encounter_rate_hr=("EncounterRateHr", "median"),
        sd_encounter_rate_hr=("EncounterRateHr", "std"),
        se_encounter_rate_hr=("EncounterRateHr", se),
        max_encounter_rate_hr=("EncounterRateHr", "max"),
    ).reset_index()
    basin["effort_standardized_rate_hr"] = np.where(basin["total_survey_hours"] > 0, basin["total_pycno"] / basin["total_survey_hours"], np.nan)
    basin["sparse_cell_flag"] = [sparse_flag(r, s, c) for r, s, c in zip(basin.rows, basin.sites, basin.total_pycno)]
    basin["Basin"] = pd.Categorical(basin["Basin"], BASIN_ORDER, ordered=True)
    basin["EelgrassStatus"] = pd.Categorical(basin["EelgrassStatus"], EELGRASS_ORDER, ordered=True)
    basin = basin.sort_values(["Basin", "EelgrassStatus"])
    basin.to_csv(STATS_DIR / "diver_fig3_basin_eelgrass_summary.csv", index=False)
    return habitat, basin


def summarize_lengths(length_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = length_df.groupby("EelgrassStatus", observed=False).agg(
        individuals=("Length_cm", "count"),
        sites=("SiteName", "nunique"),
        mean_length_cm=("Length_cm", "mean"),
        sd_length_cm=("Length_cm", "std"),
        median_length_cm=("Length_cm", "median"),
        q10_length_cm=("Length_cm", lambda x: q(x, 0.10)),
        q25_length_cm=("Length_cm", lambda x: q(x, 0.25)),
        q75_length_cm=("Length_cm", lambda x: q(x, 0.75)),
        q90_length_cm=("Length_cm", lambda x: q(x, 0.90)),
        min_length_cm=("Length_cm", "min"),
        max_length_cm=("Length_cm", "max"),
    ).reset_index()
    summary["iqr_length_cm"] = summary["q75_length_cm"] - summary["q25_length_cm"]

    site_lengths = length_df.groupby(["SiteName", "EelgrassStatus"], observed=False).agg(
        individuals=("Length_cm", "count"),
        site_mean_length_cm=("Length_cm", "mean"),
        site_median_length_cm=("Length_cm", "median"),
    ).reset_index()
    site_lengths.to_csv(STATS_DIR / "diver_fig2_length_site_summaries.csv", index=False)

    bootstrap = length_bootstrap(site_lengths, length_df)
    summary.to_csv(STATS_DIR / "diver_fig2_length_summary.csv", index=False)
    bootstrap.to_csv(STATS_DIR / "diver_length_bootstrap.csv", index=False)
    return summary, bootstrap


def length_bootstrap(site_lengths: pd.DataFrame, length_df: pd.DataFrame, n_boot: int = 20000) -> pd.DataFrame:
    statuses = set(site_lengths["EelgrassStatus"].dropna())
    if not {"Eelgrass at site", "No eelgrass at site"}.issubset(statuses):
        return pd.DataFrame()

    rows = []
    for metric in ["site_mean_length_cm", "site_median_length_cm"]:
        e = site_lengths.loc[site_lengths["EelgrassStatus"] == "Eelgrass at site", metric].dropna().to_numpy()
        n = site_lengths.loc[site_lengths["EelgrassStatus"] == "No eelgrass at site", metric].dropna().to_numpy()
        observed = float(e.mean() - n.mean())
        boots = np.empty(n_boot)
        for i in range(n_boot):
            boots[i] = RNG.choice(e, size=len(e), replace=True).mean() - RNG.choice(n, size=len(n), replace=True).mean()
        lo, hi = np.quantile(boots, [0.025, 0.975])
        p_two = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
        rows.append({
            "comparison": "Eelgrass at site - No eelgrass at site",
            "metric": metric,
            "method": f"site-cluster bootstrap ({n_boot} resamples)",
            "effect_cm": observed,
            "ci_low_cm": lo,
            "ci_high_cm": hi,
            "bootstrap_p_two_sided_about_zero": min(p_two, 1.0),
            "eelgrass_sites": len(e),
            "no_eelgrass_sites": len(n),
            "caveat": "Site-level length summaries; conditional on observed/measured individuals.",
        })

    # Individual-level tests for sensitivity only.
    e_ind = length_df.loc[length_df["EelgrassStatus"] == "Eelgrass at site", "Length_cm"].dropna()
    n_ind = length_df.loc[length_df["EelgrassStatus"] == "No eelgrass at site", "Length_cm"].dropna()
    if len(e_ind) > 0 and len(n_ind) > 0:
        u, p_u = mannwhitneyu(e_ind, n_ind, alternative="two-sided")
        ks, p_ks = ks_2samp(e_ind, n_ind, alternative="two-sided")
        rows.extend([
            {
                "comparison": "Eelgrass at site - No eelgrass at site",
                "metric": "individual_length_cm",
                "method": "Mann-Whitney U sensitivity; individuals not independent",
                "effect_cm": float(e_ind.median() - n_ind.median()),
                "ci_low_cm": np.nan,
                "ci_high_cm": np.nan,
                "bootstrap_p_two_sided_about_zero": p_u,
                "eelgrass_sites": site_lengths.loc[site_lengths["EelgrassStatus"] == "Eelgrass at site", "SiteName"].nunique(),
                "no_eelgrass_sites": site_lengths.loc[site_lengths["EelgrassStatus"] == "No eelgrass at site", "SiteName"].nunique(),
                "caveat": f"Sensitivity only; U={u:.1f}; treats individuals as independent.",
            },
            {
                "comparison": "Eelgrass at site - No eelgrass at site",
                "metric": "individual_length_distribution",
                "method": "Kolmogorov-Smirnov sensitivity; individuals not independent",
                "effect_cm": ks,
                "ci_low_cm": np.nan,
                "ci_high_cm": np.nan,
                "bootstrap_p_two_sided_about_zero": p_ks,
                "eelgrass_sites": site_lengths.loc[site_lengths["EelgrassStatus"] == "Eelgrass at site", "SiteName"].nunique(),
                "no_eelgrass_sites": site_lengths.loc[site_lengths["EelgrassStatus"] == "No eelgrass at site", "SiteName"].nunique(),
                "caveat": "Sensitivity only; treats individuals as independent.",
            },
        ])
    return pd.DataFrame(rows)


def fit_count_models(count_df: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ModelSpec(
            "M1_eelgrass_only",
            "Pycnopodia_count ~ SiteHasEelgrassInt",
            "NegativeBinomial(alpha=1)",
            "Overall eelgrass-at-site association, effort adjusted.",
        ),
        ModelSpec(
            "M2_eelgrass_plus_habitat",
            'Pycnopodia_count ~ SiteHasEelgrassInt + C(HabitatType, Treatment(reference="Natural Reef"))',
            "NegativeBinomial(alpha=1)",
            "Fig 2A model: eelgrass-at-site association adjusted for row-level habitat.",
        ),
        ModelSpec(
            "M3_eelgrass_plus_basin",
            'Pycnopodia_count ~ SiteHasEelgrassInt + C(Basin, Treatment(reference="Central"))',
            "NegativeBinomial(alpha=1)",
            "Fig 3 model: eelgrass-at-site association adjusted for basin.",
        ),
        ModelSpec(
            "M4_eelgrass_plus_basin_plus_habitat",
            'Pycnopodia_count ~ SiteHasEelgrassInt + C(Basin, Treatment(reference="Central")) + C(HabitatType, Treatment(reference="Natural Reef"))',
            "NegativeBinomial(alpha=1)",
            "Combined model adjusted for basin and row-level habitat.",
        ),
    ]
    rows = []
    for spec in specs:
        rows.extend(_fit_one_count_model(count_df, spec, sm.families.NegativeBinomial(alpha=1.0)))
        poisson_spec = ModelSpec(spec.name.replace("M", "P", 1) + "_poisson_sensitivity", spec.formula, "Poisson", "Poisson sensitivity: " + spec.note)
        rows.extend(_fit_one_count_model(count_df, poisson_spec, sm.families.Poisson()))
    results = pd.DataFrame(rows)
    results.to_csv(STATS_DIR / "diver_count_model_results.csv", index=False)
    return results


def _fit_one_count_model(count_df: pd.DataFrame, spec: ModelSpec, family) -> list[dict]:
    model_df = count_df.dropna(subset=["Pycnopodia_count", "SurveyHours", "SiteName", "Basin", "HabitatType"]).copy()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = smf.glm(
                formula=spec.formula,
                data=model_df,
                family=family,
                offset=model_df["LogSurveyHours"],
            ).fit(cov_type="cluster", cov_kwds={"groups": model_df["SiteName"]})
        warning_text = "; ".join(str(w.message) for w in caught[:5])
    except Exception as exc:
        return [{
            "model_name": spec.name,
            "formula": spec.formula,
            "family": spec.family,
            "offset": "log(Survey.Time / 60)",
            "covariance": "cluster-robust by SiteName",
            "term": "MODEL_FAILED",
            "estimate_log": np.nan,
            "se_log": np.nan,
            "z": np.nan,
            "p_value": np.nan,
            "irr": np.nan,
            "irr_ci_low": np.nan,
            "irr_ci_high": np.nan,
            "n_rows": len(model_df),
            "n_sites": model_df["SiteName"].nunique(),
            "converged": False,
            "model_aic": np.nan,
            "notes": f"{spec.note} ERROR: {exc}",
        }]

    conf = fit.conf_int()
    out = []
    for term in fit.params.index:
        est = float(fit.params[term])
        lo = float(conf.loc[term, 0])
        hi = float(conf.loc[term, 1])
        out.append({
            "model_name": spec.name,
            "formula": spec.formula,
            "family": spec.family,
            "offset": "log(Survey.Time / 60)",
            "covariance": "cluster-robust by SiteName",
            "term": term,
            "estimate_log": est,
            "se_log": float(fit.bse[term]),
            "z": float(fit.tvalues[term]),
            "p_value": float(fit.pvalues[term]),
            "irr": float(np.exp(est)),
            "irr_ci_low": float(np.exp(lo)),
            "irr_ci_high": float(np.exp(hi)),
            "n_rows": int(fit.nobs),
            "n_sites": model_df["SiteName"].nunique(),
            "converged": bool(getattr(fit, "converged", True)),
            "model_aic": float(fit.aic),
            "notes": (spec.note + (f" Warnings: {warning_text}" if warning_text else "")),
        })
    return out


def fmt_p(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def first_term(results: pd.DataFrame, model_name: str, term: str) -> pd.Series | None:
    rows = results[(results["model_name"] == model_name) & (results["term"] == term)]
    if rows.empty:
        return None
    return rows.iloc[0]


def write_report(
    audit: pd.DataFrame,
    site: pd.DataFrame,
    habitat: pd.DataFrame,
    basin: pd.DataFrame,
    length_summary: pd.DataFrame,
    length_boot: pd.DataFrame,
    model_results: pd.DataFrame,
) -> Path:
    count_audit = audit[audit["table"] == "count_rows"].iloc[0]
    len_audit = audit[audit["table"] == "positive_length_rows"].iloc[0]
    m4 = first_term(model_results, "M4_eelgrass_plus_basin_plus_habitat", "SiteHasEelgrassInt")
    m2 = first_term(model_results, "M2_eelgrass_plus_habitat", "SiteHasEelgrassInt")
    m3 = first_term(model_results, "M3_eelgrass_plus_basin", "SiteHasEelgrassInt")
    p4 = first_term(model_results, "P4_eelgrass_plus_basin_plus_habitat_poisson_sensitivity", "SiteHasEelgrassInt")

    mean_boot = length_boot[length_boot["metric"] == "site_mean_length_cm"].iloc[0]
    med_boot = length_boot[length_boot["metric"] == "site_median_length_cm"].iloc[0]

    site_e = site[site["HasEelgrass"]]
    site_n = site[~site["HasEelgrass"]]
    sparse_h = int(habitat["sparse_cell_flag"].sum())
    sparse_b = int(basin["sparse_cell_flag"].sum())

    report = f"""# Diver-only statistics report for StarMeadow Figs 1-3

Generated by `code/diver_only_stats_for_report.py`.

## Input audit

The count analysis used {int(count_audit.rows):,} survey rows from {int(count_audit.sites)} sites across {int(count_audit.basins)} basins and {int(count_audit.habitats)} habitat categories. Total diver effort was {count_audit.total_survey_hours:.1f} hours and total Pycnopodia count was {int(count_audit.total_pycno_count):,}. Counts were zero-heavy: {count_audit.zero_fraction * 100:.1f}% of survey rows had zero Pycnopodia, the median row-level encounter rate was {count_audit.encounter_rate_median:.2f} individuals hr⁻¹, and the maximum row-level encounter rate was {count_audit.encounter_rate_max:.1f} individuals hr⁻¹.

The length analysis used {int(len_audit.rows):,} positive individual length measurements from {int(len_audit.sites)} sites. These lengths are conditional on Pycnopodia being observed and measured.

## Fig 1 descriptive statistics

Fig 1 should be interpreted descriptively. The site summary contains {len(site)} survey sites. Eelgrass was recorded at {len(site_e)} sites and not recorded at {len(site_n)} sites. Site-level total-effort encounter rates ranged from {site.TotalEncounterRateHr.min():.2f} to {site.TotalEncounterRateHr.max():.2f} individuals hr⁻¹, with a median of {site.TotalEncounterRateHr.median():.2f} individuals hr⁻¹. The site-level summary table is `diver_fig1_site_summary.csv`.

## Count model methods

For formal inference, Pycnopodia counts were modeled with survey effort as an offset rather than testing raw encounter rates. Primary models used negative-binomial GLMs with `log(Survey.Time / 60)` as the effort offset and SiteName-clustered standard errors to account for repeated surveys within sites. Poisson GLMs with the same formulas, offset, and SiteName-clustered standard errors were fitted as sensitivity analyses.

Prespecified negative-binomial models were:

- M1: `count ~ SiteHasEelgrass + offset(log effort)`
- M2: `count ~ SiteHasEelgrass + HabitatType + offset(log effort)`
- M3: `count ~ SiteHasEelgrass + Basin + offset(log effort)`
- M4: `count ~ SiteHasEelgrass + Basin + HabitatType + offset(log effort)`

The script used a fixed negative-binomial alpha of 1.0 for the GLM family. Coefficients are reported as incidence-rate ratios (IRRs). Effects are observational associations, not causal estimates.

## Count model results

The combined basin- and habitat-adjusted negative-binomial model estimated that eelgrass-at-site surveys had an IRR of {m4.irr:.2f} relative to no-eelgrass-at-site surveys (95% CI {m4.irr_ci_low:.2f}-{m4.irr_ci_high:.2f}, {fmt_p(m4.p_value)}). The habitat-adjusted model without basin estimated IRR = {m2.irr:.2f} (95% CI {m2.irr_ci_low:.2f}-{m2.irr_ci_high:.2f}, {fmt_p(m2.p_value)}), and the basin-adjusted model without habitat estimated IRR = {m3.irr:.2f} (95% CI {m3.irr_ci_low:.2f}-{m3.irr_ci_high:.2f}, {fmt_p(m3.p_value)}). The Poisson sensitivity version of the combined model estimated IRR = {p4.irr:.2f} (95% CI {p4.irr_ci_low:.2f}-{p4.irr_ci_high:.2f}, {fmt_p(p4.p_value)}).

Habitat/eelgrass and basin/eelgrass descriptive summaries were written with sparse-cell flags. {sparse_h} habitat-by-eelgrass cells and {sparse_b} basin-by-eelgrass cells were flagged because they had fewer than 3 sites, fewer than 20 rows, or zero total Pycnopodia. These cells can be shown descriptively but should not support strong cell-specific inference.

## Length methods and results

Length comparisons used site-level summaries and a site-cluster bootstrap. For each bootstrap replicate, sites were resampled with replacement within eelgrass-at-site and no-eelgrass-at-site groups, and the difference between groups was recomputed. This avoids treating multiple individuals from the same site as independent site-level replicates.

Using site-level mean length, eelgrass-at-site sites differed from no-eelgrass-at-site sites by {mean_boot.effect_cm:.2f} cm (95% bootstrap CI {mean_boot.ci_low_cm:.2f} to {mean_boot.ci_high_cm:.2f}; bootstrap two-sided p about zero = {mean_boot.bootstrap_p_two_sided_about_zero:.3f}). Using site-level median length, the difference was {med_boot.effect_cm:.2f} cm (95% bootstrap CI {med_boot.ci_low_cm:.2f} to {med_boot.ci_high_cm:.2f}; bootstrap two-sided p about zero = {med_boot.bootstrap_p_two_sided_about_zero:.3f}).

Because the length data include only observed/measured individuals, these results describe size patterns conditional on detection and measurement rather than population size structure across all surveyed sites.

## Output files

- `diver_stats_input_audit.csv`
- `diver_fig1_site_summary.csv`
- `diver_fig2_habitat_eelgrass_summary.csv`
- `diver_fig2_length_summary.csv`
- `diver_fig2_length_site_summaries.csv`
- `diver_fig3_basin_eelgrass_summary.csv`
- `diver_count_model_results.csv`
- `diver_length_bootstrap.csv`
- `diver_stats_report.md`

## Manuscript-safe interpretation

The diver-only statistics support an association between site-level eelgrass status and higher Pycnopodia encounter rates after adjusting for survey effort and broad habitat/basin structure. However, the data are observational, repeated within sites, geographically uneven, and sparse in some basin/habitat cells. The length comparison is conditional on detections and should be presented as a detected-individual size pattern, not a full population size distribution.
"""
    out = STATS_DIR / "diver_stats_report.md"
    out.write_text(report, encoding="utf-8")
    return out


def main() -> None:
    count_df = load_count_data()
    length_df = load_length_data()

    audit = write_input_audit(count_df, length_df)
    site = write_fig1_site_summary(count_df)
    habitat, basin = write_group_summaries(count_df)
    length_summary, length_boot = summarize_lengths(length_df)
    model_results = fit_count_models(count_df)
    report_path = write_report(audit, site, habitat, basin, length_summary, length_boot, model_results)

    manifest = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "output_dir": str(STATS_DIR.relative_to(PROJECT_ROOT)),
        "outputs": sorted(p.name for p in STATS_DIR.glob("diver_*.csv")) + [report_path.name],
    }
    (STATS_DIR / "diver_stats_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote diver-only stats outputs to: {STATS_DIR}")
    print(f"Report: {report_path}")
    print(f"Count rows: {len(count_df)}; sites: {count_df['SiteName'].nunique()}")
    print(f"Length rows: {len(length_df)}; sites: {length_df['SiteName'].nunique()}")


if __name__ == "__main__":
    main()
