#!/usr/bin/env python3
"""Distribution-shape statistics for StarMeadow Pycnopodia length figures.

This analysis is intentionally focused on distribution shape rather than mean or
median length. It complements Fig. 2B and S2_Fig KDE curves by reporting
site-aware two-sample shape distances for site-level eelgrass contrasts and
cluster-bootstrap uncertainty for S2 habitat-curve contrasts.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import energy_distance, ks_2samp
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "outputs" / "publication_figures" / "stats"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LENGTH_FILE = DATA_DIR / "PycnoLengthCLean_12_31_2025.csv"

HABITAT_ORDER = ["Soft Bottom", "Eelgrass", "Artificial Reef", "Kelp Forest", "Natural Reef", "Sponge Garden"]
STATUS_ORDER = ["Eelgrass at site", "No eelgrass at site"]
RNG = np.random.default_rng(20260506)
N_PERM = 10000
N_BOOT = 10000


def bool_status(value: object) -> str:
    return "Eelgrass at site" if bool(value) else "No eelgrass at site"


def load_length_data() -> pd.DataFrame:
    df = pd.read_csv(LENGTH_FILE)
    df = df[df["Basin"] != "Queen Charlotte Strait"].copy()
    df["Length_cm"] = pd.to_numeric(df["Length(cm)"].astype(str).str.strip(), errors="coerce")
    df["SiteHasEelgrass"] = df.groupby("SiteName")["HabitatType"].transform(lambda x: (x == "Eelgrass").any())
    df["EelgrassStatus"] = df["SiteHasEelgrass"].map(bool_status)
    df = df[df["Length_cm"] > 0].dropna(subset=["SiteName", "HabitatType", "EelgrassStatus", "Length_cm"]).copy()
    df = df[df["HabitatType"].isin(HABITAT_ORDER)].copy()
    df["HabitatType"] = pd.Categorical(df["HabitatType"], HABITAT_ORDER, ordered=True)
    df["EelgrassStatus"] = pd.Categorical(df["EelgrassStatus"], STATUS_ORDER, ordered=True)
    return df


def two_sample_shape_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return KS D and energy distance for two length samples."""
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    ks_d = float(ks_2samp(a, b, alternative="two-sided").statistic)
    e = float(energy_distance(a, b))
    return ks_d, e


def site_status_permutation_test(df: pd.DataFrame, label: str, n_perm: int = N_PERM) -> dict:
    """Permutation test for site-level eelgrass vs no-eelgrass distribution shape.

    The grouping variable is site-level eelgrass status. The permutation shuffles
    status labels among sites while preserving the number of eelgrass/no-eelgrass
    sites and all within-site length observations.
    """
    site_status = df[["SiteName", "EelgrassStatus"]].drop_duplicates("SiteName").reset_index(drop=True)
    if len(site_status["EelgrassStatus"].unique()) < 2:
        return {}
    site_to_status = dict(zip(site_status["SiteName"], site_status["EelgrassStatus"].astype(str)))
    observed_status = df["SiteName"].map(site_to_status).to_numpy()
    lengths = df["Length_cm"].to_numpy(float)
    e = lengths[observed_status == "Eelgrass at site"]
    n = lengths[observed_status == "No eelgrass at site"]
    ks_obs, energy_obs = two_sample_shape_stats(e, n)

    sites = site_status["SiteName"].to_numpy()
    statuses = site_status["EelgrassStatus"].astype(str).to_numpy()
    site_index = {site: i for i, site in enumerate(sites)}
    row_site_idx = df["SiteName"].map(site_index).to_numpy()
    perm_ks = np.empty(n_perm)
    perm_energy = np.empty(n_perm)
    for i in range(n_perm):
        shuffled = RNG.permutation(statuses)
        row_status = shuffled[row_site_idx]
        pe = lengths[row_status == "Eelgrass at site"]
        pn = lengths[row_status == "No eelgrass at site"]
        perm_ks[i], perm_energy[i] = two_sample_shape_stats(pe, pn)

    return {
        "comparison_family": "site_eelgrass_status",
        "comparison": label,
        "group_a": "Eelgrass at site",
        "group_b": "No eelgrass at site",
        "n_a_individuals": len(e),
        "n_b_individuals": len(n),
        "n_a_sites": int((site_status["EelgrassStatus"].astype(str) == "Eelgrass at site").sum()),
        "n_b_sites": int((site_status["EelgrassStatus"].astype(str) == "No eelgrass at site").sum()),
        "ks_d": ks_obs,
        "ks_site_permutation_p": float((1 + (perm_ks >= ks_obs).sum()) / (n_perm + 1)),
        "energy_distance": energy_obs,
        "energy_site_permutation_p": float((1 + (perm_energy >= energy_obs).sum()) / (n_perm + 1)),
        "n_permutations": n_perm,
        "sparse_flag": bool(min(len(e), len(n)) < 20 or min(
            int((site_status["EelgrassStatus"].astype(str) == "Eelgrass at site").sum()),
            int((site_status["EelgrassStatus"].astype(str) == "No eelgrass at site").sum()),
        ) < 3),
        "method_note": "Site-label permutation preserves within-site length clusters and tests distribution-shape separation by site-level eelgrass status.",
    }


def site_cluster_bootstrap_stat(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    stat_name: str,
    n_boot: int = N_BOOT,
) -> tuple[float, float, float]:
    """Bootstrap a two-sample distribution-shape statistic by resampling sites."""
    subset = df[df[group_col].astype(str).isin([group_a, group_b])].copy()
    a_df = subset[subset[group_col].astype(str) == group_a]
    b_df = subset[subset[group_col].astype(str) == group_b]
    a_sites = a_df["SiteName"].drop_duplicates().to_numpy()
    b_sites = b_df["SiteName"].drop_duplicates().to_numpy()
    a_by_site = {s: a_df.loc[a_df["SiteName"] == s, "Length_cm"].to_numpy(float) for s in a_sites}
    b_by_site = {s: b_df.loc[b_df["SiteName"] == s, "Length_cm"].to_numpy(float) for s in b_sites}
    obs_ks, obs_energy = two_sample_shape_stats(a_df["Length_cm"].to_numpy(float), b_df["Length_cm"].to_numpy(float))
    observed = obs_ks if stat_name == "ks_d" else obs_energy
    boots = np.empty(n_boot)
    for i in range(n_boot):
        sampled_a = RNG.choice(a_sites, size=len(a_sites), replace=True)
        sampled_b = RNG.choice(b_sites, size=len(b_sites), replace=True)
        a_vals = np.concatenate([a_by_site[s] for s in sampled_a])
        b_vals = np.concatenate([b_by_site[s] for s in sampled_b])
        ks, ed = two_sample_shape_stats(a_vals, b_vals)
        boots[i] = ks if stat_name == "ks_d" else ed
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(observed), float(lo), float(hi)


def within_status_habitat_shape(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for status in STATUS_ORDER:
        status_df = df[df["EelgrassStatus"].astype(str) == status].copy()
        present = [h for h in HABITAT_ORDER if (status_df["HabitatType"].astype(str) == h).any()]
        for a, b in combinations(present, 2):
            pair = status_df[status_df["HabitatType"].astype(str).isin([a, b])].copy()
            a_vals = pair.loc[pair["HabitatType"].astype(str) == a, "Length_cm"].to_numpy(float)
            b_vals = pair.loc[pair["HabitatType"].astype(str) == b, "Length_cm"].to_numpy(float)
            a_sites = pair.loc[pair["HabitatType"].astype(str) == a, "SiteName"].nunique()
            b_sites = pair.loc[pair["HabitatType"].astype(str) == b, "SiteName"].nunique()
            if len(a_vals) < 2 or len(b_vals) < 2:
                continue
            ks_d, ks_lo, ks_hi = site_cluster_bootstrap_stat(pair, "HabitatType", a, b, "ks_d")
            ed, ed_lo, ed_hi = site_cluster_bootstrap_stat(pair, "HabitatType", a, b, "energy_distance")
            ks_ind = ks_2samp(a_vals, b_vals, alternative="two-sided")
            rows.append({
                "comparison_family": "within_s2_panel_habitat_pair",
                "eelgrass_status_panel": status,
                "group_a": a,
                "group_b": b,
                "n_a_individuals": len(a_vals),
                "n_b_individuals": len(b_vals),
                "n_a_sites": a_sites,
                "n_b_sites": b_sites,
                "ks_d": ks_d,
                "ks_cluster_bootstrap_ci_low": ks_lo,
                "ks_cluster_bootstrap_ci_high": ks_hi,
                "energy_distance": ed,
                "energy_cluster_bootstrap_ci_low": ed_lo,
                "energy_cluster_bootstrap_ci_high": ed_hi,
                "individual_ks_sensitivity_p": float(ks_ind.pvalue),
                "sparse_flag": bool(min(len(a_vals), len(b_vals)) < 20 or min(a_sites, b_sites) < 3),
                "method_note": "S2 within-panel shape contrast. Statistic uses individual lengths; CI resamples site clusters within each habitat curve. Individual KS p-value is sensitivity only because individuals are clustered within sites.",
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["individual_ks_sensitivity_fdr_q"] = multipletests(out["individual_ks_sensitivity_p"], method="fdr_bh")[1]
        out = out.sort_values(["sparse_flag", "eelgrass_status_panel", "ks_d", "energy_distance"], ascending=[True, True, False, False])
    return out


def site_status_shape_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(site_status_permutation_test(df, "Overall positive length distribution"))
    for habitat in HABITAT_ORDER:
        subset = df[df["HabitatType"].astype(str) == habitat].copy()
        if subset["EelgrassStatus"].astype(str).nunique() < 2:
            continue
        result = site_status_permutation_test(subset, f"{habitat} only")
        if result:
            result["survey_level_habitat"] = habitat
            rows.append(result)
    out = pd.DataFrame([r for r in rows if r])
    if not out.empty:
        out["ks_site_permutation_fdr_q"] = multipletests(out["ks_site_permutation_p"], method="fdr_bh")[1]
        out["energy_site_permutation_fdr_q"] = multipletests(out["energy_site_permutation_p"], method="fdr_bh")[1]
        out = out.sort_values(["sparse_flag", "ks_d", "energy_distance"], ascending=[True, False, False])
    return out


def write_report(df: pd.DataFrame, status_tests: pd.DataFrame, habitat_tests: pd.DataFrame) -> Path:
    lines = []
    lines.append("# Length distribution-shape statistics for StarMeadow Fig. 2B and S2_Fig")
    lines.append("")
    lines.append("Generated by `code/length_distribution_shape_stats_for_report.py`.")
    lines.append("")
    lines.append("## Rationale")
    lines.append("")
    lines.append("This analysis refocuses the length statistics on distribution shape rather than mean or median length. The KDE curves in Fig. 2B and S2_Fig can differ by modality, tail weight, or where probability mass is concentrated even when site-level means or medians are similar. Therefore, the primary shape summaries here are the two-sample Kolmogorov-Smirnov distance and the one-dimensional energy distance.")
    lines.append("")
    lines.append("For site-level eelgrass contrasts, p-values come from a site-label permutation test that preserves all within-site length observations and shuffles eelgrass status among sites. For S2 habitat-curve contrasts within each eelgrass-status panel, the table reports site-cluster bootstrap confidence intervals for the shape distances; individual-level KS p-values are included only as sensitivity values because individual animals are clustered within sites.")
    lines.append("")
    lines.append("## Input audit")
    lines.append("")
    lines.append(f"The analysis used {len(df):,} positive length measurements from {df['SiteName'].nunique()} sites. Eelgrass-at-site rows included {(df['EelgrassStatus'].astype(str) == 'Eelgrass at site').sum():,} individuals from {df.loc[df['EelgrassStatus'].astype(str) == 'Eelgrass at site', 'SiteName'].nunique()} sites; no-eelgrass-at-site rows included {(df['EelgrassStatus'].astype(str) == 'No eelgrass at site').sum():,} individuals from {df.loc[df['EelgrassStatus'].astype(str) == 'No eelgrass at site', 'SiteName'].nunique()} sites.")
    lines.append("")
    lines.append("## Site-level eelgrass distribution-shape contrasts")
    lines.append("")
    if status_tests.empty:
        lines.append("No site-level eelgrass shape tests were available.")
    else:
        for _, row in status_tests.iterrows():
            sparse = "; sparse-cell caveat" if bool(row["sparse_flag"]) else ""
            label = row["comparison"]
            lines.append(f"- {label}: KS D = {row['ks_d']:.3f}, site-permutation p = {row['ks_site_permutation_p']:.4f}, FDR q = {row['ks_site_permutation_fdr_q']:.4f}; energy distance = {row['energy_distance']:.3f}, site-permutation p = {row['energy_site_permutation_p']:.4f}, FDR q = {row['energy_site_permutation_fdr_q']:.4f} ({int(row['n_a_individuals'])} vs {int(row['n_b_individuals'])} individuals; {int(row['n_a_sites'])} vs {int(row['n_b_sites'])} sites{sparse}).")
    lines.append("")
    lines.append("## S2 within-panel habitat distribution-shape contrasts")
    lines.append("")
    if habitat_tests.empty:
        lines.append("No S2 habitat-pair shape contrasts were available.")
    else:
        nonsparse = habitat_tests[~habitat_tests["sparse_flag"]].copy()
        sparse = habitat_tests[habitat_tests["sparse_flag"]].copy()
        lines.append("Non-sparse habitat-pair contrasts:")
        if nonsparse.empty:
            lines.append("- None met the non-sparse threshold of at least 20 individuals and at least 3 sites in both curves.")
        else:
            for _, row in nonsparse.head(12).iterrows():
                lines.append(f"- {row['eelgrass_status_panel']}: {row['group_a']} vs {row['group_b']}: KS D = {row['ks_d']:.3f} (site-bootstrap 95% CI {row['ks_cluster_bootstrap_ci_low']:.3f}–{row['ks_cluster_bootstrap_ci_high']:.3f}); energy distance = {row['energy_distance']:.3f} (95% CI {row['energy_cluster_bootstrap_ci_low']:.3f}–{row['energy_cluster_bootstrap_ci_high']:.3f}); individual-KS sensitivity q = {row['individual_ks_sensitivity_fdr_q']:.4f}.")
        lines.append("")
        lines.append("Sparse but visually useful habitat-pair contrasts:")
        if sparse.empty:
            lines.append("- None.")
        else:
            for _, row in sparse.head(12).iterrows():
                lines.append(f"- {row['eelgrass_status_panel']}: {row['group_a']} vs {row['group_b']}: KS D = {row['ks_d']:.3f}; energy distance = {row['energy_distance']:.3f}; n = {int(row['n_a_individuals'])} vs {int(row['n_b_individuals'])} individuals and {int(row['n_a_sites'])} vs {int(row['n_b_sites'])} sites. Interpret cautiously because at least one curve is sparse.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The strongest shape signals should be interpreted as distributional differences among detected and measured individuals, not as complete population size-structure differences. The site-aware eelgrass-status permutation tests provide the cleanest inferential checks. The S2 habitat-pair results are most useful for prioritizing visually apparent curve differences and for describing where probability mass differs across curves; sparse curves should remain descriptive.")
    path = OUT_DIR / "diver_length_distribution_shape_report.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    df = load_length_data()
    status_tests = site_status_shape_tests(df)
    habitat_tests = within_status_habitat_shape(df)

    status_path = OUT_DIR / "diver_length_site_eelgrass_distribution_shape_tests.csv"
    habitat_path = OUT_DIR / "diver_s2_habitat_distribution_shape_tests.csv"
    status_tests.to_csv(status_path, index=False)
    habitat_tests.to_csv(habitat_path, index=False)
    report_path = write_report(df, status_tests, habitat_tests)

    manifest = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "input": str(LENGTH_FILE.relative_to(PROJECT_ROOT)),
        "outputs": [
            str(status_path.relative_to(PROJECT_ROOT)),
            str(habitat_path.relative_to(PROJECT_ROOT)),
            str(report_path.relative_to(PROJECT_ROOT)),
        ],
        "positive_length_rows": int(len(df)),
        "sites": int(df["SiteName"].nunique()),
        "n_permutations": N_PERM,
        "n_bootstrap": N_BOOT,
        "rng_seed": 20260506,
    }
    manifest_path = OUT_DIR / "diver_length_distribution_shape_manifest.json"
    manifest_path.write_text(pd.Series(manifest).to_json(indent=2))

    print(f"Wrote site eelgrass shape tests: {status_path}")
    print(f"Wrote S2 habitat shape tests: {habitat_path}")
    print(f"Wrote report: {report_path}")
    print(f"Length rows: {len(df)}; sites: {df['SiteName'].nunique()}")


if __name__ == "__main__":
    main()
