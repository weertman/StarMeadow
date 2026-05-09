"""
Figure 4 draft: ShoreZone feature prediction loss and refugia/eelgrass encounter rates.

Panel A: random-forest permutation prediction loss by dropped ShoreZone feature.
Panel B: mean site-level Pycnopodia encounter rate by ShoreZone protected-exposure
         status, dodged by continuous ShoreZone eelgrass status.
"""

from pathlib import Path
import re
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import figure_1_site_map as fig1  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "publication_figures"
SOURCE_DIR = OUTPUT_DIR / "sources"
SUBMISSION_DIR = OUTPUT_DIR / "submission"
QC_DIR = OUTPUT_DIR / "qc"

SHOREZONE_DIR = PROJECT_ROOT / "outputs" / "15_shorezone_site_analysis"
RECOVERY_DIR = PROJECT_ROOT / "outputs" / "16_shorezone_recovery_analysis"

FEATURE_IMPORTANCE_FILE = RECOVERY_DIR / "feature_importance_permutation.csv"
SITE_SUMMARY_FILE = SHOREZONE_DIR / "site_shorezone_pycno_summary.csv"
ZOS_PROPORTIONS_FILE = SHOREZONE_DIR / "shorezone_ZOS_UNIT_proportions.csv"
EXP_PROPORTIONS_FILE = SHOREZONE_DIR / "shorezone_EXP_CLASS_proportions.csv"

STATUS_COLORS = {
    "Eelgrass: continuous": "#28C340",
    "Other": "#A12CB3",
}
EELGRASS_ORDER = ["Eelgrass: continuous", "Other"]
REFUGIA_ORDER = ["Protected / very protected", "Other exposure"]


def setup_style():
    mpl.rcParams.update({
        "font.family": "Liberation Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    sns.set_theme(style="white", rc={
        "font.family": "Liberation Sans",
        "axes.edgecolor": "#666666",
        "axes.linewidth": 0.8,
    })


BC_CLASS_LABELS = {
    "30": "Sand beach",
}

HAB_CALC_LABELS = {
    # Washington ShoreZone Data Dictionary, Appendix A, Habitat and Bio-Exposure
    # Classification: codes 6-10 are mobile/partially mobile substrates.
    "7": "Low-exposure sand/gravel habitat",
    "8": "Estuarine sand/mud habitat",
}

UNIT_FIELD_LABELS = {
    "ZOS_UNIT": "Eelgrass",
    "OYS_UNIT": "Oyster beds",
    "FUC_UNIT": "Fucus band",
    "ULV_UNIT": "Ulva green algae",
    "MUS_UNIT": "Mussel-barnacle band",
    "NER_UNIT": "Bull kelp",
}

ABUNDANCE_LABELS = {
    "C": "continuous",
    "P": "patchy",
    "": "absent",
}

EXP_CLASS_LABELS = {
    "E": "exposed",
    "P": "protected",
    "SE": "semi-exposed",
    "SP": "semi-protected",
    "VE": "very exposed",
    "VP": "very protected",
    "X": "undetermined exposure",
}

SED_SOURCE_LABELS = {
    "A": "alongshore sediment source",
    "B": "backshore sediment source",
    "F": "fluvial sediment source",
    "O": "offshore sediment source",
}

SED_ABUND_LABELS = {
    "A": "abundant sediment",
    "M": "moderate sediment",
    "S": "scarce sediment",
}

ZONECOMP_LABELS = {
    "A1 B1 B2 C1": "Supratidal + two intertidal + subtidal components",
}


def _clean_code(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"\.0\b", "", value)
    return value


def pretty_feature_name(feature: str) -> str:
    """Convert compact ShoreZone feature codes to data-dictionary labels."""
    text = _clean_code(feature)
    if "_" not in text:
        return text

    for field in sorted(UNIT_FIELD_LABELS, key=len, reverse=True):
        prefix = f"{field}_"
        if text.startswith(prefix):
            code = text[len(prefix):].strip()
            status = ABUNDANCE_LABELS.get(code, code.lower())
            return f"{UNIT_FIELD_LABELS[field]}: {status}"

    if text.startswith("BC_CLASS_"):
        code = _clean_code(text.removeprefix("BC_CLASS_"))
        return f"BC shoreline type: {BC_CLASS_LABELS.get(code, code)}"

    if text.startswith("HAB_CALC_"):
        code = _clean_code(text.removeprefix("HAB_CALC_"))
        return HAB_CALC_LABELS.get(code, f"Habitat/bio-exposure class {code}")

    if text.startswith("EXP_CLASS_"):
        code = text.removeprefix("EXP_CLASS_").strip()
        return f"Wave exposure: {EXP_CLASS_LABELS.get(code, code)}"

    if text.startswith("SED_SOURCE_"):
        code = text.removeprefix("SED_SOURCE_").strip()
        return SED_SOURCE_LABELS.get(code, f"sediment source {code}")

    if text.startswith("SED_ABUND_"):
        code = text.removeprefix("SED_ABUND_").strip()
        return SED_ABUND_LABELS.get(code, f"sediment abundance {code}")

    if text.startswith("ZONECOMP_"):
        code = text.removeprefix("ZONECOMP_").strip()
        return ZONECOMP_LABELS.get(code, f"Across-shore components: {code}")

    return text.replace("_", " ").strip()


def load_panel_a(top_n: int = 15) -> pd.DataFrame:
    df = pd.read_csv(FEATURE_IMPORTANCE_FILE)
    df = df.rename(columns={"Importance_Mean": "PredictionLoss", "Importance_Std": "PredictionLossSE"})
    df = df.sort_values("PredictionLoss", ascending=False).head(top_n).copy()
    df["FeatureLabel"] = df["Feature"].map(pretty_feature_name)
    df.to_csv(QC_DIR / "Fig4_panel_A_prediction_loss_features.csv", index=False)
    return df


def load_panel_b() -> tuple[pd.DataFrame, pd.DataFrame]:
    site = pd.read_csv(SITE_SUMMARY_FILE)
    zos = pd.read_csv(ZOS_PROPORTIONS_FILE)
    exp = pd.read_csv(EXP_PROPORTIONS_FILE)

    df = site.merge(zos[["SiteName", "ZOS_UNIT_C"]], on="SiteName", how="left")
    df = df.merge(
        exp[["SiteName", "EXP_CLASS_P", "EXP_CLASS_VP"]],
        on="SiteName",
        how="left",
    )
    df = df.dropna(subset=["MeanEncounterRate"]).copy()
    df["ProtectedShoreZone"] = df[["EXP_CLASS_P", "EXP_CLASS_VP"]].fillna(False).any(axis=1)
    df["RefugiaStatus"] = np.where(df["ProtectedShoreZone"], "Protected / very protected", "Other exposure")
    df["ContinuousEelgrass"] = df["ZOS_UNIT_C"].fillna(False).astype(bool)
    df["ContinuousEelgrassStatus"] = np.where(
        df["ContinuousEelgrass"],
        "Eelgrass: continuous",
        "Other",
    )
    df["RefugiaStatus"] = pd.Categorical(df["RefugiaStatus"], REFUGIA_ORDER, ordered=True)
    df["ContinuousEelgrassStatus"] = pd.Categorical(df["ContinuousEelgrassStatus"], EELGRASS_ORDER, ordered=True)

    summary = df.groupby(["RefugiaStatus", "ContinuousEelgrassStatus"], observed=False).agg(
        sites=("SiteName", "nunique"),
        mean_encounter_rate=("MeanEncounterRate", "mean"),
        sd_encounter_rate=("MeanEncounterRate", "std"),
        se_encounter_rate=("MeanEncounterRate", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
    ).reset_index()
    summary = summary[summary["sites"] > 0].copy()
    summary = summary.sort_values(["RefugiaStatus", "ContinuousEelgrassStatus"])
    df.to_csv(QC_DIR / "Fig4_panel_B_refugia_continuous_eelgrass_plot_data.csv", index=False)
    summary.to_csv(QC_DIR / "Fig4_panel_B_refugia_continuous_eelgrass_summary.csv", index=False)
    return df, summary


def make_figure(panel_a: pd.DataFrame, panel_b_summary: pd.DataFrame) -> plt.Figure:
    setup_style()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.5, 4.4), constrained_layout=True)

    # Panel A: show highest prediction loss at the top.
    plot_a = panel_a.sort_values("PredictionLoss", ascending=True)
    ax_a.barh(
        plot_a["FeatureLabel"],
        plot_a["PredictionLoss"],
        xerr=plot_a["PredictionLossSE"],
        color="#5B8DB8",
        edgecolor="black",
        linewidth=0.7,
        capsize=2.5,
        zorder=3,
    )
    ax_a.axvline(0, color="#666666", linewidth=0.8, zorder=2)
    ax_a.set_xlabel("Prediction loss")
    ax_a.set_ylabel("Dropped ShoreZone feature")
    ax_a.tick_params(axis="both", which="major", bottom=True, left=True, length=3.5, width=0.8, color="#333333")
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.text(-0.10, 1.02, "A", transform=ax_a.transAxes, fontsize=11, fontweight="bold", va="bottom")

    # Panel B: ShoreZone protected-exposure split, continuous eelgrass dodge.
    x_centers = np.arange(len(REFUGIA_ORDER)) * 1.15
    bar_width = 0.42
    offsets = {EELGRASS_ORDER[0]: -bar_width / 2, EELGRASS_ORDER[1]: bar_width / 2}
    max_annotation_y = 0.0
    for status in EELGRASS_ORDER:
        subset = panel_b_summary[panel_b_summary["ContinuousEelgrassStatus"] == status]
        xs = np.array([x_centers[REFUGIA_ORDER.index(ref)] for ref in subset["RefugiaStatus"].astype(str)]) + offsets[status]
        ax_b.bar(
            xs,
            subset["mean_encounter_rate"],
            width=bar_width,
            color=STATUS_COLORS[status],
            edgecolor="black",
            linewidth=0.8,
            label=status,
            zorder=3,
        )
        ax_b.errorbar(
            xs,
            subset["mean_encounter_rate"],
            yerr=subset["se_encounter_rate"],
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=3,
            capthick=0.8,
            zorder=4,
        )
        for x, (_, row) in zip(xs, subset.iterrows()):
            y = float(row["mean_encounter_rate"] + row["se_encounter_rate"])
            max_annotation_y = max(max_annotation_y, y)
            ax_b.annotate(
                f"({int(row['sites'])})",
                xy=(x, y),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
                zorder=5,
            )

    ax_b.set_ylim(top=max_annotation_y * 1.18 if max_annotation_y > 0 else None)
    ax_b.set_xticks(x_centers, labels=["Protected /\nvery protected", "Other\nexposure"])
    ax_b.set_xlabel("")
    ax_b.set_ylabel("Encounter rate (Pycnopodia hr$^{-1}$)")
    ax_b.tick_params(axis="both", which="major", bottom=True, left=True, length=3.5, width=0.8, color="#333333")
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.legend(loc="upper right", frameon=False, fontsize=7, handlelength=1.0, handletextpad=0.4)
    ax_b.text(-0.10, 1.02, "B", transform=ax_b.transAxes, fontsize=11, fontweight="bold", va="bottom")

    return fig


def main():
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    panel_a = load_panel_a(top_n=15)
    _, panel_b_summary = load_panel_b()
    fig = make_figure(panel_a, panel_b_summary)

    png = SOURCE_DIR / "Fig4_shorezone_draft.png"
    pdf = SOURCE_DIR / "Fig4_shorezone_draft.pdf"
    tif = SUBMISSION_DIR / "Fig4.tif"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig1.save_plos_tiff(fig, tif, dpi=300)
    plt.close(fig)

    print(f"Saved draft PNG: {png}")
    print(f"Saved editable PDF: {pdf}")
    print(f"Saved PLOS-style TIFF: {tif}")
    print(f"Panel A features: {len(panel_a)}")
    print(f"Panel B groups: {len(panel_b_summary)}")


if __name__ == "__main__":
    main()
