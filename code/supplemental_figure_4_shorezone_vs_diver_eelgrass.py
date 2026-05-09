"""
Supplemental Figure 4: comparison of ShoreZone Zostera categorization with
human diver-recorded eelgrass at StarMeadow survey sites.

Human diver category is site-level: any survey row at the site with
HabitatType == "Eelgrass".
ShoreZone category is site-level ZOS_UNIT from nearest ShoreZone match:
continuous, patchy, or absent.
"""
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import figure_1_site_map as fig1  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "publication_figures"
SOURCE_DIR = OUTPUT_DIR / "sources"
SUBMISSION_DIR = OUTPUT_DIR / "submission"
QC_DIR = OUTPUT_DIR / "qc"
SHOREZONE_DIR = PROJECT_ROOT / "outputs" / "15_shorezone_site_analysis"

COUNT_FILE = DATA_DIR / "PycnoCountCLean_12_31_2025.csv"
ZOS_FILE = SHOREZONE_DIR / "shorezone_ZOS_UNIT_proportions.csv"

DIVER_COLORS = {
    "Eelgrass recorded by divers": "#28C340",
    "No eelgrass recorded by divers": "#A12CB3",
}
SHOREZONE_ORDER = ["Eelgrass: continuous", "Eelgrass: patchy", "Eelgrass: absent"]
DIVER_ORDER = ["No eelgrass recorded by divers", "Eelgrass recorded by divers"]


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


def bool_from_value(value):
    if pd.isna(value):
        return pd.NA
    if value is True or str(value).lower() == "true":
        return True
    if value is False or str(value).lower() == "false":
        return False
    return pd.NA


def shorezone_category(row: pd.Series) -> str:
    if pd.isna(row["ZOS_UNIT_C"]) and pd.isna(row["ZOS_UNIT_P"]) and pd.isna(row["ZOS_UNIT_ "]):
        return "No ShoreZone match"
    if row["ZOS_UNIT_C"] is True:
        return "Eelgrass: continuous"
    if row["ZOS_UNIT_P"] is True:
        return "Eelgrass: patchy"
    if row["ZOS_UNIT_ "] is True:
        return "Eelgrass: absent"
    return "No Zostera code"


def load_comparison_data() -> pd.DataFrame:
    counts = pd.read_csv(COUNT_FILE)
    diver = counts.groupby("SiteName", dropna=False).agg(
        DiverHasEelgrass=("HabitatType", lambda x: (x == "Eelgrass").any()),
        SurveyRows=("SiteName", "size"),
        TotalSurveyMinutes=("Survey.Time", "sum"),
    ).reset_index()
    diver["DiverCategory"] = np.where(
        diver["DiverHasEelgrass"],
        "Eelgrass recorded by divers",
        "No eelgrass recorded by divers",
    )

    zos = pd.read_csv(ZOS_FILE)
    for col in ["ZOS_UNIT_ ", "ZOS_UNIT_C", "ZOS_UNIT_P"]:
        zos[col] = zos[col].map(bool_from_value)
    merged = diver.merge(zos, on="SiteName", how="left")
    merged["ShoreZoneCategory"] = merged.apply(shorezone_category, axis=1)
    merged["ShoreZoneAnyEelgrass"] = merged["ShoreZoneCategory"].isin(["Eelgrass: continuous", "Eelgrass: patchy"])
    merged["ComparisonIncluded"] = merged["ShoreZoneCategory"].isin(SHOREZONE_ORDER)
    return merged


def write_qc(data: pd.DataFrame):
    QC_DIR.mkdir(parents=True, exist_ok=True)
    included = data[data["ComparisonIncluded"]].copy()
    data.to_csv(QC_DIR / "S4_shorezone_vs_diver_eelgrass_all_sites.csv", index=False)
    included.to_csv(QC_DIR / "S4_shorezone_vs_diver_eelgrass_comparison_sites.csv", index=False)
    data[~data["ComparisonIncluded"]].to_csv(QC_DIR / "S4_shorezone_vs_diver_eelgrass_excluded_sites.csv", index=False)

    count_table = pd.crosstab(included["ShoreZoneCategory"], included["DiverCategory"])
    count_table = count_table.reindex(index=SHOREZONE_ORDER, columns=DIVER_ORDER, fill_value=0)
    count_table.to_csv(QC_DIR / "S4_shorezone_vs_diver_eelgrass_count_table.csv")

    rate_summary = included.groupby("ShoreZoneCategory", observed=False).agg(
        sites=("SiteName", "count"),
        diver_eelgrass_sites=("DiverHasEelgrass", "sum"),
        diver_eelgrass_fraction=("DiverHasEelgrass", "mean"),
    ).reindex(SHOREZONE_ORDER).reset_index()
    rate_summary.to_csv(QC_DIR / "S4_shorezone_vs_diver_eelgrass_rate_summary.csv", index=False)

    binary = pd.DataFrame({
        "metric": ["TP", "TN", "FP", "FN", "agreement", "sensitivity", "specificity"],
        "value": [
            int(((included["ShoreZoneAnyEelgrass"]) & (included["DiverHasEelgrass"])).sum()),
            int(((~included["ShoreZoneAnyEelgrass"]) & (~included["DiverHasEelgrass"])).sum()),
            int(((included["ShoreZoneAnyEelgrass"]) & (~included["DiverHasEelgrass"])).sum()),
            int(((~included["ShoreZoneAnyEelgrass"]) & (included["DiverHasEelgrass"])).sum()),
            np.nan,
            np.nan,
            np.nan,
        ],
    })
    tp, tn, fp, fn = binary.loc[:3, "value"].astype(float)
    binary.loc[binary["metric"] == "agreement", "value"] = (tp + tn) / len(included)
    binary.loc[binary["metric"] == "sensitivity", "value"] = tp / (tp + fn) if tp + fn else np.nan
    binary.loc[binary["metric"] == "specificity", "value"] = tn / (tn + fp) if tn + fp else np.nan
    binary.to_csv(QC_DIR / "S4_shorezone_vs_diver_eelgrass_binary_summary.csv", index=False)


def make_figure(data: pd.DataFrame) -> plt.Figure:
    setup_style()
    included = data[data["ComparisonIncluded"]].copy()
    count_table = pd.crosstab(included["ShoreZoneCategory"], included["DiverCategory"])
    count_table = count_table.reindex(index=SHOREZONE_ORDER, columns=DIVER_ORDER, fill_value=0)
    proportions = count_table.div(count_table.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8), constrained_layout=True)
    ax_a, ax_b = axes

    sns.heatmap(
        count_table,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        ax=ax_a,
        annot_kws={"fontsize": 8},
    )
    ax_a.set_xlabel("Human diver category")
    ax_a.set_ylabel("ShoreZone Zostera category")
    ax_a.set_xticklabels(["No eelgrass", "Eelgrass"], rotation=0)
    ax_a.set_yticklabels(["Continuous", "Patchy", "Absent"], rotation=0)
    ax_a.tick_params(axis="both", which="major", bottom=True, left=True, length=3.5, width=0.8, color="#333333")
    ax_a.text(-0.12, 1.05, "A", transform=ax_a.transAxes, fontsize=11, fontweight="bold", va="top")

    y = np.arange(len(SHOREZONE_ORDER))
    left = np.zeros(len(SHOREZONE_ORDER))
    for category in DIVER_ORDER:
        values = proportions[category].to_numpy()
        ax_b.barh(
            y,
            values,
            left=left,
            color=DIVER_COLORS[category],
            edgecolor="black",
            linewidth=0.6,
            label=category.replace(" recorded by divers", ""),
        )
        for i, value in enumerate(values):
            if value >= 0.08:
                ax_b.text(
                    left[i] + value / 2,
                    i,
                    f"{value:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if category == "No eelgrass recorded by divers" else "black",
                )
        left += values

    for i, total in enumerate(count_table.sum(axis=1).to_numpy()):
        ax_b.text(1.02, i, f"({int(total)})", va="center", ha="left", fontsize=8, color="#333333")

    ax_b.set_xlim(0, 1.12)
    ax_b.set_yticks(y, ["Continuous", "Patchy", "Absent"])
    ax_b.set_xlabel("Proportion of sites")
    ax_b.set_ylabel("ShoreZone Zostera category")
    ax_b.invert_yaxis()
    ax_b.tick_params(axis="both", which="major", bottom=True, left=True, length=3.5, width=0.8, color="#333333")
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.text(-0.12, 1.05, "B", transform=ax_b.transAxes, fontsize=11, fontweight="bold", va="top")
    ax_b.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=2,
        frameon=False,
    )
    ax_b.text(
        0.5,
        -0.21,
        "Diver category",
        transform=ax_b.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#333333",
    )

    return fig


def update_caption():
    caption_path = OUTPUT_DIR / "captions.md"
    text = """
## S4 Fig

ShoreZone and diver-recorded eelgrass classifications.

Comparison of site-level ShoreZone Zostera categories and human diver-recorded eelgrass categories for survey sites with a ShoreZone match. Human diver eelgrass status indicates whether eelgrass was recorded in any survey row at a site. (A) Counts of sites in each ShoreZone-by-diver category combination. (B) Proportion of sites within each ShoreZone category where divers did or did not record eelgrass; parenthetical values indicate the number of matched survey sites in each ShoreZone category.
""".strip()
    if caption_path.exists():
        existing = caption_path.read_text()
        if "## S4 Fig" in existing:
            existing = existing.split("## S4 Fig")[0].rstrip() + "\n\n" + text + "\n"
        else:
            existing = existing.rstrip() + "\n\n" + text + "\n"
        caption_path.write_text(existing)
    else:
        caption_path.parent.mkdir(parents=True, exist_ok=True)
        caption_path.write_text("# Publication figure captions\n\n" + text + "\n")


def main():
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    data = load_comparison_data()
    write_qc(data)
    update_caption()

    fig = make_figure(data)
    png = SOURCE_DIR / "S4_shorezone_vs_diver_eelgrass_draft.png"
    pdf = SOURCE_DIR / "S4_shorezone_vs_diver_eelgrass_draft.pdf"
    tif = SUBMISSION_DIR / "S4_Fig.tif"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig1.save_plos_tiff(fig, tif, dpi=300)
    plt.close(fig)

    included = data[data["ComparisonIncluded"]]
    print(f"Saved draft PNG: {png}")
    print(f"Saved editable PDF: {pdf}")
    print(f"Saved PLOS-style TIFF: {tif}")
    print(f"Survey sites: {len(data)}")
    print(f"Comparison sites with ShoreZone category: {len(included)}")
    print(f"Excluded/no ShoreZone category: {len(data) - len(included)}")


if __name__ == "__main__":
    main()
