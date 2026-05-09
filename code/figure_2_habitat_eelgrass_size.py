"""
Figure 2 draft: encounter rates and size distributions by habitat/eelgrass status.

Panel A: survey-transect encounter rate by substrate grouping, dodged by whether
         the site had eelgrass recorded.
Panel B: individual Pycnopodia length density by whether the site had eelgrass
         recorded.
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

COUNT_FILE = DATA_DIR / "PycnoCountCLean_12_31_2025.csv"
LENGTH_FILE = DATA_DIR / "PycnoLengthCLean_12_31_2025.csv"

EELGRASS_COLORS = {
    "Eelgrass at site": "#28C340",
    "No eelgrass at site": "#A12CB3",
}
SUBSTRATE_ORDER = ["Soft Bottom", "Eelgrass", "Artificial Reef", "Kelp Forest", "Natural Reef", "Sponge Garden"]
SUBSTRATE_LABELS = ["Soft\nbottom", "Eelgrass", "Artificial\nreef", "Kelp\nforest", "Natural\nreef", "Sponge\ngarden"]
EELGRASS_ORDER = ["Eelgrass at site", "No eelgrass at site"]
PANEL_A_HABITATS = set(SUBSTRATE_ORDER)


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


def site_has_eelgrass(df: pd.DataFrame) -> pd.Series:
    return df.groupby("SiteName")["HabitatType"].transform(lambda x: (x == "Eelgrass").any())


def load_panel_a_data() -> pd.DataFrame:
    df = pd.read_csv(COUNT_FILE)
    df = df[df["Basin"] != "Queen Charlotte Strait"].copy()
    df["Survey.Time"] = pd.to_numeric(df["Survey.Time"], errors="coerce")
    df["Pycnopodia_count"] = pd.to_numeric(df["Pycnopodia_count"], errors="coerce").fillna(0)
    df["EncounterRateHr"] = np.where(
        df["Survey.Time"] > 0,
        df["Pycnopodia_count"] / df["Survey.Time"] * 60,
        np.nan,
    )
    df["SiteHasEelgrass"] = site_has_eelgrass(df)
    df["EelgrassStatus"] = np.where(df["SiteHasEelgrass"], "Eelgrass at site", "No eelgrass at site")

    panel = df[df["HabitatType"].isin(PANEL_A_HABITATS)].copy()
    panel["SubstrateGroup"] = panel["HabitatType"]
    panel = panel.dropna(subset=["EncounterRateHr", "SubstrateGroup", "EelgrassStatus"])
    return panel


def load_panel_b_data() -> pd.DataFrame:
    df = pd.read_csv(LENGTH_FILE)
    df = df[df["Basin"] != "Queen Charlotte Strait"].copy()
    df["Length_cm"] = pd.to_numeric(df["Length(cm)"].astype(str).str.strip(), errors="coerce")
    df["SiteHasEelgrass"] = site_has_eelgrass(df)
    df["EelgrassStatus"] = np.where(df["SiteHasEelgrass"], "Eelgrass at site", "No eelgrass at site")
    individuals = df[df["Length_cm"] > 0].copy()
    individuals = individuals.dropna(subset=["Length_cm", "EelgrassStatus"])
    return individuals


def write_qc(panel_a: pd.DataFrame, panel_b: pd.DataFrame):
    QC_DIR.mkdir(parents=True, exist_ok=True)
    panel_a_summary = panel_a.groupby(["SubstrateGroup", "EelgrassStatus"], observed=False).agg(
        rows=("EncounterRateHr", "count"),
        sites=("SiteName", "nunique"),
        median_encounter_rate=("EncounterRateHr", "median"),
        mean_encounter_rate=("EncounterRateHr", "mean"),
        sd_encounter_rate=("EncounterRateHr", "std"),
        se_encounter_rate=("EncounterRateHr", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
        max_encounter_rate=("EncounterRateHr", "max"),
    ).reset_index()
    panel_a_summary["SubstrateGroup"] = pd.Categorical(panel_a_summary["SubstrateGroup"], SUBSTRATE_ORDER, ordered=True)
    panel_a_summary["EelgrassStatus"] = pd.Categorical(panel_a_summary["EelgrassStatus"], EELGRASS_ORDER, ordered=True)
    panel_a_summary = panel_a_summary.sort_values(["SubstrateGroup", "EelgrassStatus"])
    panel_a_summary = panel_a_summary[panel_a_summary["rows"] > 0]
    panel_b_summary = panel_b.groupby("EelgrassStatus", observed=False).agg(
        individuals=("Length_cm", "count"),
        sites=("SiteName", "nunique"),
        mean_length_cm=("Length_cm", "mean"),
        median_length_cm=("Length_cm", "median"),
        min_length_cm=("Length_cm", "min"),
        max_length_cm=("Length_cm", "max"),
    ).reset_index()
    panel_a_summary.to_csv(QC_DIR / "Fig2_panel_A_encounter_rate_bar_summary.csv", index=False)
    panel_b_summary.to_csv(QC_DIR / "Fig2_panel_B_size_distribution_summary.csv", index=False)
    panel_a.to_csv(QC_DIR / "Fig2_panel_A_plot_data.csv", index=False)
    panel_b.to_csv(QC_DIR / "Fig2_panel_B_plot_data.csv", index=False)


def make_figure(panel_a: pd.DataFrame, panel_b: pd.DataFrame) -> plt.Figure:
    setup_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.5, 3.9),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )
    ax_a, ax_b = axes

    panel_a_summary = panel_a.groupby(["SubstrateGroup", "EelgrassStatus"], observed=False).agg(
        rows=("EncounterRateHr", "count"),
        sites=("SiteName", "nunique"),
        mean_encounter_rate=("EncounterRateHr", "mean"),
        sd_encounter_rate=("EncounterRateHr", "std"),
        se_encounter_rate=("EncounterRateHr", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
    ).reset_index()
    panel_a_summary["SubstrateGroup"] = pd.Categorical(panel_a_summary["SubstrateGroup"], SUBSTRATE_ORDER, ordered=True)
    panel_a_summary["EelgrassStatus"] = pd.Categorical(panel_a_summary["EelgrassStatus"], EELGRASS_ORDER, ordered=True)
    panel_a_summary = panel_a_summary.sort_values(["SubstrateGroup", "EelgrassStatus"])
    panel_a_summary = panel_a_summary[panel_a_summary["rows"] > 0]

    x_centers = np.arange(len(SUBSTRATE_ORDER)) * 1.25
    bar_width = 0.51
    offsets = {EELGRASS_ORDER[0]: -bar_width / 2, EELGRASS_ORDER[1]: bar_width / 2}
    max_annotation_y = 0.0
    for status in EELGRASS_ORDER:
        subset = panel_a_summary[panel_a_summary["EelgrassStatus"] == status]
        xs = np.array([x_centers[SUBSTRATE_ORDER.index(group)] for group in subset["SubstrateGroup"].astype(str)]) + offsets[status]
        ax_a.bar(
            xs,
            subset["mean_encounter_rate"],
            width=bar_width,
            color=EELGRASS_COLORS[status],
            edgecolor="black",
            linewidth=0.8,
            label=status,
            zorder=3,
        )
        ax_a.errorbar(
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
            ax_a.annotate(
                f"({int(row['rows'])})",
                xy=(x, y),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#333333",
                zorder=5,
            )

    ax_a.set_ylim(top=max_annotation_y * 1.18 if max_annotation_y > 0 else None)
    ax_a.set_xticks(x_centers, labels=SUBSTRATE_LABELS)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("Encounter rate (Pycnopodia hr$^{-1}$)")
    ax_a.tick_params(axis="x", rotation=0, bottom=True, length=3.5, width=0.8, color="#333333")
    ax_a.tick_params(axis="y", left=True, length=3.5, width=0.8, color="#333333")
    ax_a.text(0.02, 0.96, "A", transform=ax_a.transAxes, fontsize=11, fontweight="bold", va="top")
    ax_a.spines[["top", "right"]].set_visible(False)
    if ax_a.get_legend() is not None:
        ax_a.get_legend().remove()

    sns.kdeplot(
        data=panel_b,
        x="Length_cm",
        hue="EelgrassStatus",
        hue_order=EELGRASS_ORDER,
        palette=EELGRASS_COLORS,
        common_norm=False,
        fill=False,
        linewidth=2.0,
        bw_adjust=0.6,
        ax=ax_b,
    )
    ax_b.set_xlabel("Length (cm)")
    ax_b.set_ylabel("Density")
    ax_b.set_xlim(left=0)
    ax_b.tick_params(axis="both", which="major", bottom=True, left=True, length=3.5, width=0.8, color="#333333")
    ax_b.text(0.02, 0.96, "B", transform=ax_b.transAxes, fontsize=11, fontweight="bold", va="top")
    ax_b.spines[["top", "right"]].set_visible(False)
    if ax_b.get_legend() is not None:
        ax_b.get_legend().remove()

    handles = [
        plt.Line2D([0], [0], color=EELGRASS_COLORS[label], linewidth=2.5, marker="s", markersize=6, label=label)
        for label in EELGRASS_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, -0.08),
        ncol=2,
        frameon=False,
    )
    return fig


def main():
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    panel_a = load_panel_a_data()
    panel_b = load_panel_b_data()
    write_qc(panel_a, panel_b)

    fig = make_figure(panel_a, panel_b)
    png = SOURCE_DIR / "Fig2_habitat_eelgrass_size_draft.png"
    pdf = SOURCE_DIR / "Fig2_habitat_eelgrass_size_draft.pdf"
    tif = SUBMISSION_DIR / "Fig2.tif"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig1.save_plos_tiff(fig, tif, dpi=300)
    plt.close(fig)

    print(f"Saved draft PNG: {png}")
    print(f"Saved editable PDF: {pdf}")
    print(f"Saved PLOS-style TIFF: {tif}")
    print(f"Panel A rows: {len(panel_a)}")
    print(f"Panel A sites: {panel_a['SiteName'].nunique()}")
    print(f"Panel B individual lengths: {len(panel_b)}")
    print(f"Panel B sites: {panel_b['SiteName'].nunique()}")


if __name__ == "__main__":
    main()
