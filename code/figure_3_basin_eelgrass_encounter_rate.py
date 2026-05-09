"""
Figure 3 draft: encounter rates by basin and site-level eelgrass status.

Single-panel grouped bar plot. Bar heights are mean row-level Pycnopodia
encounter rates, whiskers are standard errors, and colors indicate whether a
site had eelgrass recorded during surveys.
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

EELGRASS_COLORS = {
    "Eelgrass at site": "#28C340",
    "No eelgrass at site": "#A12CB3",
}
EELGRASS_ORDER = ["Eelgrass at site", "No eelgrass at site"]

ALL_BASINS = [
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
BASIN_LABEL_MAP = {
    "Howe Sound": "Howe\nSound",
    "Strait of Georgia": "Strait of\nGeorgia",
    "San Juan": "San\nJuan",
    "Strait of Juan de Fuca": "Strait of\nJuan de Fuca",
    "Admiralty Inlet": "Admiralty\nInlet",
    "Whidbey": "Whidbey",
    "Central": "Central",
    "South": "South",
    "Hood Canal": "Hood\nCanal",
}


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


def load_plot_data() -> pd.DataFrame:
    df = pd.read_csv(COUNT_FILE)
    df = df[df["Basin"] != "Queen Charlotte Strait"].copy()
    df["Survey.Time"] = pd.to_numeric(df["Survey.Time"], errors="coerce")
    df["Pycnopodia_count"] = pd.to_numeric(df["Pycnopodia_count"], errors="coerce").fillna(0)
    df["EncounterRateHr"] = np.where(
        df["Survey.Time"] > 0,
        df["Pycnopodia_count"] / df["Survey.Time"] * 60,
        np.nan,
    )
    df["SiteHasEelgrass"] = df.groupby("SiteName")["HabitatType"].transform(lambda x: (x == "Eelgrass").any())
    df["EelgrassStatus"] = np.where(df["SiteHasEelgrass"], "Eelgrass at site", "No eelgrass at site")
    df = df.dropna(subset=["Basin", "EncounterRateHr", "EelgrassStatus"])
    df = df[df["Basin"].isin(ALL_BASINS)].copy()
    return df


def summarize(plot_data: pd.DataFrame) -> pd.DataFrame:
    summary = plot_data.groupby(["Basin", "EelgrassStatus"], observed=False).agg(
        rows=("EncounterRateHr", "count"),
        sites=("SiteName", "nunique"),
        mean_encounter_rate=("EncounterRateHr", "mean"),
        sd_encounter_rate=("EncounterRateHr", "std"),
        se_encounter_rate=("EncounterRateHr", lambda x: x.std(ddof=1) / np.sqrt(x.count())),
        max_encounter_rate=("EncounterRateHr", "max"),
    ).reset_index()
    summary["Basin"] = pd.Categorical(summary["Basin"], ALL_BASINS, ordered=True)
    summary["EelgrassStatus"] = pd.Categorical(summary["EelgrassStatus"], EELGRASS_ORDER, ordered=True)
    summary = summary.sort_values(["Basin", "EelgrassStatus"])
    summary = summary[summary["rows"] > 0].copy()
    return summary


def basin_order_by_eelgrass_rate(summary: pd.DataFrame) -> list[str]:
    """Rank basins by the eelgrass-at-site mean encounter rate, descending."""
    eelgrass_means = (
        summary[summary["EelgrassStatus"] == "Eelgrass at site"]
        .set_index("Basin")["mean_encounter_rate"]
        .astype(float)
        .to_dict()
    )
    return sorted(
        ALL_BASINS,
        key=lambda basin: (eelgrass_means.get(basin, -np.inf), -ALL_BASINS.index(basin)),
        reverse=True,
    )


def write_qc(plot_data: pd.DataFrame, summary: pd.DataFrame):
    QC_DIR.mkdir(parents=True, exist_ok=True)
    plot_data.to_csv(QC_DIR / "Fig3_basin_eelgrass_encounter_rate_plot_data.csv", index=False)
    summary.to_csv(QC_DIR / "Fig3_basin_eelgrass_encounter_rate_summary.csv", index=False)
    pd.DataFrame({"rank_order": range(1, len(basin_order_by_eelgrass_rate(summary)) + 1), "Basin": basin_order_by_eelgrass_rate(summary)}).to_csv(
        QC_DIR / "Fig3_basin_rank_order_by_eelgrass_encounter_rate.csv",
        index=False,
    )


def make_figure(summary: pd.DataFrame) -> plt.Figure:
    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)

    basin_order = basin_order_by_eelgrass_rate(summary)
    basin_labels = [BASIN_LABEL_MAP[basin] for basin in basin_order]
    x_centers = np.arange(len(basin_order)) * 1.35
    bar_width = 0.48
    offsets = {EELGRASS_ORDER[0]: -bar_width / 2, EELGRASS_ORDER[1]: bar_width / 2}

    max_annotation_y = 0.0
    for status in EELGRASS_ORDER:
        subset = summary[summary["EelgrassStatus"] == status]
        xs = np.array([x_centers[basin_order.index(basin)] for basin in subset["Basin"].astype(str)]) + offsets[status]
        ax.bar(
            xs,
            subset["mean_encounter_rate"],
            width=bar_width,
            color=EELGRASS_COLORS[status],
            edgecolor="black",
            linewidth=0.8,
            label=status,
            zorder=3,
        )
        ax.errorbar(
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
            ax.annotate(
                f"({int(row['rows'])})",
                xy=(x, y),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="#333333",
                zorder=5,
            )

    ax.set_ylim(top=max_annotation_y * 1.18 if max_annotation_y > 0 else None)
    ax.set_xticks(x_centers, labels=basin_labels)
    ax.set_xlabel("Basin", labelpad=8)
    ax.set_ylabel("Encounter rate (Pycnopodia hr$^{-1}$)")
    ax.tick_params(axis="x", rotation=45, bottom=True, length=3.5, width=0.8, color="#333333", labelsize=7)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    ax.tick_params(axis="y", left=True, length=3.5, width=0.8, color="#333333")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper right",
        frameon=False,
    )
    return fig


def main():
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)

    plot_data = load_plot_data()
    summary = summarize(plot_data)
    write_qc(plot_data, summary)

    fig = make_figure(summary)
    png = SOURCE_DIR / "Fig3_basin_eelgrass_encounter_rate_draft.png"
    pdf = SOURCE_DIR / "Fig3_basin_eelgrass_encounter_rate_draft.pdf"
    tif = SUBMISSION_DIR / "Fig3.tif"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig1.save_plos_tiff(fig, tif, dpi=300)
    plt.close(fig)

    print(f"Saved draft PNG: {png}")
    print(f"Saved editable PDF: {pdf}")
    print(f"Saved PLOS-style TIFF: {tif}")
    print(f"Rows plotted: {len(plot_data)}")
    print(f"Sites plotted: {plot_data['SiteName'].nunique()}")
    print(f"Basins plotted: {summary['Basin'].nunique()}")


if __name__ == "__main__":
    main()
