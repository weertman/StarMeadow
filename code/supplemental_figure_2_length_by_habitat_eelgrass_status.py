"""
Supplemental Figure 2: individual Pycnopodia length distributions by survey-level
habitat, split into sites with and without eelgrass recorded.

Panel A: habitat-specific length density curves for surveys at eelgrass sites.
Panel B: habitat-specific length density curves for surveys at no-eelgrass sites.
"""
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
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

LENGTH_FILE = DATA_DIR / "PycnoLengthCLean_12_31_2025.csv"

HABITAT_ORDER = ["Soft Bottom", "Eelgrass", "Artificial Reef", "Kelp Forest", "Natural Reef", "Sponge Garden"]
EELGRASS_STATUS_ORDER = ["Eelgrass at site", "No eelgrass at site"]
HABITAT_COLORS = {
    "Soft Bottom": "#B78273",
    "Eelgrass": "#28C340",
    "Artificial Reef": "#4C78A8",
    "Kelp Forest": "#0B5246",
    "Natural Reef": "#F58518",
    "Sponge Garden": "#6F4E7C",
}
SITE_STATUS_COLORS = {
    "Eelgrass at site": "#28C340",
    "No eelgrass at site": "#A12CB3",
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


def site_has_eelgrass(df: pd.DataFrame) -> pd.Series:
    return df.groupby("SiteName")["HabitatType"].transform(lambda x: (x == "Eelgrass").any())


def load_plot_data() -> pd.DataFrame:
    df = pd.read_csv(LENGTH_FILE)
    df = df[df["Basin"] != "Queen Charlotte Strait"].copy()
    df["Length_cm"] = pd.to_numeric(df["Length(cm)"].astype(str).str.strip(), errors="coerce")
    df["SiteHasEelgrass"] = site_has_eelgrass(df)
    df["EelgrassStatus"] = np.where(df["SiteHasEelgrass"], "Eelgrass at site", "No eelgrass at site")
    df = df[df["Length_cm"] > 0].copy()
    df = df[df["HabitatType"].isin(HABITAT_ORDER)].copy()
    df["HabitatType"] = pd.Categorical(df["HabitatType"], HABITAT_ORDER, ordered=True)
    df["EelgrassStatus"] = pd.Categorical(df["EelgrassStatus"], EELGRASS_STATUS_ORDER, ordered=True)
    return df.dropna(subset=["Length_cm", "HabitatType", "EelgrassStatus"])


def write_qc(plot_data: pd.DataFrame):
    QC_DIR.mkdir(parents=True, exist_ok=True)
    summary = plot_data.groupby(["EelgrassStatus", "HabitatType"], observed=True).agg(
        individuals=("Length_cm", "count"),
        sites=("SiteName", "nunique"),
        mean_length_cm=("Length_cm", "mean"),
        median_length_cm=("Length_cm", "median"),
        min_length_cm=("Length_cm", "min"),
        max_length_cm=("Length_cm", "max"),
    ).reset_index()
    summary.to_csv(QC_DIR / "S2_length_distribution_by_habitat_eelgrass_status_summary.csv", index=False)
    plot_data.to_csv(QC_DIR / "S2_length_distribution_by_habitat_eelgrass_status_plot_data.csv", index=False)


def make_figure(plot_data: pd.DataFrame) -> plt.Figure:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.05), constrained_layout=True, sharex=True, sharey=True)

    present_habitats = [hab for hab in HABITAT_ORDER if (plot_data["HabitatType"].astype(str) == hab).any()]
    x_min = 0.0
    x_max = float(np.ceil(plot_data["Length_cm"].max() / 10.0) * 10.0)

    for ax, status, panel_label in zip(axes, EELGRASS_STATUS_ORDER, ["A", "B"]):
        subset = plot_data[plot_data["EelgrassStatus"] == status]
        for habitat in present_habitats:
            habitat_data = subset[subset["HabitatType"].astype(str) == habitat]
            if len(habitat_data) < 2:
                continue
            sns.kdeplot(
                data=habitat_data,
                x="Length_cm",
                color=HABITAT_COLORS[habitat],
                linewidth=2.0,
                bw_adjust=0.6,
                common_norm=False,
                fill=False,
                clip=(x_min, x_max),
                ax=ax,
                label=f"{habitat} ({len(habitat_data)})",
            )

        ax.set_title(status, fontsize=9, pad=4, color=SITE_STATUS_COLORS[status])
        ax.set_xlabel("Length (cm)")
        ax.set_ylabel("Density")
        ax.set_xlim(x_min, x_max)
        ax.tick_params(axis="both", which="major", bottom=True, left=True, length=3.5, width=0.8, color="#333333")
        ax.text(0.02, 0.96, panel_label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.spines[["top", "right"]].set_visible(False)

    # Enforce identical y-axis limits after both KDE panels are drawn.
    y_max = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, y_max)

    legend_counts = plot_data.groupby(["HabitatType", "EelgrassStatus"], observed=True).size()

    def legend_entry(habitat: str) -> HPacker:
        eelgrass_n = int(legend_counts.get((habitat, "Eelgrass at site"), 0))
        no_eelgrass_n = int(legend_counts.get((habitat, "No eelgrass at site"), 0))
        line = DrawingArea(22, 8, 0, 0)
        line.add_artist(Line2D([1, 21], [4, 4], color=HABITAT_COLORS[habitat], linewidth=2.5))
        pieces = [
            TextArea(f" {habitat} (", textprops={"size": 8, "color": "#333333"}),
            TextArea(str(eelgrass_n), textprops={"size": 8, "color": SITE_STATUS_COLORS["Eelgrass at site"]}),
            TextArea(", ", textprops={"size": 8, "color": "#333333"}),
            TextArea(str(no_eelgrass_n), textprops={"size": 8, "color": SITE_STATUS_COLORS["No eelgrass at site"]}),
            TextArea(")", textprops={"size": 8, "color": "#333333"}),
        ]
        label = HPacker(children=pieces, align="center", pad=0, sep=0)
        return HPacker(children=[line, label], align="center", pad=0, sep=2)

    entries = [legend_entry(habitat) for habitat in present_habitats]
    rows = [
        HPacker(children=entries[:3], align="center", pad=0, sep=18),
        HPacker(children=entries[3:], align="center", pad=0, sep=18),
    ]
    legend_box = VPacker(children=rows, align="center", pad=0, sep=5)
    anchored_legend = AnchoredOffsetbox(
        loc="lower center",
        child=legend_box,
        bbox_to_anchor=(0.5, -0.16),
        bbox_transform=fig.transFigure,
        frameon=False,
        borderpad=0.0,
    )
    fig.add_artist(anchored_legend)
    return fig


def update_caption():
    caption_path = OUTPUT_DIR / "captions.md"
    text = """
## S2 Fig

Individual size distributions by survey-level habitat and site-level eelgrass status.

Kernel density distributions of individual Pycnopodia lengths by recorded survey-level habitat for (A) sites where eelgrass was recorded and (B) sites where eelgrass was not recorded. Both panels use the same x- and y-axis limits. Numbers in the legend indicate the number of individual length observations contributing to each habitat-specific density curve.
""".strip()
    if caption_path.exists():
        existing = caption_path.read_text()
        if "## S2 Fig" in existing:
            existing = existing.split("## S2 Fig")[0].rstrip() + "\n\n" + text + "\n"
        else:
            existing = existing.rstrip() + "\n\n" + text + "\n"
        caption_path.write_text(existing)
    else:
        caption_path.parent.mkdir(parents=True, exist_ok=True)
        caption_path.write_text("# Publication figure captions\n\n" + text + "\n")


def main():
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    plot_data = load_plot_data()
    write_qc(plot_data)
    update_caption()

    fig = make_figure(plot_data)
    png = SOURCE_DIR / "S2_length_distribution_by_habitat_eelgrass_status_draft.png"
    pdf = SOURCE_DIR / "S2_length_distribution_by_habitat_eelgrass_status_draft.pdf"
    tif = SUBMISSION_DIR / "S2_Fig.tif"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig1.save_plos_tiff(fig, tif, dpi=300)
    plt.close(fig)

    print(f"Saved draft PNG: {png}")
    print(f"Saved editable PDF: {pdf}")
    print(f"Saved PLOS-style TIFF: {tif}")
    print(f"Individual length rows: {len(plot_data)}")
    print(f"Sites: {plot_data['SiteName'].nunique()}")


if __name__ == "__main__":
    main()
