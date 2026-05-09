"""
Supplemental Figure 3 draft: site-level Pycnopodia encounter-rate map colored by
ShoreZone continuous eelgrass.

This reuses the finalized Figure 1 cartographic scaffold. Circle size encodes
site-level mean Pycnopodia encounter rate; circle color encodes whether the
nearest ShoreZone segment has continuous eelgrass (ZOS_UNIT_C), rather than
whether eelgrass was diver-recorded in the survey habitat field.
"""

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import figure_1_site_map as fig1  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "publication_figures"
SOURCE_DIR = OUTPUT_DIR / "sources"
SUBMISSION_DIR = OUTPUT_DIR / "submission"
QC_DIR = OUTPUT_DIR / "qc"
SHOREZONE_DIR = PROJECT_ROOT / "outputs" / "15_shorezone_site_analysis"
ZOS_PROPORTIONS_FILE = SHOREZONE_DIR / "shorezone_ZOS_UNIT_proportions.csv"

STATUS_COLORS = {
    "Eelgrass: continuous": fig1.STARMEADOW_COLORS["eelgrass"],
    "Eelgrass: patchy": "#F2A900",
    "Eelgrass: absent": fig1.STARMEADOW_COLORS["no eelgrass"],
    "Unknown / no ShoreZone match": "#8A8A8A",
}
STATUS_ORDER = [
    "Unknown / no ShoreZone match",
    "Eelgrass: absent",
    "Eelgrass: patchy",
    "Eelgrass: continuous",
]


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


def load_site_data_with_continuous_eelgrass() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plot_df, missing = fig1.load_site_data()
    zos = pd.read_csv(ZOS_PROPORTIONS_FILE)
    zos_absent_col = next(col for col in zos.columns if col.startswith("ZOS_UNIT_") and col.strip() == "ZOS_UNIT_")
    zos = zos[["SiteName", zos_absent_col, "ZOS_UNIT_C", "ZOS_UNIT_P"]].copy()
    zos = zos.rename(columns={zos_absent_col: "ZOS_UNIT_ABSENT"})
    for col in ["ZOS_UNIT_ABSENT", "ZOS_UNIT_C", "ZOS_UNIT_P"]:
        zos[col] = zos[col].fillna(False).astype(bool)

    plot_df = plot_df.merge(zos, on="SiteName", how="left")
    plot_df["ShoreZoneZOSMatched"] = plot_df[["ZOS_UNIT_ABSENT", "ZOS_UNIT_C", "ZOS_UNIT_P"]].notna().any(axis=1)
    plot_df["ShoreZoneEelgrassStatus"] = "Unknown / no ShoreZone match"
    plot_df.loc[plot_df["ZOS_UNIT_ABSENT"].fillna(False), "ShoreZoneEelgrassStatus"] = "Eelgrass: absent"
    plot_df.loc[plot_df["ZOS_UNIT_P"].fillna(False), "ShoreZoneEelgrassStatus"] = "Eelgrass: patchy"
    plot_df.loc[plot_df["ZOS_UNIT_C"].fillna(False), "ShoreZoneEelgrassStatus"] = "Eelgrass: continuous"

    missing_shorezone = plot_df[~plot_df["ShoreZoneZOSMatched"]].copy()
    return plot_df, missing, missing_shorezone


def make_figure(plot_df: pd.DataFrame) -> plt.Figure:
    setup_style()
    fig, ax = plt.subplots(figsize=(5.0, 7.9))

    min_lon, max_lon = plot_df["Long"].min(), plot_df["Long"].max()
    min_lat, max_lat = plot_df["Lat"].min(), plot_df["Lat"].max()
    lon_pad = max((max_lon - min_lon) * 0.08, 0.08)
    lat_pad = max((max_lat - min_lat) * 0.08, 0.08)
    bounds_lonlat = (min_lon - lon_pad, max_lon + lon_pad, min_lat - lat_pad, max_lat + lat_pad)
    minx, miny = fig1.lonlat_to_mercator(bounds_lonlat[0], bounds_lonlat[2])
    maxx, maxy = fig1.lonlat_to_mercator(bounds_lonlat[1], bounds_lonlat[3])

    basemap, basemap_extent = fig1.load_carto_basemap(bounds_lonlat, zoom=8)
    ax.imshow(basemap, extent=basemap_extent, origin="upper", zorder=1)
    max_mean_rate = float(plot_df["EncounterRateHr"].max())

    # Plot rare/high-interest categories last for visibility.
    for zorder, status in enumerate(STATUS_ORDER, start=3):
        subset = plot_df[plot_df["ShoreZoneEelgrassStatus"] == status]
        x, y = fig1.lonlat_to_mercator(subset["Long"], subset["Lat"])
        ax.scatter(
            x,
            y,
            s=subset["EncounterRateHr"].map(lambda rate: fig1.size_from_rate(rate, max_mean_rate)),
            c=STATUS_COLORS[status],
            edgecolor="black",
            linewidth=0.35,
            alpha=0.82,
            label=f"{status} (n={len(subset)})",
            zorder=zorder,
        )

    ax.set_aspect("equal")
    ax.set_xlim(float(minx), float(maxx))
    ax.set_ylim(float(miny), float(maxy))

    lon_ticks = np.arange(np.ceil(bounds_lonlat[0] * 2) / 2, bounds_lonlat[1], 0.5)
    lat_ticks = np.arange(np.ceil(bounds_lonlat[2] * 2) / 2, bounds_lonlat[3], 0.5)
    x_ticks, _ = fig1.lonlat_to_mercator(lon_ticks, np.full_like(lon_ticks, plot_df["Lat"].mean()))
    _, y_ticks = fig1.lonlat_to_mercator(np.full_like(lat_ticks, plot_df["Long"].mean()), lat_ticks)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{tick:.1f}°" for tick in lon_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{tick:.1f}°" for tick in lat_ticks])

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#6B6B6B")

    fig1.add_scale_bar(ax, length_km=50, mean_lat=float(plot_df["Lat"].mean()))
    fig1.add_north_arrow(ax)

    rate_values = [0, 1, 5, 15, 30, max_mean_rate]
    rate_values = sorted(set(round(rate, 1) for rate in rate_values if rate <= max_mean_rate or np.isclose(rate, max_mean_rate)))
    size_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", color="black",
            markerfacecolor="white", markeredgecolor="black",
            markersize=np.sqrt(fig1.size_from_rate(rate, max_mean_rate)),
            label=f"{rate:g}",
        )
        for rate in rate_values
    ]
    legend_handles = [
        *[
            Patch(facecolor=STATUS_COLORS[status], edgecolor="black", label=status)
            for status in ["Eelgrass: continuous", "Eelgrass: patchy", "Eelgrass: absent", "Unknown / no ShoreZone match"]
        ],
        Line2D([], [], linestyle="", label=""),
        Line2D([], [], linestyle="", label="Mean encounter rate (Pycno hr$^{-1}$)"),
        *size_handles,
    ]
    legend = ax.legend(
        handles=legend_handles,
        title="ShoreZone eelgrass and encounter rate",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        edgecolor="#555555",
        labelspacing=1.0,
        borderpad=0.8,
    )
    legend.get_frame().set_linewidth(0.7)

    return fig


def main():
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    plot_df, missing_coords, missing_shorezone = load_site_data_with_continuous_eelgrass()
    excluded = plot_df[plot_df["Lat"] >= fig1.MAP_MAX_LAT].copy()
    plot_df = plot_df[plot_df["Lat"] < fig1.MAP_MAX_LAT].copy()

    plot_df.to_csv(QC_DIR / "S3_Fig_continuous_eelgrass_encounter_map_plotted_sites.csv", index=False)
    excluded.to_csv(QC_DIR / "S3_Fig_continuous_eelgrass_encounter_map_excluded_north_vancouver_island_sites.csv", index=False)
    missing_coords.to_csv(QC_DIR / "S3_Fig_continuous_eelgrass_encounter_map_survey_sites_missing_coordinates.csv", index=False)
    missing_shorezone.to_csv(QC_DIR / "S3_Fig_continuous_eelgrass_encounter_map_sites_missing_shorezone_zos.csv", index=False)

    summary = plot_df.groupby("ShoreZoneEelgrassStatus").agg(
        sites=("SiteName", "nunique"),
        mean_encounter_rate=("EncounterRateHr", "mean"),
        max_encounter_rate=("EncounterRateHr", "max"),
    ).reset_index()
    summary.to_csv(QC_DIR / "S3_Fig_continuous_eelgrass_encounter_map_summary.csv", index=False)

    fig = make_figure(plot_df)
    png = SOURCE_DIR / "S3_Fig_continuous_eelgrass_encounter_map_draft.png"
    pdf = SOURCE_DIR / "S3_Fig_continuous_eelgrass_encounter_map_draft.pdf"
    tif = SUBMISSION_DIR / "S3_Fig.tif"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig1.save_plos_tiff(fig, tif, dpi=300)
    plt.close(fig)

    print(f"Saved draft PNG: {png}")
    print(f"Saved editable PDF: {pdf}")
    print(f"Saved PLOS-style TIFF: {tif}")
    print(f"Plotted sites: {len(plot_df)}")
    print(f"Excluded north Vancouver Island sites: {len(excluded)}")
    print(f"Survey sites missing coordinates: {len(missing_coords)}")
    print(f"Sites missing ShoreZone ZOS_UNIT_C join before map-exclusion filter: {len(missing_shorezone)}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
