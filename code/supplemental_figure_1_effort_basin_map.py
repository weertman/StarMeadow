"""
Supplemental Figure 1 draft: survey effort and basin map for StarMeadow.

Uses the same map extent and CARTO Positron no-label basemap as finalized Figure 1.
Circle size encodes site-level total survey effort in hours.
Circle color encodes basin.
"""

from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "code"))

import figure_1_site_map as fig1  # noqa: E402

SOURCE_DIR = fig1.SOURCE_DIR
SUBMISSION_DIR = fig1.SUBMISSION_DIR
QC_DIR = fig1.QC_DIR

BASIN_COLORS = {
    "Admiralty Inlet": "#0072B2",
    "Central": "#E69F00",
    "Hood Canal": "#009E73",
    "Howe Sound": "#D55E00",
    "San Juan": "#CC79A7",
    "South": "#56B4E9",
    "Strait of Georgia": "#999999",
    "Strait of Juan de Fuca": "#F0E442",
    "Whidbey": "#332288",
}
MIN_MARKER_AREA = 24.0
MAX_MARKER_AREA = 380.0


def size_from_effort(hours: float, max_hours: float | None = None) -> float:
    """Marker area in points^2, proportional to total survey effort in hours."""
    hours = max(float(hours), 0.0)
    if max_hours is None or max_hours <= 0:
        max_hours = 1.0
    scaled = min(hours / max_hours, 1.0)
    return MIN_MARKER_AREA + scaled * (MAX_MARKER_AREA - MIN_MARKER_AREA)


def make_figure(plot_df: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update({
        "font.family": "Liberation Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.5, 8.0), constrained_layout=True)
    ax.set_facecolor("#F7FBFF")

    min_lon, max_lon = plot_df["Long"].min(), plot_df["Long"].max()
    min_lat, max_lat = plot_df["Lat"].min(), plot_df["Lat"].max()
    lon_pad = max((max_lon - min_lon) * 0.08, 0.08)
    lat_pad = max((max_lat - min_lat) * 0.08, 0.08)
    bounds_lonlat = (min_lon - lon_pad, max_lon + lon_pad, min_lat - lat_pad, max_lat + lat_pad)
    minx, miny = fig1.lonlat_to_mercator(bounds_lonlat[0], bounds_lonlat[2])
    maxx, maxy = fig1.lonlat_to_mercator(bounds_lonlat[1], bounds_lonlat[3])

    basemap, basemap_extent = fig1.load_carto_basemap(bounds_lonlat, zoom=8)
    ax.imshow(basemap, extent=basemap_extent, origin="upper", zorder=1)

    max_hours = float(plot_df["TotalSurveyHours"].max())
    for basin in sorted(plot_df["Basin"].dropna().unique()):
        subset = plot_df[plot_df["Basin"] == basin]
        x, y = fig1.lonlat_to_mercator(subset["Long"], subset["Lat"])
        ax.scatter(
            x,
            y,
            s=subset["TotalSurveyHours"].map(lambda h: size_from_effort(h, max_hours)),
            c=BASIN_COLORS.get(basin, "#666666"),
            edgecolor="black",
            linewidth=0.35,
            alpha=0.82,
            zorder=4,
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

    effort_values = [1, 5, 15, 30, max_hours]
    effort_values = sorted(set(round(v, 1) for v in effort_values if v <= max_hours or np.isclose(v, max_hours)))
    basin_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", color="black",
            markerfacecolor=BASIN_COLORS.get(basin, "#666666"), markeredgecolor="black",
            markersize=6,
            label=basin,
        )
        for basin in sorted(plot_df["Basin"].dropna().unique())
    ]
    effort_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", color="black",
            markerfacecolor="white", markeredgecolor="black",
            markersize=np.sqrt(size_from_effort(hours, max_hours)),
            label=f"{hours:g}" + (" max" if np.isclose(hours, round(max_hours, 1)) else ""),
        )
        for hours in effort_values
    ]
    legend_handles = [
        *basin_handles,
        Line2D([], [], linestyle="", label=""),
        Line2D([], [], linestyle="", label="Survey effort (hr)"),
        *effort_handles,
    ]
    legend = ax.legend(
        handles=legend_handles,
        title="Basin and survey effort",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        edgecolor="#555555",
        labelspacing=0.9,
        borderpad=0.9,
        handleheight=1.8,
    )
    legend.get_frame().set_linewidth(0.7)
    return fig


def main():
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    plot_df, missing = fig1.load_site_data()
    excluded = plot_df[plot_df["Lat"] >= fig1.MAP_MAX_LAT].copy()
    plot_df = plot_df[plot_df["Lat"] < fig1.MAP_MAX_LAT].copy()
    plot_df["TotalSurveyHours"] = plot_df["TotalSurveyMinutes"] / 60.0
    excluded["TotalSurveyHours"] = excluded["TotalSurveyMinutes"] / 60.0

    plot_df.to_csv(QC_DIR / "S1_Fig_effort_basin_plotted_sites.csv", index=False)
    excluded.to_csv(QC_DIR / "S1_Fig_effort_basin_excluded_north_vancouver_island_sites.csv", index=False)
    missing.to_csv(QC_DIR / "S1_Fig_effort_basin_missing_coordinates.csv", index=False)

    fig = make_figure(plot_df)
    png = SOURCE_DIR / "S1_Fig_effort_basin_map_draft.png"
    pdf = SOURCE_DIR / "S1_Fig_effort_basin_map_draft.pdf"
    tif = SUBMISSION_DIR / "S1_Fig.tif"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig1.save_plos_tiff(fig, tif, dpi=300)
    plt.close(fig)

    print(f"Saved draft PNG: {png}")
    print(f"Saved editable PDF: {pdf}")
    print(f"Saved PLOS-style TIFF: {tif}")
    print(f"Plotted sites: {len(plot_df)}")
    print(f"Excluded north Vancouver Island sites: {len(excluded)}")
    print(f"Survey sites missing coordinates: {len(missing)}")
    print(f"Max survey effort hours: {plot_df['TotalSurveyHours'].max():.2f}")
    print(f"Basins plotted: {', '.join(sorted(plot_df['Basin'].dropna().unique()))}")


if __name__ == "__main__":
    main()
