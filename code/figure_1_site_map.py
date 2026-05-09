"""
Figure 1 draft: survey site map for StarMeadow PLOS ONE manuscript.

Circle size encodes site-level encounter rate per survey hour:
    total Pycnopodia count / total survey time minutes * 60

Circle color encodes whether eelgrass was recorded at that site in the survey data.
"""

from io import BytesIO
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "publication_figures"
SOURCE_DIR = OUTPUT_DIR / "sources"
SUBMISSION_DIR = OUTPUT_DIR / "submission"
QC_DIR = OUTPUT_DIR / "qc"

COUNT_FILE = DATA_DIR / "PycnoCountCLean_12_31_2025.csv"
COORD_FILE = DATA_DIR / "Site_LatLong.csv"
ACCEPTED_MISSING_COORD_FILE = (
    OUTPUT_DIR
    / "qc"
    / "missing_coordinate_candidates"
    / "Fig1_missing_coordinate_candidates_for_review.csv"
)
# Basemap: CARTO Positron no-label raster tiles, selected after comparing
# candidate basemaps for clean U.S.+Canada coastline definition.
CARTO_POSITRON_TILE_URL = "https://a.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}.png"
CARTO_ATTRIBUTION = "Basemap: CARTO Positron, © OpenStreetMap contributors"
USER_AGENT = "StarMeadow manuscript figure draft (contact: willemweertman)"
WEB_MERCATOR_LIMIT = 20037508.342789244
MAP_MAX_LAT = 50.0  # Exclude north Vancouver Island / Queen Charlotte Strait sites from Fig 1 extent.

STARMEADOW_COLORS = {
    "eelgrass": "#28C340",
    "no eelgrass": "#A12CB3",
    "soft bottom": "#B78273",
    "kelp or reef etc": "#0B5246",
}


def normalize_site_name(value: object) -> str:
    """Conservative normalization for joining exact site names with whitespace cleanup."""
    return " ".join(str(value).strip().replace("’", "'").split()).lower()


def load_coordinates() -> pd.DataFrame:
    """Load canonical coordinates plus reviewed Fig 1 supplemental candidates."""
    coords = pd.read_csv(COORD_FILE).rename(columns={"Site": "SiteName"})
    coords["CoordinateSource"] = "Site_LatLong.csv"
    coords["CoordinateConfidence"] = "canonical"
    coords["CoordinateNote"] = "Existing coordinate table"

    if ACCEPTED_MISSING_COORD_FILE.exists():
        accepted = pd.read_csv(ACCEPTED_MISSING_COORD_FILE)
        accepted = accepted[accepted["candidate_status"] == "candidate"].copy()
        accepted["Lat"] = pd.to_numeric(accepted["lat"], errors="coerce")
        accepted["Long"] = pd.to_numeric(accepted["lon"], errors="coerce")
        accepted["SiteName"] = accepted["SiteName"].astype(str).str.strip()
        accepted["CoordinateSource"] = "accepted Fig1 candidate"
        accepted["CoordinateConfidence"] = accepted["confidence"]
        accepted["CoordinateNote"] = accepted["note"]
        accepted = accepted[["SiteName", "Lat", "Long", "CoordinateSource", "CoordinateConfidence", "CoordinateNote"]]
        coords = pd.concat([coords, accepted], ignore_index=True)

    coords.loc[coords["Long"] > 0, "Long"] = -coords.loc[coords["Long"] > 0, "Long"]
    # Keep the last duplicate key so accepted supplemental candidates can fill
    # sites absent from the canonical coordinate table without changing it.
    coords["site_key"] = coords["SiteName"].map(normalize_site_name)
    coords = coords.drop_duplicates("site_key", keep="last")
    return coords


def load_site_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate survey rows to site-level rates and join coordinates."""
    counts = pd.read_csv(COUNT_FILE)
    coords = load_coordinates()

    counts["Survey.Time"] = pd.to_numeric(counts["Survey.Time"], errors="coerce")
    counts["Pycnopodia_count"] = pd.to_numeric(counts["Pycnopodia_count"], errors="coerce").fillna(0)
    counts["TransectEncounterRateHr"] = np.where(
        counts["Survey.Time"] > 0,
        counts["Pycnopodia_count"] / counts["Survey.Time"] * 60,
        np.nan,
    )

    site = counts.groupby("SiteName", dropna=False).agg(
        TotalPycno=("Pycnopodia_count", "sum"),
        TotalSurveyMinutes=("Survey.Time", "sum"),
        MeanEncounterRateHr=("TransectEncounterRateHr", "mean"),
        NTransects=("SiteName", "size"),
        Basin=("Basin", "first"),
        HasEelgrass=("HabitatType", lambda x: (x == "Eelgrass").any()),
    ).reset_index()
    site["TotalEncounterRateHr"] = np.where(
        site["TotalSurveyMinutes"] > 0,
        site["TotalPycno"] / site["TotalSurveyMinutes"] * 60,
        np.nan,
    )
    # Circle size uses the site mean encounter rate, normalized to the maximum
    # site mean in the plotted data. Keep the total-effort rate in QC output.
    site["EncounterRateHr"] = site["MeanEncounterRateHr"]

    site["site_key"] = site["SiteName"].map(normalize_site_name)
    coords = coords[[
        "site_key",
        "SiteName",
        "Lat",
        "Long",
        "CoordinateSource",
        "CoordinateConfidence",
        "CoordinateNote",
    ]].rename(columns={"SiteName": "CoordSiteName"})

    merged = site.merge(coords, on="site_key", how="left")
    missing = merged[merged[["Lat", "Long"]].isna().any(axis=1)].copy()
    plot_df = merged.dropna(subset=["Lat", "Long", "EncounterRateHr"]).copy()
    return plot_df, missing


def lonlat_to_mercator(lon, lat):
    """Convert lon/lat degrees to Web Mercator meters for raster tile plotting."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lat = np.clip(lat, -85.05112878, 85.05112878)
    x = lon * WEB_MERCATOR_LIMIT / 180.0
    y = np.log(np.tan((90.0 + lat) * np.pi / 360.0)) * WEB_MERCATOR_LIMIT / np.pi
    return x, y


def mercator_to_lonlat(x, y):
    """Convert Web Mercator meters to lon/lat degrees."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lon = x / WEB_MERCATOR_LIMIT * 180.0
    lat = (2.0 * np.arctan(np.exp(y / WEB_MERCATOR_LIMIT * np.pi)) - np.pi / 2.0) * 180.0 / np.pi
    return lon, lat


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    """Return XYZ tile index for lon/lat at a Web Mercator zoom level."""
    lat_rad = math.radians(float(np.clip(lat, -85.05112878, 85.05112878)))
    n = 2 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def tile_bounds_mercator(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    """Return minx, maxx, miny, maxy Web Mercator bounds for an XYZ tile."""
    n = 2 ** zoom
    tile_size = 2 * WEB_MERCATOR_LIMIT / n
    minx = -WEB_MERCATOR_LIMIT + x * tile_size
    maxx = -WEB_MERCATOR_LIMIT + (x + 1) * tile_size
    maxy = WEB_MERCATOR_LIMIT - y * tile_size
    miny = WEB_MERCATOR_LIMIT - (y + 1) * tile_size
    return minx, maxx, miny, maxy


def fetch_carto_tile(x: int, y: int, zoom: int) -> Image.Image:
    """Fetch one CARTO Positron no-label tile."""
    url = CARTO_POSITRON_TILE_URL.format(x=x, y=y, z=zoom)
    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_carto_basemap(bounds_lonlat: tuple[float, float, float, float], zoom: int = 8):
    """Fetch and mosaic CARTO Positron tiles covering the lon/lat bounds."""
    min_lon, max_lon, min_lat, max_lat = bounds_lonlat
    x0, y1 = lonlat_to_tile(min_lon, min_lat, zoom)
    x1, y0 = lonlat_to_tile(max_lon, max_lat, zoom)
    xs = range(min(x0, x1), max(x0, x1) + 1)
    ys = range(min(y0, y1), max(y0, y1) + 1)

    mosaic = Image.new("RGB", (len(xs) * 256, len(ys) * 256), "white")
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            tile = fetch_carto_tile(x, y, zoom)
            mosaic.paste(tile, (ix * 256, iy * 256))

    minx, _, _, _ = tile_bounds_mercator(min(xs), min(ys), zoom)
    _, maxx, _, _ = tile_bounds_mercator(max(xs), min(ys), zoom)
    _, _, _, maxy = tile_bounds_mercator(min(xs), min(ys), zoom)
    _, _, miny, _ = tile_bounds_mercator(min(xs), max(ys), zoom)
    return mosaic, (minx, maxx, miny, maxy)


MIN_MARKER_AREA = 22.0
MAX_MARKER_AREA = 360.0


def size_from_rate(rate: float, max_rate: float | None = None) -> float:
    """Marker area in points^2, square-scaled to emphasize high mean rates."""
    rate = max(float(rate), 0.0)
    if max_rate is None or max_rate <= 0:
        max_rate = 40.0
    scaled = min(rate / max_rate, 1.0) ** 2
    return MIN_MARKER_AREA + scaled * (MAX_MARKER_AREA - MIN_MARKER_AREA)


def save_plos_tiff(fig, out_path: Path, *, dpi: int = 300):
    """Save flattened RGB LZW-compressed TIFF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.png")
    fig.savefig(tmp, dpi=dpi, facecolor="white", edgecolor="white", bbox_inches="tight", pad_inches=0.02)
    with Image.open(tmp) as im:
        rgb = im.convert("RGB")
        rgb.save(out_path, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
    tmp.unlink()


def add_scale_bar(ax, *, length_km: float = 50.0, mean_lat: float):
    """Add an approximate Web Mercator scale bar corrected at the map's mean latitude."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    width = x1 - x0
    height = y1 - y0
    # Shift slightly west so the right-hand label has clean breathing room.
    west_shift_m = 15_000.0 / np.cos(np.deg2rad(mean_lat))
    south_shift_m = 5_000.0 / np.cos(np.deg2rad(mean_lat))
    bar_x = x0 + width * 0.68 - west_shift_m
    bar_y = y0 + height * 0.035 - south_shift_m
    bar_height = height * 0.007
    bar_width = length_km * 1000.0 / np.cos(np.deg2rad(mean_lat))
    half_width = bar_width / 2.0

    ax.add_patch(Rectangle((bar_x, bar_y), half_width, bar_height, facecolor="black", edgecolor="black", zorder=9))
    ax.add_patch(Rectangle((bar_x + half_width, bar_y), half_width, bar_height, facecolor="white", edgecolor="black", zorder=9))
    ax.plot([bar_x, bar_x], [bar_y, bar_y + bar_height * 1.8], color="black", linewidth=0.7, zorder=10)
    ax.plot([bar_x + bar_width, bar_x + bar_width], [bar_y, bar_y + bar_height * 1.8], color="black", linewidth=0.7, zorder=10)
    ax.text(
        bar_x + bar_width + width * 0.012,
        bar_y + bar_height / 2.0,
        f"{length_km:g} km",
        ha="left",
        va="center",
        fontsize=8,
        color="black",
        zorder=10,
    )


def add_north_arrow(ax):
    """Add a simple north arrow in the upper-right map corner."""
    ax.annotate(
        "",
        xy=(0.935, 0.955),
        xytext=(0.935, 0.875),
        xycoords="axes fraction",
        arrowprops={"facecolor": "black", "edgecolor": "black", "width": 2.0, "headwidth": 9.0, "headlength": 10.0},
        zorder=10,
    )
    ax.text(
        0.935,
        0.965,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="black",
        zorder=10,
    )


def make_figure(plot_df: pd.DataFrame) -> plt.Figure:
    """Render Figure 1 draft."""
    mpl.rcParams.update({
        # Arial is PLOS-accepted, but this workstation currently aliases it outside matplotlib.
        # Use Liberation Sans for draft rendering; switch to true Arial or Times for final submission if needed.
        "font.family": "Liberation Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
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
    minx, miny = lonlat_to_mercator(bounds_lonlat[0], bounds_lonlat[2])
    maxx, maxy = lonlat_to_mercator(bounds_lonlat[1], bounds_lonlat[3])

    basemap, basemap_extent = load_carto_basemap(bounds_lonlat, zoom=8)
    ax.imshow(basemap, extent=basemap_extent, origin="upper", zorder=1)
    max_mean_rate = float(plot_df["EncounterRateHr"].max())

    # Plot no-eelgrass first, then eelgrass on top for visibility.
    for has_eelgrass, label, color, zorder in [
        (False, "No eelgrass recorded", STARMEADOW_COLORS["no eelgrass"], 3),
        (True, "Eelgrass recorded", STARMEADOW_COLORS["eelgrass"], 4),
    ]:
        subset = plot_df[plot_df["HasEelgrass"] == has_eelgrass]
        x, y = lonlat_to_mercator(subset["Long"], subset["Lat"])
        ax.scatter(
            x,
            y,
            s=subset["EncounterRateHr"].map(lambda rate: size_from_rate(rate, max_mean_rate)),
            c=color,
            edgecolor="black",
            linewidth=0.35,
            alpha=0.82,
            label=f"{label} (n={len(subset)})",
            zorder=zorder,
        )

    ax.set_aspect("equal")
    ax.set_xlim(float(minx), float(maxx))
    ax.set_ylim(float(miny), float(maxy))

    lon_ticks = np.arange(np.ceil(bounds_lonlat[0] * 2) / 2, bounds_lonlat[1], 0.5)
    lat_ticks = np.arange(np.ceil(bounds_lonlat[2] * 2) / 2, bounds_lonlat[3], 0.5)
    x_ticks, _ = lonlat_to_mercator(lon_ticks, np.full_like(lon_ticks, plot_df["Lat"].mean()))
    _, y_ticks = lonlat_to_mercator(np.full_like(lat_ticks, plot_df["Long"].mean()), lat_ticks)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{tick:.1f}°" for tick in lon_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{tick:.1f}°" for tick in lat_ticks])

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#6B6B6B")

    add_scale_bar(ax, length_km=50, mean_lat=float(plot_df["Lat"].mean()))
    add_north_arrow(ax)

    rate_values = [0, 1, 5, 15, 30, max_mean_rate]
    rate_values = sorted(set(round(rate, 1) for rate in rate_values if rate <= max_mean_rate or np.isclose(rate, max_mean_rate)))
    size_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="", color="black",
            markerfacecolor="white", markeredgecolor="black",
            markersize=np.sqrt(size_from_rate(rate, max_mean_rate)),
            label=f"{rate:g}",
        )
        for rate in rate_values
    ]
    legend_handles = [
        Patch(facecolor=STARMEADOW_COLORS["eelgrass"], edgecolor="black", label="Eelgrass recorded"),
        Patch(facecolor=STARMEADOW_COLORS["no eelgrass"], edgecolor="black", label="No eelgrass recorded"),
        Line2D([], [], linestyle="", label=""),
        Line2D([], [], linestyle="", label="Mean encounter rate (Pycno hr$^{-1}$)"),
        *size_handles,
    ]
    legend = ax.legend(
        handles=legend_handles,
        title="Site status and encounter rate",
        loc="lower left",
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

    plot_df, missing = load_site_data()
    excluded = plot_df[plot_df["Lat"] >= MAP_MAX_LAT].copy()
    plot_df = plot_df[plot_df["Lat"] < MAP_MAX_LAT].copy()
    plot_df.to_csv(QC_DIR / "Fig1_site_map_plotted_sites.csv", index=False)
    excluded.to_csv(QC_DIR / "Fig1_site_map_excluded_north_vancouver_island_sites.csv", index=False)
    missing.to_csv(QC_DIR / "Fig1_site_map_survey_sites_missing_coordinates.csv", index=False)

    fig = make_figure(plot_df)
    png = SOURCE_DIR / "Fig1_site_map_draft.png"
    pdf = SOURCE_DIR / "Fig1_site_map_draft.pdf"
    tif = SUBMISSION_DIR / "Fig1.tif"
    fig.savefig(png, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    save_plos_tiff(fig, tif, dpi=300)
    plt.close(fig)

    print(f"Saved draft PNG: {png}")
    print(f"Saved editable PDF: {pdf}")
    print(f"Saved PLOS-style TIFF: {tif}")
    print(f"Plotted sites: {len(plot_df)}")
    print(f"Excluded north Vancouver Island sites: {len(excluded)}")
    if len(excluded):
        print(f"Excluded-site report: {QC_DIR / 'Fig1_site_map_excluded_north_vancouver_island_sites.csv'}")
    print(f"Survey sites missing coordinates: {len(missing)}")
    if len(missing):
        print(f"Missing-coordinate report: {QC_DIR / 'Fig1_site_map_survey_sites_missing_coordinates.csv'}")


if __name__ == "__main__":
    main()
