"""
Publication Figures for Weertman Sections

Generates four manuscript-ready figures:
  Fig 1 - Map of survey sites by habitat type (with coastline)
  Fig 2 - Encounter rate by habitat type across basins (panel)
  Fig 3 - Reef encounter rates with/without eelgrass at site
  Fig 4 - Size structure by habitat type across basins (panel)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import seaborn as sns
import geopandas as gpd

from utils import (
    get_output_dir, load_data, load_length_data_individuals,
    save_figure, DATA_DIR
)

OUTPUT_DIR = get_output_dir(__file__)

# ── Palette & constants ──────────────────────────────────────────────────────

HABITAT_ORDER = ["Eelgrass", "Soft Bottom", "Artificial Reef",
                 "Kelp Forest", "Natural Reef"]

HABITAT_COLORS = {
    "Eelgrass":       "#1b9e77",
    "Soft Bottom":    "#d95f02",
    "Artificial Reef":"#7570b3",
    "Kelp Forest":    "#66a61e",
    "Natural Reef":   "#e7298a",
}

HABITAT_MARKERS = {
    "Eelgrass": "o", "Soft Bottom": "s", "Artificial Reef": "D",
    "Kelp Forest": "^", "Natural Reef": "v",
}

BASIN_ORDER = [
    "Whidbey", "Hood Canal", "Strait of Juan de Fuca",
    "Admiralty Inlet", "Central", "San Juan", "South",
    "Howe Sound", "Queen Charlotte Strait", "Strait of Georgia",
]

FOUR_CAT_ORDER = [
    "Eelgrass", "Hard Bottom\n(Eelgrass Site)",
    "Soft Bottom", "Hard Bottom\n(No Eelgrass)",
]

FOUR_CAT_COLORS = [
    "#1b9e77", "#a6dba0", "#d95f02", "#c2a5cf",
]


def pub_style():
    """Configure matplotlib for clean publication figures."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.facecolor":"white",
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "figure.titlesize": 11,
    })


def load_coastline():
    """Load WA ShoreZone coastline reprojected to WGS84."""
    gdb = DATA_DIR / "state_DNR_ShoreZone" / "shorezone.gdb"
    shore = gpd.read_file(gdb, layer="szline")
    return shore.to_crs(epsg=4326)


def load_site_coordinates():
    coords = pd.read_csv(DATA_DIR / "Site_LatLong.csv")
    coords.columns = ["SiteName", "Lat", "Long"]
    coords.loc[coords["Long"] > 0, "Long"] *= -1
    return coords


def add_four_category(df):
    eelgrass_sites = df.groupby("SiteName")["HabitatType"].transform(
        lambda x: (x == "Eelgrass").any()
    )
    hard = df["HabitatType"].isin(
        ["Natural Reef", "Artificial Reef", "Kelp Forest", "Sponge Garden"]
    )
    df["Habitat_4Cat"] = np.select(
        [
            df["HabitatType"] == "Eelgrass",
            hard & eelgrass_sites,
            df["HabitatType"] == "Soft Bottom",
            hard & ~eelgrass_sites,
        ],
        FOUR_CAT_ORDER,
        default="Other",
    )
    return df


# ── FIGURE 1 ─────────────────────────────────────────────────────────────────

def figure1_site_map(df, coords, coastline):
    site_info = df.groupby("SiteName").agg(
        Basin=("Basin", "first"),
        Habitats=("HabitatType", lambda x: set(x.unique()) - {"Sponge Garden"}),
    ).reset_index()
    site_info = site_info.merge(coords, on="SiteName", how="left").dropna(subset=["Lat"])

    def dominant(habs):
        for h in HABITAT_ORDER:
            if h in habs:
                return h
        return list(habs)[0] if habs else "Natural Reef"

    site_info["DomHab"] = site_info["Habitats"].apply(dominant)
    site_info["HasEG"] = site_info["Habitats"].apply(lambda h: "Eelgrass" in h)

    fig, ax = plt.subplots(figsize=(8, 10))

    coastline.plot(ax=ax, color="#d9d9d9", linewidth=0.35, zorder=1)

    for _, r in site_info.iterrows():
        c = HABITAT_COLORS[r["DomHab"]]
        m = HABITAT_MARKERS[r["DomHab"]]
        ax.scatter(
            r["Long"], r["Lat"], s=40, c=c, marker=m,
            edgecolor="white" if not r["HasEG"] else "black",
            linewidth=0.4 if not r["HasEG"] else 1.3,
            alpha=0.92, zorder=3,
        )

    handles = [
        Line2D([], [], marker=HABITAT_MARKERS[h], color="w",
               markerfacecolor=HABITAT_COLORS[h], markersize=7,
               markeredgecolor="0.4", markeredgewidth=0.4, label=h)
        for h in HABITAT_ORDER
    ]
    handles.append(
        Line2D([], [], marker="o", color="w", markerfacecolor="white",
               markersize=8, markeredgecolor="black", markeredgewidth=1.3,
               label="Eelgrass present at site")
    )
    ax.legend(handles=handles, loc="upper left", title="Dominant habitat",
              title_fontsize=8, frameon=True, fancybox=False)

    ax.set_xlim(-124.9, -122.0)
    ax.set_ylim(46.9, 49.7)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_aspect("equal")
    ax.grid(True, linewidth=0.3, alpha=0.4, color="0.7")
    ax.set_title(f"Survey sites by habitat type  (n = {len(site_info)})", loc="left")

    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "fig1_site_map_by_habitat")
    plt.close()
    print("  Figure 1 done.")


# ── FIGURE 2 ─────────────────────────────────────────────────────────────────

def figure2_encounter_panel(df):
    df_plot = df[df["HabitatType"].isin(HABITAT_ORDER)].copy()

    top_basins = ["Whidbey", "Hood Canal", "Strait of Juan de Fuca",
                  "Admiralty Inlet", "Central", "San Juan"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharey=False)
    axes_flat = axes.flatten()

    def draw_bars(ax, data, title, show_ylabel=False):
        means = (
            data.groupby("HabitatType")["Encounter.Rate.Hr"]
            .agg(["mean", "sem", "size"])
            .reindex(HABITAT_ORDER)
        )
        present = means.dropna(subset=["mean"])
        if present.empty:
            ax.set_visible(False)
            return
        x = range(len(present))
        colors = [HABITAT_COLORS[h] for h in present.index]
        ax.bar(x, present["mean"], yerr=present["sem"], capsize=2.5,
               color=colors, edgecolor="white", linewidth=0.5, width=0.72)
        for i, (h, row) in enumerate(present.iterrows()):
            ax.text(i, row["mean"] + row["sem"] + ax.get_ylim()[1] * 0.02,
                    f'{int(row["size"])}', ha="center", fontsize=6.5, color="0.45")
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([h.replace(" ", "\n") for h in present.index],
                           fontsize=6.5, rotation=0, ha="center")
        ax.set_title(title, fontsize=9, pad=4)
        if show_ylabel:
            ax.set_ylabel("Encounter rate  (Pycno hr$^{-1}$)", fontsize=8)

    draw_bars(axes_flat[0], df_plot, "All basins", show_ylabel=True)

    for i, basin in enumerate(top_basins):
        bdata = df_plot[df_plot["Basin"] == basin]
        draw_bars(axes_flat[i + 1], bdata, basin,
                  show_ylabel=(i + 1) % 4 == 0)

    axes_flat[7].set_visible(False)

    fig.suptitle("Encounter rate by habitat type across Salish Sea basins",
                 fontsize=11, fontweight="bold", x=0.01, ha="left", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95], w_pad=1.8, h_pad=2.5)
    save_figure(fig, OUTPUT_DIR, "fig2_encounter_by_habitat_basin")
    plt.close()
    print("  Figure 2 done.")


# ── FIGURE 3 ─────────────────────────────────────────────────────────────────

def figure3_reef_eelgrass(df):
    df_plot = df[df["Habitat_4Cat"].isin(FOUR_CAT_ORDER)].copy()

    summary = (
        df_plot.groupby("Habitat_4Cat", sort=False)["Encounter.Rate.Hr"]
        .agg(["mean", "sem", "size"])
        .reindex(FOUR_CAT_ORDER)
    )
    detection = (
        df_plot.groupby("Habitat_4Cat", sort=False)["Pycnopodia_count"]
        .apply(lambda x: (x > 0).mean())
        .reindex(FOUR_CAT_ORDER)
    )

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    x = np.arange(len(summary))
    ax1.bar(x, summary["mean"], yerr=summary["sem"], capsize=3,
            color=FOUR_CAT_COLORS, edgecolor="white", linewidth=0.6, width=0.68)
    for i, (cat, row) in enumerate(summary.iterrows()):
        ax1.text(i, row["mean"] + row["sem"] + 0.22,
                 f'n = {int(row["size"])}', ha="center", fontsize=8, color="0.35")

    hb_eg = summary.loc["Hard Bottom\n(Eelgrass Site)", "mean"]
    hb_no = summary.loc["Hard Bottom\n(No Eelgrass)", "mean"]
    fold = hb_eg / hb_no if hb_no > 0 else 0

    mid_y = max(hb_eg, hb_no) + 1.2
    ax1.plot([1, 1, 3, 3], [hb_eg + 0.5, mid_y, mid_y, hb_no + 0.5],
             lw=1, color="0.3")
    ax1.text(2, mid_y + 0.15, f"{fold:.1f}\u00d7", ha="center", fontsize=10,
             fontweight="bold", color="#1b9e77")

    ax1.set_xticks(x)
    ax1.set_xticklabels(FOUR_CAT_ORDER, fontsize=8)
    ax1.set_ylabel("Mean encounter rate  (Pycno hr$^{-1}$)", fontsize=9)
    ax1.set_title("a)  Encounter rate", loc="left")

    ax2.bar(x, detection.values * 100, color=FOUR_CAT_COLORS,
            edgecolor="white", linewidth=0.6, width=0.68)
    for i, val in enumerate(detection.values):
        ax2.text(i, val * 100 + 0.6, f"{val*100:.1f}%", ha="center",
                 fontsize=8, color="0.35")
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.split("\n")[0] for c in FOUR_CAT_ORDER],
                        fontsize=7.5, rotation=30, ha="right")
    ax2.set_ylabel("Transects detecting Pycnopodia  (%)", fontsize=9)
    ax2.set_title("b)  Detection rate", loc="left")
    ax2.set_ylim(0, max(detection.values * 100) * 1.2)

    fig.suptitle("Effect of site-level eelgrass presence on reef encounter rates",
                 fontsize=11, fontweight="bold", x=0.01, ha="left", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, OUTPUT_DIR, "fig3_reef_eelgrass_effect")
    plt.close()
    print("  Figure 3 done.")


# ── FIGURE 4 ─────────────────────────────────────────────────────────────────

def figure4_size_panel(df_len):
    df_plot = df_len[df_len["HabitatType"].isin(HABITAT_ORDER)].copy()

    top_basins = ["Whidbey", "Hood Canal", "Strait of Juan de Fuca",
                  "Admiralty Inlet", "Central", "San Juan"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharey=True)
    axes_flat = axes.flatten()

    def draw_violins(ax, data, title, show_ylabel=False):
        present = [h for h in HABITAT_ORDER
                   if h in data["HabitatType"].unique()
                   and data.loc[data["HabitatType"] == h, "Length_cm"].size >= 3]
        if not present:
            ax.set_visible(False)
            return
        plot_data = []
        colors_used = []
        for h in present:
            vals = data.loc[data["HabitatType"] == h, "Length_cm"].dropna()
            plot_data.append(vals)
            colors_used.append(HABITAT_COLORS[h])

        parts = ax.violinplot(plot_data, positions=range(len(present)),
                              showmeans=False, showmedians=False, showextrema=False,
                              widths=0.72)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors_used[i])
            pc.set_edgecolor("white")
            pc.set_linewidth(0.5)
            pc.set_alpha(0.75)

        for i, vals in enumerate(plot_data):
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.vlines(i, q1, q3, color="black", linewidth=1.2, zorder=4)
            ax.scatter(i, med, color="white", s=14, zorder=5,
                       edgecolor="black", linewidth=0.6)
            ax.text(i, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 90,
                    f'{len(vals)}', ha="center", fontsize=6, color="0.45", va="top")

        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([h.replace(" ", "\n") for h in present],
                           fontsize=6.5, ha="center")
        ax.set_title(title, fontsize=9, pad=4)
        if show_ylabel:
            ax.set_ylabel("Body diameter  (cm)", fontsize=8)

    draw_violins(axes_flat[0], df_plot, "All basins", show_ylabel=True)

    for i, basin in enumerate(top_basins):
        bdata = df_plot[df_plot["Basin"] == basin]
        draw_violins(axes_flat[i + 1], bdata, basin,
                     show_ylabel=(i + 1) % 4 == 0)

    axes_flat[7].set_visible(False)

    fig.suptitle(
        "Size structure of  P. helianthoides  by habitat type across basins",
        fontsize=11, fontweight="bold", x=0.01, ha="left", y=0.98,
        fontstyle="italic",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95], w_pad=1.8, h_pad=2.5)
    save_figure(fig, OUTPUT_DIR, "fig4_size_by_habitat_basin")
    plt.close()
    print("  Figure 4 done.")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("PUBLICATION FIGURES  (Weertman sections)")
    print(f"{'='*60}\n")

    pub_style()

    print("Loading data...")
    df = load_data()
    df = add_four_category(df)
    coords = load_site_coordinates()
    df_len = load_length_data_individuals()

    print("Loading coastline...")
    coastline = load_coastline()

    print(f"\nOutput → {OUTPUT_DIR}\n")

    print("Figure 1  —  site map by habitat …")
    figure1_site_map(df, coords, coastline)

    print("Figure 2  —  encounter rate × habitat × basin …")
    figure2_encounter_panel(df)

    print("Figure 3  —  reef eelgrass four-category …")
    figure3_reef_eelgrass(df)

    print("Figure 4  —  size × habitat × basin …")
    figure4_size_panel(df_len)

    print(f"\n{'='*60}")
    print("DONE — all figures in", OUTPUT_DIR)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
