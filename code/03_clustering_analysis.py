"""
Hierarchical Clustering Analysis

Clusters sites and depth bins based on habitat composition,
with encounter rate overlays.

Two aggregation methods:
1. Site + Depth-Bin: Captures vertical habitat variation within sites
2. Site-Only: Emphasizes spatial differences among sites

Outputs:
- Clustergrams with habitat composition heatmaps
- Encounter rate overlays
- Separate colorbars for publication
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms

from utils import get_output_dir, load_data, set_style, save_figure

OUTPUT_DIR = get_output_dir(__file__)


def compute_habitat_matrix_by_site_depth(df: pd.DataFrame):
    """
    Compute habitat composition matrix aggregated by (SiteName, DepthBin).
    
    Returns:
        habitat_matrix: DataFrame with rows=(site, depth), columns=habitat types
        encounter_means: Series of mean encounter rates aligned to matrix index
    """
    # Habitat composition ratios
    habitat_ratios = (
        df.groupby(["SiteName", "DepthBin", "HabitatType"])
        .size()
        .reset_index(name="Count")
    )
    habitat_ratios["Ratio"] = habitat_ratios.groupby(
        ["SiteName", "DepthBin"]
    )["Count"].transform(lambda x: x / x.sum())
    
    # Pivot to wide format
    habitat_matrix = habitat_ratios.pivot_table(
        index=["SiteName", "DepthBin"],
        columns="HabitatType",
        values="Ratio",
        fill_value=0
    )
    
    # Mean encounter rate per site-depth
    encounter_means = df.groupby(["SiteName", "DepthBin"])["Encounter.Rate.Hr"].mean()
    encounter_means = encounter_means.reindex(habitat_matrix.index)
    
    return habitat_matrix, encounter_means


def compute_habitat_matrix_by_site(df: pd.DataFrame):
    """
    Compute habitat composition matrix aggregated by SiteName only.
    
    Returns:
        habitat_matrix: DataFrame with rows=sites, columns=habitat types
        encounter_means: Series of mean encounter rates aligned to matrix index
    """
    # Habitat composition ratios (collapse depth bins)
    habitat_ratios = (
        df.groupby(["SiteName", "HabitatType"])
        .size()
        .reset_index(name="Count")
    )
    habitat_ratios["Ratio"] = habitat_ratios.groupby(
        "SiteName"
    )["Count"].transform(lambda x: x / x.sum())
    
    # Pivot to wide format
    habitat_matrix = habitat_ratios.pivot_table(
        index="SiteName",
        columns="HabitatType",
        values="Ratio",
        fill_value=0
    )
    
    # Mean encounter rate per site
    encounter_means = df.groupby("SiteName")["Encounter.Rate.Hr"].mean()
    encounter_means = encounter_means.reindex(habitat_matrix.index)
    
    return habitat_matrix, encounter_means


def plot_clustermap_with_encounter(
    habitat_matrix: pd.DataFrame,
    encounter_means: pd.Series,
    title: str,
    filename: str,
    figsize=(6, 20),
    linkage_method="ward",
    cmap_main="Greens",
    cmap_encounter="Reds",
    label_format=None
):
    """
    Create clustermap with encounter rate rectangles.
    
    Args:
        habitat_matrix: Wide-format habitat composition matrix
        encounter_means: Mean encounter rates aligned to matrix index
        title: Figure title
        filename: Output filename (without extension)
        figsize: Figure size tuple
        linkage_method: Clustering linkage method
        cmap_main: Colormap for habitat ratios
        cmap_encounter: Colormap for encounter rate rectangles
        label_format: Function to format row labels (optional)
    """
    sns.set(style="whitegrid", font_scale=0.6)
    
    # Compute linkage
    link = linkage(habitat_matrix, method=linkage_method)
    
    # Create clustermap (no colorbar - we'll add separate ones)
    g = sns.clustermap(
        habitat_matrix,
        row_linkage=link,
        col_cluster=False,
        cmap=cmap_main,
        figsize=figsize,
        vmin=0, vmax=1,
        dendrogram_ratio=(0.1, 0.1),
        cbar_pos=None
    )
    
    # Get ordered indices
    heatmap_ax = g.ax_heatmap
    ordered_inds = g.dendrogram_row.reordered_ind
    ordered_index = habitat_matrix.index[ordered_inds]
    ordered_rates = encounter_means.loc[ordered_index]
    
    # Normalize encounter rates for coloring
    enc_min, enc_max = encounter_means.min(), encounter_means.max()
    norm_enc = plt.Normalize(enc_min, enc_max)
    cmap_enc = plt.get_cmap(cmap_encounter)
    
    # Add encounter rate rectangles
    xmax = len(habitat_matrix.columns)
    rect_xoffset = 0.1
    rect_width = 0.4
    rect_height = 1.0
    
    for i, rate in enumerate(ordered_rates):
        color = cmap_enc(norm_enc(rate)) if not np.isnan(rate) else (0.9, 0.9, 0.9, 1)
        rect = Rectangle(
            (xmax + rect_xoffset, i - rect_height / 2),
            rect_width, rect_height,
            facecolor=color,
            transform=heatmap_ax.transData,
            clip_on=False,
            edgecolor="none"
        )
        heatmap_ax.add_patch(rect)
    
    # Format tick labels
    n_rows = habitat_matrix.shape[0]
    heatmap_ax.set_yticks(np.arange(n_rows))
    
    if label_format:
        labels = [label_format(idx) for idx in ordered_index]
    else:
        labels = [str(idx) for idx in ordered_index]
    
    heatmap_ax.set_yticklabels(labels, fontsize=6)
    
    # Shift labels right
    trans = mtransforms.ScaledTranslation(20 / 72.0, 0, heatmap_ax.figure.dpi_scale_trans)
    for label in heatmap_ax.get_yticklabels():
        label.set_transform(label.get_transform() + trans)
    
    plt.subplots_adjust(left=0.25, right=0.9)
    
    # Save clustermap
    clustermap_path = OUTPUT_DIR / f"{filename}_clustermap.png"
    g.savefig(clustermap_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {clustermap_path}")
    plt.close()
    
    # Save separate colorbars
    save_colorbar(cmap_main, 0, 1, "Habitat Ratio (0-1)", f"{filename}_colorbar_habitat")
    save_colorbar(cmap_encounter, enc_min, enc_max, "Encounter Rate (per hr)", f"{filename}_colorbar_encounter")


def save_colorbar(cmap_name, vmin, vmax, label, filename):
    """Save a standalone colorbar figure."""
    fig, ax = plt.subplots(figsize=(1, 4))
    
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap_name))
    cb = plt.colorbar(sm, cax=ax, orientation="vertical")
    
    tick_values = np.linspace(vmin, vmax, num=6)
    cb.set_ticks(tick_values)
    cb.set_ticklabels([f"{t:.2f}" for t in tick_values])
    cb.set_label(label, fontsize=8)
    
    colorbar_path = OUTPUT_DIR / f"{filename}.png"
    fig.savefig(colorbar_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {colorbar_path}")
    plt.close()


def plot_simple_dendrogram(habitat_matrix: pd.DataFrame, title: str, filename: str):
    """Plot a simple dendrogram without heatmap."""
    fig, ax = plt.subplots(figsize=(6, 12))
    
    Z = linkage(habitat_matrix, method="ward")
    
    dendrogram(
        Z,
        labels=[str(idx) for idx in habitat_matrix.index],
        orientation="left",
        ax=ax,
        leaf_font_size=6
    )
    
    ax.set_title(title)
    ax.set_xlabel("Distance")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, filename)
    plt.close()


def main():
    print(f"\n{'='*60}")
    print("CLUSTERING ANALYSIS")
    print(f"{'='*60}\n")
    
    set_style()
    df = load_data()
    
    print(f"Loaded {len(df)} survey records")
    print(f"Unique sites: {df['SiteName'].nunique()}")
    print(f"Unique (site, depth) combinations: {df.groupby(['SiteName', 'DepthBin']).ngroups}")
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    # Site + Depth-Bin clustering
    print("Computing Site × Depth-Bin clustering...")
    habitat_matrix_sd, encounter_means_sd = compute_habitat_matrix_by_site_depth(df)
    print(f"  Matrix shape: {habitat_matrix_sd.shape}")
    
    plot_clustermap_with_encounter(
        habitat_matrix_sd,
        encounter_means_sd,
        title="Site × Depth-Bin Habitat Clustering",
        filename="site_depth_clustering",
        figsize=(6, 25),
        label_format=lambda x: f"{x[0]} ({x[1]})"
    )
    
    # Site-only clustering
    print("\nComputing Site-Only clustering...")
    habitat_matrix_s, encounter_means_s = compute_habitat_matrix_by_site(df)
    print(f"  Matrix shape: {habitat_matrix_s.shape}")
    
    plot_clustermap_with_encounter(
        habitat_matrix_s,
        encounter_means_s,
        title="Site Habitat Clustering",
        filename="site_only_clustering",
        figsize=(6, 15),
        label_format=str
    )
    
    # Simple dendrograms
    print("\nGenerating dendrograms...")
    plot_simple_dendrogram(
        habitat_matrix_s,
        "Site Clustering by Habitat Composition",
        "site_dendrogram"
    )
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()






