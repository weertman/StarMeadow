"""
Combined Size + Habitat Clustering Analysis

Creates three clustering visualizations for sites with Pycnopodia detections:

1. Combined Clustering: Cluster on both size distribution AND habitat composition
2. Size Clustering + Habitat Sidebar: Cluster on size, show habitat on right
3. Habitat Clustering + Size Sidebar: Cluster on habitat, show size on right

Requires both:
- Count data (PycnoCountClean): For habitat composition
- Length data (PycnoLengthClean): For size distributions

Only includes sites that have BOTH habitat survey data AND size measurements.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms

from utils import (
    get_output_dir, load_data, load_length_data_individuals,
    set_style, save_figure
)

OUTPUT_DIR = get_output_dir(__file__)

# Even-width size bins (5 cm increments)
SIZE_BIN_WIDTH = 5
MAX_SIZE = 90
SIZE_BINS = list(range(0, MAX_SIZE + SIZE_BIN_WIDTH + 1, SIZE_BIN_WIDTH))
SIZE_LABELS = [f"{SIZE_BINS[i]}-{SIZE_BINS[i+1]}" for i in range(len(SIZE_BINS)-1)]


def compute_site_habitat_proportions(df_count: pd.DataFrame) -> pd.DataFrame:
    """
    Compute habitat composition proportions for each site.
    
    Returns DataFrame with rows=sites, columns=habitat type proportions.
    """
    # Habitat composition ratios per site
    habitat_ratios = (
        df_count.groupby(["SiteName", "HabitatType"])
        .size()
        .reset_index(name="Count")
    )
    habitat_ratios["Ratio"] = habitat_ratios.groupby("SiteName")["Count"].transform(
        lambda x: x / x.sum()
    )
    
    # Pivot to wide format
    habitat_matrix = habitat_ratios.pivot_table(
        index="SiteName",
        columns="HabitatType",
        values="Ratio",
        fill_value=0
    )
    
    # Add prefix to columns
    habitat_matrix.columns = [f"Hab_{c}" for c in habitat_matrix.columns]
    
    return habitat_matrix


def compute_site_size_proportions(df_length: pd.DataFrame) -> pd.DataFrame:
    """
    Compute size bin proportions for each site.
    
    Returns DataFrame with rows=sites, columns=size bin proportions.
    """
    # Add size bins
    df_length = df_length.copy()
    df_length["Size_Bin"] = pd.cut(
        df_length["Length_cm"],
        bins=SIZE_BINS,
        labels=SIZE_LABELS,
        include_lowest=True,
        right=False
    )
    
    # Size bin proportions per site
    size_ratios = (
        df_length.groupby(["SiteName", "Size_Bin"], observed=False)
        .size()
        .reset_index(name="Count")
    )
    size_ratios["Ratio"] = size_ratios.groupby("SiteName")["Count"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0
    )
    
    # Pivot to wide format
    size_matrix = size_ratios.pivot_table(
        index="SiteName",
        columns="Size_Bin",
        values="Ratio",
        fill_value=0
    )
    
    # Add prefix to columns
    size_matrix.columns = [f"Size_{c}" for c in size_matrix.columns]
    
    return size_matrix


def compute_site_metadata(df_count: pd.DataFrame, df_length: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metadata for each site (eelgrass presence, mean encounter rate, mean size).
    """
    # Site eelgrass indicator
    site_eelgrass = df_count.groupby("SiteName")["HabitatType"].apply(
        lambda x: (x == "Eelgrass").any()
    )
    
    # Mean encounter rate per site
    encounter_rate = df_count.groupby("SiteName")["Encounter.Rate.Hr"].mean()
    
    # Mean size per site
    mean_size = df_length.groupby("SiteName")["Length_cm"].mean()
    
    # N individuals per site
    n_individuals = df_length.groupby("SiteName").size()
    
    metadata = pd.DataFrame({
        "Site_Has_Eelgrass": site_eelgrass,
        "Mean_Encounter_Rate": encounter_rate,
        "Mean_Size": mean_size,
        "N_Individuals": n_individuals
    })
    
    return metadata


def plot_combined_clustering(
    combined_matrix: pd.DataFrame,
    habitat_cols: list,
    size_cols: list,
    metadata: pd.DataFrame,
    min_individuals: int = 3
):
    """
    Visualization 1: Cluster on both size distribution AND habitat composition.
    """
    # Filter to sites with enough data
    valid_sites = metadata[metadata["N_Individuals"] >= min_individuals].index
    matrix = combined_matrix.loc[combined_matrix.index.isin(valid_sites)].copy()
    meta = metadata.loc[matrix.index]
    
    if len(matrix) < 5:
        print("  Not enough sites for combined clustering")
        return
    
    print(f"  Combined clustering: {len(matrix)} sites")
    
    # Cluster
    sns.set(style="whitegrid", font_scale=0.5)
    link = linkage(matrix.fillna(0), method="ward")
    
    # Row colors by eelgrass
    row_colors = meta["Site_Has_Eelgrass"].map({True: "#2ecc71", False: "#e74c3c"})
    
    # Column colors: habitat vs size
    col_colors = []
    for col in matrix.columns:
        if col.startswith("Hab_"):
            col_colors.append("#1f78b4")  # Blue for habitat
        else:
            col_colors.append("#33a02c")  # Green for size
    
    # Create clustermap
    g = sns.clustermap(
        matrix,
        row_linkage=link,
        col_cluster=False,
        cmap="YlOrRd",
        figsize=(14, max(10, len(matrix) * 0.25)),
        vmin=0, vmax=1,
        dendrogram_ratio=(0.1, 0.05),
        row_colors=row_colors,
        col_colors=col_colors,
        cbar_pos=(0.02, 0.8, 0.02, 0.15),
        xticklabels=True,
        yticklabels=True
    )
    
    # Format labels
    g.ax_heatmap.set_xticklabels(
        [c.replace("Hab_", "").replace("Size_", "") for c in matrix.columns],
        rotation=90, fontsize=6
    )
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), fontsize=5
    )
    
    g.fig.suptitle(
        "Combined Clustering: Size Distribution + Habitat Composition\n"
        "(Blue cols = Habitat, Green cols = Size; Sidebar: Green = Eelgrass Site)",
        y=1.02, fontsize=10
    )
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Eelgrass at Site"),
        Patch(facecolor="#e74c3c", label="No Eelgrass"),
        Patch(facecolor="#1f78b4", label="Habitat Features"),
        Patch(facecolor="#33a02c", label="Size Features"),
    ]
    g.ax_heatmap.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=6
    )
    
    plt.subplots_adjust(bottom=0.15)
    
    path = OUTPUT_DIR / "clustermap_combined_size_habitat.png"
    g.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_size_clustering_with_habitat_sidebar(
    size_matrix: pd.DataFrame,
    habitat_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    min_individuals: int = 3
):
    """
    Visualization 2: Cluster on size, show habitat distribution on right.
    """
    # Filter and align
    valid_sites = metadata[metadata["N_Individuals"] >= min_individuals].index
    common_sites = size_matrix.index.intersection(habitat_matrix.index).intersection(valid_sites)
    
    size_mat = size_matrix.loc[common_sites].copy()
    hab_mat = habitat_matrix.loc[common_sites].copy()
    meta = metadata.loc[common_sites]
    
    if len(size_mat) < 5:
        print("  Not enough sites for size clustering with habitat sidebar")
        return
    
    print(f"  Size clustering + habitat sidebar: {len(size_mat)} sites")
    
    # Cluster on size only
    link = linkage(size_mat.fillna(0), method="ward")
    
    # Get clustering order
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(link)
    ordered_sites = size_mat.index[order]
    
    # Reorder all matrices
    size_ordered = size_mat.loc[ordered_sites]
    hab_ordered = hab_mat.loc[ordered_sites]
    meta_ordered = meta.loc[ordered_sites]
    
    # Create figure with two heatmaps
    fig = plt.figure(figsize=(16, max(10, len(size_mat) * 0.3)))
    
    # Grid: dendrogram, row colors, size heatmap, habitat heatmap
    gs = fig.add_gridspec(1, 5, width_ratios=[0.1, 0.03, 0.5, 0.02, 0.35], wspace=0.02)
    
    ax_dendro = fig.add_subplot(gs[0, 0])
    ax_rowcol = fig.add_subplot(gs[0, 1])
    ax_size = fig.add_subplot(gs[0, 2])
    ax_gap = fig.add_subplot(gs[0, 3])
    ax_hab = fig.add_subplot(gs[0, 4])
    
    # Dendrogram
    from scipy.cluster.hierarchy import dendrogram
    dendro = dendrogram(link, orientation='left', ax=ax_dendro, no_labels=True,
                        color_threshold=0, above_threshold_color='gray')
    ax_dendro.set_axis_off()
    
    # Row colors
    colors = meta_ordered["Site_Has_Eelgrass"].map({True: "#2ecc71", False: "#e74c3c"}).values
    for i, c in enumerate(colors):
        ax_rowcol.add_patch(Rectangle((0, i), 1, 1, facecolor=c, edgecolor='none'))
    ax_rowcol.set_xlim(0, 1)
    ax_rowcol.set_ylim(0, len(colors))
    ax_rowcol.set_axis_off()
    
    # Size heatmap (main clustering)
    sns.heatmap(
        size_ordered,
        ax=ax_size,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        cbar=False,
        xticklabels=[c.replace("Size_", "") for c in size_ordered.columns],
        yticklabels=size_ordered.index
    )
    ax_size.set_xlabel("Size Bin (cm)", fontsize=9)
    ax_size.set_ylabel("")
    ax_size.tick_params(axis='x', rotation=90, labelsize=6)
    ax_size.tick_params(axis='y', labelsize=5)
    ax_size.set_title("Size Distribution (Clustered)", fontsize=10)
    
    # Gap
    ax_gap.set_axis_off()
    
    # Habitat sidebar (not clustered, follows size order)
    sns.heatmap(
        hab_ordered,
        ax=ax_hab,
        cmap="Greens",
        vmin=0, vmax=1,
        cbar=True,
        cbar_kws={"shrink": 0.5, "label": "Proportion"},
        xticklabels=[c.replace("Hab_", "") for c in hab_ordered.columns],
        yticklabels=False
    )
    ax_hab.set_xlabel("Habitat Type", fontsize=9)
    ax_hab.set_ylabel("")
    ax_hab.tick_params(axis='x', rotation=90, labelsize=7)
    ax_hab.set_title("Habitat Composition", fontsize=10)
    
    fig.suptitle(
        "Size Distribution Clustering with Habitat Sidebar\n"
        "(Clustered by size; Green sidebar = Eelgrass Site)",
        fontsize=11, y=1.02
    )
    
    path = OUTPUT_DIR / "clustermap_size_with_habitat_sidebar.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_habitat_clustering_with_size_sidebar(
    habitat_matrix: pd.DataFrame,
    size_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    min_individuals: int = 3
):
    """
    Visualization 3: Cluster on habitat, show size distribution on right.
    """
    # Filter and align
    valid_sites = metadata[metadata["N_Individuals"] >= min_individuals].index
    common_sites = habitat_matrix.index.intersection(size_matrix.index).intersection(valid_sites)
    
    hab_mat = habitat_matrix.loc[common_sites].copy()
    size_mat = size_matrix.loc[common_sites].copy()
    meta = metadata.loc[common_sites]
    
    if len(hab_mat) < 5:
        print("  Not enough sites for habitat clustering with size sidebar")
        return
    
    print(f"  Habitat clustering + size sidebar: {len(hab_mat)} sites")
    
    # Cluster on habitat only
    link = linkage(hab_mat.fillna(0), method="ward")
    
    # Get clustering order
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(link)
    ordered_sites = hab_mat.index[order]
    
    # Reorder all matrices
    hab_ordered = hab_mat.loc[ordered_sites]
    size_ordered = size_mat.loc[ordered_sites]
    meta_ordered = meta.loc[ordered_sites]
    
    # Create figure
    fig = plt.figure(figsize=(18, max(10, len(hab_mat) * 0.3)))
    
    # Grid: dendrogram, row colors, habitat heatmap, size heatmap
    gs = fig.add_gridspec(1, 5, width_ratios=[0.1, 0.03, 0.25, 0.02, 0.6], wspace=0.02)
    
    ax_dendro = fig.add_subplot(gs[0, 0])
    ax_rowcol = fig.add_subplot(gs[0, 1])
    ax_hab = fig.add_subplot(gs[0, 2])
    ax_gap = fig.add_subplot(gs[0, 3])
    ax_size = fig.add_subplot(gs[0, 4])
    
    # Dendrogram
    from scipy.cluster.hierarchy import dendrogram
    dendro = dendrogram(link, orientation='left', ax=ax_dendro, no_labels=True,
                        color_threshold=0, above_threshold_color='gray')
    ax_dendro.set_axis_off()
    
    # Row colors
    colors = meta_ordered["Site_Has_Eelgrass"].map({True: "#2ecc71", False: "#e74c3c"}).values
    for i, c in enumerate(colors):
        ax_rowcol.add_patch(Rectangle((0, i), 1, 1, facecolor=c, edgecolor='none'))
    ax_rowcol.set_xlim(0, 1)
    ax_rowcol.set_ylim(0, len(colors))
    ax_rowcol.set_axis_off()
    
    # Habitat heatmap (main clustering)
    sns.heatmap(
        hab_ordered,
        ax=ax_hab,
        cmap="Greens",
        vmin=0, vmax=1,
        cbar=False,
        xticklabels=[c.replace("Hab_", "") for c in hab_ordered.columns],
        yticklabels=hab_ordered.index
    )
    ax_hab.set_xlabel("Habitat Type", fontsize=9)
    ax_hab.set_ylabel("")
    ax_hab.tick_params(axis='x', rotation=90, labelsize=7)
    ax_hab.tick_params(axis='y', labelsize=5)
    ax_hab.set_title("Habitat Composition (Clustered)", fontsize=10)
    
    # Gap
    ax_gap.set_axis_off()
    
    # Size sidebar (not clustered, follows habitat order)
    # Only show non-empty size bins
    non_empty_cols = size_ordered.columns[size_ordered.sum() > 0]
    size_display = size_ordered[non_empty_cols]
    
    sns.heatmap(
        size_display,
        ax=ax_size,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        cbar=True,
        cbar_kws={"shrink": 0.5, "label": "Proportion"},
        xticklabels=[c.replace("Size_", "") for c in size_display.columns],
        yticklabels=False
    )
    ax_size.set_xlabel("Size Bin (cm)", fontsize=9)
    ax_size.set_ylabel("")
    ax_size.tick_params(axis='x', rotation=90, labelsize=6)
    ax_size.set_title("Size Distribution", fontsize=10)
    
    fig.suptitle(
        "Habitat Composition Clustering with Size Sidebar\n"
        "(Clustered by habitat; Green sidebar = Eelgrass Site)",
        fontsize=11, y=1.02
    )
    
    path = OUTPUT_DIR / "clustermap_habitat_with_size_sidebar.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def main():
    print(f"\n{'='*60}")
    print("COMBINED SIZE + HABITAT CLUSTERING ANALYSIS")
    print(f"{'='*60}\n")
    
    set_style()
    
    # Load data
    print("Loading count data...")
    df_count = load_data()
    print(f"  {len(df_count)} survey records from {df_count['SiteName'].nunique()} sites")
    
    print("\nLoading length data...")
    df_length = load_length_data_individuals()
    print(f"  {df_length['SiteName'].nunique()} sites with size data")
    
    # Compute site-level matrices
    print("\nComputing site-level proportions...")
    habitat_matrix = compute_site_habitat_proportions(df_count)
    print(f"  Habitat matrix: {habitat_matrix.shape}")
    
    size_matrix = compute_site_size_proportions(df_length)
    print(f"  Size matrix: {size_matrix.shape}")
    
    # Compute metadata
    metadata = compute_site_metadata(df_count, df_length)
    
    # Find common sites
    common_sites = (
        habitat_matrix.index
        .intersection(size_matrix.index)
        .intersection(metadata.index)
    )
    print(f"\nSites with both habitat AND size data: {len(common_sites)}")
    
    # Combine matrices for joint clustering
    combined_matrix = pd.concat([
        habitat_matrix.loc[common_sites],
        size_matrix.loc[common_sites]
    ], axis=1)
    
    print(f"Combined matrix shape: {combined_matrix.shape}")
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    # Generate visualizations
    habitat_cols = [c for c in combined_matrix.columns if c.startswith("Hab_")]
    size_cols = [c for c in combined_matrix.columns if c.startswith("Size_")]
    
    print("Generating visualizations...")
    
    # 1. Combined clustering
    print("\n1. Combined Size + Habitat Clustering:")
    plot_combined_clustering(
        combined_matrix, habitat_cols, size_cols, metadata, min_individuals=3
    )
    
    # 2. Size clustering with habitat sidebar
    print("\n2. Size Clustering with Habitat Sidebar:")
    plot_size_clustering_with_habitat_sidebar(
        size_matrix, habitat_matrix, metadata, min_individuals=3
    )
    
    # 3. Habitat clustering with size sidebar
    print("\n3. Habitat Clustering with Size Sidebar:")
    plot_habitat_clustering_with_size_sidebar(
        habitat_matrix, size_matrix, metadata, min_individuals=3
    )
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()






