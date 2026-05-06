"""
Size Distribution Clustering Analysis

Analyzes Pycnopodia helianthoides size distributions and incorporates
size data into clustering visualizations.

Key analyses:
- Size distributions by habitat type
- Size distributions by site eelgrass presence
- Site-level size profile clustering
- Combined habitat composition + size clustering

Data source: PycnoLengthCLean_9_24_2024.csv
- Individual-level measurements (628 individuals with size data)
- Length in cm (range: 1-85 cm)

Outputs:
- Size distribution plots by habitat/basin/season
- Size-based site clustering
- Combined habitat + size clustering
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms

from utils import (
    get_output_dir, load_length_data, load_length_data_individuals, 
    set_style, save_figure
)

OUTPUT_DIR = get_output_dir(__file__)


# Even-width size bins (5 cm increments)
SIZE_BIN_WIDTH = 5
MAX_SIZE = 90  # Upper bound for bins
SIZE_BINS = list(range(0, MAX_SIZE + SIZE_BIN_WIDTH + 1, SIZE_BIN_WIDTH))  # 0, 5, 10, ..., 95
SIZE_LABELS = [f"{SIZE_BINS[i]}-{SIZE_BINS[i+1]}" for i in range(len(SIZE_BINS)-1)]


def add_size_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Add even-width size bin column."""
    df = df.copy()
    
    # Even-width size bins (5 cm increments)
    df["Size_Bin"] = pd.cut(
        df["Length_cm"], 
        bins=SIZE_BINS,
        labels=SIZE_LABELS,
        include_lowest=True,
        right=False
    )
    
    return df


def add_site_eelgrass_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add site eelgrass category labels."""
    df = df.copy()
    df["Site_Eelgrass_Category"] = df["Site_Has_Eelgrass"].map({
        True: "Eelgrass Present at Site",
        False: "No Eelgrass at Site"
    })
    return df


def plot_overall_size_distribution(df: pd.DataFrame):
    """Overall size distribution histogram with even bins."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with even 5cm bins
    ax1 = axes[0]
    bins = np.arange(0, df["Length_cm"].max() + SIZE_BIN_WIDTH, SIZE_BIN_WIDTH)
    ax1.hist(df["Length_cm"], bins=bins, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.axvline(df["Length_cm"].mean(), color="red", linestyle="--", 
                label=f"Mean: {df['Length_cm'].mean():.1f} cm")
    ax1.axvline(df["Length_cm"].median(), color="orange", linestyle="--",
                label=f"Median: {df['Length_cm'].median():.1f} cm")
    ax1.set_xlabel("Length (cm)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Pycnopodia Size Distribution ({SIZE_BIN_WIDTH} cm bins)")
    ax1.legend()
    
    # Size bin breakdown (even width)
    ax2 = axes[1]
    size_bin_counts = df["Size_Bin"].value_counts().sort_index()
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(size_bin_counts)))
    bars = ax2.bar(range(len(size_bin_counts)), size_bin_counts.values, 
                   color=colors, edgecolor="black")
    ax2.set_xticks(range(len(size_bin_counts)))
    ax2.set_xticklabels(size_bin_counts.index, rotation=45, ha="right")
    ax2.set_xlabel(f"Size Bin ({SIZE_BIN_WIDTH} cm)")
    ax2.set_ylabel("Count")
    ax2.set_title("Size Bin Distribution")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "overall_size_distribution")
    plt.close()


def plot_size_by_habitat(df: pd.DataFrame):
    """Size distribution by habitat type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Violin plot
    ax1 = axes[0]
    habitat_order = df.groupby("HabitatType")["Length_cm"].median().sort_values(ascending=False).index
    
    sns.violinplot(
        data=df,
        x="HabitatType",
        y="Length_cm",
        order=habitat_order,
        hue="HabitatType",
        palette="viridis",
        ax=ax1,
        inner="box"
    )
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()
    
    ax1.set_xlabel("Habitat Type")
    ax1.set_ylabel("Length (cm)")
    ax1.set_title("Size Distribution by Habitat Type")
    ax1.tick_params(axis="x", rotation=45)
    
    # Mean + SE bar plot
    ax2 = axes[1]
    habitat_stats = df.groupby("HabitatType")["Length_cm"].agg(["mean", "std", "count"])
    habitat_stats["se"] = habitat_stats["std"] / np.sqrt(habitat_stats["count"])
    habitat_stats = habitat_stats.loc[habitat_order]
    
    colors = sns.color_palette("viridis", len(habitat_stats))
    ax2.bar(
        range(len(habitat_stats)),
        habitat_stats["mean"],
        yerr=habitat_stats["se"],
        capsize=4,
        color=colors,
        edgecolor="black"
    )
    ax2.set_xticks(range(len(habitat_stats)))
    ax2.set_xticklabels(habitat_stats.index, rotation=45, ha="right")
    ax2.set_xlabel("Habitat Type")
    ax2.set_ylabel("Mean Length (cm)")
    ax2.set_title("Mean Size by Habitat Type (± SE)")
    
    for i, (idx, row) in enumerate(habitat_stats.iterrows()):
        ax2.annotate(f"n={int(row['count'])}", 
                     xy=(i, row["mean"] + row["se"] + 1),
                     ha="center", fontsize=8, color="gray")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "size_by_habitat")
    plt.close()
    
    return habitat_stats


def plot_size_by_eelgrass_site(df: pd.DataFrame):
    """Size distribution by site eelgrass presence."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histogram overlay with even bins
    ax1 = axes[0]
    bins = np.arange(0, df["Length_cm"].max() + SIZE_BIN_WIDTH, SIZE_BIN_WIDTH)
    for cat, color in [("Eelgrass Present at Site", "#2ecc71"), 
                       ("No Eelgrass at Site", "#e74c3c")]:
        subset = df[df["Site_Eelgrass_Category"] == cat]
        ax1.hist(subset["Length_cm"], bins=bins, alpha=0.5, color=color, 
                 label=f"{cat} (n={len(subset)})", edgecolor="black")
    
    ax1.set_xlabel("Length (cm)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Size Distribution ({SIZE_BIN_WIDTH} cm bins)")
    ax1.legend()
    
    # Kernel density plot for smooth comparison
    ax2 = axes[1]
    for cat, color in [("Eelgrass Present at Site", "#2ecc71"), 
                       ("No Eelgrass at Site", "#e74c3c")]:
        subset = df[df["Site_Eelgrass_Category"] == cat]
        sns.kdeplot(subset["Length_cm"], ax=ax2, color=color, label=cat, fill=True, alpha=0.3)
    
    ax2.set_xlabel("Length (cm)")
    ax2.set_ylabel("Density")
    ax2.set_title("Size Distribution (Kernel Density)")
    ax2.legend()
    
    # Size bin proportions (even width)
    ax3 = axes[2]
    size_by_site = df.groupby(["Site_Eelgrass_Category", "Size_Bin"]).size().unstack(fill_value=0)
    size_by_site_pct = size_by_site.div(size_by_site.sum(axis=1), axis=0) * 100
    
    # Use colormap for the stacked bars
    n_bins = len(size_by_site_pct.columns)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_bins))
    
    size_by_site_pct.plot(
        kind="bar",
        stacked=True,
        ax=ax3,
        color=colors,
        edgecolor="black",
        width=0.7
    )
    ax3.set_xlabel("")
    ax3.set_ylabel("Percentage")
    ax3.set_title(f"Size Bin Proportions ({SIZE_BIN_WIDTH} cm bins)")
    ax3.legend(title="Size Bin (cm)", bbox_to_anchor=(1.02, 1), fontsize=7)
    ax3.tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "size_by_eelgrass_site")
    plt.close()


def plot_size_by_season(df: pd.DataFrame):
    """Size distribution by season."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    df_season = df[df["Season"].isin(season_order)]
    
    # Violin plot
    ax1 = axes[0]
    sns.violinplot(
        data=df_season,
        x="Season",
        y="Length_cm",
        order=season_order,
        hue="Season",
        palette="coolwarm",
        ax=ax1,
        inner="box"
    )
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()
    ax1.set_xlabel("Season")
    ax1.set_ylabel("Length (cm)")
    ax1.set_title("Size Distribution by Season")
    
    # Mean size over months
    ax2 = axes[1]
    monthly = df.groupby("Month")["Length_cm"].agg(["mean", "std", "count"])
    monthly["se"] = monthly["std"] / np.sqrt(monthly["count"])
    
    ax2.errorbar(monthly.index, monthly["mean"], yerr=monthly["se"],
                 marker="o", capsize=4, color="steelblue", linewidth=2)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Mean Length (cm)")
    ax2.set_title("Mean Size by Month")
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "size_by_season")
    plt.close()


def compute_site_size_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute size distribution profiles for each site.
    
    Creates a matrix where rows are sites and columns are:
    - Mean, median, std of sizes
    - Proportion in each even-width size bin
    """
    # Basic size stats per site
    size_stats = df.groupby("SiteName")["Length_cm"].agg([
        "mean", "median", "std", "count"
    ]).rename(columns={
        "mean": "Mean_Size",
        "median": "Median_Size", 
        "std": "Size_Std",
        "count": "N_Individuals"
    })
    
    # Size bin proportions per site (even width bins)
    size_bin_props = df.groupby(["SiteName", "Size_Bin"]).size().unstack(fill_value=0)
    size_bin_props = size_bin_props.div(size_bin_props.sum(axis=1), axis=0)
    size_bin_props.columns = [f"Prop_{c}" for c in size_bin_props.columns]
    
    # Combine
    site_profiles = size_stats.join(size_bin_props)
    
    # Add site eelgrass indicator
    site_eelgrass = df.groupby("SiteName")["Site_Has_Eelgrass"].first()
    site_profiles["Site_Has_Eelgrass"] = site_eelgrass
    
    return site_profiles


def plot_site_size_clustering(df: pd.DataFrame):
    """
    Cluster sites based on size distribution profiles.
    """
    site_profiles = compute_site_size_profiles(df)
    
    # Filter to sites with enough data
    site_profiles = site_profiles[site_profiles["N_Individuals"] >= 3].copy()
    
    if len(site_profiles) < 5:
        print("  Not enough sites with size data for clustering")
        return
    
    # Prepare clustering matrix (size class proportions only)
    cluster_cols = [c for c in site_profiles.columns if c.startswith("Prop_")]
    cluster_matrix = site_profiles[cluster_cols].fillna(0)
    
    # Row colors by eelgrass
    row_colors = site_profiles["Site_Has_Eelgrass"].map({
        True: "#2ecc71", False: "#e74c3c"
    })
    
    # Cluster
    sns.set(style="whitegrid", font_scale=0.7)
    link = linkage(cluster_matrix, method="ward")
    
    g = sns.clustermap(
        cluster_matrix,
        row_linkage=link,
        col_cluster=False,
        cmap="YlOrRd",
        figsize=(8, max(8, len(cluster_matrix) * 0.4)),
        vmin=0, vmax=1,
        dendrogram_ratio=(0.15, 0.1),
        row_colors=row_colors,
        cbar_pos=(0.02, 0.8, 0.03, 0.15)
    )
    
    # Add mean size as rectangles
    heatmap_ax = g.ax_heatmap
    ordered_inds = g.dendrogram_row.reordered_ind
    ordered_index = cluster_matrix.index[ordered_inds]
    ordered_sizes = site_profiles.loc[ordered_index, "Mean_Size"]
    
    size_min, size_max = ordered_sizes.min(), ordered_sizes.max()
    norm_size = plt.Normalize(size_min, size_max)
    cmap_size = plt.get_cmap("Blues")
    
    xmax = len(cluster_matrix.columns)
    for i, size in enumerate(ordered_sizes):
        color = cmap_size(norm_size(size)) if not np.isnan(size) else (0.9, 0.9, 0.9, 1)
        rect = Rectangle(
            (xmax + 0.1, i - 0.5),
            0.4, 1.0,
            facecolor=color,
            transform=heatmap_ax.transData,
            clip_on=False,
            edgecolor="none"
        )
        heatmap_ax.add_patch(rect)
    
    # Labels
    n_rows = cluster_matrix.shape[0]
    heatmap_ax.set_yticks(np.arange(n_rows))
    heatmap_ax.set_yticklabels(ordered_index, fontsize=6)
    
    trans = mtransforms.ScaledTranslation(15 / 72.0, 0, heatmap_ax.figure.dpi_scale_trans)
    for lbl in heatmap_ax.get_yticklabels():
        lbl.set_transform(lbl.get_transform() + trans)
    
    g.fig.suptitle(
        "Site Clustering by Size Class Proportions\n"
        "(Green = Eelgrass Site, Red = No Eelgrass; Blue bars = Mean Size)",
        y=1.02, fontsize=10
    )
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Eelgrass at Site"),
        Patch(facecolor="#e74c3c", label="No Eelgrass at Site")
    ]
    g.ax_heatmap.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.15, 1.0),
        fontsize=7
    )
    
    plt.subplots_adjust(left=0.02, right=0.8)
    
    clustermap_path = OUTPUT_DIR / "clustermap_site_size_profiles.png"
    g.savefig(clustermap_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {clustermap_path}")
    plt.close()


def plot_size_habitat_interaction(df: pd.DataFrame):
    """
    Size distribution by habitat, split by site eelgrass presence.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate mean sizes
    interaction_stats = df.groupby(
        ["HabitatType", "Site_Eelgrass_Category"]
    )["Length_cm"].agg(["mean", "std", "count"]).reset_index()
    interaction_stats["se"] = interaction_stats["std"] / np.sqrt(interaction_stats["count"])
    
    habitats = df["HabitatType"].unique()
    x = np.arange(len(habitats))
    width = 0.35
    
    eelgrass_sites = interaction_stats[
        interaction_stats["Site_Eelgrass_Category"] == "Eelgrass Present at Site"
    ].set_index("HabitatType").reindex(habitats)
    
    no_eelgrass_sites = interaction_stats[
        interaction_stats["Site_Eelgrass_Category"] == "No Eelgrass at Site"
    ].set_index("HabitatType").reindex(habitats)
    
    bars1 = ax.bar(
        x - width/2,
        eelgrass_sites["mean"].fillna(0),
        width,
        yerr=eelgrass_sites["se"].fillna(0),
        capsize=3,
        label="Eelgrass Present at Site",
        color="#2ecc71",
        edgecolor="black"
    )
    
    bars2 = ax.bar(
        x + width/2,
        no_eelgrass_sites["mean"].fillna(0),
        width,
        yerr=no_eelgrass_sites["se"].fillna(0),
        capsize=3,
        label="No Eelgrass at Site",
        color="#e74c3c",
        edgecolor="black"
    )
    
    ax.set_xticks(x)
    ax.set_xticklabels(habitats, rotation=45, ha="right")
    ax.set_ylabel("Mean Length (cm)")
    ax.set_title("Mean Size by Habitat Type\n(Split by Site-Level Eelgrass Presence)")
    ax.legend()
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "size_habitat_eelgrass_interaction")
    plt.close()


def run_size_statistical_tests(df: pd.DataFrame):
    """Statistical tests for size differences."""
    print("\n" + "="*50)
    print("SIZE STATISTICAL TESTS")
    print("="*50)
    
    results = []
    
    # Test 1: Size by site eelgrass presence
    print("\n--- Test 1: Size by Site Eelgrass Presence ---")
    eelgrass_sizes = df[df["Site_Has_Eelgrass"] == True]["Length_cm"]
    no_eelgrass_sizes = df[df["Site_Has_Eelgrass"] == False]["Length_cm"]
    
    u_stat, p_val = stats.mannwhitneyu(eelgrass_sizes, no_eelgrass_sizes, alternative="two-sided")
    
    print(f"Eelgrass Sites: n={len(eelgrass_sizes)}, mean={eelgrass_sizes.mean():.2f} cm")
    print(f"No-Eelgrass Sites: n={len(no_eelgrass_sizes)}, mean={no_eelgrass_sizes.mean():.2f} cm")
    print(f"Mann-Whitney U: {u_stat:.0f}, p-value: {p_val:.4f}")
    print(f"Significant at α=0.05: {'Yes' if p_val < 0.05 else 'No'}")
    
    results.append({
        "Test": "Size by Site Eelgrass (Mann-Whitney)",
        "Group1_Mean": eelgrass_sizes.mean(),
        "Group2_Mean": no_eelgrass_sizes.mean(),
        "Statistic": u_stat,
        "p-value": p_val,
        "Significant": p_val < 0.05
    })
    
    # Test 2: Size by habitat (Kruskal-Wallis)
    print("\n--- Test 2: Size by Habitat Type (Kruskal-Wallis) ---")
    habitats = df["HabitatType"].unique()
    groups = [df[df["HabitatType"] == h]["Length_cm"].values for h in habitats]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) >= 2:
        h_stat, p_val = stats.kruskal(*groups)
        print(f"H-statistic: {h_stat:.3f}")
        print(f"p-value: {p_val:.4f}")
        print(f"Significant at α=0.05: {'Yes' if p_val < 0.05 else 'No'}")
        
        results.append({
            "Test": "Size by Habitat (Kruskal-Wallis)",
            "Statistic": h_stat,
            "p-value": p_val,
            "Significant": p_val < 0.05
        })
    
    # Test 3: Size bin distribution between eelgrass/non-eelgrass sites
    print("\n--- Test 3: Size Bin Distribution (Chi-Square) ---")
    contingency = pd.crosstab(df["Site_Has_Eelgrass"], df["Size_Bin"])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    
    print(f"Chi-square: {chi2:.3f}")
    print(f"p-value: {p_val:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"Significant at α=0.05: {'Yes' if p_val < 0.05 else 'No'}")
    
    results.append({
        "Test": f"Size Bin Distribution ({SIZE_BIN_WIDTH}cm bins, Chi-Square)",
        "Statistic": chi2,
        "p-value": p_val,
        "Significant": p_val < 0.05
    })
    
    # Test 4: Kolmogorov-Smirnov test for distribution difference
    print("\n--- Test 4: Distribution Comparison (Kolmogorov-Smirnov) ---")
    ks_stat, ks_pval = stats.ks_2samp(eelgrass_sizes, no_eelgrass_sizes)
    print(f"KS statistic: {ks_stat:.3f}")
    print(f"p-value: {ks_pval:.4f}")
    print(f"Significant at α=0.05: {'Yes' if ks_pval < 0.05 else 'No'}")
    
    results.append({
        "Test": "Distribution Comparison (Kolmogorov-Smirnov)",
        "Statistic": ks_stat,
        "p-value": ks_pval,
        "Significant": ks_pval < 0.05
    })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "size_statistical_tests.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")
    
    return results_df


def generate_size_summary_tables(df: pd.DataFrame):
    """Generate summary tables."""
    
    # By habitat
    habitat_summary = df.groupby("HabitatType")["Length_cm"].agg([
        "count", "mean", "std", "median", "min", "max"
    ]).round(2)
    habitat_summary.columns = ["N", "Mean", "Std", "Median", "Min", "Max"]
    
    habitat_path = OUTPUT_DIR / "size_by_habitat_summary.csv"
    habitat_summary.to_csv(habitat_path)
    print(f"  Saved: {habitat_path}")
    
    # By site eelgrass
    site_summary = df.groupby("Site_Eelgrass_Category")["Length_cm"].agg([
        "count", "mean", "std", "median", "min", "max"
    ]).round(2)
    site_summary.columns = ["N", "Mean", "Std", "Median", "Min", "Max"]
    
    # Add size bin proportions (top bins)
    size_props = df.groupby("Site_Eelgrass_Category")["Size_Bin"].value_counts(normalize=True).unstack()
    site_summary = site_summary.join(size_props.round(3))
    
    site_path = OUTPUT_DIR / "size_by_eelgrass_site_summary.csv"
    site_summary.to_csv(site_path)
    print(f"  Saved: {site_path}")
    
    return habitat_summary, site_summary


def main():
    print(f"\n{'='*60}")
    print("SIZE DISTRIBUTION CLUSTERING ANALYSIS")
    print(f"{'='*60}\n")
    
    set_style()
    
    print("Loading length data...")
    df = load_length_data_individuals()
    
    # Add categories
    df = add_size_categories(df)
    df = add_site_eelgrass_category(df)
    
    print(f"\nSite Classification:")
    n_eelgrass = df[df["Site_Has_Eelgrass"] == True]["SiteName"].nunique()
    n_no_eelgrass = df[df["Site_Has_Eelgrass"] == False]["SiteName"].nunique()
    print(f"  Sites WITH eelgrass: {n_eelgrass}")
    print(f"  Sites WITHOUT eelgrass: {n_no_eelgrass}")
    
    print(f"\nSize Bin Distribution ({SIZE_BIN_WIDTH} cm bins):")
    bin_counts = df["Size_Bin"].value_counts().sort_index()
    for bin_label, n in bin_counts.items():
        pct = n / len(df) * 100
        print(f"  {bin_label} cm: {n} ({pct:.1f}%)")
    
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    # Generate plots
    print("Generating size distribution figures...")
    plot_overall_size_distribution(df)
    plot_size_by_habitat(df)
    plot_size_by_eelgrass_site(df)
    plot_size_by_season(df)
    plot_size_habitat_interaction(df)
    
    # Clustering
    print("\nGenerating clustering visualizations...")
    plot_site_size_clustering(df)
    
    # Statistical tests
    run_size_statistical_tests(df)
    
    # Summary tables
    print("\nGenerating summary tables...")
    habitat_summary, site_summary = generate_size_summary_tables(df)
    
    print("\n--- Size by Habitat ---")
    print(habitat_summary.to_string())
    
    print("\n--- Size by Site Eelgrass Presence ---")
    print(site_summary.to_string())
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

