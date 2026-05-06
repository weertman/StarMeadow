"""
Size Structure Analysis

Analyzes Pycnopodia helianthoides size distributions across:
- Habitat types
- Depth bins
- Seasons
- Basins

Uses two data sources:
1. Count data (PycnoCountClean): Survey-level abundance
2. Length data (PycnoLengthClean): Individual-level size measurements

Outputs:
- Size distribution histograms (even 5cm bins)
- Size by habitat/depth violin plots
- Seasonal size patterns
- Count distribution analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from utils import (
    get_output_dir, load_data, load_length_data_individuals,
    set_style, save_figure
)

OUTPUT_DIR = get_output_dir(__file__)

# Even-width size bins (5 cm increments)
SIZE_BIN_WIDTH = 5


def plot_size_distribution_overview(df_length: pd.DataFrame):
    """Overall size distribution with even bins."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram with even 5cm bins
    ax1 = axes[0]
    bins = np.arange(0, df_length["Length_cm"].max() + SIZE_BIN_WIDTH, SIZE_BIN_WIDTH)
    ax1.hist(df_length["Length_cm"], bins=bins, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.axvline(df_length["Length_cm"].mean(), color="red", linestyle="--", 
                label=f"Mean: {df_length['Length_cm'].mean():.1f} cm")
    ax1.axvline(df_length["Length_cm"].median(), color="orange", linestyle="--",
                label=f"Median: {df_length['Length_cm'].median():.1f} cm")
    ax1.set_xlabel("Length (cm)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Pycnopodia Size Distribution ({SIZE_BIN_WIDTH} cm bins, n={len(df_length)})")
    ax1.legend()
    
    # Kernel density
    ax2 = axes[1]
    sns.kdeplot(df_length["Length_cm"], ax=ax2, fill=True, color="steelblue", alpha=0.5)
    ax2.axvline(df_length["Length_cm"].mean(), color="red", linestyle="--", label="Mean")
    ax2.axvline(df_length["Length_cm"].median(), color="orange", linestyle="--", label="Median")
    ax2.set_xlabel("Length (cm)")
    ax2.set_ylabel("Density")
    ax2.set_title("Size Distribution (Kernel Density)")
    ax2.legend()
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "size_distribution_overview")
    plt.close()


def plot_size_by_habitat(df_length: pd.DataFrame):
    """Size distribution by habitat type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Filter habitats with enough data
    habitat_counts = df_length["HabitatType"].value_counts()
    valid_habitats = habitat_counts[habitat_counts >= 5].index
    df_plot = df_length[df_length["HabitatType"].isin(valid_habitats)]
    
    # Order by median size
    habitat_order = df_plot.groupby("HabitatType")["Length_cm"].median().sort_values(ascending=False).index
    
    # Violin plot
    ax1 = axes[0]
    sns.violinplot(
        data=df_plot,
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
    
    # Mean with SE
    ax2 = axes[1]
    habitat_stats = df_plot.groupby("HabitatType")["Length_cm"].agg(["mean", "std", "count"])
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
    ax2.set_title("Mean Size by Habitat (± SE)")
    
    for i, (idx, row) in enumerate(habitat_stats.iterrows()):
        ax2.annotate(f"n={int(row['count'])}", 
                     xy=(i, row["mean"] + row["se"] + 1),
                     ha="center", fontsize=8, color="gray")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "size_by_habitat")
    plt.close()
    
    return habitat_stats


def plot_size_by_depth(df_length: pd.DataFrame):
    """Size distribution by depth bin."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    depth_order = ["Shallow", "Intermediate", "Deep"]
    df_plot = df_length[df_length["DepthBin"].isin(depth_order)]
    
    # Violin plot
    ax1 = axes[0]
    sns.violinplot(
        data=df_plot,
        x="DepthBin",
        y="Length_cm",
        order=depth_order,
        hue="DepthBin",
        palette="Blues",
        ax=ax1,
        inner="box"
    )
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()
    ax1.set_xlabel("Depth Bin")
    ax1.set_ylabel("Length (cm)")
    ax1.set_title("Size Distribution by Depth")
    
    # Histograms overlaid
    ax2 = axes[1]
    bins = np.arange(0, df_plot["Length_cm"].max() + SIZE_BIN_WIDTH, SIZE_BIN_WIDTH)
    colors = {"Shallow": "#a6cee3", "Intermediate": "#1f78b4", "Deep": "#08306b"}
    
    for depth in depth_order:
        subset = df_plot[df_plot["DepthBin"] == depth]
        if len(subset) > 0:
            ax2.hist(subset["Length_cm"], bins=bins, alpha=0.5, 
                     label=f"{depth} (n={len(subset)})", color=colors[depth], edgecolor="black")
    
    ax2.set_xlabel("Length (cm)")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Size Distribution by Depth ({SIZE_BIN_WIDTH} cm bins)")
    ax2.legend()
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "size_by_depth")
    plt.close()


def plot_size_by_basin(df_length: pd.DataFrame):
    """Size distribution by basin."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Filter basins with enough data
    basin_counts = df_length["Basin"].value_counts()
    valid_basins = basin_counts[basin_counts >= 10].index
    df_plot = df_length[df_length["Basin"].isin(valid_basins)]
    
    if len(df_plot) == 0:
        print("  Not enough data per basin for size analysis")
        plt.close()
        return
    
    # Order by median size
    basin_order = df_plot.groupby("Basin")["Length_cm"].median().sort_values(ascending=False).index
    
    sns.boxplot(
        data=df_plot,
        x="Basin",
        y="Length_cm",
        order=basin_order,
        hue="Basin",
        palette="mako",
        ax=ax
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    ax.set_xlabel("Basin")
    ax.set_ylabel("Length (cm)")
    ax.set_title("Size Distribution by Basin")
    ax.tick_params(axis="x", rotation=45)
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "size_by_basin")
    plt.close()


def plot_size_by_season(df_length: pd.DataFrame):
    """Size distribution by season."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    df_plot = df_length[df_length["Season"].isin(season_order)]
    
    # Violin plot
    ax1 = axes[0]
    season_colors = {"Winter": "#4575b4", "Spring": "#91cf60", "Summer": "#fc8d59", "Autumn": "#d73027"}
    
    sns.violinplot(
        data=df_plot,
        x="Season",
        y="Length_cm",
        order=season_order,
        hue="Season",
        palette=season_colors,
        ax=ax1,
        inner="box"
    )
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()
    ax1.set_xlabel("Season")
    ax1.set_ylabel("Length (cm)")
    ax1.set_title("Size Distribution by Season")
    
    # Monthly trend
    ax2 = axes[1]
    monthly = df_length.groupby("Month")["Length_cm"].agg(["mean", "std", "count"])
    monthly["se"] = monthly["std"] / np.sqrt(monthly["count"])
    monthly = monthly[monthly["count"] >= 3]  # Filter months with few data
    
    ax2.errorbar(monthly.index, monthly["mean"], yerr=monthly["se"],
                 marker="o", capsize=4, color="steelblue", linewidth=2)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Mean Length (cm)")
    ax2.set_title("Mean Size by Month (± SE)")
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "size_by_season")
    plt.close()


def plot_count_distribution(df_count: pd.DataFrame):
    """Distribution of Pycnopodia counts per survey."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram of counts
    ax1 = axes[0]
    counts = df_count["Pycnopodia_count"]
    ax1.hist(counts, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.axvline(counts.mean(), color="red", linestyle="--", label=f"Mean: {counts.mean():.2f}")
    ax1.axvline(counts.median(), color="orange", linestyle="--", label=f"Median: {counts.median():.1f}")
    ax1.set_xlabel("Pycnopodia Count per Survey")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Counts per Survey")
    ax1.legend()
    
    # Log-transformed for better visualization
    ax2 = axes[1]
    counts_nonzero = counts[counts > 0]
    ax2.hist(np.log10(counts_nonzero + 1), bins=30, color="coral", edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Log10(Count + 1)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Log-Transformed Distribution (non-zero only)")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "count_distribution")
    plt.close()


def plot_seasonal_encounter_patterns(df_count: pd.DataFrame):
    """Seasonal patterns in encounter rates."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    season_colors = {"Winter": "#4575b4", "Spring": "#91cf60", "Summer": "#fc8d59", "Autumn": "#d73027"}
    
    # Encounter rate by season
    ax1 = axes[0]
    season_stats = df_count.groupby("Season")["Encounter.Rate.Hr"].agg(["mean", "std", "count"])
    season_stats["se"] = season_stats["std"] / np.sqrt(season_stats["count"])
    season_stats = season_stats.reindex(season_order)
    
    ax1.bar(
        season_stats.index,
        season_stats["mean"],
        yerr=season_stats["se"],
        capsize=5,
        color=[season_colors[s] for s in season_stats.index],
        edgecolor="black"
    )
    ax1.set_xlabel("Season")
    ax1.set_ylabel("Mean Encounter Rate (per hour)")
    ax1.set_title("Encounter Rate by Season")
    
    for i, (idx, row) in enumerate(season_stats.iterrows()):
        ax1.annotate(f"n={int(row['count'])}", xy=(i, row["mean"] + row["se"] + 0.2),
                     ha="center", fontsize=9, color="gray")
    
    # Monthly trend
    ax2 = axes[1]
    monthly = df_count.groupby("Month")["Encounter.Rate.Hr"].agg(["mean", "sem"])
    ax2.errorbar(monthly.index, monthly["mean"], yerr=monthly["sem"], 
                 marker="o", capsize=3, color="steelblue", linewidth=2)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Mean Encounter Rate (per hour)")
    ax2.set_title("Monthly Encounter Rate Trend")
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "seasonal_encounter_patterns")
    plt.close()


def plot_habitat_season_heatmap(df_count: pd.DataFrame):
    """Heatmap of encounter rate by habitat and season."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot = df_count.pivot_table(
        values="Encounter.Rate.Hr",
        index="HabitatType",
        columns="Season",
        aggfunc="mean"
    )
    
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    pivot = pivot.reindex(columns=season_order)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        ax=ax,
        cbar_kws={"label": "Mean Encounter Rate (per hour)"}
    )
    
    ax.set_title("Encounter Rate by Habitat Type and Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Habitat Type")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "habitat_season_heatmap")
    plt.close()


def plot_size_season_heatmap(df_length: pd.DataFrame):
    """Heatmap of mean size by habitat and season."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter for valid habitats
    habitat_counts = df_length.groupby("HabitatType").size()
    valid_habitats = habitat_counts[habitat_counts >= 5].index
    df_plot = df_length[df_length["HabitatType"].isin(valid_habitats)]
    
    pivot = df_plot.pivot_table(
        values="Length_cm",
        index="HabitatType",
        columns="Season",
        aggfunc="mean"
    )
    
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    available_seasons = [s for s in season_order if s in pivot.columns]
    pivot = pivot.reindex(columns=available_seasons)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Mean Length (cm)"}
    )
    
    ax.set_title("Mean Size by Habitat Type and Season")
    ax.set_xlabel("Season")
    ax.set_ylabel("Habitat Type")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "size_season_heatmap")
    plt.close()


def generate_size_summary(df_length: pd.DataFrame):
    """Generate size summary statistics."""
    # By habitat
    habitat_summary = df_length.groupby("HabitatType")["Length_cm"].agg([
        "count", "mean", "std", "median", "min", "max"
    ]).round(2)
    habitat_summary.columns = ["N", "Mean", "Std", "Median", "Min", "Max"]
    habitat_summary = habitat_summary.sort_values("Mean", ascending=False)
    
    habitat_path = OUTPUT_DIR / "size_summary_by_habitat.csv"
    habitat_summary.to_csv(habitat_path)
    print(f"  Saved: {habitat_path}")
    
    # By season
    season_summary = df_length.groupby("Season")["Length_cm"].agg([
        "count", "mean", "std", "median", "min", "max"
    ]).round(2)
    season_summary.columns = ["N", "Mean", "Std", "Median", "Min", "Max"]
    
    season_path = OUTPUT_DIR / "size_summary_by_season.csv"
    season_summary.to_csv(season_path)
    print(f"  Saved: {season_path}")
    
    # By depth
    depth_summary = df_length.groupby("DepthBin")["Length_cm"].agg([
        "count", "mean", "std", "median", "min", "max"
    ]).round(2)
    depth_summary.columns = ["N", "Mean", "Std", "Median", "Min", "Max"]
    
    depth_path = OUTPUT_DIR / "size_summary_by_depth.csv"
    depth_summary.to_csv(depth_path)
    print(f"  Saved: {depth_path}")
    
    return habitat_summary, season_summary, depth_summary


def main():
    print(f"\n{'='*60}")
    print("SIZE STRUCTURE ANALYSIS")
    print(f"{'='*60}\n")
    
    set_style()
    
    # Load count data
    print("Loading count data...")
    df_count = load_data()
    print(f"  {len(df_count)} survey records")
    print(f"  Surveys with Pycnopodia present: {(df_count['Pycnopodia_count'] > 0).sum()}")
    print(f"  Overall detection rate: {(df_count['Pycnopodia_count'] > 0).mean():.1%}")
    
    # Load length data
    print("\nLoading length data...")
    df_length = load_length_data_individuals()
    
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    # Size distribution analysis (from length data)
    print("Generating size distribution figures...")
    plot_size_distribution_overview(df_length)
    plot_size_by_habitat(df_length)
    plot_size_by_depth(df_length)
    plot_size_by_basin(df_length)
    plot_size_by_season(df_length)
    plot_size_season_heatmap(df_length)
    
    # Count/encounter analysis (from count data)
    print("\nGenerating count/encounter figures...")
    plot_count_distribution(df_count)
    plot_seasonal_encounter_patterns(df_count)
    plot_habitat_season_heatmap(df_count)
    
    # Summary tables
    print("\nGenerating summary tables...")
    habitat_summary, season_summary, depth_summary = generate_size_summary(df_length)
    
    print("\n--- Size Summary by Habitat ---")
    print(habitat_summary.to_string())
    
    print("\n--- Size Summary by Season ---")
    print(season_summary.to_string())
    
    print("\n--- Size Summary by Depth ---")
    print(depth_summary.to_string())
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
