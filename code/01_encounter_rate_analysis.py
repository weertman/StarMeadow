"""
Encounter Rate Analysis by Habitat and Basin

Analyzes Pycnopodia helianthoides encounter rates across:
- Habitat types (eelgrass, natural reef, artificial reef, kelp forest, soft bottom)
- Geographic basins within the Salish Sea
- Depth bins (shallow, intermediate, deep)

Outputs:
- Bar plots of encounter rates by habitat
- Basin × habitat heatmaps
- Statistical summary tables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from utils import get_output_dir, load_data, set_style, save_figure

OUTPUT_DIR = get_output_dir(__file__)


def plot_encounter_by_habitat(df: pd.DataFrame):
    """Bar plot of mean encounter rate by habitat type."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    habitat_stats = df.groupby("HabitatType")["Encounter.Rate.Hr"].agg(["mean", "std", "count"])
    habitat_stats["se"] = habitat_stats["std"] / np.sqrt(habitat_stats["count"])
    habitat_stats = habitat_stats.sort_values("mean", ascending=False)
    
    colors = sns.color_palette("viridis", len(habitat_stats))
    bars = ax.bar(
        habitat_stats.index, 
        habitat_stats["mean"], 
        yerr=habitat_stats["se"],
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.8
    )
    
    ax.set_xlabel("Habitat Type")
    ax.set_ylabel("Mean Encounter Rate (per hour)")
    ax.set_title("Pycnopodia helianthoides Encounter Rate by Habitat Type")
    ax.tick_params(axis="x", rotation=45)
    
    # Add sample sizes
    for i, (idx, row) in enumerate(habitat_stats.iterrows()):
        ax.annotate(
            f"n={int(row['count'])}", 
            xy=(i, row["mean"] + row["se"] + 0.5),
            ha="center", fontsize=8, color="gray"
        )
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "encounter_rate_by_habitat")
    plt.close()


def plot_encounter_by_basin(df: pd.DataFrame):
    """Bar plot of mean encounter rate by basin."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    basin_stats = df.groupby("Basin")["Encounter.Rate.Hr"].agg(["mean", "std", "count"])
    basin_stats["se"] = basin_stats["std"] / np.sqrt(basin_stats["count"])
    basin_stats = basin_stats.sort_values("mean", ascending=False)
    
    colors = sns.color_palette("mako", len(basin_stats))
    ax.bar(
        basin_stats.index,
        basin_stats["mean"],
        yerr=basin_stats["se"],
        capsize=4,
        color=colors,
        edgecolor="black",
        linewidth=0.8
    )
    
    ax.set_xlabel("Basin")
    ax.set_ylabel("Mean Encounter Rate (per hour)")
    ax.set_title("Pycnopodia helianthoides Encounter Rate by Basin")
    ax.tick_params(axis="x", rotation=45)
    
    for i, (idx, row) in enumerate(basin_stats.iterrows()):
        ax.annotate(
            f"n={int(row['count'])}",
            xy=(i, row["mean"] + row["se"] + 0.3),
            ha="center", fontsize=8, color="gray"
        )
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "encounter_rate_by_basin")
    plt.close()


def plot_basin_habitat_heatmap(df: pd.DataFrame):
    """Heatmap of encounter rates by basin and habitat."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    pivot = df.pivot_table(
        values="Encounter.Rate.Hr",
        index="Basin",
        columns="HabitatType",
        aggfunc="mean"
    )
    
    # Sort by total encounter rate
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Mean Encounter Rate (per hour)"}
    )
    
    ax.set_title("Encounter Rate by Basin and Habitat Type")
    ax.set_xlabel("Habitat Type")
    ax.set_ylabel("Basin")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "basin_habitat_heatmap")
    plt.close()


def plot_encounter_by_depth(df: pd.DataFrame):
    """Box plot of encounter rates by depth bin."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    depth_order = ["Shallow", "Intermediate", "Deep"]
    df_plot = df[df["DepthBin"].isin(depth_order)]
    
    sns.boxplot(
        data=df_plot,
        x="DepthBin",
        y="Encounter.Rate.Hr",
        order=depth_order,
        hue="DepthBin",
        palette="Blues",
        ax=ax
    )
    # Remove legend (hue creates one automatically in older seaborn)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    
    sns.stripplot(
        data=df_plot,
        x="DepthBin",
        y="Encounter.Rate.Hr",
        order=depth_order,
        color="black",
        alpha=0.3,
        size=3,
        ax=ax
    )
    
    ax.set_xlabel("Depth Bin")
    ax.set_ylabel("Encounter Rate (per hour)")
    ax.set_title("Pycnopodia helianthoides Encounter Rate by Depth")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "encounter_rate_by_depth")
    plt.close()


def plot_habitat_depth_facet(df: pd.DataFrame):
    """Faceted bar plot: habitat type by depth bin."""
    depth_order = ["Shallow", "Intermediate", "Deep"]
    df_plot = df[df["DepthBin"].isin(depth_order)].copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
    
    for ax, depth in zip(axes, depth_order):
        subset = df_plot[df_plot["DepthBin"] == depth]
        habitat_stats = subset.groupby("HabitatType")["Encounter.Rate.Hr"].agg(["mean", "std", "count"])
        habitat_stats["se"] = habitat_stats["std"] / np.sqrt(habitat_stats["count"])
        habitat_stats = habitat_stats.sort_values("mean", ascending=False)
        
        colors = sns.color_palette("viridis", len(habitat_stats))
        ax.bar(
            range(len(habitat_stats)),
            habitat_stats["mean"],
            yerr=habitat_stats["se"],
            capsize=3,
            color=colors,
            edgecolor="black",
            linewidth=0.5
        )
        ax.set_xticks(range(len(habitat_stats)))
        ax.set_xticklabels(habitat_stats.index, rotation=45, ha="right")
        ax.set_title(f"{depth}")
        ax.set_xlabel("")
    
    axes[0].set_ylabel("Mean Encounter Rate (per hour)")
    fig.suptitle("Encounter Rate by Habitat Type Across Depth Bins", fontsize=14)
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "habitat_by_depth_facet")
    plt.close()


def generate_summary_table(df: pd.DataFrame):
    """Generate summary statistics table."""
    summary = df.groupby("HabitatType").agg({
        "Encounter.Rate.Hr": ["mean", "std", "median", "min", "max", "count"],
        "Pycnopodia_count": "sum",
        "Survey.Time": "sum"
    }).round(3)
    
    summary.columns = [
        "Mean Rate", "Std Dev", "Median Rate", "Min Rate", "Max Rate", 
        "N Surveys", "Total Pycnos", "Total Time (min)"
    ]
    
    summary_path = OUTPUT_DIR / "encounter_rate_summary.csv"
    summary.to_csv(summary_path)
    print(f"  Saved: {summary_path}")
    
    return summary


def main():
    print(f"\n{'='*60}")
    print("ENCOUNTER RATE ANALYSIS")
    print(f"{'='*60}\n")
    
    set_style()
    df = load_data()
    
    print(f"Loaded {len(df)} survey records")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Unique sites: {df['SiteName'].nunique()}")
    print(f"Total Pycnopodia observed: {df['Pycnopodia_count'].sum()}")
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    print("Generating figures...")
    plot_encounter_by_habitat(df)
    plot_encounter_by_basin(df)
    plot_basin_habitat_heatmap(df)
    plot_encounter_by_depth(df)
    plot_habitat_depth_facet(df)
    
    print("\nGenerating summary table...")
    summary = generate_summary_table(df)
    print("\n" + summary.to_string())
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()






