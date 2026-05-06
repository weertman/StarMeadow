"""
Statistical Summary & Modeling

Comprehensive statistical analysis including:
- Descriptive statistics
- Zero-inflated model considerations
- Habitat effect sizes
- ANOVA / Kruskal-Wallis tests

Outputs:
- Statistical test results
- Effect size calculations
- Publication-ready summary tables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

from utils import get_output_dir, load_data, set_style, save_figure

OUTPUT_DIR = get_output_dir(__file__)


def descriptive_statistics(df: pd.DataFrame):
    """Generate comprehensive descriptive statistics."""
    print("\n" + "="*50)
    print("DESCRIPTIVE STATISTICS")
    print("="*50)
    
    # Overall stats
    print("\n--- Overall Dataset ---")
    print(f"Total surveys: {len(df)}")
    print(f"Total Pycnopodia observed: {df['Pycnopodia_count'].sum()}")
    print(f"Surveys with presence: {(df['Pycnopodia_count'] > 0).sum()} ({(df['Pycnopodia_count'] > 0).mean():.1%})")
    print(f"Total survey time: {df['Survey.Time'].sum():.0f} minutes ({df['Survey.Time'].sum()/60:.1f} hours)")
    
    print(f"\n--- Pycnopodia Count per Survey ---")
    print(f"Mean: {df['Pycnopodia_count'].mean():.2f}")
    print(f"Median: {df['Pycnopodia_count'].median():.1f}")
    print(f"Std Dev: {df['Pycnopodia_count'].std():.2f}")
    print(f"Range: {df['Pycnopodia_count'].min()} - {df['Pycnopodia_count'].max()}")
    print(f"IQR: {df['Pycnopodia_count'].quantile(0.25):.1f} - {df['Pycnopodia_count'].quantile(0.75):.1f}")
    
    print(f"\n--- Encounter Rate (per hour) ---")
    print(f"Mean: {df['Encounter.Rate.Hr'].mean():.2f}")
    print(f"Median: {df['Encounter.Rate.Hr'].median():.2f}")
    print(f"Std Dev: {df['Encounter.Rate.Hr'].std():.2f}")
    print(f"Range: {df['Encounter.Rate.Hr'].min():.2f} - {df['Encounter.Rate.Hr'].max():.2f}")
    
    # Geographic coverage
    print(f"\n--- Geographic Coverage ---")
    print(f"Unique sites: {df['SiteName'].nunique()}")
    print(f"Unique basins: {df['Basin'].nunique()}")
    print(f"Regions: {df['Region'].unique().tolist()}")
    
    # Temporal coverage
    print(f"\n--- Temporal Coverage ---")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Years: {sorted(df['Year'].unique())}")
    
    return df.describe()


def test_habitat_differences(df: pd.DataFrame):
    """Statistical tests for habitat differences."""
    print("\n" + "="*50)
    print("HABITAT EFFECT TESTS")
    print("="*50)
    
    results = []
    
    # Kruskal-Wallis test (non-parametric ANOVA)
    habitats = df["HabitatType"].unique()
    groups = [df[df["HabitatType"] == h]["Encounter.Rate.Hr"].values for h in habitats]
    
    h_stat, p_value = stats.kruskal(*groups)
    print(f"\n--- Kruskal-Wallis Test (Encounter Rate by Habitat) ---")
    print(f"H-statistic: {h_stat:.3f}")
    print(f"p-value: {p_value:.2e}")
    print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    
    results.append({
        "Test": "Kruskal-Wallis",
        "Variable": "Encounter Rate by Habitat",
        "Statistic": h_stat,
        "p-value": p_value,
        "Significant": p_value < 0.05
    })
    
    # Pairwise Mann-Whitney U tests
    print(f"\n--- Pairwise Mann-Whitney U Tests ---")
    from itertools import combinations
    
    pairwise_results = []
    for h1, h2 in combinations(habitats, 2):
        g1 = df[df["HabitatType"] == h1]["Encounter.Rate.Hr"]
        g2 = df[df["HabitatType"] == h2]["Encounter.Rate.Hr"]
        
        u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(g1), len(g2)
        effect_size = 1 - (2 * u_stat) / (n1 * n2)
        
        pairwise_results.append({
            "Comparison": f"{h1} vs {h2}",
            "U-statistic": u_stat,
            "p-value": p_val,
            "Effect Size (r)": effect_size,
            "Significant": p_val < 0.05
        })
    
    pairwise_df = pd.DataFrame(pairwise_results)
    
    # Bonferroni correction
    n_tests = len(pairwise_df)
    pairwise_df["p-adjusted"] = pairwise_df["p-value"] * n_tests
    pairwise_df["Sig (Bonferroni)"] = pairwise_df["p-adjusted"] < 0.05
    
    print(pairwise_df.to_string(index=False))
    
    pairwise_path = OUTPUT_DIR / "pairwise_habitat_tests.csv"
    pairwise_df.to_csv(pairwise_path, index=False)
    print(f"\nSaved: {pairwise_path}")
    
    return pairwise_df


def test_depth_differences(df: pd.DataFrame):
    """Statistical tests for depth bin differences."""
    print("\n" + "="*50)
    print("DEPTH EFFECT TESTS")
    print("="*50)
    
    depth_bins = ["Shallow", "Intermediate", "Deep"]
    df_depth = df[df["DepthBin"].isin(depth_bins)]
    
    groups = [df_depth[df_depth["DepthBin"] == d]["Encounter.Rate.Hr"].values for d in depth_bins]
    
    h_stat, p_value = stats.kruskal(*groups)
    print(f"\n--- Kruskal-Wallis Test (Encounter Rate by Depth) ---")
    print(f"H-statistic: {h_stat:.3f}")
    print(f"p-value: {p_value:.2e}")
    print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Depth bin summary
    depth_summary = df_depth.groupby("DepthBin")["Encounter.Rate.Hr"].agg(
        ["mean", "median", "std", "count"]
    ).round(3)
    print(f"\n--- Depth Bin Summary ---")
    print(depth_summary)
    
    return depth_summary


def test_basin_differences(df: pd.DataFrame):
    """Statistical tests for basin differences."""
    print("\n" + "="*50)
    print("BASIN EFFECT TESTS")
    print("="*50)
    
    basins = df["Basin"].unique()
    groups = [df[df["Basin"] == b]["Encounter.Rate.Hr"].values for b in basins]
    
    h_stat, p_value = stats.kruskal(*groups)
    print(f"\n--- Kruskal-Wallis Test (Encounter Rate by Basin) ---")
    print(f"H-statistic: {h_stat:.3f}")
    print(f"p-value: {p_value:.2e}")
    print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Basin summary
    basin_summary = df.groupby("Basin")["Encounter.Rate.Hr"].agg(
        ["mean", "median", "std", "count"]
    ).round(3).sort_values("mean", ascending=False)
    print(f"\n--- Basin Summary ---")
    print(basin_summary)
    
    return basin_summary


def analyze_zero_inflation(df: pd.DataFrame):
    """Analyze zero-inflation in the data."""
    print("\n" + "="*50)
    print("ZERO-INFLATION ANALYSIS")
    print("="*50)
    
    zero_prop = (df["Pycnopodia_count"] == 0).mean()
    print(f"\nProportion of zero counts: {zero_prop:.1%}")
    
    # Zero proportions by habitat
    print(f"\n--- Zero Proportion by Habitat ---")
    zero_by_habitat = df.groupby("HabitatType").apply(
        lambda x: (x["Pycnopodia_count"] == 0).mean()
    ).sort_values()
    for habitat, prop in zero_by_habitat.items():
        print(f"  {habitat}: {prop:.1%}")
    
    # Zero proportions by basin
    print(f"\n--- Zero Proportion by Basin ---")
    zero_by_basin = df.groupby("Basin").apply(
        lambda x: (x["Pycnopodia_count"] == 0).mean()
    ).sort_values()
    for basin, prop in zero_by_basin.items():
        print(f"  {basin}: {prop:.1%}")
    
    return zero_by_habitat, zero_by_basin


def plot_statistical_summary(df: pd.DataFrame):
    """Create statistical summary visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Distribution of encounter rates
    ax1 = axes[0, 0]
    df["Encounter.Rate.Hr"].hist(bins=50, ax=ax1, color="steelblue", edgecolor="black", alpha=0.7)
    ax1.axvline(df["Encounter.Rate.Hr"].mean(), color="red", linestyle="--", label=f"Mean: {df['Encounter.Rate.Hr'].mean():.2f}")
    ax1.axvline(df["Encounter.Rate.Hr"].median(), color="orange", linestyle="--", label=f"Median: {df['Encounter.Rate.Hr'].median():.2f}")
    ax1.set_xlabel("Encounter Rate (per hour)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Encounter Rates")
    ax1.legend()
    
    # QQ plot
    ax2 = axes[0, 1]
    stats.probplot(df["Encounter.Rate.Hr"], dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot (Normal)")
    
    # Box plots by habitat
    ax3 = axes[1, 0]
    habitat_order = df.groupby("HabitatType")["Encounter.Rate.Hr"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="HabitatType", y="Encounter.Rate.Hr", order=habitat_order, ax=ax3, palette="viridis")
    ax3.set_xlabel("Habitat Type")
    ax3.set_ylabel("Encounter Rate (per hour)")
    ax3.set_title("Encounter Rate Distribution by Habitat")
    ax3.tick_params(axis="x", rotation=45)
    
    # Zero vs non-zero by habitat
    ax4 = axes[1, 1]
    df["Has_Pycno"] = df["Pycnopodia_count"] > 0
    presence_by_habitat = df.groupby("HabitatType")["Has_Pycno"].mean().sort_values(ascending=False)
    ax4.bar(presence_by_habitat.index, presence_by_habitat.values, color="coral", edgecolor="black")
    ax4.set_xlabel("Habitat Type")
    ax4.set_ylabel("Detection Rate")
    ax4.set_title("Pycnopodia Detection Rate by Habitat")
    ax4.tick_params(axis="x", rotation=45)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "statistical_summary")
    plt.close()


def generate_publication_table(df: pd.DataFrame):
    """Generate publication-ready summary table."""
    
    # Summary by habitat
    habitat_summary = df.groupby("HabitatType").agg({
        "Pycnopodia_count": ["count", "sum", "mean"],
        "Encounter.Rate.Hr": ["mean", "std"],
        "Survey.Time": "sum"
    })
    habitat_summary.columns = [
        "N Surveys", "Total Pycno", "Mean Count", 
        "Mean Rate (hr⁻¹)", "Rate SD", "Total Time (min)"
    ]
    habitat_summary["Detection Rate"] = df.groupby("HabitatType").apply(
        lambda x: (x["Pycnopodia_count"] > 0).mean()
    )
    habitat_summary = habitat_summary.round(3)
    
    # Sort by mean rate
    habitat_summary = habitat_summary.sort_values("Mean Rate (hr⁻¹)", ascending=False)
    
    # Save
    table_path = OUTPUT_DIR / "publication_summary_table.csv"
    habitat_summary.to_csv(table_path)
    print(f"\nSaved publication table: {table_path}")
    
    # Also save as formatted text
    print("\n--- Publication Summary Table ---")
    print(habitat_summary.to_string())
    
    return habitat_summary


def main():
    print(f"\n{'='*60}")
    print("STATISTICAL SUMMARY & MODELING")
    print(f"{'='*60}\n")
    
    set_style()
    df = load_data()
    
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Run analyses
    descriptive_statistics(df)
    test_habitat_differences(df)
    test_depth_differences(df)
    test_basin_differences(df)
    analyze_zero_inflation(df)
    
    print("\nGenerating visualizations...")
    plot_statistical_summary(df)
    
    print("\nGenerating publication table...")
    generate_publication_table(df)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()






