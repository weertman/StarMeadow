"""
Temporal Analysis

Analyzes temporal patterns in Pycnopodia helianthoides observations:
- Year-over-year trends
- Monthly patterns
- Seasonal variation by habitat
- Survey effort over time

Outputs:
- Time series plots
- Effort distribution visualizations
- Trend analyses
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from utils import get_output_dir, load_data, set_style, save_figure

OUTPUT_DIR = get_output_dir(__file__)


def plot_annual_trends(df: pd.DataFrame):
    """Year-over-year encounter rate trends."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Survey effort by year
    ax1 = axes[0, 0]
    effort_by_year = df.groupby("Year").size()
    ax1.bar(effort_by_year.index, effort_by_year.values, color="steelblue", edgecolor="black")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Surveys")
    ax1.set_title("Survey Effort by Year")
    for i, (year, count) in enumerate(effort_by_year.items()):
        ax1.annotate(str(count), xy=(year, count + 20), ha="center", fontsize=9)
    
    # Mean encounter rate by year
    ax2 = axes[0, 1]
    year_stats = df.groupby("Year")["Encounter.Rate.Hr"].agg(["mean", "std", "count"])
    year_stats["se"] = year_stats["std"] / np.sqrt(year_stats["count"])
    ax2.errorbar(
        year_stats.index, year_stats["mean"], yerr=year_stats["se"],
        marker="o", capsize=5, color="darkgreen", linewidth=2, markersize=8
    )
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Mean Encounter Rate (per hour)")
    ax2.set_title("Annual Encounter Rate Trend")
    ax2.set_xticks(year_stats.index)
    
    # Total Pycnopodia observed by year
    ax3 = axes[1, 0]
    total_by_year = df.groupby("Year")["Pycnopodia_count"].sum()
    ax3.bar(total_by_year.index, total_by_year.values, color="coral", edgecolor="black")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Total Pycnopodia Observed")
    ax3.set_title("Total Observations by Year")
    for i, (year, count) in enumerate(total_by_year.items()):
        ax3.annotate(str(count), xy=(year, count + 50), ha="center", fontsize=9)
    
    # Detection rate by year
    ax4 = axes[1, 1]
    df["Presence"] = (df["Pycnopodia_count"] > 0).astype(int)
    detection_by_year = df.groupby("Year")["Presence"].mean()
    ax4.bar(detection_by_year.index, detection_by_year.values, color="purple", edgecolor="black", alpha=0.7)
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Detection Rate")
    ax4.set_title("Proportion of Surveys with Pycnopodia Present")
    ax4.set_ylim(0, 1)
    for i, (year, rate) in enumerate(detection_by_year.items()):
        ax4.annotate(f"{rate:.1%}", xy=(year, rate + 0.03), ha="center", fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "annual_trends")
    plt.close()


def plot_monthly_heatmap(df: pd.DataFrame):
    """Heatmap of encounter rates by month and year."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_valid = df.dropna(subset=["Year", "Month"]).copy()
    df_valid["Month"] = df_valid["Month"].astype(int)
    df_valid["Year"] = df_valid["Year"].astype(int)
    pivot = df_valid.pivot_table(
        values="Encounter.Rate.Hr",
        index="Year",
        columns="Month",
        aggfunc="mean"
    )
    
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Mean Encounter Rate (per hour)"},
        xticklabels=[month_labels[int(i)-1] for i in pivot.columns]
    )
    
    ax.set_title("Encounter Rate by Month and Year")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "monthly_heatmap")
    plt.close()


def plot_effort_heatmap(df: pd.DataFrame):
    """Heatmap of survey effort by month and year."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_valid = df.dropna(subset=["Year", "Month"]).copy()
    df_valid["Month"] = df_valid["Month"].astype(int)
    df_valid["Year"] = df_valid["Year"].astype(int)
    pivot = df_valid.pivot_table(
        values="Pycnopodia_count",
        index="Year",
        columns="Month",
        aggfunc="count"
    )
    
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Number of Surveys"},
        xticklabels=[month_labels[int(i)-1] for i in pivot.columns]
    )
    
    ax.set_title("Survey Effort by Month and Year")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "effort_heatmap")
    plt.close()


def plot_basin_temporal(df: pd.DataFrame):
    """Temporal patterns by basin."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Get top basins by survey count
    top_basins = df.groupby("Basin").size().nlargest(6).index.tolist()
    df_top = df[df["Basin"].isin(top_basins)]
    
    # Annual trend by basin
    ax1 = axes[0]
    basin_year = df_top.groupby(["Year", "Basin"])["Encounter.Rate.Hr"].mean().unstack()
    basin_year.plot(ax=ax1, marker="o", linewidth=2)
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Mean Encounter Rate (per hour)")
    ax1.set_title("Annual Encounter Rate Trend by Basin")
    ax1.legend(title="Basin", bbox_to_anchor=(1.02, 1), loc="upper left")
    
    # Monthly pattern by basin
    ax2 = axes[1]
    basin_month = df_top.groupby(["Month", "Basin"])["Encounter.Rate.Hr"].mean().unstack()
    basin_month.plot(ax=ax2, marker="o", linewidth=2)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Mean Encounter Rate (per hour)")
    ax2.set_title("Monthly Encounter Rate Pattern by Basin")
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax2.legend(title="Basin", bbox_to_anchor=(1.02, 1), loc="upper left")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "basin_temporal_trends")
    plt.close()


def plot_cumulative_observations(df: pd.DataFrame):
    """Cumulative observations over time."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_sorted = df.sort_values("Date")
    df_sorted["Cumulative_Count"] = df_sorted["Pycnopodia_count"].cumsum()
    df_sorted["Cumulative_Surveys"] = range(1, len(df_sorted) + 1)
    
    ax.plot(df_sorted["Date"], df_sorted["Cumulative_Count"], 
            label="Cumulative Pycnopodia", color="coral", linewidth=2)
    
    ax2 = ax.twinx()
    ax2.plot(df_sorted["Date"], df_sorted["Cumulative_Surveys"], 
             label="Cumulative Surveys", color="steelblue", linewidth=2, linestyle="--")
    
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Pycnopodia Observed", color="coral")
    ax2.set_ylabel("Cumulative Surveys", color="steelblue")
    ax.set_title("Cumulative Survey Effort and Observations")
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "cumulative_observations")
    plt.close()


def plot_site_temporal_coverage(df: pd.DataFrame):
    """Visualize temporal coverage across sites."""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Get sites with most surveys
    top_sites = df.groupby("SiteName").size().nlargest(20).index.tolist()
    df_top = df[df["SiteName"].isin(top_sites)]
    
    # Create year-month for x-axis
    df_top = df_top.copy()
    df_top["YearMonth"] = df_top["Date"].dt.to_period("M")
    
    pivot = df_top.pivot_table(
        values="Pycnopodia_count",
        index="SiteName",
        columns="YearMonth",
        aggfunc="count",
        fill_value=0
    )
    
    # Order sites by total surveys
    site_order = pivot.sum(axis=1).sort_values(ascending=True).index
    pivot = pivot.loc[site_order]
    
    sns.heatmap(
        pivot > 0,  # Binary: surveyed or not
        cmap="YlGnBu",
        ax=ax,
        cbar_kws={"label": "Survey Conducted"}
    )
    
    ax.set_title("Site Survey Coverage Over Time (Top 20 Sites)")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Site")
    
    # Reduce x-tick density
    n_cols = len(pivot.columns)
    ax.set_xticks(range(0, n_cols, max(1, n_cols // 12)))
    ax.set_xticklabels([str(pivot.columns[i]) for i in range(0, n_cols, max(1, n_cols // 12))], 
                       rotation=45, ha="right")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "site_temporal_coverage")
    plt.close()


def generate_temporal_summary(df: pd.DataFrame):
    """Generate temporal summary statistics."""
    # Annual summary
    annual = df.groupby("Year").agg({
        "Pycnopodia_count": ["sum", "mean"],
        "Encounter.Rate.Hr": ["mean", "std"],
        "SiteName": "nunique",
        "Date": "count"
    }).round(3)
    annual.columns = ["Total Count", "Mean Count", "Mean Rate", "Rate Std", "Unique Sites", "N Surveys"]
    
    annual_path = OUTPUT_DIR / "annual_summary.csv"
    annual.to_csv(annual_path)
    print(f"  Saved: {annual_path}")
    
    # Monthly summary (across all years)
    monthly = df.groupby("Month").agg({
        "Encounter.Rate.Hr": ["mean", "std", "count"],
        "Pycnopodia_count": "sum"
    }).round(3)
    monthly.columns = ["Mean Rate", "Rate Std", "N Surveys", "Total Count"]
    
    monthly_path = OUTPUT_DIR / "monthly_summary.csv"
    monthly.to_csv(monthly_path)
    print(f"  Saved: {monthly_path}")
    
    return annual, monthly


def main():
    print(f"\n{'='*60}")
    print("TEMPORAL ANALYSIS")
    print(f"{'='*60}\n")
    
    set_style()
    df = load_data()
    
    print(f"Loaded {len(df)} survey records")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Years covered: {sorted(df['Year'].unique())}")
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    print("Generating figures...")
    plot_annual_trends(df)
    plot_monthly_heatmap(df)
    plot_effort_heatmap(df)
    plot_basin_temporal(df)
    plot_cumulative_observations(df)
    plot_site_temporal_coverage(df)
    
    print("\nGenerating summary tables...")
    annual, monthly = generate_temporal_summary(df)
    
    print("\nAnnual Summary:")
    print(annual.to_string())
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()






