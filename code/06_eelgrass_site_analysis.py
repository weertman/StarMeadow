"""
Eelgrass Site-Level Analysis

Investigates Pycnopodia helianthoides encounter rates by classifying
sites based on eelgrass presence. This differs from transect-level
habitat analysis by asking: "Do sites that CONTAIN eelgrass (anywhere)
have higher sunflower star encounter rates?"

Site Classification:
- Eelgrass Site: Any site where ≥1 transect has HabitatType == "Eelgrass"
- Non-Eelgrass Site: Sites with no eelgrass transects

Four-Category Habitat Scheme:
1. Eelgrass (transect in eelgrass habitat)
2. Hard Bottom + Eelgrass at Site (reef transect at a site with eelgrass)
3. Hard Bottom (reef transect at a site without eelgrass)
4. Soft Bottom

Outputs:
- Site-level eelgrass presence comparisons
- Four-category habitat analysis
- Interaction plots (habitat × site eelgrass presence)
- Statistical tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib.patches import Rectangle
import matplotlib.transforms as mtransforms

from utils import get_output_dir, load_data, set_style, save_figure

OUTPUT_DIR = get_output_dir(__file__)


def add_site_eelgrass_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add site-level eelgrass presence indicator.
    
    For each site, determine if ANY transect has HabitatType == "Eelgrass",
    then map this back to all transects at that site.
    """
    df = df.copy()
    
    # Identify sites with eelgrass
    sites_with_eelgrass = df.groupby("SiteName")["HabitatType"].apply(
        lambda x: (x == "Eelgrass").any()
    )
    
    # Map to transect level
    df["Site_Has_Eelgrass"] = df["SiteName"].map(sites_with_eelgrass)
    df["Site_Eelgrass_Category"] = df["Site_Has_Eelgrass"].map({
        True: "Eelgrass Present at Site",
        False: "No Eelgrass at Site"
    })
    
    return df


def add_four_category_habitat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create four-category habitat classification based on Rilee's email.
    
    Categories:
    1. Eelgrass - transect habitat is eelgrass
    2. Hard Bottom + Eelgrass Site - reef transect at site WITH eelgrass
    3. Hard Bottom - reef transect at site WITHOUT eelgrass  
    4. Soft Bottom - soft bottom transect (regardless of site eelgrass)
    """
    df = df.copy()
    
    # Define hard bottom habitats
    hard_bottom = ["Natural Reef", "Artificial Reef", "Kelp Forest"]
    
    def classify_habitat(row):
        if row["HabitatType"] == "Eelgrass":
            return "Eelgrass"
        elif row["HabitatType"] == "Soft Bottom":
            return "Soft Bottom"
        elif row["HabitatType"] in hard_bottom:
            if row["Site_Has_Eelgrass"]:
                return "Hard Bottom (Eelgrass Site)"
            else:
                return "Hard Bottom (No Eelgrass)"
        else:
            return "Other"
    
    df["Habitat_4Category"] = df.apply(classify_habitat, axis=1)
    
    return df


def plot_site_eelgrass_comparison(df: pd.DataFrame):
    """Compare encounter rates between sites with/without eelgrass."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Bar plot with error bars
    ax1 = axes[0]
    site_stats = df.groupby("Site_Eelgrass_Category")["Encounter.Rate.Hr"].agg(["mean", "std", "count"])
    site_stats["se"] = site_stats["std"] / np.sqrt(site_stats["count"])
    
    colors = ["#2ecc71", "#e74c3c"]  # Green for eelgrass, red for no eelgrass
    bars = ax1.bar(
        site_stats.index,
        site_stats["mean"],
        yerr=site_stats["se"],
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=1.2
    )
    
    ax1.set_ylabel("Mean Encounter Rate (per hour)")
    ax1.set_title("Encounter Rate: Sites With vs Without Eelgrass")
    
    for i, (idx, row) in enumerate(site_stats.iterrows()):
        ax1.annotate(
            f"n={int(row['count'])}\n({row['mean']:.2f} ± {row['se']:.2f})",
            xy=(i, row["mean"] + row["se"] + 0.3),
            ha="center", fontsize=10
        )
    
    # Box plot
    ax2 = axes[1]
    order = ["Eelgrass Present at Site", "No Eelgrass at Site"]
    sns.boxplot(
        data=df,
        x="Site_Eelgrass_Category",
        y="Encounter.Rate.Hr",
        order=order,
        hue="Site_Eelgrass_Category",
        palette={"Eelgrass Present at Site": "#2ecc71", "No Eelgrass at Site": "#e74c3c"},
        ax=ax2
    )
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()
    
    ax2.set_xlabel("")
    ax2.set_ylabel("Encounter Rate (per hour)")
    ax2.set_title("Distribution of Encounter Rates")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "site_eelgrass_comparison")
    plt.close()
    
    return site_stats


def plot_four_category_habitat(df: pd.DataFrame):
    """Bar plot of encounter rates by four-category habitat scheme."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out "Other" category
    df_plot = df[df["Habitat_4Category"] != "Other"]
    
    # Order categories logically
    category_order = [
        "Eelgrass",
        "Hard Bottom (Eelgrass Site)",
        "Soft Bottom",
        "Hard Bottom (No Eelgrass)"
    ]
    
    habitat_stats = df_plot.groupby("Habitat_4Category")["Encounter.Rate.Hr"].agg(["mean", "std", "count"])
    habitat_stats["se"] = habitat_stats["std"] / np.sqrt(habitat_stats["count"])
    habitat_stats = habitat_stats.reindex(category_order)
    
    # Color scheme: eelgrass-related in greens, non-eelgrass in oranges
    colors = ["#27ae60", "#2ecc71", "#f39c12", "#e67e22"]
    
    bars = ax.bar(
        range(len(habitat_stats)),
        habitat_stats["mean"],
        yerr=habitat_stats["se"],
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=1
    )
    
    ax.set_xticks(range(len(habitat_stats)))
    ax.set_xticklabels(habitat_stats.index, rotation=20, ha="right")
    ax.set_ylabel("Mean Encounter Rate (per hour)")
    ax.set_title("Pycnopodia Encounter Rate by Habitat Category\n(Four-Category Classification)")
    
    for i, (idx, row) in enumerate(habitat_stats.iterrows()):
        if pd.notna(row["mean"]):
            ax.annotate(
                f"n={int(row['count'])}",
                xy=(i, row["mean"] + row["se"] + 0.3),
                ha="center", fontsize=9, color="gray"
            )
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "four_category_habitat")
    plt.close()
    
    return habitat_stats


def plot_habitat_by_site_eelgrass_interaction(df: pd.DataFrame):
    """
    Interaction plot: transect habitat type × site eelgrass presence.
    
    Shows encounter rates for each transect habitat type, split by
    whether the site contains eelgrass or not.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate stats for each habitat × site-eelgrass combination
    interaction_stats = df.groupby(
        ["HabitatType", "Site_Eelgrass_Category"]
    )["Encounter.Rate.Hr"].agg(["mean", "std", "count"]).reset_index()
    interaction_stats["se"] = interaction_stats["std"] / np.sqrt(interaction_stats["count"])
    
    # Pivot for grouped bar plot
    habitats = df["HabitatType"].unique()
    x = np.arange(len(habitats))
    width = 0.35
    
    eelgrass_sites = interaction_stats[
        interaction_stats["Site_Eelgrass_Category"] == "Eelgrass Present at Site"
    ].set_index("HabitatType")
    
    no_eelgrass_sites = interaction_stats[
        interaction_stats["Site_Eelgrass_Category"] == "No Eelgrass at Site"
    ].set_index("HabitatType")
    
    # Reindex to ensure all habitats present
    eelgrass_sites = eelgrass_sites.reindex(habitats)
    no_eelgrass_sites = no_eelgrass_sites.reindex(habitats)
    
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
    ax.set_ylabel("Mean Encounter Rate (per hour)")
    ax.set_title("Encounter Rate by Transect Habitat Type\n(Split by Site-Level Eelgrass Presence)")
    ax.legend()
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "habitat_site_eelgrass_interaction")
    plt.close()


def plot_site_level_summary(df: pd.DataFrame):
    """
    Aggregate to site level and compare sites with/without eelgrass.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Aggregate to site level
    site_summary = df.groupby("SiteName").agg({
        "Encounter.Rate.Hr": "mean",
        "Pycnopodia_count": "sum",
        "Site_Has_Eelgrass": "first",
        "Basin": "first"
    }).reset_index()
    
    site_summary["Site_Eelgrass_Category"] = site_summary["Site_Has_Eelgrass"].map({
        True: "Eelgrass Present",
        False: "No Eelgrass"
    })
    
    # Scatter plot of site-level encounter rates
    ax1 = axes[0]
    for cat, color in [("Eelgrass Present", "#2ecc71"), ("No Eelgrass", "#e74c3c")]:
        subset = site_summary[site_summary["Site_Eelgrass_Category"] == cat]
        ax1.scatter(
            subset.index,
            subset["Encounter.Rate.Hr"],
            c=color,
            label=cat,
            alpha=0.7,
            s=50,
            edgecolor="black"
        )
    
    ax1.set_xlabel("Site Index")
    ax1.set_ylabel("Mean Site Encounter Rate (per hour)")
    ax1.set_title("Site-Level Encounter Rates")
    ax1.legend()
    
    # Box plot at site level
    ax2 = axes[1]
    sns.boxplot(
        data=site_summary,
        x="Site_Eelgrass_Category",
        y="Encounter.Rate.Hr",
        hue="Site_Eelgrass_Category",
        palette={"Eelgrass Present": "#2ecc71", "No Eelgrass": "#e74c3c"},
        ax=ax2
    )
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()
    
    # Add individual points
    sns.stripplot(
        data=site_summary,
        x="Site_Eelgrass_Category",
        y="Encounter.Rate.Hr",
        color="black",
        alpha=0.5,
        size=6,
        ax=ax2
    )
    
    ax2.set_xlabel("")
    ax2.set_ylabel("Mean Site Encounter Rate (per hour)")
    ax2.set_title("Site-Level Encounter Rate Distribution")
    
    # Add stats
    for i, cat in enumerate(["Eelgrass Present", "No Eelgrass"]):
        subset = site_summary[site_summary["Site_Eelgrass_Category"] == cat]
        n_sites = len(subset)
        mean_rate = subset["Encounter.Rate.Hr"].mean()
        ax2.annotate(
            f"n={n_sites} sites\nmean={mean_rate:.2f}",
            xy=(i, site_summary["Encounter.Rate.Hr"].max() * 0.9),
            ha="center", fontsize=9
        )
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "site_level_summary")
    plt.close()
    
    return site_summary


def run_statistical_tests(df: pd.DataFrame):
    """Run statistical tests comparing site categories."""
    print("\n" + "="*50)
    print("STATISTICAL TESTS")
    print("="*50)
    
    results = []
    
    # Test 1: Site-level eelgrass presence (transect-level data)
    print("\n--- Test 1: Transect-Level Encounter Rate by Site Eelgrass Presence ---")
    eelgrass_sites = df[df["Site_Has_Eelgrass"]]["Encounter.Rate.Hr"]
    no_eelgrass_sites = df[~df["Site_Has_Eelgrass"]]["Encounter.Rate.Hr"]
    
    u_stat, p_val = stats.mannwhitneyu(eelgrass_sites, no_eelgrass_sites, alternative="two-sided")
    
    print(f"Eelgrass Sites: n={len(eelgrass_sites)}, mean={eelgrass_sites.mean():.3f}")
    print(f"No-Eelgrass Sites: n={len(no_eelgrass_sites)}, mean={no_eelgrass_sites.mean():.3f}")
    print(f"Mann-Whitney U: {u_stat:.0f}, p-value: {p_val:.2e}")
    print(f"Significant at α=0.05: {'Yes' if p_val < 0.05 else 'No'}")
    
    # Effect size (rank-biserial correlation)
    n1, n2 = len(eelgrass_sites), len(no_eelgrass_sites)
    effect_size = 1 - (2 * u_stat) / (n1 * n2)
    print(f"Effect size (r): {effect_size:.3f}")
    
    results.append({
        "Test": "Site Eelgrass Presence (transect-level)",
        "Statistic": u_stat,
        "p-value": p_val,
        "Effect Size": effect_size,
        "Significant": p_val < 0.05
    })
    
    # Test 2: Four-category Kruskal-Wallis
    print("\n--- Test 2: Four-Category Habitat (Kruskal-Wallis) ---")
    df_4cat = df[df["Habitat_4Category"] != "Other"]
    categories = df_4cat["Habitat_4Category"].unique()
    groups = [df_4cat[df_4cat["Habitat_4Category"] == cat]["Encounter.Rate.Hr"].values for cat in categories]
    
    h_stat, p_val = stats.kruskal(*groups)
    print(f"H-statistic: {h_stat:.3f}")
    print(f"p-value: {p_val:.2e}")
    print(f"Significant at α=0.05: {'Yes' if p_val < 0.05 else 'No'}")
    
    results.append({
        "Test": "Four-Category Habitat (Kruskal-Wallis)",
        "Statistic": h_stat,
        "p-value": p_val,
        "Effect Size": np.nan,
        "Significant": p_val < 0.05
    })
    
    # Test 3: Site-level comparison
    print("\n--- Test 3: Site-Level Mean Encounter Rate ---")
    site_summary = df.groupby("SiteName").agg({
        "Encounter.Rate.Hr": "mean",
        "Site_Has_Eelgrass": "first"
    })
    
    eelgrass_site_means = site_summary[site_summary["Site_Has_Eelgrass"]]["Encounter.Rate.Hr"]
    no_eelgrass_site_means = site_summary[~site_summary["Site_Has_Eelgrass"]]["Encounter.Rate.Hr"]
    
    u_stat, p_val = stats.mannwhitneyu(eelgrass_site_means, no_eelgrass_site_means, alternative="two-sided")
    
    print(f"Sites with Eelgrass: n={len(eelgrass_site_means)}, mean={eelgrass_site_means.mean():.3f}")
    print(f"Sites without Eelgrass: n={len(no_eelgrass_site_means)}, mean={no_eelgrass_site_means.mean():.3f}")
    print(f"Mann-Whitney U: {u_stat:.0f}, p-value: {p_val:.2e}")
    print(f"Significant at α=0.05: {'Yes' if p_val < 0.05 else 'No'}")
    
    n1, n2 = len(eelgrass_site_means), len(no_eelgrass_site_means)
    effect_size = 1 - (2 * u_stat) / (n1 * n2)
    print(f"Effect size (r): {effect_size:.3f}")
    
    results.append({
        "Test": "Site-Level Mean Encounter Rate",
        "Statistic": u_stat,
        "p-value": p_val,
        "Effect Size": effect_size,
        "Significant": p_val < 0.05
    })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "statistical_tests.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")
    
    return results_df


def compute_site_habitat_matrix_by_eelgrass(df: pd.DataFrame):
    """
    Compute habitat composition matrices split by site eelgrass presence.
    
    Returns separate matrices for eelgrass sites and non-eelgrass sites.
    """
    # Compute habitat ratios per site
    habitat_ratios = (
        df.groupby(["SiteName", "HabitatType"])
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
    
    # Mean encounter rate per site
    encounter_means = df.groupby("SiteName")["Encounter.Rate.Hr"].mean()
    
    # Site eelgrass indicator
    site_eelgrass = df.groupby("SiteName")["Site_Has_Eelgrass"].first()
    
    return habitat_matrix, encounter_means, site_eelgrass


def plot_clustermap_by_eelgrass_status(
    habitat_matrix: pd.DataFrame,
    encounter_means: pd.Series,
    site_eelgrass: pd.Series,
    title: str,
    filename: str,
    figsize=(6, 15),
    cmap_main="Greens",
    cmap_encounter="Reds"
):
    """
    Create clustermap with sites colored by eelgrass presence.
    """
    sns.set(style="whitegrid", font_scale=0.6)
    
    # Add eelgrass indicator as row colors
    row_colors = site_eelgrass.map({True: "#2ecc71", False: "#e74c3c"})
    row_colors = row_colors.reindex(habitat_matrix.index)
    
    # Compute linkage
    link = linkage(habitat_matrix, method="ward")
    
    # Create clustermap with row colors
    g = sns.clustermap(
        habitat_matrix,
        row_linkage=link,
        col_cluster=False,
        cmap=cmap_main,
        figsize=figsize,
        vmin=0, vmax=1,
        dendrogram_ratio=(0.15, 0.1),
        row_colors=row_colors,
        cbar_pos=(0.02, 0.8, 0.03, 0.15)
    )
    
    # Get ordered indices
    heatmap_ax = g.ax_heatmap
    ordered_inds = g.dendrogram_row.reordered_ind
    ordered_index = habitat_matrix.index[ordered_inds]
    ordered_rates = encounter_means.reindex(ordered_index)
    
    # Add encounter rate rectangles
    enc_min, enc_max = encounter_means.min(), encounter_means.max()
    norm_enc = plt.Normalize(enc_min, enc_max)
    cmap_enc = plt.get_cmap(cmap_encounter)
    
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
    
    # Format labels
    n_rows = habitat_matrix.shape[0]
    heatmap_ax.set_yticks(np.arange(n_rows))
    heatmap_ax.set_yticklabels(ordered_index, fontsize=5)
    
    # Shift labels
    trans = mtransforms.ScaledTranslation(15 / 72.0, 0, heatmap_ax.figure.dpi_scale_trans)
    for label in heatmap_ax.get_yticklabels():
        label.set_transform(label.get_transform() + trans)
    
    g.fig.suptitle(title, y=1.02, fontsize=10)
    
    # Add legend for row colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Eelgrass at Site"),
        Patch(facecolor="#e74c3c", label="No Eelgrass at Site")
    ]
    g.ax_heatmap.legend(
        handles=legend_elements, 
        loc="upper left",
        bbox_to_anchor=(1.1, 1.0),
        fontsize=7
    )
    
    plt.subplots_adjust(left=0.02, right=0.85)
    
    # Save
    clustermap_path = OUTPUT_DIR / f"{filename}.png"
    g.savefig(clustermap_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {clustermap_path}")
    plt.close()


def plot_split_clustermaps(df: pd.DataFrame):
    """
    Create separate clustermaps for eelgrass sites and non-eelgrass sites.
    """
    # Get sites in each category
    eelgrass_sites = df[df["Site_Has_Eelgrass"]]["SiteName"].unique()
    no_eelgrass_sites = df[~df["Site_Has_Eelgrass"]]["SiteName"].unique()
    
    for sites, label, color in [
        (eelgrass_sites, "eelgrass_sites", "YlGn"),
        (no_eelgrass_sites, "no_eelgrass_sites", "OrRd")
    ]:
        df_subset = df[df["SiteName"].isin(sites)]
        
        # Compute habitat matrix
        habitat_ratios = (
            df_subset.groupby(["SiteName", "HabitatType"])
            .size()
            .reset_index(name="Count")
        )
        habitat_ratios["Ratio"] = habitat_ratios.groupby("SiteName")["Count"].transform(
            lambda x: x / x.sum()
        )
        
        habitat_matrix = habitat_ratios.pivot_table(
            index="SiteName",
            columns="HabitatType",
            values="Ratio",
            fill_value=0
        )
        
        encounter_means = df_subset.groupby("SiteName")["Encounter.Rate.Hr"].mean()
        encounter_means = encounter_means.reindex(habitat_matrix.index)
        
        if len(habitat_matrix) < 2:
            print(f"  Skipping {label}: not enough sites")
            continue
        
        # Create clustermap
        sns.set(style="whitegrid", font_scale=0.6)
        link = linkage(habitat_matrix, method="ward")
        
        figsize = (6, max(8, len(habitat_matrix) * 0.3))
        
        g = sns.clustermap(
            habitat_matrix,
            row_linkage=link,
            col_cluster=False,
            cmap=color,
            figsize=figsize,
            vmin=0, vmax=1,
            dendrogram_ratio=(0.15, 0.1),
            cbar_pos=(0.02, 0.8, 0.03, 0.15)
        )
        
        # Add encounter rate rectangles
        heatmap_ax = g.ax_heatmap
        ordered_inds = g.dendrogram_row.reordered_ind
        ordered_index = habitat_matrix.index[ordered_inds]
        ordered_rates = encounter_means.loc[ordered_index]
        
        enc_min, enc_max = encounter_means.min(), encounter_means.max()
        if enc_max > enc_min:
            norm_enc = plt.Normalize(enc_min, enc_max)
        else:
            norm_enc = plt.Normalize(0, 1)
        cmap_enc = plt.get_cmap("Reds")
        
        xmax = len(habitat_matrix.columns)
        for i, rate in enumerate(ordered_rates):
            color_rect = cmap_enc(norm_enc(rate)) if not np.isnan(rate) else (0.9, 0.9, 0.9, 1)
            rect = Rectangle(
                (xmax + 0.1, i - 0.5),
                0.4, 1.0,
                facecolor=color_rect,
                transform=heatmap_ax.transData,
                clip_on=False,
                edgecolor="none"
            )
            heatmap_ax.add_patch(rect)
        
        # Labels
        n_rows = habitat_matrix.shape[0]
        heatmap_ax.set_yticks(np.arange(n_rows))
        heatmap_ax.set_yticklabels(ordered_index, fontsize=6)
        
        trans = mtransforms.ScaledTranslation(15 / 72.0, 0, heatmap_ax.figure.dpi_scale_trans)
        for lbl in heatmap_ax.get_yticklabels():
            lbl.set_transform(lbl.get_transform() + trans)
        
        title = "Sites WITH Eelgrass" if "eelgrass_sites" == label else "Sites WITHOUT Eelgrass"
        g.fig.suptitle(f"Habitat Clustering: {title}\n(n={len(habitat_matrix)} sites)", y=1.02)
        
        plt.subplots_adjust(left=0.02, right=0.85)
        
        clustermap_path = OUTPUT_DIR / f"clustermap_{label}.png"
        g.savefig(clustermap_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {clustermap_path}")
        plt.close()


def plot_four_category_clustering(df: pd.DataFrame):
    """
    Cluster sites based on four-category habitat composition.
    """
    df_4cat = df[df["Habitat_4Category"] != "Other"].copy()
    
    # Compute four-category ratios per site
    habitat_ratios = (
        df_4cat.groupby(["SiteName", "Habitat_4Category"])
        .size()
        .reset_index(name="Count")
    )
    habitat_ratios["Ratio"] = habitat_ratios.groupby("SiteName")["Count"].transform(
        lambda x: x / x.sum()
    )
    
    habitat_matrix = habitat_ratios.pivot_table(
        index="SiteName",
        columns="Habitat_4Category",
        values="Ratio",
        fill_value=0
    )
    
    # Reorder columns logically
    col_order = [
        "Eelgrass",
        "Hard Bottom (Eelgrass Site)",
        "Soft Bottom",
        "Hard Bottom (No Eelgrass)"
    ]
    col_order = [c for c in col_order if c in habitat_matrix.columns]
    habitat_matrix = habitat_matrix[col_order]
    
    encounter_means = df_4cat.groupby("SiteName")["Encounter.Rate.Hr"].mean()
    encounter_means = encounter_means.reindex(habitat_matrix.index)
    
    site_eelgrass = df.groupby("SiteName")["Site_Has_Eelgrass"].first()
    row_colors = site_eelgrass.reindex(habitat_matrix.index).map({
        True: "#2ecc71", False: "#e74c3c"
    })
    
    # Create clustermap
    sns.set(style="whitegrid", font_scale=0.6)
    link = linkage(habitat_matrix, method="ward")
    
    g = sns.clustermap(
        habitat_matrix,
        row_linkage=link,
        col_cluster=False,
        cmap="viridis",
        figsize=(8, 18),
        vmin=0, vmax=1,
        dendrogram_ratio=(0.15, 0.1),
        row_colors=row_colors,
        cbar_pos=(0.02, 0.8, 0.03, 0.15)
    )
    
    # Add encounter rate rectangles
    heatmap_ax = g.ax_heatmap
    ordered_inds = g.dendrogram_row.reordered_ind
    ordered_index = habitat_matrix.index[ordered_inds]
    ordered_rates = encounter_means.loc[ordered_index]
    
    enc_min, enc_max = encounter_means.min(), encounter_means.max()
    norm_enc = plt.Normalize(enc_min, enc_max)
    cmap_enc = plt.get_cmap("inferno")
    
    xmax = len(habitat_matrix.columns)
    for i, rate in enumerate(ordered_rates):
        color_rect = cmap_enc(norm_enc(rate)) if not np.isnan(rate) else (0.9, 0.9, 0.9, 1)
        rect = Rectangle(
            (xmax + 0.1, i - 0.5),
            0.4, 1.0,
            facecolor=color_rect,
            transform=heatmap_ax.transData,
            clip_on=False,
            edgecolor="none"
        )
        heatmap_ax.add_patch(rect)
    
    # Labels
    n_rows = habitat_matrix.shape[0]
    heatmap_ax.set_yticks(np.arange(n_rows))
    heatmap_ax.set_yticklabels(ordered_index, fontsize=5)
    
    trans = mtransforms.ScaledTranslation(15 / 72.0, 0, heatmap_ax.figure.dpi_scale_trans)
    for lbl in heatmap_ax.get_yticklabels():
        lbl.set_transform(lbl.get_transform() + trans)
    
    g.fig.suptitle("Site Clustering by Four-Category Habitat Composition\n(Green sidebar = Eelgrass Site, Red = No Eelgrass)", y=1.02, fontsize=10)
    
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
    
    clustermap_path = OUTPUT_DIR / "clustermap_four_category.png"
    g.savefig(clustermap_path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {clustermap_path}")
    plt.close()


def plot_substrate_by_eelgrass_site(df: pd.DataFrame):
    """
    Compare substrate encounter rates split by site eelgrass presence.
    Shows both soft bottom and hard bottom comparisons.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Soft bottom comparison
    ax1 = axes[0]
    soft_bottom = df[df["HabitatType"] == "Soft Bottom"]
    
    sb_stats = soft_bottom.groupby("Site_Eelgrass_Category")["Encounter.Rate.Hr"].agg(
        ["mean", "std", "count"]
    )
    sb_stats["se"] = sb_stats["std"] / np.sqrt(sb_stats["count"])
    
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax1.bar(
        sb_stats.index,
        sb_stats["mean"],
        yerr=sb_stats["se"],
        capsize=5,
        color=colors,
        edgecolor="black"
    )
    
    ax1.set_ylabel("Mean Encounter Rate (per hour)")
    ax1.set_title("SOFT BOTTOM Transects\nby Site Eelgrass Presence")
    
    for i, (idx, row) in enumerate(sb_stats.iterrows()):
        ax1.annotate(
            f"n={int(row['count'])}\n{row['mean']:.2f} ± {row['se']:.2f}",
            xy=(i, row["mean"] + row["se"] + 0.5),
            ha="center", fontsize=9
        )
    
    # Hard bottom comparison
    ax2 = axes[1]
    hard_bottom = df[df["HabitatType"].isin(["Natural Reef", "Artificial Reef", "Kelp Forest"])]
    
    hb_stats = hard_bottom.groupby("Site_Eelgrass_Category")["Encounter.Rate.Hr"].agg(
        ["mean", "std", "count"]
    )
    hb_stats["se"] = hb_stats["std"] / np.sqrt(hb_stats["count"])
    
    bars = ax2.bar(
        hb_stats.index,
        hb_stats["mean"],
        yerr=hb_stats["se"],
        capsize=5,
        color=colors,
        edgecolor="black"
    )
    
    ax2.set_ylabel("Mean Encounter Rate (per hour)")
    ax2.set_title("HARD BOTTOM Transects\n(Reef + Kelp) by Site Eelgrass Presence")
    
    for i, (idx, row) in enumerate(hb_stats.iterrows()):
        ax2.annotate(
            f"n={int(row['count'])}\n{row['mean']:.2f} ± {row['se']:.2f}",
            xy=(i, row["mean"] + row["se"] + 0.2),
            ha="center", fontsize=9
        )
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "substrate_by_eelgrass_site")
    plt.close()
    
    return sb_stats, hb_stats


def generate_summary_tables(df: pd.DataFrame):
    """Generate summary tables for publication."""
    
    # Site-level summary
    site_summary = df.groupby("Site_Eelgrass_Category").agg({
        "Encounter.Rate.Hr": ["mean", "std", "count"],
        "Pycnopodia_count": "sum",
        "SiteName": "nunique"
    }).round(3)
    site_summary.columns = ["Mean Rate", "Std Dev", "N Transects", "Total Pycno", "N Sites"]
    
    site_path = OUTPUT_DIR / "site_eelgrass_summary.csv"
    site_summary.to_csv(site_path)
    print(f"  Saved: {site_path}")
    
    # Four-category summary
    df_4cat = df[df["Habitat_4Category"] != "Other"]
    four_cat_summary = df_4cat.groupby("Habitat_4Category").agg({
        "Encounter.Rate.Hr": ["mean", "std", "count"],
        "Pycnopodia_count": "sum"
    }).round(3)
    four_cat_summary.columns = ["Mean Rate", "Std Dev", "N Transects", "Total Pycno"]
    
    # Detection rate
    four_cat_summary["Detection Rate"] = df_4cat.groupby("Habitat_4Category").apply(
        lambda x: (x["Pycnopodia_count"] > 0).mean()
    ).round(3)
    
    four_cat_path = OUTPUT_DIR / "four_category_summary.csv"
    four_cat_summary.to_csv(four_cat_path)
    print(f"  Saved: {four_cat_path}")
    
    return site_summary, four_cat_summary


def main():
    print(f"\n{'='*60}")
    print("EELGRASS SITE-LEVEL ANALYSIS")
    print(f"{'='*60}\n")
    
    set_style()
    df = load_data()
    
    # Add site-level classifications
    print("Adding site-level eelgrass indicator...")
    df = add_site_eelgrass_indicator(df)
    
    print("Creating four-category habitat classification...")
    df = add_four_category_habitat(df)
    
    # Summary stats
    n_sites_eelgrass = df[df["Site_Has_Eelgrass"]]["SiteName"].nunique()
    n_sites_no_eelgrass = df[~df["Site_Has_Eelgrass"]]["SiteName"].nunique()
    
    print(f"\nSite Classification:")
    print(f"  Sites WITH eelgrass: {n_sites_eelgrass}")
    print(f"  Sites WITHOUT eelgrass: {n_sites_no_eelgrass}")
    print(f"  Total sites: {df['SiteName'].nunique()}")
    
    print(f"\nFour-Category Breakdown:")
    for cat in df["Habitat_4Category"].value_counts().index:
        n = (df["Habitat_4Category"] == cat).sum()
        print(f"  {cat}: {n} transects")
    
    print(f"\nOutput directory: {OUTPUT_DIR}\n")
    
    # Generate plots
    print("Generating figures...")
    plot_site_eelgrass_comparison(df)
    plot_four_category_habitat(df)
    plot_habitat_by_site_eelgrass_interaction(df)
    plot_site_level_summary(df)
    
    print("\nGenerating substrate comparison (soft/hard bottom by eelgrass site)...")
    sb_stats, hb_stats = plot_substrate_by_eelgrass_site(df)
    
    print("\n--- Soft Bottom by Site Eelgrass ---")
    print(sb_stats.round(3).to_string())
    print("\n--- Hard Bottom by Site Eelgrass ---")
    print(hb_stats.round(3).to_string())
    
    # Clustering visualizations
    print("\nGenerating clustering visualizations...")
    habitat_matrix, encounter_means, site_eelgrass = compute_site_habitat_matrix_by_eelgrass(df)
    
    plot_clustermap_by_eelgrass_status(
        habitat_matrix, encounter_means, site_eelgrass,
        title="Site Clustering by Habitat Composition\n(Colored by Eelgrass Presence)",
        filename="clustermap_all_sites_by_eelgrass"
    )
    
    plot_split_clustermaps(df)
    plot_four_category_clustering(df)
    
    # Statistical tests
    run_statistical_tests(df)
    
    # Summary tables
    print("\nGenerating summary tables...")
    site_summary, four_cat_summary = generate_summary_tables(df)
    
    print("\n--- Site Eelgrass Summary ---")
    print(site_summary.to_string())
    
    print("\n--- Four-Category Summary ---")
    print(four_cat_summary.to_string())
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

