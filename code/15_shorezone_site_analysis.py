"""
ShoreZone Site Analysis

Performs buffer-based spatial join between Pycnopodia survey sites and
Washington State ShoreZone inventory data. Characterizes the local shoreline
habitat diversity within 250m of each dive site.

Spatial Join Approach:
- Each site point is buffered by 250m radius
- All ShoreZone line segments intersecting the buffer are captured
- Segment lengths within buffer are calculated for weighting
- Categorical attributes are aggregated to site level preserving diversity

IMPORTANT DATA LIMITATIONS:
- ShoreZone data collected 1994-2000; survey data from 2020-2024
- Biotic features (eelgrass, kelp, algae) may have changed significantly
- Physical features (exposure, sediment, substrate) are more temporally stable
- Features are classified as STABLE vs DYNAMIC accordingly

Coordinate Reference Systems:
- ShoreZone native CRS: EPSG:2927 (WA State Plane South, feet)
- Site coordinates: WGS84 (EPSG:4326) from diver GPS
- Processing CRS: UTM 10N (EPSG:32610) for metric buffering
- All outputs in WGS84 for compatibility

Outputs:
- Joined site-shorezone data (GeoPackage and CSV)
- Site-level ShoreZone category proportions
- Habitat diversity indices per site (STABLE features prioritized)
- Feature correlation matrix and composite features
- Distance-to-shore for each site
- Visualizations of shorezone habitat composition
- Integration with Pycnopodia encounter rate data

Dependencies:
    Requires geospatial packages: geopandas, fiona, pyproj, shapely
    Install: conda env update -f code/environment.yml

Author: Star Meadow Project
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster

# Local imports
from utils import get_output_dir, load_data, set_style, save_figure, DATA_DIR

# Import shorezone utilities (will raise ImportError if geospatial deps missing)
try:
    from shorezone_utils import (
        load_site_points,
        load_shorezone_lines,
        list_shorezone_layers,
        buffer_spatial_join,
        nearest_shorezone_join,
        aggregate_shorezone_by_site,
        compute_shorezone_diversity,
        export_shorezone_join,
        run_shorezone_join_pipeline,
        HAS_GEOSPATIAL,
        SHOREZONE_GDB,
        SHOREZONE_THEMES_GDB,
        CRS_WGS84,
        CRS_UTM10N,
    )
except ImportError as e:
    HAS_GEOSPATIAL = False
    CRS_WGS84 = "EPSG:4326"
    CRS_UTM10N = "EPSG:32610"
    print(f"Warning: Could not import shorezone_utils: {e}")

OUTPUT_DIR = get_output_dir(__file__)

# Maximum distance to ShoreZone for matching (in meters)
# Sites beyond this distance will have no ShoreZone data
MAX_DISTANCE_M = 3000  # 3 km - captures offshore dive sites near shore

# =============================================================================
# FEATURE CLASSIFICATION
# =============================================================================
# Features are classified by temporal stability given 24-30 year data lag

# STABLE FEATURES: Physical characteristics unlikely to change over decades
STABLE_FEATURES = [
    "EXP_CLASS",    # Wave exposure class - determined by geomorphology
    "SED_SOURCE",   # Sediment source - geological
    "SED_ABUND",    # Sediment abundance - geological
    "ZONECOMP",     # Zone composition - structural
]

# DYNAMIC FEATURES: Biotic features that may have changed since 1994-2000
# Use with caution - primarily for exploratory analysis
DYNAMIC_FEATURES = [
    "BC_CLASS",     # Biotic class code - biological
    "NRDA_CLASS",   # NRDA habitat class - includes biotic
    "HAB_CALC",     # Calculated habitat type - includes biotic
    "ZOS_UNIT",     # Zostera (eelgrass) - HIGHLY DYNAMIC (declines documented)
    "VER_UNIT",     # Verrucaria (lichen) - biological
    "GRA_UNIT",     # Gracilaria (algae) - biological
    "FUC_UNIT",     # Fucus (rockweed) - biological
    "ULV_UNIT",     # Ulva (sea lettuce) - biological
    "MUS_UNIT",     # Mussel - biological
    "OYS_UNIT",     # Oyster - biological
    "MAC_UNIT",     # Macrocystis (giant kelp) - biological
    "NER_UNIT",     # Nereocystis (bull kelp) - biological
]

# METADATA: Not useful for ecological analysis
METADATA_COLS = ["CHNG_TYPE", "GEO_MAPPER", "UNIT_TYPE"]


def validate_crs_alignment(sites, shorezone):
    """
    Validate and report CRS alignment between sites and ShoreZone.
    
    Critical for ensuring spatial joins are accurate.
    """
    print("\n" + "=" * 60)
    print("CRS VALIDATION")
    print("=" * 60)
    
    print(f"\nShoreZone native CRS: {shorezone.crs}")
    print(f"  EPSG: {shorezone.crs.to_epsg()}")
    
    print(f"\nSite points CRS: {sites.crs}")
    print(f"  EPSG: {sites.crs.to_epsg()}")
    
    # Check bounds overlap
    sz_bounds = shorezone.to_crs(CRS_WGS84).total_bounds
    site_bounds = sites.to_crs(CRS_WGS84).total_bounds
    
    print(f"\nShoreZone bounds (WGS84):")
    print(f"  Lon: {sz_bounds[0]:.4f} to {sz_bounds[2]:.4f}")
    print(f"  Lat: {sz_bounds[1]:.4f} to {sz_bounds[3]:.4f}")
    
    print(f"\nSite bounds (WGS84):")
    print(f"  Lon: {site_bounds[0]:.4f} to {site_bounds[2]:.4f}")
    print(f"  Lat: {site_bounds[1]:.4f} to {site_bounds[3]:.4f}")
    
    # Check which sites are outside ShoreZone bounding box
    sites_wgs = sites.to_crs(CRS_WGS84)
    in_bounds = (
        (sites_wgs.geometry.x >= sz_bounds[0]) & 
        (sites_wgs.geometry.x <= sz_bounds[2]) &
        (sites_wgs.geometry.y >= sz_bounds[1]) & 
        (sites_wgs.geometry.y <= sz_bounds[3])
    )
    
    n_outside = (~in_bounds).sum()
    if n_outside > 0:
        print(f"\n⚠️  {n_outside} sites outside ShoreZone bounding box")
        outside_sites = sites_wgs[~in_bounds]["SiteName"].tolist()
        print(f"    Sites: {outside_sites[:5]}{'...' if len(outside_sites) > 5 else ''}")
    else:
        print(f"\n✓ All {len(sites)} sites within ShoreZone coverage area")
    
    return in_bounds


def compute_distance_to_shore(sites, shorezone):
    """
    Calculate minimum distance from each site to nearest ShoreZone segment.
    
    Sites far from shore (>250m) may have unreliable ShoreZone associations
    because the buffer artificially connects them to distant shoreline features.
    
    Returns DataFrame with SiteName and distance_to_shore_m.
    """
    print("\n" + "=" * 60)
    print("COMPUTING DISTANCE TO SHORE")
    print("=" * 60)
    
    # Project to UTM for metric distances
    sites_proj = sites.to_crs(CRS_UTM10N)
    sz_proj = shorezone.to_crs(CRS_UTM10N)
    
    # Union all shorezone segments for distance calculation
    from shapely.ops import unary_union
    shore_union = unary_union(sz_proj.geometry)
    
    distances = []
    for idx, row in sites_proj.iterrows():
        dist = row.geometry.distance(shore_union)
        distances.append({
            "SiteName": row["SiteName"],
            "distance_to_shore_m": dist
        })
    
    dist_df = pd.DataFrame(distances)
    
    # Summary stats
    print(f"\nDistance to shore statistics:")
    print(f"  Min: {dist_df['distance_to_shore_m'].min():.0f} m")
    print(f"  Max: {dist_df['distance_to_shore_m'].max():.0f} m")
    print(f"  Mean: {dist_df['distance_to_shore_m'].mean():.0f} m")
    print(f"  Median: {dist_df['distance_to_shore_m'].median():.0f} m")
    
    # Flag sites beyond max distance (no ShoreZone match possible)
    n_beyond_max = (dist_df["distance_to_shore_m"] > MAX_DISTANCE_M).sum()
    print(f"\n⚠️  {n_beyond_max} sites > {MAX_DISTANCE_M/1000:.1f}km from shore (no match)")
    print(f"    These sites have weak ShoreZone signal (buffer extends beyond data)")
    
    return dist_df


def analyze_feature_correlations(merged: pd.DataFrame, diversity_cols: list):
    """
    Analyze correlations between ShoreZone diversity features.
    
    Identifies highly correlated feature groups that could be combined
    into composite features to reduce dimensionality.
    
    Returns correlation matrix and suggested feature groups.
    """
    print("\n" + "=" * 60)
    print("FEATURE CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Get only diversity columns that exist and have variance
    valid_cols = []
    for col in diversity_cols:
        if col in merged.columns:
            if merged[col].notna().sum() > 10 and merged[col].std() > 0.01:
                valid_cols.append(col)
    
    if len(valid_cols) < 2:
        print("  Insufficient features for correlation analysis")
        return None, {}
    
    # Compute correlation matrix
    corr_matrix = merged[valid_cols].corr(method="spearman")
    
    print(f"\nAnalyzing {len(valid_cols)} features...")
    
    # Find highly correlated pairs (|r| > 0.7)
    high_corr_pairs = []
    for i, col1 in enumerate(valid_cols):
        for col2 in valid_cols[i+1:]:
            r = corr_matrix.loc[col1, col2]
            if abs(r) > 0.7:
                high_corr_pairs.append((col1, col2, r))
    
    if high_corr_pairs:
        print(f"\nHighly correlated feature pairs (|ρ| > 0.7):")
        for col1, col2, r in sorted(high_corr_pairs, key=lambda x: -abs(x[2]))[:10]:
            print(f"  {col1} ↔ {col2}: ρ = {r:.3f}")
    else:
        print("\n  No highly correlated pairs found (|ρ| > 0.7)")
    
    # Group features by base category (e.g., EXP_CLASS_Richness, EXP_CLASS_Shannon)
    feature_groups = {}
    for col in valid_cols:
        # Extract base name (everything before _Richness/_Shannon/_Simpson/_Evenness)
        for suffix in ["_Richness", "_Shannon", "_Simpson", "_Evenness"]:
            if col.endswith(suffix):
                base = col.replace(suffix, "")
                if base not in feature_groups:
                    feature_groups[base] = []
                feature_groups[base].append(col)
                break
    
    print(f"\nFeature groups by category:")
    for base, cols in feature_groups.items():
        print(f"  {base}: {len(cols)} metrics")
    
    return corr_matrix, feature_groups


def create_composite_features(merged: pd.DataFrame, feature_groups: dict):
    """
    Create composite features by averaging highly correlated metrics.
    
    For each category, creates:
    - <category>_Diversity: mean of Shannon and Simpson (both measure evenness)
    - Keeps Richness separate (measures different concept)
    
    Also creates ecosystem-level composites:
    - Physical_Diversity: mean across stable physical features
    - Biotic_Diversity: mean across dynamic biotic features
    """
    print("\n" + "=" * 60)
    print("CREATING COMPOSITE FEATURES")
    print("=" * 60)
    
    composite_df = merged[["SiteName"]].copy()
    
    # Per-category composites: combine Shannon + Simpson
    for base, cols in feature_groups.items():
        shannon_col = f"{base}_Shannon"
        simpson_col = f"{base}_Simpson"
        richness_col = f"{base}_Richness"
        
        # Create diversity composite (Shannon + Simpson average)
        if shannon_col in merged.columns and simpson_col in merged.columns:
            composite_df[f"{base}_Diversity"] = merged[[shannon_col, simpson_col]].mean(axis=1)
        
        # Keep richness as-is
        if richness_col in merged.columns:
            composite_df[f"{base}_Richness"] = merged[richness_col]
    
    # Ecosystem-level composites
    stable_diversity_cols = [f"{feat}_Diversity" for feat in STABLE_FEATURES 
                             if f"{feat}_Diversity" in composite_df.columns]
    dynamic_diversity_cols = [f"{feat}_Diversity" for feat in DYNAMIC_FEATURES 
                              if f"{feat}_Diversity" in composite_df.columns]
    
    if stable_diversity_cols:
        composite_df["Physical_Diversity"] = composite_df[stable_diversity_cols].mean(axis=1)
        print(f"  Created Physical_Diversity from {len(stable_diversity_cols)} stable features")
    
    if dynamic_diversity_cols:
        composite_df["Biotic_Diversity"] = composite_df[dynamic_diversity_cols].mean(axis=1)
        print(f"  Created Biotic_Diversity from {len(dynamic_diversity_cols)} dynamic features")
    
    # Richness composites
    stable_richness_cols = [f"{feat}_Richness" for feat in STABLE_FEATURES 
                            if f"{feat}_Richness" in composite_df.columns]
    dynamic_richness_cols = [f"{feat}_Richness" for feat in DYNAMIC_FEATURES 
                             if f"{feat}_Richness" in composite_df.columns]
    
    if stable_richness_cols:
        composite_df["Physical_Richness"] = composite_df[stable_richness_cols].mean(axis=1)
        print(f"  Created Physical_Richness from {len(stable_richness_cols)} stable features")
    
    if dynamic_richness_cols:
        composite_df["Biotic_Richness"] = composite_df[dynamic_richness_cols].mean(axis=1)
        print(f"  Created Biotic_Richness from {len(dynamic_richness_cols)} dynamic features")
    
    n_composites = len([c for c in composite_df.columns if c != "SiteName"])
    print(f"\nTotal composite features: {n_composites}")
    
    return composite_df


def explore_shorezone_schema():
    """
    Explore and document the ShoreZone geodatabase schema.
    
    Prints available layers and their column names.
    """
    print("\n" + "=" * 60)
    print("SHOREZONE GEODATABASE SCHEMA EXPLORATION")
    print("=" * 60)
    
    layers = list_shorezone_layers()
    
    for gdb_name, layer_list in layers.items():
        print(f"\n{gdb_name}:")
        print("-" * 40)
        
        if isinstance(layer_list, list):
            for layer_name in layer_list:
                print(f"  • {layer_name}")
        else:
            print(f"  {layer_list}")
    
    return layers


def load_and_explore_shorezone():
    """
    Load ShoreZone data and explore its structure.
    """
    print("\n" + "=" * 60)
    print("LOADING SHOREZONE DATA")
    print("=" * 60)
    
    # Load primary line layer
    shorezone = load_shorezone_lines(layer_name="szline")
    
    print(f"\nShoreZone Line Data Overview:")
    print(f"  Total features: {len(shorezone)}")
    print(f"  Geometry type: {shorezone.geom_type.unique()}")
    print(f"  CRS: {shorezone.crs}")
    
    print(f"\nColumns ({len(shorezone.columns)}):")
    for col in shorezone.columns:
        if col != "geometry":
            dtype = shorezone[col].dtype
            n_unique = shorezone[col].nunique()
            print(f"  {col}: {dtype} ({n_unique} unique values)")
    
    # Identify categorical columns
    categorical_cols = []
    for col in shorezone.columns:
        if col != "geometry":
            if shorezone[col].dtype == "object" and shorezone[col].nunique() < 50:
                categorical_cols.append(col)
    
    print(f"\nPotential categorical columns for analysis:")
    for col in categorical_cols[:10]:  # Show first 10
        print(f"  {col}: {list(shorezone[col].unique()[:5])}...")
    
    return shorezone, categorical_cols


def run_nearest_join(shorezone, max_distance_m: float = MAX_DISTANCE_M):
    """
    Run the nearest-neighbor spatial join between sites and ShoreZone.
    
    Each site is matched to its single nearest ShoreZone segment,
    up to max_distance_m away.
    """
    print("\n" + "=" * 60)
    print(f"NEAREST-NEIGHBOR SPATIAL JOIN (max distance = {max_distance_m/1000:.1f} km)")
    print("=" * 60)
    
    # Load sites
    sites = load_site_points()
    
    # Perform nearest-neighbor join
    joined = nearest_shorezone_join(sites, shorezone, max_distance_m=max_distance_m)
    
    # Summary stats
    n_matched = joined["distance_to_shore_m"].notna().sum()
    n_total = len(sites)
    
    print(f"\nJoin Results:")
    print(f"  Sites with ShoreZone match: {n_matched}")
    print(f"  Sites without match (>{max_distance_m/1000:.1f}km): {n_total - n_matched}")
    
    if n_matched > 0:
        matched = joined[joined["distance_to_shore_m"].notna()]
        print(f"  Distance to shore - min: {matched['distance_to_shore_m'].min():.0f}m")
        print(f"  Distance to shore - max: {matched['distance_to_shore_m'].max():.0f}m")
        print(f"  Distance to shore - median: {matched['distance_to_shore_m'].median():.0f}m")
    
    return joined, sites


def aggregate_shorezone_attributes(joined, category_columns: list):
    """
    Extract ShoreZone categorical attributes at site level.
    
    With nearest-neighbor join, each site has exactly 1 ShoreZone segment,
    so we directly extract the attributes (no aggregation needed).
    """
    print("\n" + "=" * 60)
    print("EXTRACTING SHOREZONE ATTRIBUTES")
    print("=" * 60)
    
    # Check if this is nearest-neighbor join (1 row per site) or buffer join (multiple rows)
    is_nearest_neighbor = "distance_to_shore_m" in joined.columns and "intersect_length_m" not in joined.columns
    
    if is_nearest_neighbor:
        print("  Mode: Nearest-neighbor (1 segment per site)")
        return _extract_nearest_neighbor_attributes(joined, category_columns)
    else:
        print("  Mode: Buffer join (multiple segments per site)")
        return _aggregate_buffer_join_attributes(joined, category_columns)


def _extract_nearest_neighbor_attributes(joined, category_columns: list):
    """
    Extract ShoreZone attributes for nearest-neighbor join.
    Each site has exactly 1 matched segment.
    """
    all_extractions = {}
    
    # Filter to matched sites only
    matched = joined[joined["distance_to_shore_m"].notna()].copy()
    print(f"  Matched sites: {len(matched)}")
    
    for col in category_columns:
        if col not in matched.columns:
            continue
            
        print(f"\n--- {col} ---")
        
        # Get unique values for this column
        unique_vals = matched[col].dropna().unique()
        print(f"  Unique values: {list(unique_vals)[:10]}...")
        
        # Create a simple pivot: 1 row per site, 1 column per category (binary presence)
        # For nearest-neighbor, each site only has 1 value, so this is simpler
        site_values = matched[["SiteName", col]].drop_duplicates()
        
        # Store the direct value
        all_extractions[f"{col}_value"] = site_values
        
        # Create one-hot encoding for compatibility with downstream analysis
        if len(unique_vals) > 0:
            dummies = pd.get_dummies(matched[["SiteName", col]], columns=[col], prefix=col)
            dummies = dummies.groupby("SiteName").max().reset_index()  # In case of duplicates
            all_extractions[f"{col}_proportions"] = dummies
            
            # Create "diversity" metrics (all will be 1 since 1 segment per site)
            diversity_df = pd.DataFrame({
                "SiteName": matched["SiteName"],
                f"{col}_Richness": 1,  # Only 1 segment, so richness = 1
                f"{col}_Shannon": 0.0,  # No diversity with 1 item
                f"{col}_Simpson": 0.0,
                f"{col}_Evenness": 1.0
            }).drop_duplicates()
            all_extractions[f"{col}_diversity"] = diversity_df
        
        print(f"  Extracted for {len(site_values)} sites")
    
    return all_extractions


def _aggregate_buffer_join_attributes(joined, category_columns: list):
    """
    Aggregate ShoreZone attributes for buffer join (original behavior).
    Each site may have multiple matched segments.
    """
    all_aggregations = {}
    
    for col in category_columns:
        if col not in joined.columns:
            continue
            
        print(f"\n--- {col} ---")
        
        # Get proportions (length-weighted)
        try:
            props = aggregate_shorezone_by_site(
                joined, col, 
                weight_column="intersect_length_m",
                aggregation="proportion"
            )
            all_aggregations[f"{col}_proportions"] = props
            
            # Get diversity metrics
            diversity = compute_shorezone_diversity(joined, col, weight_column="intersect_length_m")
            all_aggregations[f"{col}_diversity"] = diversity
            
        except Exception as e:
            print(f"  Error aggregating {col}: {e}")
    
    return all_aggregations


def merge_with_pycno_data(site_summary: pd.DataFrame):
    """
    Merge ShoreZone site summary with Pycnopodia survey data.
    """
    print("\n" + "=" * 60)
    print("MERGING WITH PYCNOPODIA DATA")
    print("=" * 60)
    
    # Load Pycnopodia data
    pycno = load_data()
    
    # Aggregate to site level
    pycno_site = pycno.groupby("SiteName").agg({
        "Encounter.Rate.Hr": "mean",
        "Pycnopodia_count": ["sum", "mean"],
        "Survey.Time": "sum",
        "HabitatType": lambda x: (x == "Eelgrass").any(),
        "Basin": "first"
    }).reset_index()
    
    pycno_site.columns = [
        "SiteName", "MeanEncounterRate", "TotalPycnoCount", "MeanPycnoCount",
        "TotalSurveyTime", "HasEelgrass", "Basin"
    ]
    
    # Detection rate
    detection = pycno.groupby("SiteName")["Pycnopodia_count"].apply(lambda x: (x > 0).mean())
    pycno_site["DetectionRate"] = pycno_site["SiteName"].map(detection)
    
    # Merge
    merged = site_summary.merge(pycno_site, on="SiteName", how="left")
    
    print(f"  Sites with Pycno data: {merged['MeanEncounterRate'].notna().sum()}")
    print(f"  Sites without Pycno data: {merged['MeanEncounterRate'].isna().sum()}")
    
    return merged


def plot_shorezone_distribution(joined, category_column: str):
    """
    Visualize ShoreZone category distribution across all sites.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count by category
    ax1 = axes[0]
    category_counts = joined[category_column].value_counts()
    
    bars = ax1.barh(category_counts.index[:15], category_counts.values[:15], 
                    color="steelblue", edgecolor="black")
    ax1.set_xlabel("Number of Intersections")
    ax1.set_title(f"ShoreZone '{category_column}' Distribution\n(Top 15 categories)")
    ax1.invert_yaxis()
    
    # Sites per category
    ax2 = axes[1]
    sites_per_cat = joined.groupby(category_column)["SiteName"].nunique().sort_values(ascending=False)
    
    bars = ax2.barh(sites_per_cat.index[:15], sites_per_cat.values[:15],
                    color="coral", edgecolor="black")
    ax2.set_xlabel("Number of Sites")
    ax2.set_title(f"Sites per '{category_column}' Category\n(Top 15 categories)")
    ax2.invert_yaxis()
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, f"shorezone_{category_column}_distribution")
    plt.close()


def plot_diversity_vs_encounter_rate(merged: pd.DataFrame, diversity_col: str):
    """
    Scatter plot of ShoreZone diversity vs Pycnopodia encounter rate.
    """
    # Filter to sites with both metrics
    df_plot = merged.dropna(subset=[diversity_col, "MeanEncounterRate"])
    
    if len(df_plot) == 0:
        print(f"  No data for {diversity_col} vs encounter rate plot")
        return
    
    # Check if diversity column has sufficient variance
    if df_plot[diversity_col].nunique() < 3:
        print(f"  Skipping {diversity_col}: insufficient variance ({df_plot[diversity_col].nunique()} unique values)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    ax1 = axes[0]
    
    # Handle HasEelgrass being NaN
    colors = df_plot["HasEelgrass"].map({True: "#2ecc71", False: "#e74c3c"}).fillna("gray")
    
    scatter = ax1.scatter(
        df_plot[diversity_col],
        df_plot["MeanEncounterRate"],
        c=colors,
        s=60, alpha=0.7, edgecolor="black", linewidth=0.5
    )
    
    ax1.set_xlabel(diversity_col.replace("_", " "))
    ax1.set_ylabel("Mean Encounter Rate (Pycno/hr)")
    ax1.set_title(f"ShoreZone Diversity vs Pycnopodia Encounter Rate")
    
    # Add correlation (handle constant input)
    from scipy import stats
    try:
        r, p = stats.pearsonr(df_plot[diversity_col], df_plot["MeanEncounterRate"])
        ax1.annotate(f"r = {r:.3f}\np = {p:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                     fontsize=10, va="top")
    except Exception:
        ax1.annotate("r = N/A", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=10, va="top")
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="Eelgrass Site"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Non-Eelgrass Site")
    ]
    ax1.legend(handles=legend_elements, loc="upper right")
    
    # Box plot by diversity quartiles
    ax2 = axes[1]
    df_plot = df_plot.copy()
    
    # Handle case where quartiles can't be formed (constant or near-constant values)
    try:
        df_plot["DiversityQuartile"] = pd.qcut(
            df_plot[diversity_col], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
        )
        sns.boxplot(data=df_plot, x="DiversityQuartile", y="MeanEncounterRate", 
                    hue="DiversityQuartile", palette="viridis", ax=ax2)
        if ax2.get_legend() is not None:
            ax2.get_legend().remove()
        ax2.set_xlabel(f"{diversity_col.replace('_', ' ')} Quartile")
    except Exception:
        # Fall back to histogram of encounter rates
        ax2.hist(df_plot["MeanEncounterRate"], bins=20, color="steelblue", edgecolor="black")
        ax2.set_xlabel("Mean Encounter Rate (Pycno/hr)")
    
    ax2.set_ylabel("Mean Encounter Rate (Pycno/hr)")
    ax2.set_title("Encounter Rate by Diversity Quartile")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, f"diversity_vs_encounter_{diversity_col}")
    plt.close()


def plot_site_shorezone_heatmap(proportions: pd.DataFrame, site_order: list = None):
    """
    Heatmap of ShoreZone category proportions by site.
    """
    # Get category columns (exclude SiteName)
    cat_cols = [c for c in proportions.columns if c != "SiteName"]
    
    if len(cat_cols) == 0:
        print("  No category columns for heatmap")
        return
    
    # Pivot if needed
    matrix = proportions.set_index("SiteName")[cat_cols]
    
    # Filter to categories present in at least 5 sites
    min_sites = 5
    cols_to_keep = matrix.columns[(matrix > 0).sum() >= min_sites]
    matrix = matrix[cols_to_keep]
    
    if len(matrix.columns) == 0:
        print("  No categories with sufficient coverage for heatmap")
        return
    
    # Sort sites by total diversity (row sum)
    if site_order is None:
        site_order = matrix.sum(axis=1).sort_values(ascending=False).index
    matrix = matrix.reindex(site_order)
    
    # Limit to manageable size
    max_sites = 50
    max_cats = 20
    matrix = matrix.iloc[:max_sites, :max_cats]
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(matrix) * 0.2)))
    
    sns.heatmap(matrix, cmap="YlGnBu", ax=ax, 
                cbar_kws={"label": "Proportion"},
                linewidths=0.1)
    
    ax.set_xlabel("ShoreZone Category")
    ax.set_ylabel("Site")
    ax.set_title(f"ShoreZone Category Proportions by Site\n(Top {len(matrix)} sites × {len(matrix.columns)} categories)")
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, "shorezone_site_heatmap")
    plt.close()


def generate_summary_tables(
    joined,
    aggregations: dict,
    merged: pd.DataFrame
):
    """
    Generate and save summary tables.
    """
    print("\n" + "=" * 60)
    print("GENERATING SUMMARY TABLES")
    print("=" * 60)
    
    # Save joined data
    joined_path = OUTPUT_DIR / "shorezone_joined_data.csv"
    joined_df = joined.drop(columns=["geometry"]).copy()
    if "buffer_geometry" in joined_df.columns:
        joined_df = joined_df.drop(columns=["buffer_geometry"])
    if "point_geometry" in joined_df.columns:
        joined_df = joined_df.drop(columns=["point_geometry"])
    joined_df.to_csv(joined_path, index=False)
    print(f"  Saved: {joined_path}")
    
    # Save aggregations
    for name, df in aggregations.items():
        agg_path = OUTPUT_DIR / f"shorezone_{name}.csv"
        df.to_csv(agg_path, index=False)
        print(f"  Saved: {agg_path}")
    
    # Save merged site summary
    merged_path = OUTPUT_DIR / "site_shorezone_pycno_summary.csv"
    merged.to_csv(merged_path, index=False)
    print(f"  Saved: {merged_path}")
    
    return {
        "joined": joined_path,
        "merged": merged_path,
        "aggregations": aggregations
    }


def run_statistical_analysis(merged: pd.DataFrame, diversity_cols: list):
    """
    Statistical analysis of ShoreZone diversity vs Pycnopodia metrics.
    """
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)
    
    from scipy import stats
    
    results = []
    
    for div_col in diversity_cols:
        if div_col not in merged.columns:
            continue
            
        df_valid = merged.dropna(subset=[div_col, "MeanEncounterRate"])
        
        if len(df_valid) < 10:
            continue
        
        # Pearson correlation
        r_pearson, p_pearson = stats.pearsonr(df_valid[div_col], df_valid["MeanEncounterRate"])
        
        # Spearman correlation
        r_spearman, p_spearman = stats.spearmanr(df_valid[div_col], df_valid["MeanEncounterRate"])
        
        print(f"\n{div_col}:")
        print(f"  n = {len(df_valid)}")
        print(f"  Pearson r = {r_pearson:.3f}, p = {p_pearson:.4f}")
        print(f"  Spearman ρ = {r_spearman:.3f}, p = {p_spearman:.4f}")
        
        results.append({
            "Diversity_Metric": div_col,
            "N": len(df_valid),
            "Pearson_r": r_pearson,
            "Pearson_p": p_pearson,
            "Spearman_rho": r_spearman,
            "Spearman_p": p_spearman
        })
    
    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "shorezone_correlation_tests.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n  Saved: {results_path}")
    
    return results_df


def plot_correlation_heatmap(corr_matrix: pd.DataFrame):
    """Plot feature correlation heatmap."""
    if corr_matrix is None or len(corr_matrix) < 2:
        return
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Use only a subset if too many features
    if len(corr_matrix) > 30:
        corr_matrix = corr_matrix.iloc[:30, :30]
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        cmap="RdBu_r", 
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5, "label": "Spearman ρ"},
        ax=ax
    )
    
    ax.set_title("ShoreZone Feature Correlations\n(Spearman ρ)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    
    plt.tight_layout()
    save_figure(plt.gcf(), OUTPUT_DIR, "feature_correlation_matrix")
    plt.close()


def main():
    """Main analysis pipeline."""
    
    print("\n" + "=" * 70)
    print("SHOREZONE SITE ANALYSIS")
    print(f"Join method: Nearest-neighbor (max distance = {MAX_DISTANCE_M/1000:.1f} km)")
    print("=" * 70)
    
    print("\n⚠️  DATA TEMPORAL MISMATCH WARNING")
    print("    ShoreZone data: 1994-2000")
    print("    Survey data: 2020-2024")
    print("    STABLE features (exposure, sediment) prioritized over DYNAMIC (biotic)")
    
    # Check for geospatial dependencies
    if not HAS_GEOSPATIAL:
        print("\n⚠️  GEOSPATIAL DEPENDENCIES NOT INSTALLED")
        print("This script requires: geopandas, fiona, pyproj, shapely")
        print("\nTo install, run:")
        print("  conda env update -f code/environment.yml")
        print("  # or")
        print("  pip install geopandas fiona pyproj shapely")
        return
    
    set_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Step 1: Explore ShoreZone schema
    layers = explore_shorezone_schema()
    
    # Step 2: Load and explore ShoreZone data
    shorezone, categorical_cols = load_and_explore_shorezone()
    
    # Step 3: Load sites and validate CRS alignment
    sites = load_site_points()
    in_bounds = validate_crs_alignment(sites, shorezone)
    
    # Step 4: Compute distance to shore
    distance_df = compute_distance_to_shore(sites, shorezone)
    
    # Step 5: Run nearest-neighbor spatial join (each site → nearest shoreline)
    joined, sites = run_nearest_join(shorezone, max_distance_m=MAX_DISTANCE_M)
    
    # Step 6: Aggregate ShoreZone attributes
    # PRIORITIZE STABLE FEATURES (physical characteristics)
    # Then include DYNAMIC FEATURES with clear warnings
    all_features = STABLE_FEATURES + DYNAMIC_FEATURES
    
    # Use features that exist in the data
    cols_to_aggregate = [c for c in all_features if c in joined.columns]
    if not cols_to_aggregate:
        cols_to_aggregate = categorical_cols[:5] if categorical_cols else []
    
    print(f"\n" + "=" * 60)
    print("FEATURE CLASSIFICATION")
    print("=" * 60)
    stable_used = [c for c in cols_to_aggregate if c in STABLE_FEATURES]
    dynamic_used = [c for c in cols_to_aggregate if c in DYNAMIC_FEATURES]
    print(f"  STABLE features (reliable): {stable_used}")
    print(f"  DYNAMIC features (use with caution): {dynamic_used}")
    
    aggregations = aggregate_shorezone_attributes(joined, cols_to_aggregate)
    
    # Step 7: Build site summary
    print("\n" + "=" * 60)
    print("BUILDING SITE SUMMARY")
    print("=" * 60)
    
    # Start with sites
    site_summary = sites[["SiteName", "Lat", "Long"]].copy()
    
    # Add distance to shore
    site_summary = site_summary.merge(distance_df, on="SiteName", how="left")
    
    # Add diversity metrics
    diversity_cols = []
    for key, df in aggregations.items():
        if "diversity" in key:
            site_summary = site_summary.merge(df, on="SiteName", how="left")
            diversity_cols.extend([c for c in df.columns if c != "SiteName"])
    
    # Step 8: Merge with Pycnopodia data
    merged = merge_with_pycno_data(site_summary)
    
    # Step 9: Analyze feature correlations
    corr_matrix, feature_groups = analyze_feature_correlations(merged, diversity_cols)
    
    # Step 10: Create composite features
    composite_df = create_composite_features(merged, feature_groups)
    
    # Merge composites into main dataframe
    merged = merged.merge(
        composite_df.drop(columns=[c for c in composite_df.columns 
                                   if c in merged.columns and c != "SiteName"]),
        on="SiteName", how="left"
    )
    
    # Step 11: Generate visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Plot correlation heatmap
    if corr_matrix is not None:
        print("\nPlotting feature correlation matrix...")
        plot_correlation_heatmap(corr_matrix)
    
    # Plot ShoreZone distributions for STABLE columns first
    plotted_distributions = 0
    for col in stable_used + dynamic_used[:2]:  # Stable + first 2 dynamic
        if col in joined.columns and joined[col].nunique() > 1:
            print(f"\nPlotting distribution for {col}...")
            plot_shorezone_distribution(joined, col)
            plotted_distributions += 1
            if plotted_distributions >= 4:
                break
    
    # Plot diversity vs encounter rate for STABLE features first
    stable_diversity_cols = [c for c in diversity_cols 
                             if any(s in c for s in STABLE_FEATURES)]
    dynamic_diversity_cols = [c for c in diversity_cols 
                              if any(s in c for s in DYNAMIC_FEATURES)]
    
    plotted_diversity = 0
    for div_col in stable_diversity_cols[:2] + dynamic_diversity_cols[:2]:
        if div_col in merged.columns:
            valid_data = merged[div_col].dropna()
            if valid_data.nunique() >= 3:
                print(f"\nPlotting {div_col} vs encounter rate...")
                plot_diversity_vs_encounter_rate(merged, div_col)
                plotted_diversity += 1
    
    # Plot heatmap
    if aggregations:
        first_prop_key = [k for k in aggregations.keys() if "proportion" in k]
        if first_prop_key:
            print("\nPlotting site-category heatmap...")
            plot_site_shorezone_heatmap(aggregations[first_prop_key[0]])
    
    # Step 12: Statistical analysis
    stats_results = run_statistical_analysis(merged, diversity_cols)
    
    # Step 13: Save results
    generate_summary_tables(joined, aggregations, merged)
    
    # Save composite features separately
    composite_path = OUTPUT_DIR / "composite_features.csv"
    composite_df.to_csv(composite_path, index=False)
    print(f"  Saved: {composite_path}")
    
    # Save distance to shore
    distance_path = OUTPUT_DIR / "site_distance_to_shore.csv"
    distance_df.to_csv(distance_path, index=False)
    print(f"  Saved: {distance_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    n_matched = joined["distance_to_shore_m"].notna().sum() if "distance_to_shore_m" in joined.columns else joined['SiteName'].nunique()
    n_unmatched = len(sites) - n_matched
    
    print(f"""
Summary:
  • Sites analyzed: {len(sites)}
  • Sites with ShoreZone match: {n_matched}
  • Sites without match (>{MAX_DISTANCE_M/1000:.1f}km from shore): {n_unmatched}
  • Join method: Nearest-neighbor (max distance = {MAX_DISTANCE_M/1000:.1f} km)
  
Features analyzed:
  • STABLE (physical, reliable): {len(stable_used)}
  • DYNAMIC (biotic, use with caution): {len(dynamic_used)}
  • Composite features created: {len([c for c in composite_df.columns if c != 'SiteName'])}
  
Output files saved to: {OUTPUT_DIR}
  • site_shorezone_pycno_summary.csv - Main analysis file
  • composite_features.csv - Reduced feature set
  • site_distance_to_shore.csv - QC for offshore sites
  • feature_correlation_matrix.png - Feature redundancy
""")


if __name__ == "__main__":
    main()


