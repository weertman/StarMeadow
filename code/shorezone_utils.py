"""
ShoreZone Spatial Join Utilities

Provides functions for performing buffer-based spatial joins between
Pycnopodia survey site coordinates and Washington State ShoreZone inventory data.

The ShoreZone inventory characterizes Washington's saltwater shorelines with
attributes including habitat type, substrate, biotic cover, and wave exposure.

Key Functions:
    - load_shorezone_data(): Load ShoreZone geodatabase layers
    - load_site_points(): Load site coordinates as GeoDataFrame
    - buffer_spatial_join(): Perform 250m buffer join preserving habitat diversity
    - aggregate_shorezone_categories(): Summarize ShoreZone attributes per site

Dependencies:
    - geopandas>=0.14
    - fiona>=1.9
    - pyproj>=3.6
    - shapely>=2.0

Usage:
    from shorezone_utils import (
        load_site_points,
        load_shorezone_lines,
        buffer_spatial_join,
        compute_shorezone_diversity
    )
    
    sites = load_site_points()
    shorezone = load_shorezone_lines()
    joined = buffer_spatial_join(sites, shorezone, buffer_m=250)
    diversity = compute_shorezone_diversity(joined)

Author: Star Meadow Project
Date: December 2024
"""

from pathlib import Path
from typing import Optional, Union, Literal
import warnings

import numpy as np
import pandas as pd

# Geospatial imports - will raise ImportError if not installed
try:
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.ops import unary_union
    import fiona
    HAS_GEOSPATIAL = True
except ImportError as e:
    HAS_GEOSPATIAL = False
    _IMPORT_ERROR = str(e)


# =============================================================================
# Path Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ShoreZone geodatabase paths
SHOREZONE_GDB = DATA_DIR / "state_DNR_ShoreZone" / "shorezone.gdb"
SHOREZONE_THEMES_GDB = DATA_DIR / "state_DNR_ShoreZone" / "shorezone_themes.gdb"
SHOREZONE_CODES_GDB = DATA_DIR / "state_DNR_ShoreZone" / "shorezone_code_tables.gdb"

# Site coordinates
SITE_COORDS_FILE = DATA_DIR / "Site_LatLong.csv"

# Output directory for processed shorezone data
SHOREZONE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "shorezone_data"

# Coordinate Reference Systems
CRS_WGS84 = "EPSG:4326"  # Input CRS (lat/long)
CRS_WA_SOUTH = "EPSG:2927"  # WA State Plane South (feet) - for accurate buffering
CRS_UTM10N = "EPSG:32610"  # UTM Zone 10N (meters) - alternative projected CRS


def _check_geospatial_deps():
    """Raise ImportError if geospatial dependencies are not installed."""
    if not HAS_GEOSPATIAL:
        raise ImportError(
            f"Geospatial dependencies not installed: {_IMPORT_ERROR}\n"
            "Install with: conda env update -f code/environment.yml\n"
            "Or: pip install geopandas fiona pyproj shapely"
        )


# =============================================================================
# Data Loading Functions
# =============================================================================

def list_shorezone_layers(gdb_path: Optional[Path] = None) -> dict:
    """
    List all available layers in the ShoreZone geodatabases.
    
    Parameters
    ----------
    gdb_path : Path, optional
        Specific geodatabase to query. If None, lists all three geodatabases.
    
    Returns
    -------
    dict
        Dictionary mapping geodatabase names to lists of layer names.
    """
    _check_geospatial_deps()
    
    gdbs = {
        "shorezone.gdb": SHOREZONE_GDB,
        "shorezone_themes.gdb": SHOREZONE_THEMES_GDB,
        "shorezone_code_tables.gdb": SHOREZONE_CODES_GDB,
    }
    
    if gdb_path is not None:
        gdbs = {"custom": gdb_path}
    
    result = {}
    for name, path in gdbs.items():
        if path.exists():
            try:
                layers = fiona.listlayers(str(path))
                result[name] = layers
            except Exception as e:
                result[name] = f"Error: {e}"
        else:
            result[name] = "File not found"
    
    return result


def load_shorezone_lines(
    layer_name: str = "szline",
    gdb_path: Optional[Path] = None
) -> "gpd.GeoDataFrame":
    """
    Load ShoreZone line features from the geodatabase.
    
    The primary ShoreZone data is stored as line segments representing
    homogeneous shoreline units (average length ~0.5 miles).
    
    Parameters
    ----------
    layer_name : str
        Name of the layer to load. Default "szline" is the main line layer.
    gdb_path : Path, optional
        Path to geodatabase. Defaults to shorezone.gdb.
    
    Returns
    -------
    gpd.GeoDataFrame
        ShoreZone line features with all attributes.
    """
    _check_geospatial_deps()
    
    gdb = gdb_path or SHOREZONE_GDB
    
    print(f"Loading ShoreZone layer '{layer_name}' from {gdb.name}...")
    gdf = gpd.read_file(str(gdb), layer=layer_name)
    print(f"  Loaded {len(gdf)} features")
    print(f"  CRS: {gdf.crs}")
    print(f"  Columns: {list(gdf.columns)}")
    
    return gdf


def load_shorezone_themes(
    theme_name: str
) -> "gpd.GeoDataFrame":
    """
    Load a thematic layer from shorezone_themes.gdb.
    
    Available themes typically include:
    - Habitat classifications
    - Substrate types
    - Biotic cover
    - Wave exposure classes
    
    Parameters
    ----------
    theme_name : str
        Name of the theme layer to load.
    
    Returns
    -------
    gpd.GeoDataFrame
        Thematic ShoreZone features.
    """
    _check_geospatial_deps()
    
    print(f"Loading ShoreZone theme '{theme_name}'...")
    gdf = gpd.read_file(str(SHOREZONE_THEMES_GDB), layer=theme_name)
    print(f"  Loaded {len(gdf)} features")
    
    return gdf


def load_shorezone_code_table(table_name: str) -> pd.DataFrame:
    """
    Load a code lookup table from shorezone_code_tables.gdb.
    
    These tables provide human-readable descriptions for coded
    attribute values in the ShoreZone data.
    
    Parameters
    ----------
    table_name : str
        Name of the code table to load.
    
    Returns
    -------
    pd.DataFrame
        Code lookup table.
    """
    _check_geospatial_deps()
    
    # Code tables are non-spatial, but stored in geodatabase
    gdf = gpd.read_file(str(SHOREZONE_CODES_GDB), layer=table_name)
    
    # Drop geometry column if present (tables shouldn't have geometry)
    if "geometry" in gdf.columns:
        gdf = gdf.drop(columns=["geometry"])
    
    return pd.DataFrame(gdf)


def load_site_points(
    coords_file: Optional[Path] = None,
    crs: str = CRS_WGS84
) -> "gpd.GeoDataFrame":
    """
    Load site coordinates as a GeoDataFrame.
    
    Parameters
    ----------
    coords_file : Path, optional
        Path to CSV with Site, Lat, Long columns. Defaults to Site_LatLong.csv.
    crs : str
        Coordinate reference system. Default is WGS84 (EPSG:4326).
    
    Returns
    -------
    gpd.GeoDataFrame
        Site points with geometry column.
    """
    _check_geospatial_deps()
    
    csv_path = coords_file or SITE_COORDS_FILE
    
    print(f"Loading site coordinates from {csv_path.name}...")
    df = pd.read_csv(csv_path)
    df.columns = ["SiteName", "Lat", "Long"]
    
    # Fix longitude sign (should be negative for Western Hemisphere)
    df.loc[df["Long"] > 0, "Long"] = -df.loc[df["Long"] > 0, "Long"]
    
    # Create geometry
    geometry = [Point(xy) for xy in zip(df["Long"], df["Lat"])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=crs)
    
    print(f"  Loaded {len(gdf)} sites")
    print(f"  Lat range: {gdf['Lat'].min():.3f} to {gdf['Lat'].max():.3f}")
    print(f"  Long range: {gdf['Long'].min():.3f} to {gdf['Long'].max():.3f}")
    
    return gdf


# =============================================================================
# Spatial Join Functions
# =============================================================================

def buffer_spatial_join(
    sites: "gpd.GeoDataFrame",
    shorezone: "gpd.GeoDataFrame",
    buffer_m: float = 250.0,
    projected_crs: str = CRS_UTM10N
) -> "gpd.GeoDataFrame":
    """
    Perform buffer-based spatial join between sites and ShoreZone features.
    
    Each site point is buffered by the specified radius, and all ShoreZone
    features intersecting the buffer are returned. This preserves the
    local habitat diversity around each site.
    
    Parameters
    ----------
    sites : gpd.GeoDataFrame
        Site point features (must have 'SiteName' column).
    shorezone : gpd.GeoDataFrame
        ShoreZone line/polygon features.
    buffer_m : float
        Buffer radius in meters. Default 250m.
    projected_crs : str
        Projected CRS to use for buffering. Default UTM 10N (meters).
    
    Returns
    -------
    gpd.GeoDataFrame
        Joined data with one row per site-shorezone intersection.
        Contains all columns from both inputs plus:
        - buffer_geometry: The buffer polygon used for intersection
        - intersect_length_m: Length of ShoreZone segment within buffer
    """
    _check_geospatial_deps()
    
    print(f"\nPerforming buffer spatial join (buffer = {buffer_m}m)...")
    
    # Reproject to projected CRS for accurate buffering
    print(f"  Reprojecting to {projected_crs}...")
    sites_proj = sites.to_crs(projected_crs)
    shorezone_proj = shorezone.to_crs(projected_crs)
    
    # Create buffers around sites
    print(f"  Creating {buffer_m}m buffers around {len(sites_proj)} sites...")
    sites_proj["buffer_geometry"] = sites_proj.geometry.buffer(buffer_m)
    
    # Store original point geometry
    sites_proj["point_geometry"] = sites_proj.geometry
    
    # Replace geometry with buffer for spatial join
    sites_buffered = sites_proj.set_geometry("buffer_geometry")
    
    # Perform spatial join (predicate='intersects')
    print("  Performing spatial join...")
    joined = gpd.sjoin(
        shorezone_proj,
        sites_buffered[["SiteName", "Lat", "Long", "buffer_geometry", "point_geometry"]].set_geometry("buffer_geometry"),
        how="inner",
        predicate="intersects"
    )
    
    print(f"  Found {len(joined)} intersections across {joined['SiteName'].nunique()} sites")
    
    # Calculate intersection length (for line features)
    if joined.geom_type.iloc[0] in ["LineString", "MultiLineString"]:
        print("  Calculating intersection lengths...")
        
        # Create a lookup for buffer geometries
        buffer_lookup = sites_buffered.set_index("SiteName")["buffer_geometry"]
        
        def calc_intersect_length(row):
            buffer = buffer_lookup[row["SiteName"]]
            intersection = row.geometry.intersection(buffer)
            return intersection.length
        
        joined["intersect_length_m"] = joined.apply(calc_intersect_length, axis=1)
    
    # Reproject back to WGS84
    joined = joined.to_crs(CRS_WGS84)
    
    # Clean up index
    joined = joined.reset_index(drop=True)
    
    # Drop the spatial join index column
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])
    
    print(f"  Join complete: {len(joined)} rows")
    
    return joined


def nearest_shorezone_join(
    sites: "gpd.GeoDataFrame",
    shorezone: "gpd.GeoDataFrame",
    max_distance_m: float = 5000.0,
    projected_crs: str = CRS_UTM10N
) -> "gpd.GeoDataFrame":
    """
    Join each site to its nearest ShoreZone segment within max distance.
    
    Unlike buffer_spatial_join which may match multiple segments,
    this returns exactly ONE match per site (the nearest segment).
    
    Parameters
    ----------
    sites : gpd.GeoDataFrame
        Site point features (must have 'SiteName' column).
    shorezone : gpd.GeoDataFrame
        ShoreZone line features.
    max_distance_m : float
        Maximum distance in meters. Sites beyond this have no match.
    projected_crs : str
        Projected CRS for distance calculations. Default UTM 10N (meters).
    
    Returns
    -------
    gpd.GeoDataFrame
        Joined data with one row per site that has a match.
        Contains all columns from both inputs plus:
        - distance_to_shore_m: Distance to nearest ShoreZone segment
    """
    _check_geospatial_deps()
    
    print(f"\nPerforming nearest-neighbor join (max distance = {max_distance_m/1000:.1f} km)...")
    
    # Reproject to projected CRS for accurate distance calculations
    print(f"  Reprojecting to {projected_crs}...")
    sites_proj = sites.to_crs(projected_crs)
    shorezone_proj = shorezone.to_crs(projected_crs)
    
    # Build spatial index for shorezone
    print(f"  Building spatial index for {len(shorezone_proj)} ShoreZone segments...")
    
    # Use geopandas sjoin_nearest (requires geopandas >= 0.10)
    try:
        print("  Finding nearest segment for each site...")
        joined = gpd.sjoin_nearest(
            sites_proj,
            shorezone_proj,
            how="left",
            max_distance=max_distance_m,
            distance_col="distance_to_shore_m"
        )
        
        # Count matches
        n_matched = joined["distance_to_shore_m"].notna().sum()
        n_unmatched = joined["distance_to_shore_m"].isna().sum()
        
        print(f"  Matched: {n_matched} sites")
        print(f"  No match (>{max_distance_m/1000:.1f}km): {n_unmatched} sites")
        
        if n_matched > 0:
            matched = joined[joined["distance_to_shore_m"].notna()]
            print(f"  Distance range: {matched['distance_to_shore_m'].min():.0f}m - {matched['distance_to_shore_m'].max():.0f}m")
            print(f"  Median distance: {matched['distance_to_shore_m'].median():.0f}m")
        
    except AttributeError:
        # Fallback for older geopandas versions
        print("  Using manual nearest-neighbor calculation (older geopandas)...")
        
        from shapely.ops import nearest_points
        
        results = []
        shorezone_union = shorezone_proj.unary_union
        
        for idx, site_row in sites_proj.iterrows():
            nearest_geom = nearest_points(site_row.geometry, shorezone_union)[1]
            distance = site_row.geometry.distance(nearest_geom)
            
            if distance <= max_distance_m:
                # Find which segment contains the nearest point
                for sz_idx, sz_row in shorezone_proj.iterrows():
                    if sz_row.geometry.distance(nearest_geom) < 1:  # Within 1m tolerance
                        result = {**site_row.to_dict(), **sz_row.to_dict()}
                        result["distance_to_shore_m"] = distance
                        results.append(result)
                        break
        
        joined = gpd.GeoDataFrame(results, crs=projected_crs)
        print(f"  Matched: {len(joined)} sites")
    
    # Reproject back to WGS84
    if len(joined) > 0:
        joined = joined.to_crs(CRS_WGS84)
    
    # Clean up index columns
    joined = joined.reset_index(drop=True)
    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])
    
    return joined


def aggregate_shorezone_by_site(
    joined: "gpd.GeoDataFrame",
    category_column: str,
    weight_column: Optional[str] = "intersect_length_m",
    aggregation: Literal["count", "proportion", "length", "presence"] = "proportion"
) -> pd.DataFrame:
    """
    Aggregate ShoreZone categories to site level.
    
    Parameters
    ----------
    joined : gpd.GeoDataFrame
        Output from buffer_spatial_join().
    category_column : str
        Column containing categorical values to aggregate.
    weight_column : str, optional
        Column to use for weighting (e.g., segment length). 
        Set to None for unweighted aggregation.
    aggregation : str
        Aggregation method:
        - "count": Number of features per category
        - "proportion": Proportion of total (weighted if weight_column provided)
        - "length": Total length per category (requires weight_column)
        - "presence": Binary presence/absence (1/0)
    
    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame with sites as rows, categories as columns.
    """
    print(f"\nAggregating '{category_column}' by site using '{aggregation}' method...")
    
    if category_column not in joined.columns:
        available = [c for c in joined.columns if c not in ["geometry", "buffer_geometry", "point_geometry"]]
        raise ValueError(f"Column '{category_column}' not found. Available: {available}")
    
    if aggregation == "presence":
        # Binary presence/absence
        pivot = pd.crosstab(joined["SiteName"], joined[category_column])
        pivot = (pivot > 0).astype(int)
    
    elif aggregation == "count":
        # Simple count
        pivot = pd.crosstab(joined["SiteName"], joined[category_column])
    
    elif aggregation == "length":
        if weight_column is None or weight_column not in joined.columns:
            raise ValueError(f"'length' aggregation requires weight_column. Got: {weight_column}")
        
        pivot = joined.pivot_table(
            index="SiteName",
            columns=category_column,
            values=weight_column,
            aggfunc="sum",
            fill_value=0
        )
    
    elif aggregation == "proportion":
        if weight_column and weight_column in joined.columns:
            # Length-weighted proportion
            pivot = joined.pivot_table(
                index="SiteName",
                columns=category_column,
                values=weight_column,
                aggfunc="sum",
                fill_value=0
            )
        else:
            # Count-based proportion
            pivot = pd.crosstab(joined["SiteName"], joined[category_column])
        
        # Normalize to proportions
        pivot = pivot.div(pivot.sum(axis=1), axis=0)
    
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    pivot = pivot.reset_index()
    print(f"  Aggregated to {len(pivot)} sites × {len(pivot.columns)-1} categories")
    
    return pivot


def compute_shorezone_diversity(
    joined: "gpd.GeoDataFrame",
    category_column: str,
    weight_column: Optional[str] = "intersect_length_m"
) -> pd.DataFrame:
    """
    Compute diversity metrics for ShoreZone categories per site.
    
    Calculates:
    - Richness: Number of unique categories
    - Shannon Index: -Σ(p * ln(p))
    - Simpson Index: 1 - Σ(p²)
    - Evenness: Shannon / ln(Richness)
    
    Parameters
    ----------
    joined : gpd.GeoDataFrame
        Output from buffer_spatial_join().
    category_column : str
        Column containing categorical values.
    weight_column : str, optional
        Column for weighting proportions (e.g., segment length).
    
    Returns
    -------
    pd.DataFrame
        Diversity metrics per site.
    """
    print(f"\nComputing diversity metrics for '{category_column}'...")
    
    # Get proportions
    props = aggregate_shorezone_by_site(
        joined, category_column, weight_column, aggregation="proportion"
    )
    
    site_col = props.columns[0]  # "SiteName"
    category_cols = props.columns[1:]
    
    diversity_data = []
    
    for _, row in props.iterrows():
        site_name = row[site_col]
        p = row[category_cols].values.astype(float)
        p = p[p > 0]  # Remove zeros for log calculation
        
        richness = len(p)
        
        if richness == 0:
            shannon = 0
            simpson = 0
            evenness = 0
        else:
            shannon = -np.sum(p * np.log(p))
            simpson = 1 - np.sum(p ** 2)
            evenness = shannon / np.log(richness) if richness > 1 else 1.0
        
        diversity_data.append({
            "SiteName": site_name,
            f"{category_column}_Richness": richness,
            f"{category_column}_Shannon": shannon,
            f"{category_column}_Simpson": simpson,
            f"{category_column}_Evenness": evenness
        })
    
    diversity_df = pd.DataFrame(diversity_data)
    print(f"  Computed diversity for {len(diversity_df)} sites")
    print(f"  Mean richness: {diversity_df[f'{category_column}_Richness'].mean():.2f}")
    print(f"  Mean Shannon: {diversity_df[f'{category_column}_Shannon'].mean():.3f}")
    
    return diversity_df


# =============================================================================
# Export Functions
# =============================================================================

def export_shorezone_join(
    joined: "gpd.GeoDataFrame",
    output_dir: Optional[Path] = None,
    prefix: str = "shorezone_join"
) -> dict:
    """
    Export joined ShoreZone data in multiple formats.
    
    Parameters
    ----------
    joined : gpd.GeoDataFrame
        Output from buffer_spatial_join().
    output_dir : Path, optional
        Output directory. Defaults to outputs/shorezone_data/.
    prefix : str
        Filename prefix.
    
    Returns
    -------
    dict
        Paths to exported files.
    """
    _check_geospatial_deps()
    
    out_dir = output_dir or SHOREZONE_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    
    # Export as GeoPackage (preserves geometry)
    gpkg_path = out_dir / f"{prefix}.gpkg"
    joined.to_file(gpkg_path, driver="GPKG")
    paths["gpkg"] = gpkg_path
    print(f"  Saved: {gpkg_path}")
    
    # Export as CSV (drop geometry)
    csv_path = out_dir / f"{prefix}.csv"
    df = joined.drop(columns=["geometry"]).copy()
    if "buffer_geometry" in df.columns:
        df = df.drop(columns=["buffer_geometry"])
    if "point_geometry" in df.columns:
        df = df.drop(columns=["point_geometry"])
    df.to_csv(csv_path, index=False)
    paths["csv"] = csv_path
    print(f"  Saved: {csv_path}")
    
    return paths


def save_site_shorezone_summary(
    sites: "gpd.GeoDataFrame",
    joined: "gpd.GeoDataFrame",
    category_columns: list,
    output_dir: Optional[Path] = None,
    filename: str = "site_shorezone_summary.csv"
) -> Path:
    """
    Create and save a comprehensive site-level summary with ShoreZone attributes.
    
    Parameters
    ----------
    sites : gpd.GeoDataFrame
        Original site points.
    joined : gpd.GeoDataFrame
        Output from buffer_spatial_join().
    category_columns : list
        List of ShoreZone columns to aggregate.
    output_dir : Path, optional
        Output directory.
    filename : str
        Output filename.
    
    Returns
    -------
    Path
        Path to saved CSV.
    """
    out_dir = output_dir or SHOREZONE_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Start with site info
    summary = sites[["SiteName", "Lat", "Long"]].copy()
    
    # Add aggregated categories for each column
    for col in category_columns:
        if col in joined.columns:
            # Get proportions
            props = aggregate_shorezone_by_site(joined, col, aggregation="proportion")
            summary = summary.merge(props, on="SiteName", how="left")
            
            # Get diversity
            diversity = compute_shorezone_diversity(joined, col)
            summary = summary.merge(diversity, on="SiteName", how="left")
    
    # Save
    output_path = out_dir / filename
    summary.to_csv(output_path, index=False)
    print(f"\nSaved site summary: {output_path}")
    
    return output_path


# =============================================================================
# Convenience Function
# =============================================================================

def run_shorezone_join_pipeline(
    buffer_m: float = 250.0,
    shorezone_layer: str = "szline",
    output_dir: Optional[Path] = None
) -> dict:
    """
    Run the complete ShoreZone spatial join pipeline.
    
    This is a convenience function that:
    1. Loads site coordinates
    2. Loads ShoreZone data
    3. Performs buffer spatial join
    4. Computes diversity metrics
    5. Exports results
    
    Parameters
    ----------
    buffer_m : float
        Buffer radius in meters.
    shorezone_layer : str
        ShoreZone layer to load.
    output_dir : Path, optional
        Output directory for results.
    
    Returns
    -------
    dict
        Dictionary containing:
        - sites: Site GeoDataFrame
        - shorezone: ShoreZone GeoDataFrame
        - joined: Joined GeoDataFrame
        - diversity: Diversity metrics DataFrame
        - paths: Exported file paths
    """
    _check_geospatial_deps()
    
    print("=" * 60)
    print("SHOREZONE SPATIAL JOIN PIPELINE")
    print("=" * 60)
    print(f"Buffer radius: {buffer_m}m")
    print(f"ShoreZone layer: {shorezone_layer}")
    
    # Load data
    sites = load_site_points()
    shorezone = load_shorezone_lines(layer_name=shorezone_layer)
    
    # Perform join
    joined = buffer_spatial_join(sites, shorezone, buffer_m=buffer_m)
    
    # Identify category columns (non-numeric, non-geometry)
    category_cols = []
    for col in joined.columns:
        if col not in ["SiteName", "Lat", "Long", "geometry", "buffer_geometry", 
                       "point_geometry", "intersect_length_m", "index_right"]:
            if joined[col].dtype == "object" or str(joined[col].dtype).startswith("category"):
                category_cols.append(col)
    
    print(f"\nIdentified category columns: {category_cols[:5]}...")  # Show first 5
    
    # Export joined data
    print("\nExporting results...")
    out_dir = output_dir or SHOREZONE_OUTPUT_DIR
    paths = export_shorezone_join(joined, out_dir, prefix=f"shorezone_join_{int(buffer_m)}m")
    
    # Compute diversity for first category column if available
    diversity = None
    if category_cols:
        primary_col = category_cols[0]
        diversity = compute_shorezone_diversity(joined, primary_col)
        diversity_path = out_dir / f"shorezone_diversity_{int(buffer_m)}m.csv"
        diversity.to_csv(diversity_path, index=False)
        paths["diversity"] = diversity_path
        print(f"  Saved: {diversity_path}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    
    return {
        "sites": sites,
        "shorezone": shorezone,
        "joined": joined,
        "diversity": diversity,
        "paths": paths,
        "category_columns": category_cols
    }


if __name__ == "__main__":
    # Quick test / example usage
    print("\nShoreZone Utilities - Module Test")
    print("-" * 40)
    
    if HAS_GEOSPATIAL:
        # List available layers
        layers = list_shorezone_layers()
        for gdb, lyrs in layers.items():
            print(f"\n{gdb}:")
            if isinstance(lyrs, list):
                for lyr in lyrs:
                    print(f"  - {lyr}")
            else:
                print(f"  {lyrs}")
    else:
        print(f"\nGeospatial dependencies not installed.")
        print(f"Install with: conda env update -f code/environment.yml")


