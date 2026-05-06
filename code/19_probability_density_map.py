"""
Pycnopodia Probability Density Map

Colors ShoreZone line segments by predicted Pycnopodia encounter rate.
Uses native Folium GeoJSON rendering - no alignment issues.

Author: Star Meadow Project
Date: December 2024
"""

import numpy as np
import pandas as pd
from pathlib import Path
import joblib

import geopandas as gpd

import folium
from branca.colormap import LinearColormap

from utils import get_output_dir, DATA_DIR

# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = get_output_dir(__file__)

SHOREZONE_GDB = DATA_DIR / "state_DNR_ShoreZone" / "shorezone.gdb"
SITE_SUMMARY_PATH = Path(__file__).parent.parent / "outputs" / "15_shorezone_site_analysis" / "site_shorezone_pycno_summary.csv"
MODEL_PATH = Path(__file__).parent.parent / "outputs" / "16_shorezone_recovery_analysis" / "trained_model.joblib"

CRS_WGS84 = "EPSG:4326"

SHOREZONE_CATEGORIES = [
    "EXP_CLASS", "SED_SOURCE", "SED_ABUND", "ZONECOMP", "BC_CLASS",
    "NRDA_CLASS", "HAB_CALC", "ZOS_UNIT", "FUC_UNIT", "ULV_UNIT",
    "NER_UNIT", "OYS_UNIT", "MUS_UNIT"
]


# =============================================================================
# Data Loading
# =============================================================================

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    print(f"Loading model...")
    data = joblib.load(MODEL_PATH)
    print(f"  Features: {len(data['feature_cols'])}")
    return data["model"], data["feature_cols"]


def load_shorezone():
    print(f"\nLoading ShoreZone...")
    szline = gpd.read_file(str(SHOREZONE_GDB), layer="szline")
    if szline.crs != CRS_WGS84:
        szline = szline.to_crs(CRS_WGS84)
    print(f"  Segments: {len(szline)}")
    return szline


def load_site_summary():
    df = pd.read_csv(SITE_SUMMARY_PATH)
    print(f"  Sites: {len(df)}")
    return df


# =============================================================================
# Feature Extraction & Prediction
# =============================================================================

def extract_features(szline: gpd.GeoDataFrame, feature_cols: list):
    """One-hot encode segment attributes to match training features."""
    print(f"\nExtracting features for {len(szline)} segments...")
    
    feature_dict = {}
    for cat in SHOREZONE_CATEGORIES:
        if cat not in szline.columns:
            continue
        values = szline[cat].values
        for val in [v for v in np.unique(values) if pd.notna(v)]:
            feature_dict[f"{cat}_{val}"] = (values == val).astype(int)
    
    features = pd.DataFrame(feature_dict)
    
    # Align to training features
    aligned = pd.DataFrame({
        col: features[col].values if col in features.columns else 0
        for col in feature_cols
    })
    
    print(f"  Aligned to {len(feature_cols)} features")
    return aligned


def predict_segments(szline: gpd.GeoDataFrame, model, feature_cols: list):
    """Add predicted encounter rate to each segment."""
    features = extract_features(szline, feature_cols)
    
    print(f"\nApplying model...")
    X = np.nan_to_num(features.values, nan=0)
    predictions = np.maximum(model.predict(X), 0)
    
    szline = szline.copy()
    szline["predicted_rate"] = predictions
    
    print(f"  Range: {predictions.min():.2f} - {predictions.max():.2f}")
    print(f"  Mean: {predictions.mean():.2f}")
    
    return szline


# =============================================================================
# Map
# =============================================================================

def get_color(rate, max_rate=10.0):
    """YlOrRd color scale."""
    norm = min(rate / max_rate, 1.0)
    
    # Simple gradient: yellow -> orange -> red
    if norm < 0.25:
        return "#ffffb2"  # light yellow
    elif norm < 0.5:
        return "#fecc5c"  # yellow-orange
    elif norm < 0.75:
        return "#fd8d3c"  # orange
    elif norm < 0.9:
        return "#f03b20"  # red-orange
    else:
        return "#bd0026"  # dark red


def create_map(szline: gpd.GeoDataFrame, site_summary: pd.DataFrame, max_rate: float):
    print(f"\nCreating map...")
    
    # Center on data
    bounds = szline.total_bounds
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="CartoDB positron")
    
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite"
    ).add_to(m)
    
    # Add colored segments as GeoJSON
    print(f"  Adding {len(szline)} segments...")
    
    def style_function(feature):
        rate = feature["properties"].get("predicted_rate", 0)
        return {
            "color": get_color(rate, max_rate),
            "weight": 3,
            "opacity": 0.9
        }
    
    # Simplify for performance
    szline_simple = szline.copy()
    szline_simple["geometry"] = szline_simple.geometry.simplify(0.0001)
    
    # Keep only needed columns for GeoJSON
    szline_simple = szline_simple[["geometry", "predicted_rate"]]
    
    geojson = folium.GeoJson(
        szline_simple.__geo_interface__,
        style_function=style_function,
        name="Predicted Density"
    )
    geojson.add_to(m)
    
    # Survey sites
    sites_group = folium.FeatureGroup(name="Survey Sites")
    for _, row in site_summary.dropna(subset=["Lat", "Long"]).iterrows():
        rate = row.get("MeanEncounterRate", 0) or 0
        folium.Marker(
            location=[row["Lat"], row["Long"]],
            popup=f"<b>{row['SiteName']}</b><br>Observed: {rate:.2f}/hr",
            icon=folium.Icon(color="green" if rate > 0 else "blue", 
                           icon="star" if rate > 0 else "info-sign")
        ).add_to(sites_group)
    sites_group.add_to(m)
    
    # Legend
    colormap = LinearColormap(
        colors=["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
        vmin=0, vmax=max_rate,
        caption="Predicted Rate (Pycno/hr)"
    )
    colormap.add_to(m)
    
    folium.LayerControl().add_to(m)
    
    return m


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("PYCNOPODIA PROBABILITY DENSITY MAP")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    model, feature_cols = load_model()
    szline = load_shorezone()
    site_summary = load_site_summary()
    
    # Predict for each segment
    szline = predict_segments(szline, model, feature_cols)
    
    max_rate = min(szline["predicted_rate"].max(), 10.0)
    
    # Save predictions
    pred_df = szline[["predicted_rate"]].copy()
    pred_df.to_csv(OUTPUT_DIR / "segment_predictions.csv", index=False)
    print(f"\nSaved: segment_predictions.csv")
    
    # Create map
    m = create_map(szline, site_summary, max_rate)
    
    map_path = OUTPUT_DIR / "probability_density_map.html"
    m.save(str(map_path))
    
    print(f"\n{'=' * 70}")
    print(f"COMPLETE: {map_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
