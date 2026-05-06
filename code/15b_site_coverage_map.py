"""
Site Coverage Diagnostic Map

Creates an interactive map showing which sites have ShoreZone coverage
and why some sites are excluded from analysis.

Categories:
- Green: Sites with ShoreZone match (nearest segment within MAX_DISTANCE_M)
- Red: Sites within range but no ShoreZone data (coverage gap)
- Blue: Sites outside ShoreZone dataset extent (likely BC/Canada or outside WA coverage)

Also displays the ShoreZone line segments so you can see the actual coverage.

Author: Star Meadow Project
Date: December 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List

# Interactive mapping
try:
    import folium
    from folium import plugins
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    print("folium not installed. Run: pip install folium")

# Geospatial
try:
    import geopandas as gpd
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("geopandas not installed")

# Model loading
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("joblib not installed")

from utils import DATA_DIR

# Paths
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "15_shorezone_site_analysis"
RESULTS_DIR = Path(__file__).parent.parent / "outputs" / "16_shorezone_recovery_analysis"
RESULTS_MD = RESULTS_DIR / "results.md"
SITE_SUMMARY = OUTPUT_DIR / "site_shorezone_pycno_summary.csv"
DISTANCE_FILE = OUTPUT_DIR / "site_distance_to_shore.csv"
SITE_COORDS = DATA_DIR / "Site_LatLong.csv"
SHOREZONE_GDB = DATA_DIR / "state_DNR_ShoreZone" / "shorezone.gdb"

# Thresholds
MAX_DISTANCE_M = 3000  # 3km - matches the nearest-neighbor join threshold in 15_shorezone_site_analysis.py

# CRS
CRS_WGS84 = "EPSG:4326"
CRS_PROJECTED = "EPSG:32610"  # UTM Zone 10N for WA - used for geometry simplification

# Geometry simplification tolerance (meters in projected CRS)
# Higher value = smaller file size but less detail
SIMPLIFY_TOLERANCE = 50  # 50m tolerance for coastline - reduces vertices significantly

# Model path for predictions
MODEL_PATH = Path(__file__).parent.parent / "outputs" / "16_shorezone_recovery_analysis" / "trained_model.joblib"

# ShoreZone categories for feature extraction (prediction model)
SHOREZONE_PREDICTION_CATS = [
    "EXP_CLASS", "SED_SOURCE", "SED_ABUND", "ZONECOMP", "BC_CLASS",
    "NRDA_CLASS", "HAB_CALC", "ZOS_UNIT", "FUC_UNIT", "ULV_UNIT",
    "NER_UNIT", "OYS_UNIT", "MUS_UNIT"
]


# ShoreZone categories to display in popups
SHOREZONE_DISPLAY_COLS = [
    ("EXP_CLASS", "Wave Exposure"),
    ("BC_CLASS", "Biotic Class"),
    ("ZOS_UNIT", "Eelgrass"),
    ("NER_UNIT", "Bull Kelp"),
    ("FUC_UNIT", "Rockweed"),
    ("ZONECOMP", "Zone Comp"),
    ("HAB_CALC", "Habitat"),
]


def find_latest_pycno_count_file() -> Optional[Path]:
    """
    Find the most recently modified PycnoCount CSV in DATA_DIR.

    We intentionally avoid hard-coding a specific date-stamped filename because
    new data drops update these files over time.
    """
    patterns = [
        "PycnoCountCLean_*.csv",  # historical naming in this repo
        "PycnoCountClean_*.csv",  # alternate capitalization
    ]

    candidates: List[Path] = []
    for pat in patterns:
        candidates.extend(DATA_DIR.glob(pat))

    if not candidates:
        return None

    # Prefer newest by modified time (robust to filename date changes)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def get_shorezone_bounds_wgs84(shorezone_layers: Dict) -> Optional[Tuple[float, float, float, float]]:
    """
    Compute ShoreZone WGS84 bounds (minx, miny, maxx, maxy) from loaded layers.

    Prefer `szline` (shoreline segments). Falls back to any available layer.
    """
    if not shorezone_layers:
        return None

    preferred = ["szline", "szpoly", "szpt"]
    for key in preferred:
        gdf = shorezone_layers.get(key)
        if gdf is not None and len(gdf) > 0:
            try:
                gdf_wgs = gdf.to_crs(CRS_WGS84) if getattr(gdf, "crs", None) != CRS_WGS84 else gdf
                minx, miny, maxx, maxy = gdf_wgs.total_bounds
                return (float(minx), float(miny), float(maxx), float(maxy))
            except Exception:
                continue

    return None


def is_within_bounds(
    lat: float,
    lon: float,
    bounds: Optional[Tuple[float, float, float, float]],
) -> Optional[bool]:
    """Return True/False if point is within bounds; None if bounds unavailable."""
    if bounds is None or pd.isna(lat) or pd.isna(lon):
        return None
    minx, miny, maxx, maxy = bounds
    return (minx <= lon <= maxx) and (miny <= lat <= maxy)


def load_results_html() -> str:
    """
    Build a comprehensive analysis report from all script outputs.
    Returns HTML content for the results panel.
    """
    from datetime import datetime
    
    # Base outputs directory
    OUTPUTS_BASE = Path(__file__).parent.parent / "outputs"
    
    def _fmt(val, precision=2):
        """Format numeric values for display."""
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        if isinstance(val, float):
            return f"{val:.{precision}f}"
        return str(val)
    
    def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
        """Safely read CSV, returning None if not found."""
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                return None
        return None
    
    # Build HTML content directly
    html_parts: List[str] = []
    
    # =========================================================================
    # HEADER
    # =========================================================================
    html_parts.append(f"""
    <h1>Pycnopodia helianthoides Analysis Report</h1>
    <p class="meta">Comprehensive analysis of sunflower sea star distribution in the Salish Sea</p>
    <p class="meta">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <hr>
    """)
    
    # =========================================================================
    # 0. DATASET OVERVIEW
    # =========================================================================
    html_parts.append("""
    <h2>📊 Dataset Overview</h2>
    <div class="summary-box">
        <table class="summary-table">
            <tr><td>Survey records</td><td><strong>3,998</strong></td></tr>
            <tr><td>Unique sites</td><td><strong>139</strong></td></tr>
            <tr><td>Geographic basins</td><td><strong>8</strong></td></tr>
            <tr><td>Total Pycnopodia observed</td><td><strong>2,654</strong></td></tr>
            <tr><td>Date range</td><td>July 2020 – December 2025</td></tr>
            <tr><td>Overall detection rate</td><td>14.7%</td></tr>
        </table>
    </div>
    """)
    
    # =========================================================================
    # 1. ENCOUNTER RATE ANALYSIS
    # =========================================================================
    html_parts.append("""
    <h2>01 — Encounter Rate Analysis</h2>
    <p class="script-info">Script: <code>01_encounter_rate_analysis.py</code></p>
    <h3>Methods</h3>
    <p>Analyzed encounter rates (Pycnopodia per hour) across habitat types, geographic basins, 
    and depth bins using survey-level data. Bar plots with standard error, heatmaps, and 
    box plots visualize patterns.</p>
    """)
    
    enc_path = OUTPUTS_BASE / "01_encounter_rate_analysis" / "encounter_rate_summary.csv"
    enc_df = _read_csv_safe(enc_path)
    if enc_df is not None:
        html_parts.append("<h3>Results: Encounter Rate by Habitat</h3>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Habitat</th><th>Mean Rate</th><th>Std Dev</th><th>N Surveys</th><th>Total Pycno</th></tr>")
        for _, row in enc_df.iterrows():
            html_parts.append(f"""<tr>
                <td>{row.get('HabitatType', '—')}</td>
                <td><strong>{_fmt(row.get('Mean Rate'))}</strong>/hr</td>
                <td>±{_fmt(row.get('Std Dev'))}</td>
                <td>{int(row.get('N Surveys', 0))}</td>
                <td>{int(row.get('Total Pycnos', 0))}</td>
            </tr>""")
        html_parts.append("</table>")
    
    html_parts.append("""
    <div class="finding">
        <strong>Key Finding:</strong> Soft bottom habitat shows highest mean encounter rate (8.67/hr), 
        followed by eelgrass (4.10/hr). Natural reef has lowest rates (0.88/hr).
    </div>
    """)
    
    # =========================================================================
    # 2. SIZE STRUCTURE ANALYSIS
    # =========================================================================
    html_parts.append("""
    <h2>02 — Size Structure Analysis</h2>
    <p class="script-info">Script: <code>02_size_structure_analysis.py</code></p>
    <h3>Methods</h3>
    <p>Analyzed individual size measurements (arm length in cm) using 5cm bins. 
    Kernel density estimation and violin plots compare distributions across 
    habitats, depths, seasons, and basins.</p>
    """)
    
    size_path = OUTPUTS_BASE / "02_size_structure_analysis" / "size_summary_by_habitat.csv"
    size_df = _read_csv_safe(size_path)
    if size_df is not None:
        html_parts.append("<h3>Results: Size by Habitat</h3>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Habitat</th><th>N</th><th>Mean (cm)</th><th>Median</th><th>Range</th></tr>")
        for _, row in size_df.iterrows():
            html_parts.append(f"""<tr>
                <td>{row.get('HabitatType', '—')}</td>
                <td>{int(row.get('N', 0))}</td>
                <td><strong>{_fmt(row.get('Mean'))}</strong></td>
                <td>{_fmt(row.get('Median'))}</td>
                <td>{_fmt(row.get('Min'))}–{_fmt(row.get('Max'))}</td>
            </tr>""")
        html_parts.append("</table>")
    
    html_parts.append("""
    <div class="finding">
        <strong>Key Finding:</strong> Mean size is 19.6 cm overall. Artificial reef individuals 
        are largest (20.7 cm mean), while eelgrass sites have smaller individuals (12.1 cm).
    </div>
    """)
    
    # =========================================================================
    # 3. CLUSTERING ANALYSIS
    # =========================================================================
    html_parts.append("""
    <h2>03 — Hierarchical Clustering</h2>
    <p class="script-info">Script: <code>03_clustering_analysis.py</code></p>
    <h3>Methods</h3>
    <p>Hierarchical clustering (Ward's method) groups sites by habitat composition ratios. 
    Clustergrams show habitat similarity with encounter rate overlay rectangles.</p>
    <h3>Results</h3>
    <p>Two aggregation approaches: (1) Site × Depth-Bin captures vertical variation; 
    (2) Site-only emphasizes spatial differences. Clusters reveal habitat guilds 
    associated with different encounter rates.</p>
    <div class="finding">
        <strong>Key Finding:</strong> Sites cluster into distinct habitat guilds. High-encounter 
        clusters tend to be dominated by soft bottom and eelgrass habitats.
    </div>
    """)
    
    # =========================================================================
    # 4. TEMPORAL ANALYSIS
    # =========================================================================
    html_parts.append("""
    <h2>04 — Temporal Analysis</h2>
    <p class="script-info">Script: <code>04_temporal_analysis.py</code></p>
    <h3>Methods</h3>
    <p>Year-over-year and monthly trend analysis using time series plots, heatmaps 
    of encounter rates by month × year, and cumulative observation curves.</p>
    <h3>Results</h3>
    <p>Survey effort and observations tracked from 2020–2025. Peak survey months are 
    summer (June–August). Detection rates vary by year as survey effort expanded.</p>
    <div class="finding">
        <strong>Key Finding:</strong> Observations concentrated in summer months. Year-to-year 
        variation reflects expanding survey coverage rather than population trends.
    </div>
    """)
    
    # =========================================================================
    # 5. STATISTICAL SUMMARY
    # =========================================================================
    html_parts.append("""
    <h2>05 — Statistical Summary</h2>
    <p class="script-info">Script: <code>05_statistical_summary.py</code></p>
    <h3>Methods</h3>
    <p>Kruskal-Wallis tests for habitat, basin, and depth effects. Pairwise Mann-Whitney U 
    tests with Bonferroni correction. Effect sizes (rank-biserial correlation) calculated.</p>
    """)
    
    stats_path = OUTPUTS_BASE / "05_statistical_summary" / "pairwise_habitat_tests.csv"
    stats_df = _read_csv_safe(stats_path)
    if stats_df is not None:
        html_parts.append("<h3>Pairwise Habitat Comparisons</h3>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Comparison</th><th>p-value</th><th>Effect Size</th><th>Sig?</th></tr>")
        for _, row in stats_df.head(6).iterrows():
            sig = "✓" if row.get('Significant', False) else "—"
            p_val = row.get('p-value', 0)
            p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
            html_parts.append(f"""<tr>
                <td>{row.get('Comparison', '—')}</td>
                <td>{p_str}</td>
                <td>{_fmt(row.get('Effect Size (r)'))}</td>
                <td>{sig}</td>
            </tr>""")
        html_parts.append("</table>")
    
    html_parts.append("""
    <div class="finding">
        <strong>Key Finding:</strong> Habitat significantly affects encounter rate (Kruskal-Wallis 
        H = 81.5, p < 0.0001). Natural Reef vs Soft Bottom shows largest effect (r = 0.22).
    </div>
    """)
    
    # =========================================================================
    # 6. EELGRASS SITE ANALYSIS
    # =========================================================================
    html_parts.append("""
    <h2>06 — Eelgrass Site-Level Analysis</h2>
    <p class="script-info">Script: <code>06_eelgrass_site_analysis.py</code></p>
    <h3>Methods</h3>
    <p>Sites classified by presence/absence of eelgrass habitat at the site level 
    (not transect level). Four-category habitat scheme: Eelgrass, Hard Bottom + Eelgrass Site, 
    Hard Bottom (No Eelgrass), Soft Bottom. Mann-Whitney and Kruskal-Wallis tests.</p>
    """)
    
    eelgrass_path = OUTPUTS_BASE / "06_eelgrass_site_analysis" / "site_eelgrass_summary.csv"
    eelgrass_df = _read_csv_safe(eelgrass_path)
    if eelgrass_df is not None:
        html_parts.append("<h3>Results: Site Eelgrass Effect</h3>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Site Category</th><th>Mean Rate</th><th>N Transects</th><th>N Sites</th></tr>")
        for _, row in eelgrass_df.iterrows():
            html_parts.append(f"""<tr>
                <td>{row.get('Site_Eelgrass_Category', '—')}</td>
                <td><strong>{_fmt(row.get('Mean Rate'))}</strong>/hr</td>
                <td>{int(row.get('N Transects', 0))}</td>
                <td>{int(row.get('N Sites', 0))}</td>
            </tr>""")
        html_parts.append("</table>")
    
    stats6_path = OUTPUTS_BASE / "06_eelgrass_site_analysis" / "statistical_tests.csv"
    stats6_df = _read_csv_safe(stats6_path)
    if stats6_df is not None:
        html_parts.append("<h3>Statistical Tests</h3>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Test</th><th>Statistic</th><th>p-value</th><th>Effect Size</th></tr>")
        for _, row in stats6_df.iterrows():
            p_val = row.get('p-value', 0)
            p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.4f}"
            html_parts.append(f"""<tr>
                <td>{row.get('Test', '—')}</td>
                <td>{_fmt(row.get('Statistic'), 0)}</td>
                <td>{p_str}</td>
                <td>{_fmt(row.get('Effect Size'))}</td>
            </tr>""")
        html_parts.append("</table>")
    
    html_parts.append("""
    <div class="finding">
        <strong>Key Finding:</strong> Sites with eelgrass present show 8× higher encounter rates 
        (5.87/hr vs 0.71/hr, p < 0.001). Effect size r = -0.35 indicates strong association.
    </div>
    """)
    
    # =========================================================================
    # 07. SIZE DISTRIBUTION CLUSTERING
    # =========================================================================
    html_parts.append("""
    <h2>07 — Size Distribution Clustering</h2>
    <p class="script-info">Script: <code>07_size_clustering_analysis.py</code></p>
    <h3>Methods</h3>
    <p>Analyzes size distributions (5cm bins) across habitat types and site-level 
    eelgrass presence. Hierarchical clustering groups sites by size profile similarity. 
    Uses individual length measurements (n≈628).</p>
    <h3>Results</h3>
    <p>Size distributions reveal population structure differences between habitat types. 
    Sites cluster into groups with similar size profiles, which may indicate different 
    recruitment patterns or survival rates.</p>
    <div class="finding">
        <strong>Key Finding:</strong> Size profiles vary significantly by habitat. 
        Eelgrass-associated sites tend to have distinct size distributions compared 
        to non-eelgrass sites.
    </div>
    """)
    
    # =========================================================================
    # 08. COMBINED CLUSTERING
    # =========================================================================
    html_parts.append("""
    <h2>08 — Combined Size + Habitat Clustering</h2>
    <p class="script-info">Script: <code>08_combined_clustering_analysis.py</code></p>
    <h3>Methods</h3>
    <p>Creates three clustering approaches: (1) Combined clustering on both size and 
    habitat, (2) Size clustering with habitat sidebar, (3) Habitat clustering with 
    size sidebar. Only sites with both habitat surveys AND size measurements included.</p>
    <h3>Results</h3>
    <p>Combined clustering reveals whether sites similar in habitat composition also 
    have similar size distributions. Sidebars enable visual comparison of how one 
    characteristic maps onto clusters formed by the other.</p>
    """)
    
    # =========================================================================
    # 09. SIZE PREDICTION ANALYSIS
    # =========================================================================
    html_parts.append("""
    <h2>09 — Size Prediction from Habitat</h2>
    <p class="script-info">Script: <code>09_size_prediction_analysis.py</code></p>
    <h3>Methods</h3>
    <p>Regression models (Linear, Ridge, Random Forest) predict size metrics from 
    habitat composition. Predictors: habitat type proportions at each site. 
    Response variables: mean size, median size, std, and size bin proportions.</p>
    <h3>Results</h3>
    <p>Model performance evaluated via cross-validation. Feature importance identifies 
    which habitat types best predict size distributions.</p>
    <div class="finding">
        <strong>Key Finding:</strong> Habitat composition has limited predictive power 
        for size metrics (low R²), suggesting size is driven by other factors 
        (recruitment, disease history, prey availability).
    </div>
    """)
    
    # =========================================================================
    # 10. EELGRASS SIZE RELATIONSHIP
    # =========================================================================
    html_parts.append("""
    <h2>10 — Eelgrass–Size Relationship</h2>
    <p class="script-info">Script: <code>10_eelgrass_size_relationship.py</code></p>
    <h3>Methods</h3>
    <p>Tests whether Pycnopodia are larger at eelgrass sites. Site-level analysis 
    compares mean/median size between eelgrass vs non-eelgrass sites. Individual-level 
    analysis compares all measurements. Effect sizes (Cohen's d) calculated.</p>
    <h3>Results</h3>
    <table>
        <tr><th>Level</th><th>Eelgrass Sites</th><th>Non-Eelgrass</th><th>p-value</th></tr>
        <tr><td>Site mean size</td><td><strong>19.5 cm</strong></td><td>15.2 cm</td><td>&lt;0.001</td></tr>
        <tr><td>Individual size</td><td><strong>18.0 cm</strong> (median)</td><td>9.0 cm</td><td>&lt;0.0001</td></tr>
    </table>
    <div class="finding">
        <strong>Key Finding:</strong> Individuals at eelgrass sites are significantly larger. 
        Effect size Cohen's d = 0.42 (individual), 0.63 (site-level). Distribution shapes 
        differ (KS test D = 0.48, p < 0.0001).
    </div>
    """)
    
    # =========================================================================
    # 11. HABITAT DIVERSITY ANALYSIS
    # =========================================================================
    html_parts.append("""
    <h2>11 — Habitat Diversity Analysis</h2>
    <p class="script-info">Script: <code>11_habitat_diversity_analysis.py</code></p>
    <h3>Methods</h3>
    <p>Tests hypothesis that habitat heterogeneity predicts abundance. Metrics: 
    Habitat Richness (# types), Shannon Diversity Index. Critical confound control: 
    sites with more transect locations detect more habitats (spatial design artifact). 
    Partial correlations control for spatial coverage.</p>
    <h3>Results</h3>
    <table>
        <tr><th>Analysis</th><th>Correlation</th><th>p-value</th></tr>
        <tr><td>Raw richness vs rate</td><td>r = 0.25</td><td>0.01</td></tr>
        <tr><td><strong>Partial (controlling coverage)</strong></td><td><strong>r = -0.01</strong></td><td><strong>0.92</strong></td></tr>
    </table>
    <div class="finding">
        <strong>Key Finding:</strong> Habitat diversity does NOT predict Pycnopodia abundance 
        after controlling for sampling effort (partial r = -0.01, p = 0.92). The raw 
        correlation was entirely a sampling artifact.
    </div>
    """)
    
    # =========================================================================
    # 12. EELGRASS BASIN ANALYSIS
    # =========================================================================
    html_parts.append("""
    <h2>12 — Eelgrass Effect (Basin-Controlled)</h2>
    <p class="script-info">Script: <code>12_eelgrass_basin_analysis.py</code></p>
    <h3>Methods</h3>
    <p>Tests eelgrass–encounter rate relationship controlling for basin effects. 
    Approaches: (1) Stratified within-basin tests, (2) Two-way ANOVA, 
    (3) Partial correlation controlling for basin. Addresses confound that both 
    eelgrass and Pycnopodia may vary by basin.</p>
    <h3>Results</h3>
    <table>
        <tr><th>Analysis</th><th>Result</th><th>p-value</th></tr>
        <tr><td>Overall eelgrass effect</td><td>Significant</td><td>&lt;0.001</td></tr>
        <tr><td>Basin effect</td><td>Significant</td><td>&lt;0.001</td></tr>
        <tr><td><strong>Partial r (basin-controlled)</strong></td><td><strong>r = 0.21</strong></td><td><strong>0.029</strong></td></tr>
    </table>
    <div class="finding">
        <strong>Key Finding:</strong> Eelgrass effect remains significant (partial r = 0.21, 
        p = 0.029) after controlling for basin. Effect consistent in 5 of 7 basins tested.
    </div>
    """)
    
    # =========================================================================
    # 13. STATIC MAPS
    # =========================================================================
    html_parts.append("""
    <h2>13 — Static Map Visualizations</h2>
    <p class="script-info">Script: <code>13_static_maps.py</code></p>
    <h3>Methods</h3>
    <p>Matplotlib-based maps showing: site distribution, survey effort (transects, hours), 
    encounter rates by site, detection probability, eelgrass site distribution, and 
    basin-level summaries. Site coordinates from GPS data.</p>
    <h3>Outputs</h3>
    <ul>
        <li>Survey site distribution map</li>
        <li>Encounter rate by site (bubble map)</li>
        <li>Detection rate by site</li>
        <li>Eelgrass site highlighting</li>
        <li>Basin-level summary map</li>
    </ul>
    """)
    
    # =========================================================================
    # 14. INTERACTIVE MAPS
    # =========================================================================
    html_parts.append("""
    <h2>14 — Interactive Map Visualizations</h2>
    <p class="script-info">Script: <code>14_interactive_maps.py</code></p>
    <h3>Methods</h3>
    <p>Folium-based interactive HTML maps with: site popups showing survey metrics, 
    encounter rate heatmap layer, eelgrass site highlighting, basin color coding, 
    and multiple basemap options.</p>
    <h3>Outputs</h3>
    <ul>
        <li>Interactive site map with popups</li>
        <li>Encounter rate heatmap</li>
        <li>Clustered marker map</li>
        <li>Basin-colored site map</li>
        <li>Eelgrass highlight map</li>
    </ul>
    <div class="finding">
        <strong>Note:</strong> Interactive maps can be opened in any web browser. 
        Click sites for detailed survey information.
    </div>
    """)
    
    # =========================================================================
    # 15. SHOREZONE SITE ANALYSIS
    # =========================================================================
    html_parts.append("""
    <h2>15 — ShoreZone Spatial Analysis</h2>
    <p class="script-info">Script: <code>15_shorezone_site_analysis.py</code></p>
    <h3>Methods</h3>
    <p>Nearest-neighbor spatial join (max 3km) between dive sites and Washington State 
    ShoreZone inventory data (1994–2000). Extracts shoreline habitat characteristics 
    including wave exposure, sediment type, and biotic indicators.</p>
    <h3>Results</h3>
    <table>
        <tr><td>Sites with ShoreZone match</td><td><strong>104</strong> (74.8%)</td></tr>
        <tr><td>Sites beyond 3km threshold</td><td>35 (25.2%)</td></tr>
        <tr><td>Join method</td><td>Nearest segment within 3km</td></tr>
    </table>
    <div class="warning">
        <strong>⚠️ Data Caveat:</strong> ShoreZone data collected 1994–2000; dive surveys from 
        2020–2025. Biotic features (eelgrass, kelp) may have changed significantly. 
        Physical features (wave exposure, sediment) are more temporally stable.
    </div>
    """)
    
    # =========================================================================
    # 16. SHOREZONE RECOVERY ANALYSIS
    # =========================================================================
    html_parts.append("""
    <h2>16 — Habitat Suitability Model</h2>
    <p class="script-info">Script: <code>16_shorezone_recovery_analysis.py</code></p>
    <h3>Methods</h3>
    <p>Random Forest regression predicting encounter rate from one-hot encoded ShoreZone 
    categorical features. 5-fold cross-validation with permutation importance. 
    Model predictions used to identify priority unsurveyed sites.</p>
    """)
    
    perm_path = RESULTS_DIR / "feature_importance_permutation.csv"
    perm_df = _read_csv_safe(perm_path)
    if perm_df is not None:
        perm_df = perm_df.sort_values("Importance_Mean", ascending=False)
        html_parts.append("<h3>Top Predictors (Permutation Importance)</h3>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Feature</th><th>Importance</th><th>Std</th></tr>")
        for _, row in perm_df.head(10).iterrows():
            imp = row.get('Importance_Mean', 0)
            html_parts.append(f"""<tr>
                <td>{row.get('Feature', '—')}</td>
                <td><strong>{_fmt(imp, 3)}</strong></td>
                <td>±{_fmt(row.get('Importance_Std'), 3)}</td>
            </tr>""")
        html_parts.append("</table>")
    
    html_parts.append("""
    <div class="finding">
        <strong>Key Finding:</strong> ZOS_UNIT_C (continuous eelgrass) dominates predictions 
        (importance = 0.29). Model R² = 0.08 indicates weak predictive power from 
        ShoreZone features alone—other factors (depth, prey, disease history) matter more.
    </div>
    """)
    
    # Priority Sites
    prio_path = RESULTS_DIR / "priority_sites_with_uncertainty.csv"
    prio_df = _read_csv_safe(prio_path)
    if prio_df is not None:
        html_parts.append("<h3>Priority Sites for Future Surveys</h3>")
        
        # Tier summary
        if "Priority_Tier" in prio_df.columns:
            tier_counts = prio_df["Priority_Tier"].value_counts()
            html_parts.append("<table>")
            html_parts.append("<tr><th>Priority Tier</th><th>Count</th></tr>")
            for tier in ["Very High", "High", "Medium", "Low"]:
                if tier in tier_counts:
                    html_parts.append(f"<tr><td>{tier}</td><td>{tier_counts[tier]}</td></tr>")
            html_parts.append("</table>")
        
        # Top candidates
        if "Priority_Score" in prio_df.columns:
            prio_df = prio_df.sort_values("Priority_Score", ascending=False)
        html_parts.append("<h4>Top 10 Candidates</h4>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Site</th><th>Predicted Rate</th><th>95% CI</th><th>Tier</th></tr>")
        for _, row in prio_df.head(10).iterrows():
            ci_low = row.get('CI_95_Low', 0)
            ci_high = row.get('CI_95_High', 0)
            html_parts.append(f"""<tr>
                <td>{row.get('SiteName', '—')}</td>
                <td><strong>{_fmt(row.get('Predicted_Rate'))}</strong>/hr</td>
                <td>[{_fmt(ci_low)}–{_fmt(ci_high)}]</td>
                <td>{row.get('Priority_Tier', '—')}</td>
            </tr>""")
        html_parts.append("</table>")
    
    # Refugia
        ref_path = RESULTS_DIR / "refugia_comparison.csv"
    ref_df = _read_csv_safe(ref_path)
    if ref_df is not None:
        html_parts.append("<h3>Refugia Habitat Comparison</h3>")
        html_parts.append("<p>Comparing habitat features between high-detection refugia sites and other sites:</p>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Feature</th><th>Refugia Mean</th><th>Other Mean</th><th>p-value</th></tr>")
        ref_df = ref_df.sort_values("P_Value")
        for _, row in ref_df.head(8).iterrows():
            p_val = row.get('P_Value', 1)
            sig_mark = " *" if p_val < 0.1 else ""
            html_parts.append(f"""<tr>
                <td>{row.get('Feature', '—')}</td>
                <td>{_fmt(row.get('Refugia_Mean'))}</td>
                <td>{_fmt(row.get('NonRefugia_Mean'))}</td>
                <td>{_fmt(p_val, 3)}{sig_mark}</td>
            </tr>""")
        html_parts.append("</table>")
    
    # =========================================================================
    # SUMMARY OF FINDINGS
    # =========================================================================
    html_parts.append("""
    <h2>Summary of Key Findings</h2>
    <div class="conclusions">
        <h3>✓ Supported Hypotheses</h3>
        <ul>
            <li><strong>Habitat affects encounter rate</strong> — Kruskal-Wallis H = 81.5, p < 0.0001</li>
            <li><strong>Eelgrass sites have higher encounter rates</strong> — 8× higher, p < 0.001</li>
            <li><strong>Strong geographic (basin) effects</strong> — Whidbey Basin highest rates</li>
            <li><strong>Larger individuals at eelgrass sites</strong> — p < 0.0001, d = 0.42</li>
        </ul>
        <h3>✗ Not Supported</h3>
        <ul>
            <li><strong>Habitat diversity predicts abundance</strong> — partial r = -0.01, p = 0.92</li>
            <li><strong>ShoreZone features alone predict encounter rate</strong> — Model R² = 0.08</li>
        </ul>
        <h3>Implications</h3>
        <p>Eelgrass-associated sites appear to function as refugia for Pycnopodia recovery 
        following SSWD. The ShoreZone model's weak predictive power suggests local-scale 
        factors (prey availability, depth, disease history) are more important than 
        regional shoreline characteristics alone.</p>
    </div>
    """)
    
    # =========================================================================
    # OUTPUTS INDEX
    # =========================================================================
    html_parts.append("""
    <h2>Analysis Outputs</h2>
    <div class="outputs-list">
        <h4>Figures (PNG/PDF)</h4>
        <ul>
            <li>01: Encounter rate by habitat, basin, depth; heatmaps</li>
            <li>02: Size distribution histograms, violin plots by habitat/season</li>
            <li>03: Site clustering dendrograms and clustergrams</li>
            <li>04: Annual/monthly trends, cumulative observations</li>
            <li>05: Statistical summary plots, Q-Q plots</li>
            <li>06: Eelgrass site comparisons, interaction plots</li>
            <li>15: ShoreZone distribution plots, correlation heatmaps</li>
            <li>16: Feature importance, priority site maps, refugia profiles</li>
        </ul>
        <h4>Data Tables (CSV)</h4>
        <ul>
            <li>Encounter rate summaries by habitat, basin, depth</li>
            <li>Size statistics by habitat, season, depth</li>
            <li>Statistical test results (pairwise comparisons)</li>
            <li>Site-ShoreZone joined data with habitat features</li>
            <li>Model predictions and priority site rankings</li>
        </ul>
    </div>
    """)
    
    return "\n".join(html_parts)


def create_results_panel_html(results_html: str) -> str:
    """Create the toggleable results panel HTML/CSS/JS."""
    if not results_html:
        results_html = "<h2>Analysis Results</h2><p>(No results content available.)</p>"
    
    return f'''
    <style>
        #results-toggle-btn {{
            position: fixed;
            bottom: 20px;
            left: 20px;
            z-index: 1000;
            background: linear-gradient(135deg, #1e3a5f 0%, #2c5f8d 100%);
            color: white;
            border: none;
            padding: 14px 24px;
            border-radius: 10px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
            letter-spacing: 0.5px;
        }}
        #results-toggle-btn:hover {{
            background: linear-gradient(135deg, #2c5f8d 0%, #3d7cb8 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.4);
        }}
        #results-panel {{
            position: fixed;
            top: 0;
            right: -680px;
            width: 660px;
            height: 100vh;
            background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
            color: #c9d1d9;
            z-index: 999;
            overflow-y: auto;
            transition: right 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            box-shadow: -8px 0 30px rgba(0,0,0,0.6);
            padding: 25px 35px;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 13px;
            line-height: 1.7;
        }}
        #results-panel.open {{
            right: 0;
        }}
        #results-panel h1 {{
            color: #58a6ff;
            font-size: 24px;
            font-weight: 700;
            border-bottom: 3px solid #58a6ff;
            padding-bottom: 12px;
            margin-top: 0;
            margin-bottom: 8px;
            letter-spacing: -0.5px;
        }}
        #results-panel .meta {{
            color: #8b949e;
            font-size: 12px;
            margin: 4px 0;
        }}
        #results-panel h2 {{
            color: #3fb950;
            font-size: 17px;
            font-weight: 600;
            margin-top: 30px;
            margin-bottom: 12px;
            border-left: 4px solid #3fb950;
            padding-left: 12px;
            background: linear-gradient(90deg, rgba(63,185,80,0.1) 0%, transparent 100%);
            padding: 8px 12px;
            border-radius: 0 6px 6px 0;
        }}
        #results-panel h3 {{
            color: #f0883e;
            font-size: 14px;
            font-weight: 600;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        #results-panel h4 {{
            color: #a371f7;
            font-size: 13px;
            font-weight: 600;
            margin-top: 16px;
            margin-bottom: 8px;
        }}
        #results-panel .script-info {{
            color: #8b949e;
            font-size: 11px;
            margin: -8px 0 12px 0;
            padding-left: 16px;
        }}
        #results-panel table {{
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            margin: 14px 0;
            font-size: 11.5px;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }}
        #results-panel th {{
            background: linear-gradient(180deg, #21262d 0%, #1c2128 100%);
            color: #f0f6fc;
            padding: 10px 8px;
            text-align: left;
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 2px solid #30363d;
        }}
        #results-panel td {{
            padding: 8px;
            border-bottom: 1px solid #21262d;
            background: #0d1117;
        }}
        #results-panel tr:hover td {{
            background: #161b22;
        }}
        #results-panel .summary-box {{
            background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 16px;
            margin: 12px 0;
        }}
        #results-panel .summary-table {{
            margin: 0;
            box-shadow: none;
        }}
        #results-panel .summary-table td {{
            background: transparent;
            border-bottom: 1px solid #30363d;
        }}
        #results-panel .summary-table td:first-child {{
            color: #8b949e;
        }}
        #results-panel .finding {{
            background: linear-gradient(135deg, rgba(88,166,255,0.1) 0%, rgba(88,166,255,0.05) 100%);
            border-left: 4px solid #58a6ff;
            padding: 12px 16px;
            margin: 16px 0;
            border-radius: 0 8px 8px 0;
            font-size: 12.5px;
        }}
        #results-panel .warning {{
            background: linear-gradient(135deg, rgba(240,136,62,0.15) 0%, rgba(240,136,62,0.05) 100%);
            border-left: 4px solid #f0883e;
            padding: 12px 16px;
            margin: 16px 0;
            border-radius: 0 8px 8px 0;
            font-size: 12px;
        }}
        #results-panel .conclusions {{
            background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
        }}
        #results-panel .conclusions h3 {{
            margin-top: 0;
        }}
        #results-panel .conclusions ul {{
            margin: 8px 0;
            padding-left: 20px;
        }}
        #results-panel .outputs-list {{
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 10px;
            padding: 16px;
            margin: 12px 0;
        }}
        #results-panel .outputs-list ul {{
            margin: 4px 0;
            padding-left: 18px;
        }}
        #results-panel .outputs-list li {{
            font-size: 11.5px;
            color: #8b949e;
            margin: 3px 0;
        }}
        #results-panel code {{
            background: #21262d;
            padding: 2px 7px;
            border-radius: 4px;
            font-family: 'SF Mono', 'Fira Code', Monaco, monospace;
            font-size: 11px;
            color: #79c0ff;
            border: 1px solid #30363d;
        }}
        #results-panel hr {{
            border: none;
            border-top: 1px solid #30363d;
            margin: 24px 0;
        }}
        #results-panel li {{
            margin: 5px 0;
        }}
        #results-panel p {{
            margin: 10px 0;
        }}
        #results-panel strong {{
            color: #f0f6fc;
        }}
        #results-close {{
            position: fixed;
            top: 20px;
            right: 25px;
            background: linear-gradient(135deg, #f85149 0%, #da3633 100%);
            color: white;
            border: none;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 20px;
            line-height: 1;
            box-shadow: 0 2px 8px rgba(248,81,73,0.4);
            transition: all 0.2s ease;
            z-index: 1001;
        }}
        #results-close:hover {{
            background: linear-gradient(135deg, #da3633 0%, #b62324 100%);
            transform: scale(1.1);
        }}
        /* Scrollbar styling */
        #results-panel::-webkit-scrollbar {{
            width: 8px;
        }}
        #results-panel::-webkit-scrollbar-track {{
            background: #0d1117;
        }}
        #results-panel::-webkit-scrollbar-thumb {{
            background: #30363d;
            border-radius: 4px;
        }}
        #results-panel::-webkit-scrollbar-thumb:hover {{
            background: #484f58;
        }}
    </style>
    
    <button id="results-toggle-btn" onclick="toggleResults()">📊 Analysis Report</button>
    
    <div id="results-panel">
        <button id="results-close" onclick="toggleResults()">×</button>
        {results_html}
    </div>
    
    <script>
        function toggleResults() {{
            var panel = document.getElementById('results-panel');
            panel.classList.toggle('open');
        }}
        // Close panel with Escape key
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Escape') {{
                document.getElementById('results-panel').classList.remove('open');
            }}
        }});
    </script>
    '''


def load_data():
    """Load site data, distance information, and ShoreZone codes."""
    # Load all sites
    sites = pd.read_csv(SITE_COORDS)
    sites.columns = ["SiteName", "Lat", "Long"]
    sites.loc[sites["Long"] > 0, "Long"] = -sites.loc[sites["Long"] > 0, "Long"]
    
    # Load distance to shore
    distances = pd.read_csv(DISTANCE_FILE)
    
    # Load summary (has ShoreZone match info and encounter rate)
    summary = pd.read_csv(SITE_SUMMARY)
    
    # Merge
    sites = sites.merge(distances, on="SiteName", how="left")
    
    # Check if site has ShoreZone data (non-NaN in any diversity column)
    diversity_cols = [c for c in summary.columns if "_Diversity" in c or "_Richness" in c]
    if diversity_cols:
        summary["has_shorezone"] = summary[diversity_cols].notna().any(axis=1)
    else:
        summary["has_shorezone"] = False
    
    # Get MeanEncounterRate and other pycno data
    pycno_cols = ["SiteName", "has_shorezone", "MeanEncounterRate", "TotalPycnoCount", "DetectionRate"]
    pycno_cols = [c for c in pycno_cols if c in summary.columns]
    
    sites = sites.merge(
        summary[pycno_cols].drop_duplicates(),
        on="SiteName", how="left"
    )
    sites["has_shorezone"] = sites["has_shorezone"].fillna(False)
    
    # Load ShoreZone codes from value files
    for col, display_name in SHOREZONE_DISPLAY_COLS:
        value_path = OUTPUT_DIR / f"shorezone_{col}_value.csv"
        if value_path.exists():
            values = pd.read_csv(value_path)
            sites = sites.merge(values, on="SiteName", how="left")
    
    print(f"  Loaded ShoreZone codes for {len([c for c, _ in SHOREZONE_DISPLAY_COLS if c in sites.columns])} categories")
    
    # Load dive survey habitat data (from pycno count data)
    pycno_count_file = find_latest_pycno_count_file()
    if pycno_count_file is not None and pycno_count_file.exists():
        pycno_df = pd.read_csv(pycno_count_file)
        print(f"  Using Pycno count file: {pycno_count_file.name}")
        
        # Calculate habitat proportions per site
        if "HabitatType" in pycno_df.columns and "SiteName" in pycno_df.columns:
            # Count transects by habitat type per site
            hab_counts = pycno_df.groupby(["SiteName", "HabitatType"]).size().unstack(fill_value=0)
            
            # Calculate proportions
            hab_props = hab_counts.div(hab_counts.sum(axis=1), axis=0)
            hab_props.columns = [f"hab_{c.replace(' ', '_')}" for c in hab_props.columns]
            hab_props = hab_props.reset_index()
            
            # Also keep total transects per site
            hab_props["TotalTransects"] = hab_counts.sum(axis=1).values
            
            sites = sites.merge(hab_props, on="SiteName", how="left")
            print(f"  Loaded dive habitat proportions for {len(hab_props)} sites")
    else:
        print("  Warning: No PycnoCount* CSV found in data/. Skipping dive habitat proportions.")
    
    return sites


def count_vertices(geom):
    """Count vertices in a geometry, handling multi-part geometries."""
    if geom is None or geom.is_empty:
        return 0
    
    geom_type = geom.geom_type
    
    if geom_type == "Point":
        return 1
    elif geom_type == "LineString":
        return len(geom.coords)
    elif geom_type == "Polygon":
        # Exterior ring + interior rings
        count = len(geom.exterior.coords)
        for interior in geom.interiors:
            count += len(interior.coords)
        return count
    elif geom_type in ("MultiPoint", "MultiLineString", "MultiPolygon", "GeometryCollection"):
        return sum(count_vertices(part) for part in geom.geoms)
    else:
        return 0


def simplify_geometry(gdf: gpd.GeoDataFrame, tolerance: float = SIMPLIFY_TOLERANCE) -> gpd.GeoDataFrame:
    """
    Simplify geometries to reduce file size while preserving shape.
    
    Projects to UTM, simplifies with tolerance in meters, then reprojects to WGS84.
    """
    if len(gdf) == 0:
        return gdf
    
    # Project to UTM for meter-based simplification
    gdf_proj = gdf.to_crs(CRS_PROJECTED)
    
    # Count original vertices
    orig_vertices = sum(count_vertices(g) for g in gdf_proj.geometry)
    
    # Simplify
    gdf_proj['geometry'] = gdf_proj.geometry.simplify(tolerance, preserve_topology=True)
    
    # Count simplified vertices
    simp_vertices = sum(count_vertices(g) for g in gdf_proj.geometry)
    
    reduction = (1 - simp_vertices / orig_vertices) * 100 if orig_vertices > 0 else 0
    print(f"    Simplified: {orig_vertices:,} → {simp_vertices:,} vertices ({reduction:.1f}% reduction)")
    
    # Reproject back to WGS84
    return gdf_proj.to_crs(CRS_WGS84)


def load_shorezone(sample_fraction: float = 0.1, simplify: bool = True):
    """
    Load ShoreZone layers for visualization.
    
    Parameters
    ----------
    sample_fraction : float
        Fraction of features to load (0.1 = 10% for faster rendering)
    simplify : bool
        If True, simplify geometries to reduce file size (recommended for web)
    
    Returns
    -------
    dict of layer_name -> gpd.GeoDataFrame
    """
    if not HAS_GEOPANDAS:
        print("  geopandas not available, skipping ShoreZone layer")
        return {}
    
    if not SHOREZONE_GDB.exists():
        print(f"  ShoreZone GDB not found: {SHOREZONE_GDB}")
        return {}
    
    layers = {}
    
    # Load line segments
    print(f"  Loading ShoreZone szline (shoreline segments)...")
    szline = gpd.read_file(str(SHOREZONE_GDB), layer="szline")
    print(f"    Loaded {len(szline)} segments")
    if sample_fraction < 1.0:
        n_sample = int(len(szline) * sample_fraction)
        szline = szline.sample(n=n_sample, random_state=42)
        print(f"    Sampled to {len(szline)} segments ({sample_fraction*100:.0f}%)")
    if szline.crs != CRS_WGS84:
        szline = szline.to_crs(CRS_WGS84)
    if simplify:
        szline = simplify_geometry(szline)
    layers["szline"] = szline
    
    # Load polygons (habitat zones)
    print(f"  Loading ShoreZone szpoly (habitat polygons)...")
    szpoly = gpd.read_file(str(SHOREZONE_GDB), layer="szpoly")
    print(f"    Loaded {len(szpoly)} polygons")
    if sample_fraction < 1.0:
        n_sample = int(len(szpoly) * sample_fraction)
        szpoly = szpoly.sample(n=n_sample, random_state=42)
        print(f"    Sampled to {len(szpoly)} polygons ({sample_fraction*100:.0f}%)")
    if szpoly.crs != CRS_WGS84:
        szpoly = szpoly.to_crs(CRS_WGS84)
    if simplify:
        szpoly = simplify_geometry(szpoly)
    layers["szpoly"] = szpoly
    
    # Load points
    print(f"  Loading ShoreZone szpt (observation points)...")
    szpt = gpd.read_file(str(SHOREZONE_GDB), layer="szpt")
    print(f"    Loaded {len(szpt)} points")
    if sample_fraction < 1.0:
        n_sample = int(len(szpt) * sample_fraction)
        szpt = szpt.sample(n=n_sample, random_state=42)
        print(f"    Sampled to {len(szpt)} points ({sample_fraction*100:.0f}%)")
    if szpt.crs != CRS_WGS84:
        szpt = szpt.to_crs(CRS_WGS84)
    # Points don't need simplification
    layers["szpt"] = szpt
    
    return layers


def load_prediction_model():
    """Load the trained habitat suitability model."""
    if not HAS_JOBLIB:
        print("  joblib not available, skipping predictions")
        return None, None
    
    if not MODEL_PATH.exists():
        print(f"  Model not found: {MODEL_PATH}")
        return None, None
    
    try:
        data = joblib.load(MODEL_PATH)
        print(f"  Loaded prediction model ({len(data['feature_cols'])} features)")
        return data["model"], data["feature_cols"]
    except Exception as e:
        print(f"  Error loading model: {e}")
        return None, None


def predict_segment_rates(szline: gpd.GeoDataFrame, model, feature_cols: list) -> gpd.GeoDataFrame:
    """Add predicted encounter rate to each ShoreZone segment."""
    if model is None or feature_cols is None:
        return szline
    
    print(f"  Predicting rates for {len(szline)} segments...")
    
    # One-hot encode segment attributes
    feature_dict = {}
    for cat in SHOREZONE_PREDICTION_CATS:
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
    
    # Predict
    X = np.nan_to_num(aligned.values, nan=0)
    predictions = np.maximum(model.predict(X), 0)
    
    szline = szline.copy()
    szline["predicted_rate"] = predictions
    
    print(f"    Prediction range: {predictions.min():.2f} - {predictions.max():.2f}")
    
    return szline


def get_prediction_color(rate, max_rate=10.0):
    """YlOrRd color scale for predictions."""
    norm = min(rate / max_rate, 1.0)
    
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


def categorize_sites(
    sites: pd.DataFrame,
    shorezone_bounds_wgs84: Optional[Tuple[float, float, float, float]] = None,
) -> pd.DataFrame:
    """
    Categorize sites by coverage status.
    
    With nearest-neighbor join (max distance = MAX_DISTANCE_M), categories are:
    - matched: Site has ShoreZone data (joined to nearest segment)
    - outside_shorezone_extent: Site is outside ShoreZone layer bounds (likely BC/Canada or outside WA coverage)
    - beyond_max_dist: Site is >MAX_DISTANCE_M from ShoreZone segments (no match possible)
    - no_shorezone_match: Site is within range but no ShoreZone data (coverage gap)
    """
    
    def get_category(row):
        dist = row["distance_to_shore_m"]
        has_sz = row["has_shorezone"]
        in_sz_bounds = is_within_bounds(row.get("Lat"), row.get("Long"), shorezone_bounds_wgs84)
        
        if pd.isna(dist):
            return "no_distance_calc"
        # If the site is outside the ShoreZone dataset extent, classify explicitly.
        elif in_sz_bounds is False:
            return "outside_shorezone_extent"
        elif dist > MAX_DISTANCE_M:
            return "beyond_max_dist"
        elif has_sz:
            return "matched"
        else:
            return "no_shorezone_match"  # Within range but gap in ShoreZone coverage
    
    sites["category"] = sites.apply(get_category, axis=1)
    
    return sites


def create_coverage_map(sites: pd.DataFrame, shorezone_layers=None) -> "folium.Map":
    """Create interactive map showing site coverage and ShoreZone data.
    
    Uses GeoJSON FeatureCollections for efficient rendering instead of 
    individual markers/polylines. This reduces file size and improves
    cross-browser compatibility (especially on Windows).
    """
    
    # Center on study area
    center_lat = sites["Lat"].mean()
    center_lon = sites["Long"].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles="CartoDB positron"
    )
    
    
    # Add satellite tile option
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite"
    ).add_to(m)
    
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    
    if shorezone_layers is None:
        shorezone_layers = {}

    shorezone_bounds = get_shorezone_bounds_wgs84(shorezone_layers)
    
    # Color by BC_CLASS (biotic classification)
    bc_colors = {
        "R": "#1abc9c",    # Rocky
        "S": "#f1c40f",    # Sand
        "M": "#8b4513",    # Mud
        "G": "#7f8c8d",    # Gravel
        "C": "#2c3e50",    # Cobble
        "B": "#34495e",    # Boulder
    }
    
    # Exposure class colors
    exp_colors = {
        "P": "#1a5276",    # Protected - dark blue
        "SP": "#2980b9",   # Semi-protected - blue
        "SE": "#f39c12",   # Semi-exposed - orange
        "E": "#e74c3c",    # Exposed - red
        "VE": "#8e44ad",   # Very exposed - purple
    }
    
    # Add szpoly (habitat polygons) layer using GeoJSON FeatureCollection
    if "szpoly" in shorezone_layers and len(shorezone_layers["szpoly"]) > 0:
        szpoly = shorezone_layers["szpoly"].copy()
        print(f"  Adding {len(szpoly)} habitat polygons as GeoJSON...")
        
        # Prepare properties for styling and popups
        szpoly["_bc_first"] = szpoly["BC_CLASS"].apply(
            lambda x: str(x)[:1] if pd.notna(x) else ""
        )
        
        def poly_style(feature):
            bc = feature["properties"].get("_bc_first", "")
            return {
                "fillColor": bc_colors.get(bc, "#3498db"),
                "color": "#2c3e50",
                "weight": 1,
                "fillOpacity": 0.4,
            }
        
        def poly_popup(feature):
            props = feature["properties"]
            parts = ["<b>Habitat Polygon</b>"]
            for col in ["BC_CLASS", "NRDA_CLASS", "UNIT_TYPE"]:
                if col in props and props[col] and pd.notna(props[col]):
                    parts.append(f"<b>{col}:</b> {props[col]}")
            return "<br>".join(parts)
        
        # Keep only needed columns for smaller file size
        keep_cols = ["geometry", "BC_CLASS", "NRDA_CLASS", "UNIT_TYPE", "_bc_first"]
        szpoly_slim = szpoly[[c for c in keep_cols if c in szpoly.columns]]
        
        poly_geojson = folium.GeoJson(
            szpoly_slim.__geo_interface__,
            name="Habitat Polygons (szpoly)",
            style_function=poly_style,
            popup=folium.GeoJsonPopup(fields=["BC_CLASS", "NRDA_CLASS"], aliases=["Biotic Class", "NRDA Class"]),
            tooltip=folium.GeoJsonTooltip(fields=["BC_CLASS"], aliases=["BC:"]),
            show=True
        )
        poly_geojson.add_to(m)
    
    # Add szline (shoreline segments) layer using GeoJSON FeatureCollection
    if "szline" in shorezone_layers and len(shorezone_layers["szline"]) > 0:
        szline = shorezone_layers["szline"].copy()
        print(f"  Adding {len(szline)} shoreline segments as GeoJSON...")
        
        def line_style(feature):
            exp = feature["properties"].get("EXP_CLASS", "")
            return {
                "color": exp_colors.get(exp, "#7f8c8d"),
                "weight": 3,
                "opacity": 0.8,
            }
        
        # Keep only needed columns for smaller file size
        keep_cols = ["geometry", "EXP_CLASS", "BC_CLASS", "ZONECOMP", "HAB_CALC",
                     "ZOS_UNIT", "FUC_UNIT", "ULV_UNIT", "NER_UNIT", "MAC_UNIT", "GRA_UNIT"]
        szline_slim = szline[[c for c in keep_cols if c in szline.columns]]
        
        # Fill NaN with empty string for JSON serialization
        for col in szline_slim.columns:
            if col != "geometry":
                szline_slim[col] = szline_slim[col].fillna("")
        
        line_geojson = folium.GeoJson(
            szline_slim.__geo_interface__,
            name="Shoreline (by Exposure)",
            style_function=line_style,
            popup=folium.GeoJsonPopup(
                fields=["EXP_CLASS", "BC_CLASS", "ZOS_UNIT", "NER_UNIT"],
                aliases=["Exposure", "Biotic", "Eelgrass", "Bull Kelp"]
            ),
            tooltip=folium.GeoJsonTooltip(fields=["EXP_CLASS"], aliases=["Exposure:"]),
            show=True
        )
        line_geojson.add_to(m)
        
        # Add separate vegetation layers using GeoJSON (toggle on/off)
        veg_config = {
            "ZOS_UNIT": {"name": "Eelgrass (Zostera)", "color": "#27ae60", "emoji": "🌿"},
            "NER_UNIT": {"name": "Bull Kelp (Nereocystis)", "color": "#8b4513", "emoji": "🌊"},
            "MAC_UNIT": {"name": "Giant Kelp (Macrocystis)", "color": "#d35400", "emoji": "🌊"},
            "FUC_UNIT": {"name": "Rockweed (Fucus)", "color": "#7d6608", "emoji": "🪨"},
            "ULV_UNIT": {"name": "Sea Lettuce (Ulva)", "color": "#2ecc71", "emoji": "🥬"},
            "GRA_UNIT": {"name": "Red Algae (Gracilaria)", "color": "#c0392b", "emoji": "🔴"},
        }
        
        for veg_col, config in veg_config.items():
            if veg_col not in szline.columns:
                continue
            
            # Filter to segments with this vegetation (P or C)
            veg_segments = szline[szline[veg_col].isin(["P", "C"])].copy()
            if len(veg_segments) == 0:
                continue
            
            print(f"  Adding {len(veg_segments)} segments with {config['name']} as GeoJSON...")
            
            def veg_style(feature, col=veg_col, color=config["color"]):
                code = feature["properties"].get(col, "")
                weight = 5 if code == "C" else 3
                opacity = 0.9 if code == "C" else 0.6
                return {
                    "color": color,
                    "weight": weight,
                    "opacity": opacity,
                }
            
            veg_slim = veg_segments[["geometry", veg_col]].copy()
            veg_slim[veg_col] = veg_slim[veg_col].fillna("")
            
            veg_geojson = folium.GeoJson(
                veg_slim.__geo_interface__,
                name=f"{config['emoji']} {config['name']}",
                style_function=veg_style,
                tooltip=folium.GeoJsonTooltip(fields=[veg_col], aliases=["Coverage:"]),
                show=False
            )
            veg_geojson.add_to(m)
    
    # Add predicted density layer using GeoJSON (if model predictions available)
    if "szline" in shorezone_layers and "predicted_rate" in shorezone_layers["szline"].columns:
        szline = shorezone_layers["szline"].copy()
        max_rate = min(szline["predicted_rate"].max(), 10.0)
        
        print(f"  Adding predicted density layer as GeoJSON...")
        
        def pred_style(feature, max_r=max_rate):
            rate = feature["properties"].get("predicted_rate", 0)
            return {
                "color": get_prediction_color(rate, max_r),
                "weight": 4,
                "opacity": 0.9,
            }
        
        # Round predictions to reduce JSON size
        szline["predicted_rate"] = szline["predicted_rate"].round(2)
        pred_slim = szline[["geometry", "predicted_rate"]].copy()
        
        pred_geojson = folium.GeoJson(
            pred_slim.__geo_interface__,
            name="🔥 Predicted Pycno Density",
            style_function=pred_style,
            tooltip=folium.GeoJsonTooltip(fields=["predicted_rate"], aliases=["Predicted/hr:"]),
            show=False
        )
        pred_geojson.add_to(m)
        
        # Add custom colorbar with disclaimer (hidden by default, shown when layer toggled)
        # Position on the right side, below the layer control
        colorbar_html = f'''
        <div id="prediction-colorbar" style="
            display: none;
            position: fixed;
            top: 120px;
            right: 15px;
            z-index: 1000;
            background: rgba(255,255,255,0.95);
            padding: 12px 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 11px;
            max-width: 220px;
        ">
            <div style="font-weight: bold; margin-bottom: 8px; font-size: 12px;">
                Predicted Pycno Rate
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span style="margin-right: 5px;">0</span>
                <div style="
                    flex: 1;
                    height: 12px;
                    background: linear-gradient(to right, #ffffb2, #fecc5c, #fd8d3c, #f03b20, #bd0026);
                    border-radius: 2px;
                "></div>
                <span style="margin-left: 5px;">{max_rate:.0f}/hr</span>
            </div>
            <div style="
                background: #fff3cd;
                border: 1px solid #ffc107;
                border-radius: 4px;
                padding: 8px;
                margin-top: 8px;
                color: #856404;
                font-size: 10px;
                line-height: 1.4;
            ">
                <b>⚠️ Model Limitations:</b><br>
                • CV R² = 0.08 (weak predictive power)<br>
                • Based on 24-30 yr old ShoreZone data<br>
                • ZOS_UNIT_C drives most predictions<br>
                • Use for exploration, not decision-making
            </div>
        </div>
        
        <script>
        document.addEventListener('DOMContentLoaded', function() {{
            // Find the layer control checkbox for predicted density
            var checkInterval = setInterval(function() {{
                var labels = document.querySelectorAll('.leaflet-control-layers-overlays label');
                labels.forEach(function(label) {{
                    if (label.textContent.includes('Predicted Pycno Density')) {{
                        var checkbox = label.querySelector('input');
                        if (checkbox) {{
                            checkbox.addEventListener('change', function() {{
                                var colorbar = document.getElementById('prediction-colorbar');
                                if (colorbar) {{
                                    colorbar.style.display = this.checked ? 'block' : 'none';
                                }}
                            }});
                            clearInterval(checkInterval);
                        }}
                    }}
                }});
            }}, 500);
        }});
        </script>
        '''
        m.get_root().html.add_child(folium.Element(colorbar_html))
    
    # Add szpt (observation points) layer using GeoJSON
    if "szpt" in shorezone_layers and len(shorezone_layers["szpt"]) > 0:
        szpt = shorezone_layers["szpt"].copy()
        print(f"  Adding {len(szpt)} observation points as GeoJSON...")
        
        # Keep only needed columns
        keep_cols = ["geometry", "BC_CLASS", "NRDA_CLASS", "UNIT_TYPE"]
        szpt_slim = szpt[[c for c in keep_cols if c in szpt.columns]].copy()
        for col in szpt_slim.columns:
            if col != "geometry":
                szpt_slim[col] = szpt_slim[col].fillna("")
        
        pt_geojson = folium.GeoJson(
            szpt_slim.__geo_interface__,
            name="Observation Points (szpt)",
            marker=folium.CircleMarker(radius=3, color="#2c3e50", fill=True, fill_color="#3498db", fill_opacity=0.7),
            popup=folium.GeoJsonPopup(fields=["BC_CLASS", "NRDA_CLASS"], aliases=["Biotic", "NRDA"]),
            tooltip=folium.GeoJsonTooltip(fields=["BC_CLASS"], aliases=["BC:"]),
            show=False
        )
        pt_geojson.add_to(m)
    
    # Color scheme for nearest-neighbor join
    colors = {
        "matched": "#2ecc71",              # Green - has ShoreZone data
        "no_shorezone_match": "#e74c3c",   # Red - within range but coverage gap
        "beyond_max_dist": "#9b59b6",      # Purple - beyond MAX_DISTANCE_M threshold
        "outside_shorezone_extent": "#3498db",  # Blue - outside ShoreZone dataset extent
        "no_distance_calc": "#95a5a6"      # Gray
    }
    
    labels = {
        "matched": f"ShoreZone Match (<{MAX_DISTANCE_M/1000:.0f}km)",
        "no_shorezone_match": "No Match (coverage gap)",
        "beyond_max_dist": f"Beyond {MAX_DISTANCE_M/1000:.0f}km (no match)",
        "outside_shorezone_extent": "Outside ShoreZone extent",
        "no_distance_calc": "No Distance Calculated"
    }
    
    # Create feature groups for each category
    # All hidden by default - observation rate layer looks better
    groups = {}
    for cat in colors.keys():
        groups[cat] = folium.FeatureGroup(name=labels[cat], show=False)
    
    # Create separate group for unsurveyed sites (hidden by default)
    unsurveyed_group = folium.FeatureGroup(name="✕ Unsurveyed Sites (no Pycno data)", show=False)
    
    # Add markers
    for idx, row in sites.iterrows():
        cat = row["category"]
        color = colors.get(cat, "#95a5a6")
        
        dist_str = f"{row['distance_to_shore_m']:.0f}m" if pd.notna(row['distance_to_shore_m']) else "N/A"
        has_survey = pd.notna(row.get('MeanEncounterRate'))
        rate_str = f"{row['MeanEncounterRate']:.2f}/hr" if has_survey else "NOT SURVEYED"
        pycno_count = f"{int(row['TotalPycnoCount'])}" if pd.notna(row.get('TotalPycnoCount')) else "0"
        detect_rate = f"{row['DetectionRate']*100:.0f}%" if pd.notna(row.get('DetectionRate')) else "N/A"
        
        # Build ShoreZone codes section
        sz_codes_html = ""
        if row.get("has_shorezone", False):
            sz_codes_html = "<hr style='margin: 5px 0;'><b>ShoreZone Data:</b><br>"
            for col, display_name in SHOREZONE_DISPLAY_COLS:
                if col in row.index and pd.notna(row[col]):
                    val = row[col]
                    # Format value nicely
                    if val == " " or val == "":
                        val_str = "None"
                    elif val == "P":
                        val_str = "Patchy"
                    elif val == "C":
                        val_str = "Continuous"
                    else:
                        val_str = str(val)
                    sz_codes_html += f"  <b>{display_name}:</b> {val_str}<br>"
        
        # Different styling for unsurveyed sites
        if has_survey:
            survey_html = f"""
            <b>Pycno Observations:</b><br>
            &nbsp;&nbsp;Encounter rate: <b>{rate_str}</b><br>
            &nbsp;&nbsp;Total count: {pycno_count}<br>
            &nbsp;&nbsp;Detection rate: {detect_rate}
            """
        else:
            survey_html = """
            <div style="background: #fff3cd; padding: 5px; border-radius: 3px; margin: 5px 0;">
                <b style="color: #856404;">⚠ NOT SURVEYED</b><br>
                <span style="font-size: 10px;">No Pycnopodia data available.<br>
                Not used in habitat modeling.</span>
            </div>
            """
        
        popup_html = f"""
        <div style="width: 220px; font-size: 11px;">
            <b style="font-size: 13px;">{row['SiteName']}</b><br>
            <hr style="margin: 5px 0;">
            <b>Status:</b> {labels[cat]}<br>
            <b>Distance to shore:</b> {dist_str}<br>
            <hr style="margin: 5px 0;">
            {survey_html}
            {sz_codes_html}
            <hr style="margin: 5px 0;">
            <span style="color: #666; font-size: 10px;">
                {row['Lat']:.4f}°N, {abs(row['Long']):.4f}°W
            </span>
        </div>
        """
        
        # Tooltip shows key info on hover
        if has_survey:
            tooltip_str = f"<b>{row['SiteName']}</b><br>Rate: {rate_str}"
        else:
            tooltip_str = f"<b>{row['SiteName']}</b><br><i>Not surveyed</i>"
        if row.get("has_shorezone", False) and "EXP_CLASS" in row.index and pd.notna(row.get("EXP_CLASS")):
            tooltip_str += f"<br>Exposure: {row['EXP_CLASS']}"
        
        # UNSURVEYED sites: Use black X marker and add to separate group
        if not has_survey:
            # Create a black X marker using DivIcon
            x_icon = folium.DivIcon(
                html='''<div style="
                    font-size: 18px; 
                    font-weight: bold; 
                    color: #000000; 
                    text-shadow: 1px 1px 1px white, -1px -1px 1px white, 1px -1px 1px white, -1px 1px 1px white;
                ">✕</div>''',
                icon_size=(20, 20),
                icon_anchor=(10, 10)
            )
            marker = folium.Marker(
                location=[row["Lat"], row["Long"]],
                icon=x_icon,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=folium.Tooltip(tooltip_str)
            )
            marker.add_to(unsurveyed_group)
        else:
            # SURVEYED sites: Use circle marker (small fixed size)
            radius = 5
            
            marker = folium.CircleMarker(
                location=[row["Lat"], row["Long"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=folium.Tooltip(tooltip_str)
            )
            marker.add_to(groups[cat])
    # Add groups to map
    for group in groups.values():
        group.add_to(m)
    
    # Add unsurveyed sites group (toggleable)
    unsurveyed_group.add_to(m)
    
    # Add observation rate layer (sized and colored by measured encounter rate)
    # This is the default view - shows surveyed sites by their measured rates
    rate_group = folium.FeatureGroup(name="📊 Sites by Observation Rate", show=True)
    
    # Get max rate for normalization
    surveyed_sites = sites[sites["MeanEncounterRate"].notna()]
    if len(surveyed_sites) > 0:
        max_observed_rate = surveyed_sites["MeanEncounterRate"].max()
        max_observed_rate = max(max_observed_rate, 1.0)  # Avoid division by zero
        
        for idx, row in surveyed_sites.iterrows():
            rate = row["MeanEncounterRate"]
            
            # Size: scale radius from 3 (rate=0) to 15 (max rate)
            norm_rate = rate / max_observed_rate
            radius = 3 + norm_rate * 12
            
            # Color: YlOrRd gradient based on rate
            if norm_rate < 0.2:
                color = "#ffffb2"  # light yellow
            elif norm_rate < 0.4:
                color = "#fecc5c"  # yellow-orange
            elif norm_rate < 0.6:
                color = "#fd8d3c"  # orange
            elif norm_rate < 0.8:
                color = "#f03b20"  # red-orange
            else:
                color = "#bd0026"  # dark red
            
            # Build full popup with ShoreZone info
            cat = row.get("category", "unknown")
            dist_str = f"{row['distance_to_shore_m']:.0f}m" if pd.notna(row.get('distance_to_shore_m')) else "N/A"
            pycno_count = f"{int(row['TotalPycnoCount'])}" if pd.notna(row.get('TotalPycnoCount')) else "0"
            detect_rate = f"{row['DetectionRate']*100:.0f}%" if pd.notna(row.get('DetectionRate')) else "N/A"
            
            # ShoreZone codes section
            sz_codes_html = ""
            if row.get("has_shorezone", False):
                sz_codes_html = "<hr style='margin: 5px 0;'><b>ShoreZone Data:</b><br>"
                for col, display_name in SHOREZONE_DISPLAY_COLS:
                    if col in row.index and pd.notna(row[col]):
                        val = row[col]
                        if val == " " or val == "":
                            val_str = "None"
                        elif val == "P":
                            val_str = "Patchy"
                        elif val == "C":
                            val_str = "Continuous"
                        else:
                            val_str = str(val)
                        sz_codes_html += f"  <b>{display_name}:</b> {val_str}<br>"
            
            popup_html = f"""
            <div style="width: 220px; font-size: 11px;">
                <b style="font-size: 13px;">{row['SiteName']}</b><br>
                <hr style="margin: 5px 0;">
                <b>Distance to shore:</b> {dist_str}<br>
                <hr style="margin: 5px 0;">
                <b>Pycno Observations:</b><br>
                &nbsp;&nbsp;Encounter rate: <b>{rate:.2f}/hr</b><br>
                &nbsp;&nbsp;Total count: {pycno_count}<br>
                &nbsp;&nbsp;Detection rate: {detect_rate}
                {sz_codes_html}
                <hr style="margin: 5px 0;">
                <span style="color: #666; font-size: 10px;">
                    {row['Lat']:.4f}°N, {abs(row['Long']):.4f}°W
                </span>
            </div>
            """
            
            # Tooltip with dive site habitat info
            tooltip_str = f"<b>{row['SiteName']}</b><br>Rate: {rate:.2f}/hr"
            if row.get("has_shorezone", False) and "EXP_CLASS" in row.index and pd.notna(row.get("EXP_CLASS")):
                tooltip_str += f"<br>Exposure: {row['EXP_CLASS']}"
            
            # Add dive survey habitat breakdown
            hab_cols = [c for c in row.index if c.startswith("hab_")]
            if hab_cols:
                hab_parts = []
                for col in sorted(hab_cols):
                    prop = row[col]
                    if pd.notna(prop) and prop > 0:
                        hab_name = col.replace("hab_", "").replace("_", " ")
                        hab_parts.append(f"{hab_name}: {prop*100:.0f}%")
                if hab_parts:
                    tooltip_str += "<br><b>Dive Habitats:</b><br>" + "<br>".join(hab_parts)
            
            # Use DivIcon for proper z-index control (always on top)
            size = int(radius * 2)
            icon_html = f'''
                <div style="
                    width: {size}px; 
                    height: {size}px; 
                    background-color: {color}; 
                    border: 1.5px solid #000; 
                    border-radius: 50%; 
                    opacity: 0.9;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                "></div>
            '''
            icon = folium.DivIcon(
                html=icon_html,
                icon_size=(size, size),
                icon_anchor=(size//2, size//2)
            )
            folium.Marker(
                location=[row["Lat"], row["Long"]],
                icon=icon,
                popup=folium.Popup(popup_html, max_width=280),
                tooltip=folium.Tooltip(tooltip_str)
            ).add_to(rate_group)
    
    # Add ShoreZone bounds (derived from loaded dataset, not hard-coded)
    if shorezone_bounds is not None:
        minx, miny, maxx, maxy = shorezone_bounds
        bounds_poly = [
            [miny, minx],  # SW (lat, lon)
            [miny, maxx],  # SE
            [maxy, maxx],  # NE
            [maxy, minx],  # NW
            [miny, minx],  # Close polygon
        ]
        folium.PolyLine(
            locations=bounds_poly,
            color="#e67e22",
            weight=3,
            dash_array="10, 10",
            popup="ShoreZone dataset bounds (derived)",
            tooltip="ShoreZone dataset extent"
        ).add_to(m)
    
    # Add rate_group LAST so observation circles are always on top of all other layers
    rate_group.add_to(m)
    
    # Legend
    legend_html = f"""
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background-color: white; padding: 15px; border-radius: 8px;
                border: 2px solid #666; font-size: 11px; font-family: Arial, sans-serif;
                box-shadow: 2px 2px 6px rgba(0,0,0,0.3); max-width: 360px; max-height: 85vh; overflow-y: auto;">
        <div style="font-weight: bold; margin-bottom: 8px; font-size: 13px;">
            Site Coverage Status
        </div>
        <div style="font-size: 10px; color: #666; margin-bottom: 6px;">
            Join method: Nearest segment within {MAX_DISTANCE_M/1000:.0f}km
        </div>
        <div style="margin-bottom: 4px;">
            <span style="display: inline-block; width: 12px; height: 12px; 
                         background: #2ecc71; border-radius: 50%; margin-right: 6px;"></span>
            ShoreZone Match (&lt;{MAX_DISTANCE_M/1000:.0f}km)
        </div>
        <div style="margin-bottom: 4px;">
            <span style="display: inline-block; width: 12px; height: 12px; 
                         background: #e74c3c; border-radius: 50%; margin-right: 6px;"></span>
            No Match (coverage gap)
        </div>
        <div style="margin-bottom: 4px;">
            <span style="display: inline-block; width: 12px; height: 12px; 
                         background: #3498db; border-radius: 50%; margin-right: 6px;"></span>
            Outside ShoreZone extent
        </div>
        <div style="margin-bottom: 4px;">
            <span style="display: inline-block; width: 12px; height: 12px; 
                         background: #9b59b6; border-radius: 50%; margin-right: 6px;"></span>
            Beyond {MAX_DISTANCE_M/1000:.0f}km (no match)
        </div>
        <hr style="margin: 6px 0;">
        <div style="margin-bottom: 4px;">
            <span style="display: inline-block; width: 14px; font-weight: bold; 
                         color: #000; margin-right: 4px;">✕</span>
            Not surveyed for Pycno
        </div>
        <hr style="margin: 6px 0;">
        <div style="font-weight: bold; margin-bottom: 4px;">Wave Exposure</div>
        <div style="margin-bottom: 3px; padding-left: 8px;">
            <span style="display: inline-block; width: 18px; height: 3px; 
                         background: #1a5276; margin-right: 6px;"></span>
            Protected (P)
        </div>
        <div style="margin-bottom: 3px; padding-left: 8px;">
            <span style="display: inline-block; width: 18px; height: 3px; 
                         background: #2980b9; margin-right: 6px;"></span>
            Semi-Protected (SP)
        </div>
        <div style="margin-bottom: 3px; padding-left: 8px;">
            <span style="display: inline-block; width: 18px; height: 3px; 
                         background: #f39c12; margin-right: 6px;"></span>
            Semi-Exposed (SE)
        </div>
        <div style="margin-bottom: 3px; padding-left: 8px;">
            <span style="display: inline-block; width: 18px; height: 3px; 
                         background: #e74c3c; margin-right: 6px;"></span>
            Exposed (E)
        </div>
        <hr style="margin: 6px 0;">
        <div style="font-weight: bold; margin-bottom: 4px;">🌿 Vegetation Layers</div>
        <div style="font-size: 10px; color: #666; margin-bottom: 4px;">
            Toggle in layer control (top right)
        </div>
        <div style="margin-bottom: 3px; padding-left: 8px;">
            <span style="display: inline-block; width: 18px; height: 4px; 
                         background: #27ae60; margin-right: 6px;"></span>
            Eelgrass (Zostera)
        </div>
        <div style="margin-bottom: 3px; padding-left: 8px;">
            <span style="display: inline-block; width: 18px; height: 4px; 
                         background: #8b4513; margin-right: 6px;"></span>
            Bull Kelp (Nereocystis)
        </div>
        <div style="margin-bottom: 3px; padding-left: 8px;">
            <span style="display: inline-block; width: 18px; height: 4px; 
                         background: #d35400; margin-right: 6px;"></span>
            Giant Kelp (Macrocystis)
        </div>
        <div style="margin-bottom: 3px; padding-left: 8px;">
            <span style="display: inline-block; width: 18px; height: 4px; 
                         background: #7d6608; margin-right: 6px;"></span>
            Rockweed (Fucus)
        </div>
        <div style="margin-bottom: 3px; padding-left: 8px;">
            <span style="display: inline-block; width: 18px; height: 4px; 
                         background: #2ecc71; margin-right: 6px;"></span>
            Sea Lettuce (Ulva)
        </div>
        <div style="margin-bottom: 3px; padding-left: 8px;">
            <span style="display: inline-block; width: 18px; height: 4px; 
                         background: #c0392b; margin-right: 6px;"></span>
            Red Algae (Gracilaria)
        </div>
        <hr style="margin: 6px 0;">
        <div style="font-weight: bold; margin-bottom: 4px;">📊 Observation Rate</div>
        <div style="font-size: 10px; color: #666; margin-bottom: 4px;">
            Measured Pycno encounter rate (toggle in layers)
        </div>
        <div style="margin-bottom: 3px; padding-left: 8px;">
            <span style="display: inline-block; width: 6px; height: 6px; 
                         background: #ffffb2; border-radius: 50%; margin-right: 3px;"></span>
            <span style="display: inline-block; width: 10px; height: 10px; 
                         background: #fd8d3c; border-radius: 50%; margin-right: 3px;"></span>
            <span style="display: inline-block; width: 14px; height: 14px; 
                         background: #bd0026; border-radius: 50%; margin-right: 3px;"></span>
            Size + color = rate
        </div>
        <hr style="margin: 6px 0;">
        <div style="font-weight: bold; margin-bottom: 4px;">🔥 Predicted Density</div>
        <div style="font-size: 10px; color: #666; margin-bottom: 4px;">
            Model-predicted Pycno encounter rate
        </div>
        <div style="margin-bottom: 3px; padding-left: 8px;">
            <span style="display: inline-block; width: 18px; height: 4px; 
                         background: linear-gradient(to right, #ffffb2, #fecc5c, #fd8d3c, #f03b20, #bd0026); margin-right: 6px;"></span>
            Low → High
        </div>
        <hr style="margin: 6px 0;">
        <div style="font-size: 10px; color: #666;">
            <b>Vegetation:</b> C = Continuous (thick), P = Patchy (thin)<br>
            <b>Join:</b> Nearest shoreline within {MAX_DISTANCE_M/1000:.0f}km<br>
            Click features for full habitat info
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add results panel (always). Content comes from results.md if present,
    # otherwise a CSV-derived fallback summary is used.
    results_panel = create_results_panel_html(load_results_html())
    m.get_root().html.add_child(folium.Element(results_panel))
    print("  Added results panel (analysis summary)")
    
    # Add controls
    folium.LayerControl().add_to(m)
    plugins.Fullscreen().add_to(m)
    plugins.MeasureControl(position='topright').add_to(m)
    
    return m


def main():
    """Generate site coverage diagnostic map."""
    
    print("=" * 60)
    print("SITE COVERAGE DIAGNOSTIC MAP")
    print("=" * 60)
    
    if not HAS_FOLIUM:
        print("ERROR: folium not installed")
        return
    
    # Load data
    print("\nLoading data...")
    sites = load_data()
    print(f"  Total sites: {len(sites)}")
    
    # Load ShoreZone (100% to see full coverage)
    shorezone = load_shorezone(sample_fraction=1.0)
    
    # Load prediction model and add predictions to szline
    print("\nLoading prediction model...")
    model, feature_cols = load_prediction_model()
    if model is not None and "szline" in shorezone:
        shorezone["szline"] = predict_segment_rates(shorezone["szline"], model, feature_cols)
    
    # Categorize
    print("\nCategorizing sites...")
    shorezone_bounds = get_shorezone_bounds_wgs84(shorezone)
    sites = categorize_sites(sites, shorezone_bounds_wgs84=shorezone_bounds)
    
    # Summary
    print("\nCategory breakdown:")
    for cat, count in sites["category"].value_counts().items():
        pct = 100 * count / len(sites)
        print(f"  {cat}: {count} ({pct:.1f}%)")
    
    # Create map
    print("\nCreating map...")
    m = create_coverage_map(sites, shorezone_layers=shorezone)
    
    # Save
    output_path = OUTPUT_DIR / "site_coverage_diagnostic_map.html"
    m.save(str(output_path))
    print(f"\nSaved: {output_path}")
    
    # Also save the categorized data
    sites_path = OUTPUT_DIR / "site_coverage_categories.csv"
    sites.to_csv(sites_path, index=False)
    print(f"Saved: {sites_path}")
    
    # Print problem sites
    print("\n" + "=" * 60)
    print("SITES WITHOUT SHOREZONE COVERAGE")
    print("=" * 60)
    
    problem_sites = sites[sites["category"].isin(["outside_shorezone_extent", "beyond_max_dist", "no_shorezone_match"])]
    problem_sites = problem_sites.sort_values("distance_to_shore_m", ascending=False)
    
    print(f"\n{len(problem_sites)} sites excluded from ShoreZone analysis:\n")
    
    for idx, row in problem_sites.head(20).iterrows():
        dist_km = row["distance_to_shore_m"] / 1000 if pd.notna(row["distance_to_shore_m"]) else 0
        print(f"  {row['SiteName']}: {dist_km:.1f} km from shore ({row['category']})")
    
    if len(problem_sites) > 20:
        print(f"  ... and {len(problem_sites) - 20} more")


if __name__ == "__main__":
    main()

