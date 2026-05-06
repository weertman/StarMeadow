"""
Static Map Visualizations of Pycnopodia Survey Data

Creates matplotlib-based static maps showing:
1. Survey site distribution
2. Survey effort by site (transects, hours)
3. Pycnopodia encounter rates by site
4. Detection probability by site
5. Eelgrass site distribution
6. Basin-level summaries

Outputs:
- PNG and PDF map figures
- Site-level summary CSV with coordinates
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import seaborn as sns

from utils import get_output_dir, load_data, set_style, save_figure, DATA_DIR

OUTPUT_DIR = get_output_dir(__file__)


def load_site_coordinates() -> pd.DataFrame:
    """Load site latitude/longitude data."""
    coords = pd.read_csv(DATA_DIR / "Site_LatLong.csv")
    coords.columns = ['SiteName', 'Lat', 'Long']
    
    # Fix any longitude sign issues (should be negative for Western Hemisphere)
    coords.loc[coords['Long'] > 0, 'Long'] = -coords.loc[coords['Long'] > 0, 'Long']
    
    return coords


def prepare_site_summary(df: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """
    Create site-level summary with coordinates and survey metrics.
    """
    # Aggregate survey data by site
    site_summary = df.groupby('SiteName').agg({
        'Pycnopodia_count': ['sum', 'mean'],
        'Encounter.Rate.Hr': 'mean',
        'Survey.Time': 'sum',
        'Date': ['nunique', 'min', 'max'],
        'HabitatType': lambda x: (x == 'Eelgrass').any(),
        'Basin': 'first'
    }).reset_index()
    
    # Flatten column names
    site_summary.columns = [
        'SiteName', 'TotalPycno', 'MeanCount', 'MeanEncounterRate',
        'TotalSurveyTime', 'NSurveyDates', 'FirstSurvey', 'LastSurvey',
        'HasEelgrass', 'Basin'
    ]
    
    # Add transect count
    transect_counts = df.groupby('SiteName').size().reset_index(name='NTransects')
    site_summary = site_summary.merge(transect_counts, on='SiteName')
    
    # Calculate detection rate
    detection = df.groupby('SiteName')['Pycnopodia_count'].apply(lambda x: (x > 0).mean())
    site_summary['DetectionRate'] = site_summary['SiteName'].map(detection)
    
    # Calculate survey hours
    site_summary['SurveyHours'] = site_summary['TotalSurveyTime'] / 60
    
    # Merge with coordinates
    site_summary = site_summary.merge(coords, on='SiteName', how='left')
    
    # Report missing coordinates
    missing = site_summary[site_summary['Lat'].isna()]['SiteName'].tolist()
    if missing:
        print(f"  Warning: {len(missing)} sites missing coordinates")
    
    return site_summary


def plot_site_distribution(site_summary: pd.DataFrame):
    """
    Basic map showing all survey site locations.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Filter to sites with coordinates
    df_plot = site_summary.dropna(subset=['Lat', 'Long'])
    
    # Plot all sites
    ax.scatter(df_plot['Long'], df_plot['Lat'], 
               s=50, c='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Customize
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Pycnopodia Survey Sites (n={len(df_plot)})')
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, OUTPUT_DIR, 'map_site_distribution')
    plt.close()


def plot_survey_effort(site_summary: pd.DataFrame):
    """
    Map showing survey effort by site (bubble size = transects, color = hours).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    df_plot = site_summary.dropna(subset=['Lat', 'Long'])
    
    # Left: Bubble size by number of transects
    ax1 = axes[0]
    sizes = df_plot['NTransects'] * 3  # Scale for visibility
    scatter1 = ax1.scatter(df_plot['Long'], df_plot['Lat'], 
                           s=sizes, c='steelblue', alpha=0.6, 
                           edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Survey Effort: Bubble Size = N Transects')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Add legend for sizes
    for n in [5, 25, 100]:
        ax1.scatter([], [], s=n*3, c='steelblue', alpha=0.6, 
                   edgecolor='black', label=f'{n} transects')
    ax1.legend(loc='lower left', title='Transects')
    
    # Right: Color by survey hours
    ax2 = axes[1]
    hours = df_plot['SurveyHours'].fillna(0)
    scatter2 = ax2.scatter(df_plot['Long'], df_plot['Lat'],
                           s=60, c=hours, cmap='YlOrRd',
                           alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title('Survey Effort: Color = Total Survey Hours')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter2, ax=ax2, shrink=0.7)
    cbar.set_label('Survey Hours')
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, 'map_survey_effort')
    plt.close()


def plot_encounter_rate_map(site_summary: pd.DataFrame):
    """
    Map showing Pycnopodia encounter rates by site.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    df_plot = site_summary.dropna(subset=['Lat', 'Long'])
    
    # Left: Color by mean encounter rate
    ax1 = axes[0]
    rates = df_plot['MeanEncounterRate'].fillna(0)
    
    # Use log scale for better visualization (many zeros/low values)
    rates_log = np.log10(rates + 0.01)  # Add small constant to handle zeros
    
    scatter1 = ax1.scatter(df_plot['Long'], df_plot['Lat'],
                           s=80, c=rates_log, cmap='RdYlGn',
                           alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Mean Encounter Rate (Pycno/hr)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter1, ax=ax1, shrink=0.7)
    cbar.set_label('log₁₀(Rate + 0.01)')
    
    # Right: Bubble size by total Pycnopodia observed
    ax2 = axes[1]
    total = df_plot['TotalPycno'].fillna(0)
    sizes = np.sqrt(total + 1) * 10  # Square root scale for better visibility
    
    # Color by detection rate
    detection = df_plot['DetectionRate'].fillna(0)
    
    scatter2 = ax2.scatter(df_plot['Long'], df_plot['Lat'],
                           s=sizes, c=detection, cmap='RdYlGn',
                           alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title('Total Pycnopodia (size) & Detection Rate (color)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter2, ax=ax2, shrink=0.7)
    cbar.set_label('Detection Rate')
    
    # Add legend for sizes
    for n in [1, 10, 100]:
        ax2.scatter([], [], s=np.sqrt(n+1)*10, c='gray', alpha=0.7,
                   edgecolor='black', label=f'{n} individuals')
    ax2.legend(loc='lower left', title='Total Pycno')
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, 'map_encounter_rates')
    plt.close()


def plot_eelgrass_site_map(site_summary: pd.DataFrame):
    """
    Map highlighting eelgrass vs non-eelgrass sites.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    df_plot = site_summary.dropna(subset=['Lat', 'Long'])
    
    # Split by eelgrass presence
    eelgrass = df_plot[df_plot['HasEelgrass'] == True]
    no_eelgrass = df_plot[df_plot['HasEelgrass'] == False]
    
    # Plot non-eelgrass sites first (background)
    ax.scatter(no_eelgrass['Long'], no_eelgrass['Lat'],
               s=60, c='#e74c3c', alpha=0.6, edgecolor='black', linewidth=0.5,
               label=f'Non-Eelgrass Sites (n={len(no_eelgrass)})')
    
    # Plot eelgrass sites on top
    ax.scatter(eelgrass['Long'], eelgrass['Lat'],
               s=80, c='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5,
               label=f'Eelgrass Sites (n={len(eelgrass)})')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Sites by Eelgrass Presence')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')
    
    save_figure(fig, OUTPUT_DIR, 'map_eelgrass_sites')
    plt.close()


def plot_basin_map(site_summary: pd.DataFrame):
    """
    Map with sites colored by basin.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    df_plot = site_summary.dropna(subset=['Lat', 'Long', 'Basin'])
    
    # Get unique basins and assign colors
    basins = df_plot['Basin'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(basins)))
    basin_colors = dict(zip(basins, colors))
    
    # Plot each basin
    for basin in basins:
        basin_data = df_plot[df_plot['Basin'] == basin]
        ax.scatter(basin_data['Long'], basin_data['Lat'],
                   s=70, c=[basin_colors[basin]], alpha=0.7,
                   edgecolor='black', linewidth=0.5,
                   label=f'{basin} (n={len(basin_data)})')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Sites by Geographic Basin')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=8)
    
    save_figure(fig, OUTPUT_DIR, 'map_basins')
    plt.close()


def plot_hotspot_map(site_summary: pd.DataFrame):
    """
    Map highlighting Pycnopodia hotspots (top encounter rate sites).
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    df_plot = site_summary.dropna(subset=['Lat', 'Long']).copy()
    
    # Classify sites
    rate_threshold = df_plot['MeanEncounterRate'].quantile(0.9)  # Top 10%
    df_plot['IsHotspot'] = df_plot['MeanEncounterRate'] >= rate_threshold
    
    # Plot non-hotspots
    non_hotspots = df_plot[~df_plot['IsHotspot']]
    ax.scatter(non_hotspots['Long'], non_hotspots['Lat'],
               s=40, c='lightgray', alpha=0.5, edgecolor='gray', linewidth=0.3,
               label=f'Other Sites (n={len(non_hotspots)})')
    
    # Plot hotspots with size by encounter rate
    hotspots = df_plot[df_plot['IsHotspot']]
    sizes = hotspots['MeanEncounterRate'] * 5
    ax.scatter(hotspots['Long'], hotspots['Lat'],
               s=sizes, c='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1,
               label=f'Hotspots (n={len(hotspots)}, ≥{rate_threshold:.1f}/hr)')
    
    # Add labels for top sites
    top_sites = hotspots.nlargest(5, 'MeanEncounterRate')
    for _, row in top_sites.iterrows():
        ax.annotate(row['SiteName'][:20], (row['Long'], row['Lat']),
                   fontsize=8, alpha=0.8,
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Pycnopodia Hotspots (Top 10% Encounter Rate Sites)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left')
    
    save_figure(fig, OUTPUT_DIR, 'map_hotspots')
    plt.close()


def plot_combined_summary_map(site_summary: pd.DataFrame):
    """
    Four-panel summary map showing multiple variables.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    df_plot = site_summary.dropna(subset=['Lat', 'Long'])
    
    # Panel 1: Site distribution by basin
    ax1 = axes[0, 0]
    basins = df_plot['Basin'].dropna().unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(basins)))
    basin_colors = dict(zip(basins, colors))
    
    for basin in basins:
        basin_data = df_plot[df_plot['Basin'] == basin]
        ax1.scatter(basin_data['Long'], basin_data['Lat'],
                   s=50, c=[basin_colors[basin]], alpha=0.7,
                   edgecolor='black', linewidth=0.3)
    ax1.set_title('A) Sites by Basin')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Eelgrass presence
    ax2 = axes[0, 1]
    eelgrass_colors = df_plot['HasEelgrass'].map({True: '#2ecc71', False: '#e74c3c'})
    ax2.scatter(df_plot['Long'], df_plot['Lat'],
               s=60, c=eelgrass_colors, alpha=0.7,
               edgecolor='black', linewidth=0.3)
    ax2.set_title('B) Eelgrass Presence (Green = Yes)')
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Survey effort
    ax3 = axes[1, 0]
    sizes = df_plot['NTransects'] * 2
    ax3.scatter(df_plot['Long'], df_plot['Lat'],
               s=sizes, c='steelblue', alpha=0.6,
               edgecolor='black', linewidth=0.3)
    ax3.set_title('C) Survey Effort (size = transects)')
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Latitude')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Encounter rate
    ax4 = axes[1, 1]
    rates = df_plot['MeanEncounterRate'].fillna(0)
    rates_log = np.log10(rates + 0.01)
    scatter = ax4.scatter(df_plot['Long'], df_plot['Lat'],
                         s=60, c=rates_log, cmap='RdYlGn',
                         alpha=0.8, edgecolor='black', linewidth=0.3)
    ax4.set_title('D) Mean Encounter Rate')
    ax4.set_xlabel('Longitude')
    ax4.set_ylabel('Latitude')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax4, shrink=0.7)
    cbar.set_label('log₁₀(Rate + 0.01)')
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, 'map_combined_summary')
    plt.close()


def main():
    print(f"\n{'='*60}")
    print("STATIC MAP VISUALIZATIONS")
    print(f"{'='*60}\n")
    
    set_style()
    
    # Load data
    print("Loading data...")
    df = load_data()
    coords = load_site_coordinates()
    
    print(f"  Survey records: {len(df)}")
    print(f"  Sites with coordinates: {len(coords)}")
    
    # Prepare site summary
    print("\nPreparing site-level summary...")
    site_summary = prepare_site_summary(df, coords)
    
    sites_with_coords = site_summary.dropna(subset=['Lat', 'Long'])
    print(f"  Sites matched with coordinates: {len(sites_with_coords)}")
    
    # Save site summary
    site_summary.to_csv(OUTPUT_DIR / 'site_summary_with_coords.csv', index=False)
    print(f"  Saved: {OUTPUT_DIR / 'site_summary_with_coords.csv'}")
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Generate maps
    print("\n" + "="*60)
    print("GENERATING STATIC MAPS")
    print("="*60)
    
    print("\n1. Site distribution map...")
    plot_site_distribution(site_summary)
    
    print("2. Survey effort maps...")
    plot_survey_effort(site_summary)
    
    print("3. Encounter rate maps...")
    plot_encounter_rate_map(site_summary)
    
    print("4. Eelgrass site map...")
    plot_eelgrass_site_map(site_summary)
    
    print("5. Basin map...")
    plot_basin_map(site_summary)
    
    print("6. Hotspot map...")
    plot_hotspot_map(site_summary)
    
    print("7. Combined summary map...")
    plot_combined_summary_map(site_summary)
    
    # Summary statistics
    print("\n" + "="*60)
    print("MAP SUMMARY")
    print("="*60)
    
    print(f"""
Sites mapped: {len(sites_with_coords)}
Geographic extent:
  Latitude:  {sites_with_coords['Lat'].min():.3f}° to {sites_with_coords['Lat'].max():.3f}°
  Longitude: {sites_with_coords['Long'].min():.3f}° to {sites_with_coords['Long'].max():.3f}°

Eelgrass sites: {sites_with_coords['HasEelgrass'].sum()}
Basins represented: {sites_with_coords['Basin'].nunique()}

Top 5 sites by encounter rate:
""")
    
    top_sites = sites_with_coords.nlargest(5, 'MeanEncounterRate')[
        ['SiteName', 'MeanEncounterRate', 'TotalPycno', 'Basin']
    ]
    print(top_sites.to_string(index=False))
    
    print(f"\n{'='*60}")
    print("STATIC MAPS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()






