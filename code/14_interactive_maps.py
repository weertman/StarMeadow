"""
Interactive Map Visualizations of Pycnopodia Survey Data

Creates folium-based interactive HTML maps showing:
1. Survey site distribution with popups
2. Encounter rate heatmap
3. Eelgrass site highlighting
4. Basin-colored sites
5. Time-series animation data

Outputs:
- Interactive HTML map files
- Can be opened in any web browser

Requires: folium (pip install folium)
"""

import pandas as pd
import numpy as np
import warnings

# Check for folium availability
try:
    import folium
    from folium.plugins import MarkerCluster, HeatMap
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False
    warnings.warn("folium not installed. Run: pip install folium")

from utils import get_output_dir, load_data, DATA_DIR

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
    
    # Get habitat list for each site
    habitats = df.groupby('SiteName')['HabitatType'].apply(lambda x: ', '.join(sorted(x.unique())))
    site_summary['Habitats'] = site_summary['SiteName'].map(habitats)
    
    # Merge with coordinates
    site_summary = site_summary.merge(coords, on='SiteName', how='left')
    
    return site_summary


def get_color_by_rate(rate, max_rate):
    """Return color based on encounter rate (green to red gradient)."""
    if pd.isna(rate) or rate == 0:
        return '#808080'  # Gray for zeros
    
    # Normalize to 0-1
    norm_rate = min(rate / max_rate, 1.0)
    
    # Color gradient: green (low) -> yellow -> red (high)
    if norm_rate < 0.5:
        # Green to yellow
        r = int(255 * (norm_rate * 2))
        g = 200
        b = 50
    else:
        # Yellow to red
        r = 255
        g = int(200 * (1 - (norm_rate - 0.5) * 2))
        b = 50
    
    return f'#{r:02x}{g:02x}{b:02x}'


def create_basic_site_map(site_summary: pd.DataFrame) -> folium.Map:
    """
    Create basic interactive map with all survey sites.
    Click markers for site details.
    """
    df_plot = site_summary.dropna(subset=['Lat', 'Long'])
    
    # Center map on data
    center_lat = df_plot['Lat'].mean()
    center_lon = df_plot['Long'].mean()
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='OpenStreetMap'
    )
    
    # Add markers for each site
    for _, row in df_plot.iterrows():
        popup_html = f"""
        <b>{row['SiteName']}</b><br>
        <hr>
        Basin: {row['Basin']}<br>
        Habitats: {row['Habitats']}<br>
        Eelgrass: {'Yes' if row['HasEelgrass'] else 'No'}<br>
        <hr>
        Transects: {row['NTransects']}<br>
        Survey Hours: {row['SurveyHours']:.1f}<br>
        Survey Dates: {row['NSurveyDates']}<br>
        <hr>
        Total Pycno: {int(row['TotalPycno'])}<br>
        Mean Rate: {row['MeanEncounterRate']:.2f}/hr<br>
        Detection Rate: {row['DetectionRate']:.1%}
        """
        
        folium.CircleMarker(
            location=[row['Lat'], row['Long']],
            radius=6,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=row['SiteName'],
            color='steelblue',
            fill=True,
            fill_color='steelblue',
            fill_opacity=0.7
        ).add_to(m)
    
    # Add title
    title_html = '''
    <div style="position: fixed; top: 10px; left: 50px; z-index: 9999; 
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray;">
        <h4 style="margin: 0;">Pycnopodia Survey Sites</h4>
        <p style="margin: 0; font-size: 12px;">Click markers for details</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m


def create_encounter_rate_map(site_summary: pd.DataFrame) -> folium.Map:
    """
    Create map with markers colored by encounter rate.
    """
    df_plot = site_summary.dropna(subset=['Lat', 'Long'])
    
    center_lat = df_plot['Lat'].mean()
    center_lon = df_plot['Long'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='CartoDB positron'
    )
    
    max_rate = df_plot['MeanEncounterRate'].max()
    
    for _, row in df_plot.iterrows():
        rate = row['MeanEncounterRate']
        color = get_color_by_rate(rate, max_rate)
        
        # Size by total Pycno observed
        radius = max(4, min(20, np.sqrt(row['TotalPycno'] + 1) * 2))
        
        popup_html = f"""
        <b>{row['SiteName']}</b><br>
        <hr>
        Mean Rate: <b>{rate:.2f}/hr</b><br>
        Total Pycno: {int(row['TotalPycno'])}<br>
        Detection Rate: {row['DetectionRate']:.1%}<br>
        <hr>
        Transects: {row['NTransects']}<br>
        Basin: {row['Basin']}
        """
        
        folium.CircleMarker(
            location=[row['Lat'], row['Long']],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['SiteName']}: {rate:.2f}/hr",
            color='black',
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.8
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; z-index: 9999;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray;">
        <h4 style="margin: 0 0 5px 0;">Encounter Rate</h4>
        <p style="margin: 2px;"><span style="background-color: #808080; padding: 2px 10px;">  </span> Zero</p>
        <p style="margin: 2px;"><span style="background-color: #00c832; padding: 2px 10px;">  </span> Low</p>
        <p style="margin: 2px;"><span style="background-color: #ffc800; padding: 2px 10px;">  </span> Medium</p>
        <p style="margin: 2px;"><span style="background-color: #ff3232; padding: 2px 10px;">  </span> High</p>
        <p style="margin: 5px 0 0 0; font-size: 10px;">Size = Total Pycno</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    title_html = '''
    <div style="position: fixed; top: 10px; left: 50px; z-index: 9999;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray;">
        <h4 style="margin: 0;">Pycnopodia Encounter Rates</h4>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m


def create_eelgrass_map(site_summary: pd.DataFrame) -> folium.Map:
    """
    Create map highlighting eelgrass vs non-eelgrass sites.
    """
    df_plot = site_summary.dropna(subset=['Lat', 'Long'])
    
    center_lat = df_plot['Lat'].mean()
    center_lon = df_plot['Long'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='CartoDB positron'
    )
    
    # Create feature groups for layer control
    eelgrass_group = folium.FeatureGroup(name='Eelgrass Sites')
    no_eelgrass_group = folium.FeatureGroup(name='Non-Eelgrass Sites')
    
    for _, row in df_plot.iterrows():
        has_eelgrass = row['HasEelgrass']
        color = '#2ecc71' if has_eelgrass else '#e74c3c'
        group = eelgrass_group if has_eelgrass else no_eelgrass_group
        
        popup_html = f"""
        <b>{row['SiteName']}</b><br>
        <hr>
        Eelgrass: <b>{'Yes' if has_eelgrass else 'No'}</b><br>
        Habitats: {row['Habitats']}<br>
        <hr>
        Mean Rate: {row['MeanEncounterRate']:.2f}/hr<br>
        Total Pycno: {int(row['TotalPycno'])}<br>
        Detection Rate: {row['DetectionRate']:.1%}
        """
        
        folium.CircleMarker(
            location=[row['Lat'], row['Long']],
            radius=8 if has_eelgrass else 6,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['SiteName']} ({'Eelgrass' if has_eelgrass else 'No Eelgrass'})",
            color='black',
            weight=1,
            fill=True,
            fill_color=color,
            fill_opacity=0.8
        ).add_to(group)
    
    eelgrass_group.add_to(m)
    no_eelgrass_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; z-index: 9999;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray;">
        <h4 style="margin: 0 0 5px 0;">Site Type</h4>
        <p style="margin: 2px;"><span style="background-color: #2ecc71; padding: 2px 10px;">  </span> Eelgrass Site</p>
        <p style="margin: 2px;"><span style="background-color: #e74c3c; padding: 2px 10px;">  </span> No Eelgrass</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    title_html = '''
    <div style="position: fixed; top: 10px; left: 50px; z-index: 9999;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray;">
        <h4 style="margin: 0;">Eelgrass Site Distribution</h4>
        <p style="margin: 0; font-size: 12px;">Use layer control to toggle</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m


def create_basin_map(site_summary: pd.DataFrame) -> folium.Map:
    """
    Create map with sites colored by basin.
    """
    df_plot = site_summary.dropna(subset=['Lat', 'Long', 'Basin'])
    
    center_lat = df_plot['Lat'].mean()
    center_lon = df_plot['Long'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='CartoDB positron'
    )
    
    # Basin colors
    basins = df_plot['Basin'].unique()
    basin_colors = {
        'Whidbey': '#e41a1c',
        'Hood Canal': '#377eb8',
        'Central': '#4daf4a',
        'San Juan': '#984ea3',
        'South': '#ff7f00',
        'Admiralty Inlet': '#ffff33',
        'Strait of Juan de Fuca': '#a65628',
        'Howe Sound': '#f781bf'
    }
    
    # Create feature group for each basin
    for basin in basins:
        basin_data = df_plot[df_plot['Basin'] == basin]
        color = basin_colors.get(basin, '#808080')
        
        group = folium.FeatureGroup(name=f'{basin} (n={len(basin_data)})')
        
        for _, row in basin_data.iterrows():
            popup_html = f"""
            <b>{row['SiteName']}</b><br>
            <hr>
            Basin: <b>{basin}</b><br>
            Habitats: {row['Habitats']}<br>
            <hr>
            Mean Rate: {row['MeanEncounterRate']:.2f}/hr<br>
            Total Pycno: {int(row['TotalPycno'])}
            """
            
            folium.CircleMarker(
                location=[row['Lat'], row['Long']],
                radius=7,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{row['SiteName']} ({basin})",
                color='black',
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.8
            ).add_to(group)
        
        group.add_to(m)
    
    folium.LayerControl().add_to(m)
    
    title_html = '''
    <div style="position: fixed; top: 10px; left: 50px; z-index: 9999;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray;">
        <h4 style="margin: 0;">Sites by Geographic Basin</h4>
        <p style="margin: 0; font-size: 12px;">Use layer control to filter</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m


def create_heatmap(site_summary: pd.DataFrame) -> folium.Map:
    """
    Create heatmap of Pycnopodia observations.
    """
    df_plot = site_summary.dropna(subset=['Lat', 'Long'])
    
    center_lat = df_plot['Lat'].mean()
    center_lon = df_plot['Long'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles='CartoDB dark_matter'
    )
    
    # Prepare heatmap data: [lat, lon, weight]
    # Weight by total Pycnopodia observed (log scale to handle outliers)
    heat_data = []
    for _, row in df_plot.iterrows():
        weight = np.log10(row['TotalPycno'] + 1)  # Log scale
        if weight > 0:
            heat_data.append([row['Lat'], row['Long'], weight])
    
    if heat_data:
        HeatMap(
            heat_data,
            min_opacity=0.3,
            max_zoom=13,
            radius=25,
            blur=15,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}
        ).add_to(m)
    
    title_html = '''
    <div style="position: fixed; top: 10px; left: 50px; z-index: 9999;
                background-color: rgba(0,0,0,0.7); color: white;
                padding: 10px; border-radius: 5px;">
        <h4 style="margin: 0;">Pycnopodia Observation Heatmap</h4>
        <p style="margin: 0; font-size: 12px;">Intensity = log(Total Pycno)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m


def create_clustered_map(site_summary: pd.DataFrame) -> folium.Map:
    """
    Create map with clustered markers (auto-groups nearby sites at low zoom).
    """
    df_plot = site_summary.dropna(subset=['Lat', 'Long'])
    
    center_lat = df_plot['Lat'].mean()
    center_lon = df_plot['Long'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=7,
        tiles='OpenStreetMap'
    )
    
    # Create marker cluster
    marker_cluster = MarkerCluster(name='Survey Sites')
    
    for _, row in df_plot.iterrows():
        has_pycno = row['TotalPycno'] > 0
        icon_color = 'green' if has_pycno else 'red'
        
        popup_html = f"""
        <b>{row['SiteName']}</b><br>
        <hr>
        Basin: {row['Basin']}<br>
        Habitats: {row['Habitats']}<br>
        <hr>
        Total Pycno: {int(row['TotalPycno'])}<br>
        Mean Rate: {row['MeanEncounterRate']:.2f}/hr<br>
        Transects: {row['NTransects']}
        """
        
        folium.Marker(
            location=[row['Lat'], row['Long']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=row['SiteName'],
            icon=folium.Icon(color=icon_color, icon='info-sign')
        ).add_to(marker_cluster)
    
    marker_cluster.add_to(m)
    
    title_html = '''
    <div style="position: fixed; top: 10px; left: 50px; z-index: 9999;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray;">
        <h4 style="margin: 0;">Clustered Survey Sites</h4>
        <p style="margin: 0; font-size: 12px;">Green = Pycno detected, Red = None</p>
        <p style="margin: 0; font-size: 12px;">Zoom in to expand clusters</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m


def main():
    print(f"\n{'='*60}")
    print("INTERACTIVE MAP VISUALIZATIONS")
    print(f"{'='*60}\n")
    
    if not FOLIUM_AVAILABLE:
        print("ERROR: folium is not installed.")
        print("Install with: pip install folium")
        print("Then re-run this script.")
        return
    
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
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Generate interactive maps
    print("\n" + "="*60)
    print("GENERATING INTERACTIVE MAPS")
    print("="*60)
    
    print("\n1. Basic site map...")
    m1 = create_basic_site_map(site_summary)
    m1.save(OUTPUT_DIR / 'map_interactive_sites.html')
    print(f"  Saved: {OUTPUT_DIR / 'map_interactive_sites.html'}")
    
    print("\n2. Encounter rate map...")
    m2 = create_encounter_rate_map(site_summary)
    m2.save(OUTPUT_DIR / 'map_interactive_encounter_rates.html')
    print(f"  Saved: {OUTPUT_DIR / 'map_interactive_encounter_rates.html'}")
    
    print("\n3. Eelgrass site map...")
    m3 = create_eelgrass_map(site_summary)
    m3.save(OUTPUT_DIR / 'map_interactive_eelgrass.html')
    print(f"  Saved: {OUTPUT_DIR / 'map_interactive_eelgrass.html'}")
    
    print("\n4. Basin map...")
    m4 = create_basin_map(site_summary)
    m4.save(OUTPUT_DIR / 'map_interactive_basins.html')
    print(f"  Saved: {OUTPUT_DIR / 'map_interactive_basins.html'}")
    
    print("\n5. Heatmap...")
    m5 = create_heatmap(site_summary)
    m5.save(OUTPUT_DIR / 'map_interactive_heatmap.html')
    print(f"  Saved: {OUTPUT_DIR / 'map_interactive_heatmap.html'}")
    
    print("\n6. Clustered marker map...")
    m6 = create_clustered_map(site_summary)
    m6.save(OUTPUT_DIR / 'map_interactive_clustered.html')
    print(f"  Saved: {OUTPUT_DIR / 'map_interactive_clustered.html'}")
    
    print(f"\n{'='*60}")
    print("INTERACTIVE MAPS COMPLETE")
    print(f"{'='*60}")
    print("\nOpen the HTML files in a web browser to explore the maps.")
    print(f"\n")


if __name__ == "__main__":
    main()






