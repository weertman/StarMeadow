"""
Habitat Diversity and Pycnopodia Abundance Analysis

Tests the hypothesis that sites with greater habitat heterogeneity
(multiple distinct habitat types) have higher Pycnopodia encounter rates.

Key Questions:
1. Does habitat richness (# of habitat types) predict encounter rate?
2. Do multi-habitat sites have more Pycnopodia than single-habitat sites?
3. Does habitat diversity (Shannon index) correlate with abundance?

Metrics:
- Habitat Richness: Number of unique habitat types at a site
- Shannon Diversity Index: H' = -Σ(p_i * ln(p_i))
- Encounter Rate: Pycnopodia per hour (site mean)

Confound Controls:
- Spatial coverage (number of unique transect LOCATIONS per site)
  NOTE: More transect locations = more habitats detected (spatial design artifact)
- Normalized metrics: Habitats per transect location

Outputs:
- Correlation and regression analyses (raw and effort-controlled)
- Group comparisons (single vs multi-habitat sites)
- Visualizations of diversity-abundance relationships
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, pearsonr

from utils import (
    get_output_dir, load_data, set_style, save_figure
)

OUTPUT_DIR = get_output_dir(__file__)


def calculate_shannon_diversity(habitat_counts: pd.Series) -> float:
    """
    Calculate Shannon diversity index: H' = -Σ(p_i * ln(p_i))
    where p_i is the proportion of transects in habitat i.
    """
    proportions = habitat_counts / habitat_counts.sum()
    # Filter out zeros to avoid log(0)
    proportions = proportions[proportions > 0]
    return -np.sum(proportions * np.log(proportions))


def calculate_simpson_diversity(habitat_counts: pd.Series) -> float:
    """
    Calculate Simpson's diversity index: D = 1 - Σ(p_i^2)
    """
    proportions = habitat_counts / habitat_counts.sum()
    return 1 - np.sum(proportions ** 2)


def prepare_site_diversity_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate habitat diversity metrics and encounter rates at site level.
    
    Includes spatial coverage metrics to control for the confound that
    more transect locations = more habitat types detected.
    """
    site_data = []
    
    for site_name, site_df in df.groupby('SiteName'):
        # Habitat metrics
        habitat_counts = site_df['HabitatType'].value_counts()
        habitat_richness = len(habitat_counts)
        shannon_div = calculate_shannon_diversity(habitat_counts)
        simpson_div = calculate_simpson_diversity(habitat_counts)
        
        # List of habitats present
        habitats_present = sorted(habitat_counts.index.tolist())
        
        # Encounter rate metrics
        mean_encounter_rate = site_df['Encounter.Rate.Hr'].mean()
        total_pycno_count = site_df['Pycnopodia_count'].sum()
        has_detection = total_pycno_count > 0
        
        # Spatial coverage metrics (CRITICAL for controlling confound)
        n_records = len(site_df)  # Total survey records
        n_transect_locations = site_df['Transect..'].nunique()  # Unique transect locations
        n_survey_dates = site_df['Date'].nunique()  # Unique survey dates
        total_survey_time = site_df['Survey.Time'].sum()
        
        # Normalized habitat richness (controls for spatial coverage)
        # Habitats per transect location - how "diverse" is each transect location?
        habitats_per_transect = habitat_richness / n_transect_locations if n_transect_locations > 0 else 0
        
        # Eelgrass presence
        has_eelgrass = 'Eelgrass' in habitats_present
        
        site_data.append({
            'SiteName': site_name,
            'HabitatRichness': habitat_richness,
            'HabitatsPerTransect': habitats_per_transect,  # Normalized metric
            'ShannonDiversity': shannon_div,
            'SimpsonDiversity': simpson_div,
            'HabitatsPresent': ', '.join(habitats_present),
            'MeanEncounterRate': mean_encounter_rate,
            'TotalPycnoCount': total_pycno_count,
            'HasDetection': has_detection,
            'NRecords': n_records,
            'NTransectLocations': n_transect_locations,  # Spatial coverage
            'NSurveyDates': n_survey_dates,
            'TotalSurveyTime': total_survey_time,
            'HasEelgrass': has_eelgrass,
            'SiteType': 'Multi-Habitat' if habitat_richness > 1 else 'Single-Habitat'
        })
    
    return pd.DataFrame(site_data)


def run_correlation_analysis(site_data: pd.DataFrame) -> dict:
    """
    Run correlation analyses between diversity metrics and encounter rates.
    Includes both raw metrics and spatial-coverage-controlled analyses.
    """
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS (RAW METRICS)")
    print("="*60)
    
    results = {}
    
    # Metrics to correlate with encounter rate
    diversity_metrics = ['HabitatRichness', 'ShannonDiversity', 'SimpsonDiversity']
    
    for metric in diversity_metrics:
        # Pearson correlation
        r_pearson, p_pearson = pearsonr(site_data[metric], site_data['MeanEncounterRate'])
        
        # Spearman correlation (rank-based, better for non-normal data)
        r_spearman, p_spearman = spearmanr(site_data[metric], site_data['MeanEncounterRate'])
        
        print(f"\n{metric} vs Mean Encounter Rate:")
        print(f"  Pearson:  r = {r_pearson:.4f}, p = {p_pearson:.4f}")
        print(f"  Spearman: ρ = {r_spearman:.4f}, p = {p_spearman:.4f}")
        
        results[metric] = {
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_r': r_spearman,
            'spearman_p': p_spearman
        }
    
    # SPATIAL COVERAGE CONFOUND
    print("\n" + "="*60)
    print("SPATIAL COVERAGE CONFOUND CHECK")
    print("="*60)
    
    r_spatial, p_spatial = pearsonr(site_data['NTransectLocations'], site_data['HabitatRichness'])
    print(f"\nN Transect Locations vs Habitat Richness:")
    print(f"  r = {r_spatial:.4f}, p = {p_spatial:.4f}")
    if p_spatial < 0.05:
        print("  ⚠️  CONFOUND: More transect locations → more habitats detected")
    
    r_spatial_enc, p_spatial_enc = pearsonr(site_data['NTransectLocations'], site_data['MeanEncounterRate'])
    print(f"\nN Transect Locations vs Encounter Rate:")
    print(f"  r = {r_spatial_enc:.4f}, p = {p_spatial_enc:.4f}")
    
    results['spatial_confound'] = {
        'transects_habitats_r': r_spatial,
        'transects_habitats_p': p_spatial,
        'transects_encounter_r': r_spatial_enc,
        'transects_encounter_p': p_spatial_enc
    }
    
    # NORMALIZED METRIC: Habitats per transect location
    print("\n" + "="*60)
    print("NORMALIZED ANALYSIS (Controlling for Spatial Coverage)")
    print("="*60)
    
    r_norm, p_norm = pearsonr(site_data['HabitatsPerTransect'], site_data['MeanEncounterRate'])
    rho_norm, p_rho_norm = spearmanr(site_data['HabitatsPerTransect'], site_data['MeanEncounterRate'])
    
    print(f"\nHabitats Per Transect Location vs Mean Encounter Rate:")
    print(f"  Pearson:  r = {r_norm:.4f}, p = {p_norm:.4f}")
    print(f"  Spearman: ρ = {rho_norm:.4f}, p = {p_rho_norm:.4f}")
    
    results['HabitatsPerTransect'] = {
        'pearson_r': r_norm,
        'pearson_p': p_norm,
        'spearman_r': rho_norm,
        'spearman_p': p_rho_norm
    }
    
    # Detection probability correlations
    print("\n--- Detection Probability ---")
    for metric in diversity_metrics + ['HabitatsPerTransect']:
        r, p = pearsonr(site_data[metric], site_data['HasDetection'].astype(int))
        print(f"{metric} vs Has Detection: r = {r:.4f}, p = {p:.4f}")
        results[f'{metric}_detection'] = {'r': r, 'p': p}
    
    return results


def run_regression_analysis(site_data: pd.DataFrame) -> dict:
    """
    Run linear regression: Encounter Rate ~ Habitat Richness
    With proper controls for spatial coverage confound.
    """
    print("\n" + "="*60)
    print("REGRESSION ANALYSIS")
    print("="*60)
    
    results = {}
    
    # Simple regression: Richness → Encounter Rate
    X = site_data['HabitatRichness'].values
    y = site_data['MeanEncounterRate'].values
    
    slope, intercept, r, p, se = stats.linregress(X, y)
    r2 = r ** 2
    
    print(f"\n1. Simple Regression: Encounter Rate ~ Habitat Richness")
    print(f"   R² = {r2:.4f} ({r2*100:.1f}% variance explained)")
    print(f"   Slope = {slope:.4f} (change in encounter rate per additional habitat)")
    print(f"   p = {p:.4f}")
    
    results['simple'] = {
        'r2': r2, 'slope': slope, 'intercept': intercept, 'p': p, 'se': se
    }
    
    # Regression with NORMALIZED metric (habitats per transect)
    print(f"\n2. Regression with Normalized Metric: Encounter Rate ~ Habitats/Transect")
    X_norm = site_data['HabitatsPerTransect'].values
    slope_n, intercept_n, r_n, p_n, se_n = stats.linregress(X_norm, y)
    r2_n = r_n ** 2
    
    print(f"   R² = {r2_n:.4f} ({r2_n*100:.1f}% variance explained)")
    print(f"   Slope = {slope_n:.4f}")
    print(f"   p = {p_n:.4f}")
    
    results['normalized'] = {
        'r2': r2_n, 'slope': slope_n, 'intercept': intercept_n, 'p': p_n
    }
    
    # Partial correlation: Richness vs Encounter Rate, controlling for Spatial Coverage
    print(f"\n3. Partial Correlation (Controlling for N Transect Locations)")
    
    # Residualize encounter rate on n_transect_locations
    slope_effort, intercept_effort, _, _, _ = stats.linregress(
        site_data['NTransectLocations'], site_data['MeanEncounterRate']
    )
    residuals_encounter = site_data['MeanEncounterRate'] - (
        slope_effort * site_data['NTransectLocations'] + intercept_effort
    )
    
    # Residualize richness on n_transect_locations
    slope_rich, intercept_rich, _, _, _ = stats.linregress(
        site_data['NTransectLocations'], site_data['HabitatRichness']
    )
    residuals_richness = site_data['HabitatRichness'] - (
        slope_rich * site_data['NTransectLocations'] + intercept_rich
    )
    
    # Partial correlation
    r_partial, p_partial = pearsonr(residuals_richness, residuals_encounter)
    print(f"   Partial r = {r_partial:.4f}, p = {p_partial:.4f}")
    print(f"   (Effect of habitat richness BEYOND what's explained by spatial coverage)")
    
    results['partial_spatial'] = {'r': r_partial, 'p': p_partial}
    
    # Stratified analysis: Sites with similar spatial coverage
    print(f"\n4. Stratified Analysis: Sites with Similar Spatial Coverage")
    
    # Split into low/medium/high spatial coverage
    terciles = site_data['NTransectLocations'].quantile([0.33, 0.67])
    low_coverage = site_data[site_data['NTransectLocations'] <= terciles[0.33]]
    mid_coverage = site_data[(site_data['NTransectLocations'] > terciles[0.33]) & 
                             (site_data['NTransectLocations'] <= terciles[0.67])]
    high_coverage = site_data[site_data['NTransectLocations'] > terciles[0.67]]
    
    for label, subset in [('Low', low_coverage), ('Medium', mid_coverage), ('High', high_coverage)]:
        if len(subset) > 5:
            r_s, p_s = pearsonr(subset['HabitatRichness'], subset['MeanEncounterRate'])
            print(f"   {label} coverage (n={len(subset)}, transects≤{subset['NTransectLocations'].max():.0f}): r = {r_s:.3f}, p = {p_s:.3f}")
            results[f'stratified_{label.lower()}'] = {'r': r_s, 'p': p_s, 'n': len(subset)}
        else:
            print(f"   {label} coverage: n={len(subset)} (too few for analysis)")
    
    return results


def run_group_comparison(site_data: pd.DataFrame) -> dict:
    """
    Compare encounter rates between single-habitat and multi-habitat sites.
    Includes spatial-coverage-controlled comparison.
    """
    print("\n" + "="*60)
    print("GROUP COMPARISON: Single vs Multi-Habitat Sites")
    print("="*60)
    
    single = site_data[site_data['HabitatRichness'] == 1]
    multi = site_data[site_data['HabitatRichness'] > 1]
    
    print(f"\nSample sizes:")
    print(f"  Single-habitat sites: {len(single)}")
    print(f"  Multi-habitat sites: {len(multi)}")
    
    results = {
        'n_single': len(single),
        'n_multi': len(multi)
    }
    
    if len(single) < 2 or len(multi) < 2:
        print("\n  ⚠️  Insufficient sample size for meaningful comparison")
        return results
    
    # Check spatial coverage difference
    print(f"\n--- Spatial Coverage Comparison ---")
    print(f"Single-habitat: {single['NTransectLocations'].mean():.1f} ± {single['NTransectLocations'].std():.1f} transect locations")
    print(f"Multi-habitat:  {multi['NTransectLocations'].mean():.1f} ± {multi['NTransectLocations'].std():.1f} transect locations")
    t_spatial, p_spatial = stats.ttest_ind(single['NTransectLocations'], multi['NTransectLocations'])
    print(f"T-test: t = {t_spatial:.3f}, p = {p_spatial:.4f}")
    if p_spatial < 0.05:
        print("⚠️  CONFOUND: Multi-habitat sites have significantly more transect locations")
    
    results['spatial_coverage_diff'] = {'t': t_spatial, 'p': p_spatial}
    
    # Encounter rate comparison (raw)
    print(f"\n--- Mean Encounter Rate (Raw) ---")
    print(f"Single-habitat: {single['MeanEncounterRate'].mean():.4f} ± {single['MeanEncounterRate'].std():.4f}")
    print(f"Multi-habitat:  {multi['MeanEncounterRate'].mean():.4f} ± {multi['MeanEncounterRate'].std():.4f}")
    
    # T-test
    t_stat, t_pval = stats.ttest_ind(single['MeanEncounterRate'], multi['MeanEncounterRate'])
    print(f"T-test: t = {t_stat:.3f}, p = {t_pval:.4f}")
    
    # Mann-Whitney
    u_stat, u_pval = stats.mannwhitneyu(single['MeanEncounterRate'], multi['MeanEncounterRate'])
    print(f"Mann-Whitney U: U = {u_stat:.0f}, p = {u_pval:.4f}")
    
    # Effect size
    n1, n2 = len(single), len(multi)
    pooled_std = np.sqrt(
        ((n1-1)*single['MeanEncounterRate'].var() + (n2-1)*multi['MeanEncounterRate'].var()) 
        / (n1 + n2 - 2)
    )
    if pooled_std > 0:
        cohens_d = (multi['MeanEncounterRate'].mean() - single['MeanEncounterRate'].mean()) / pooled_std
    else:
        cohens_d = 0
    print(f"Cohen's d: {cohens_d:.3f}")
    
    results.update({
        'mean_single': single['MeanEncounterRate'].mean(),
        'mean_multi': multi['MeanEncounterRate'].mean(),
        't_stat': t_stat,
        't_pval': t_pval,
        'u_stat': u_stat,
        'u_pval': u_pval,
        'cohens_d': cohens_d
    })
    
    # MATCHED COMPARISON: Compare only sites with similar spatial coverage
    print(f"\n--- Matched Comparison (Similar Spatial Coverage) ---")
    
    # Find overlapping range of transect locations
    min_shared = max(single['NTransectLocations'].min(), multi['NTransectLocations'].min())
    max_shared = min(single['NTransectLocations'].max(), multi['NTransectLocations'].max())
    
    single_matched = single[(single['NTransectLocations'] >= min_shared) & 
                           (single['NTransectLocations'] <= max_shared)]
    multi_matched = multi[(multi['NTransectLocations'] >= min_shared) & 
                         (multi['NTransectLocations'] <= max_shared)]
    
    print(f"Matched range: {min_shared:.0f} - {max_shared:.0f} transect locations")
    print(f"Single-habitat (matched): n={len(single_matched)}, mean={single_matched['MeanEncounterRate'].mean():.4f}")
    print(f"Multi-habitat (matched):  n={len(multi_matched)}, mean={multi_matched['MeanEncounterRate'].mean():.4f}")
    
    if len(single_matched) >= 3 and len(multi_matched) >= 3:
        t_match, p_match = stats.ttest_ind(single_matched['MeanEncounterRate'], 
                                           multi_matched['MeanEncounterRate'])
        u_match, p_u_match = stats.mannwhitneyu(single_matched['MeanEncounterRate'],
                                                 multi_matched['MeanEncounterRate'])
        print(f"T-test (matched): t = {t_match:.3f}, p = {p_match:.4f}")
        print(f"Mann-Whitney (matched): U = {u_match:.0f}, p = {p_u_match:.4f}")
        results['matched'] = {'t': t_match, 'p': p_match, 
                             'n_single': len(single_matched), 'n_multi': len(multi_matched)}
    else:
        print("(Insufficient sample size for matched comparison)")
    
    # Detection probability comparison
    print(f"\n--- Detection Probability ---")
    detect_single = single['HasDetection'].mean()
    detect_multi = multi['HasDetection'].mean()
    print(f"Single-habitat: {detect_single:.1%} of sites have detections")
    print(f"Multi-habitat:  {detect_multi:.1%} of sites have detections")
    
    # Chi-square test for detection
    contingency = pd.crosstab(site_data['SiteType'], site_data['HasDetection'])
    if contingency.shape == (2, 2):
        chi2, chi_p, dof, expected = stats.chi2_contingency(contingency)
        print(f"Chi-square: χ² = {chi2:.3f}, p = {chi_p:.4f}")
        results['chi2'] = chi2
        results['chi_p'] = chi_p
    
    results['detect_single'] = detect_single
    results['detect_multi'] = detect_multi
    
    return results


def run_richness_category_analysis(site_data: pd.DataFrame) -> dict:
    """
    Analyze encounter rates by habitat richness category.
    """
    print("\n" + "="*60)
    print("ANALYSIS BY HABITAT RICHNESS CATEGORY")
    print("="*60)
    
    # Group by richness
    richness_stats = site_data.groupby('HabitatRichness').agg({
        'MeanEncounterRate': ['mean', 'std', 'count'],
        'HasDetection': ['sum', 'mean'],
        'TotalPycnoCount': 'sum'
    }).round(4)
    
    print("\nEncounter Rate by Habitat Richness:")
    print(richness_stats.to_string())
    
    # Kruskal-Wallis test
    groups = [group['MeanEncounterRate'].values for name, group in site_data.groupby('HabitatRichness')]
    if len(groups) >= 2 and all(len(g) >= 1 for g in groups):
        h_stat, kw_p = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis H-test: H = {h_stat:.3f}, p = {kw_p:.4f}")
    
    return {'richness_stats': richness_stats}


def plot_diversity_encounter_relationship(site_data: pd.DataFrame):
    """
    Visualize relationship between habitat diversity and encounter rate.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Scatter: Richness vs Encounter Rate
    ax1 = axes[0, 0]
    ax1.scatter(site_data['HabitatRichness'], site_data['MeanEncounterRate'], 
                s=100, alpha=0.7, c='steelblue', edgecolor='black')
    
    # Add regression line
    slope, intercept, r, p, se = stats.linregress(
        site_data['HabitatRichness'], site_data['MeanEncounterRate']
    )
    x_line = np.array([site_data['HabitatRichness'].min(), site_data['HabitatRichness'].max()])
    ax1.plot(x_line, slope * x_line + intercept, 'r--', linewidth=2, 
             label=f'R² = {r**2:.3f}, p = {p:.3f}')
    
    ax1.set_xlabel('Habitat Richness (# of habitat types)')
    ax1.set_ylabel('Mean Encounter Rate (Pycno/hr)')
    ax1.set_title('Habitat Richness vs Encounter Rate')
    ax1.legend()
    
    # Add site labels for context
    for idx, row in site_data.iterrows():
        if row['MeanEncounterRate'] > site_data['MeanEncounterRate'].quantile(0.9):
            ax1.annotate(row['SiteName'][:15], (row['HabitatRichness'], row['MeanEncounterRate']),
                        fontsize=8, alpha=0.7)
    
    # 2. Scatter: Shannon Diversity vs Encounter Rate
    ax2 = axes[0, 1]
    ax2.scatter(site_data['ShannonDiversity'], site_data['MeanEncounterRate'],
                s=100, alpha=0.7, c='forestgreen', edgecolor='black')
    
    slope, intercept, r, p, se = stats.linregress(
        site_data['ShannonDiversity'], site_data['MeanEncounterRate']
    )
    x_line = np.array([site_data['ShannonDiversity'].min(), site_data['ShannonDiversity'].max()])
    ax2.plot(x_line, slope * x_line + intercept, 'r--', linewidth=2,
             label=f'R² = {r**2:.3f}, p = {p:.3f}')
    
    ax2.set_xlabel("Shannon Diversity Index (H')")
    ax2.set_ylabel('Mean Encounter Rate (Pycno/hr)')
    ax2.set_title('Shannon Diversity vs Encounter Rate')
    ax2.legend()
    
    # 3. Box plot: Single vs Multi-Habitat
    ax3 = axes[1, 0]
    colors = {'Single-Habitat': '#9b59b6', 'Multi-Habitat': '#3498db'}  # Purple/Blue for habitat diversity
    
    sns.boxplot(data=site_data, x='SiteType', y='MeanEncounterRate',
                hue='SiteType', palette=colors, ax=ax3)
    if ax3.get_legend() is not None:
        ax3.get_legend().remove()
    sns.stripplot(data=site_data, x='SiteType', y='MeanEncounterRate',
                  color='black', alpha=0.5, size=8, ax=ax3)
    
    ax3.set_xlabel('')
    ax3.set_ylabel('Mean Encounter Rate (Pycno/hr)')
    ax3.set_title('Single vs Multi-Habitat Sites')
    
    # Add sample sizes
    for i, st in enumerate(['Single-Habitat', 'Multi-Habitat']):
        n = len(site_data[site_data['SiteType'] == st])
        ax3.annotate(f'n={n}', xy=(i, ax3.get_ylim()[1] * 0.95), ha='center', fontsize=10)
    
    # 4. Bar plot: Encounter rate by richness category
    ax4 = axes[1, 1]
    richness_means = site_data.groupby('HabitatRichness')['MeanEncounterRate'].agg(['mean', 'std', 'count'])
    richness_means['se'] = richness_means['std'] / np.sqrt(richness_means['count'])
    
    bars = ax4.bar(richness_means.index.astype(str), richness_means['mean'],
                   yerr=richness_means['se'], capsize=5,
                   color='mediumpurple', edgecolor='black', alpha=0.8)
    
    ax4.set_xlabel('Habitat Richness')
    ax4.set_ylabel('Mean Encounter Rate (± SE)')
    ax4.set_title('Encounter Rate by Habitat Richness')
    
    for i, (idx, row) in enumerate(richness_means.iterrows()):
        ax4.annotate(f'n={int(row["count"])}', xy=(i, row['mean'] + row['se'] + 0.01),
                     ha='center', fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, 'diversity_encounter_relationship')
    plt.close()


def plot_confound_analysis(site_data: pd.DataFrame):
    """
    Visualize spatial coverage confound and controlled analyses.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. N Transect Locations vs Habitat Richness (THE CONFOUND)
    ax1 = axes[0, 0]
    ax1.scatter(site_data['NTransectLocations'], site_data['HabitatRichness'],
                s=100, alpha=0.7, c='coral', edgecolor='black')
    
    r, p = pearsonr(site_data['NTransectLocations'], site_data['HabitatRichness'])
    slope, intercept, _, _, _ = stats.linregress(site_data['NTransectLocations'], site_data['HabitatRichness'])
    x_line = np.array([site_data['NTransectLocations'].min(), site_data['NTransectLocations'].max()])
    ax1.plot(x_line, slope * x_line + intercept, 'r--', linewidth=2)
    
    ax1.set_xlabel('N Transect Locations (Spatial Coverage)')
    ax1.set_ylabel('Habitat Richness')
    ax1.set_title(f'THE CONFOUND: Spatial Coverage → Habitats Detected\nr = {r:.3f}, p = {p:.4f}')
    
    # 2. N Transect Locations vs Encounter Rate
    ax2 = axes[0, 1]
    ax2.scatter(site_data['NTransectLocations'], site_data['MeanEncounterRate'],
                s=100, alpha=0.7, c='steelblue', edgecolor='black')
    
    r, p = pearsonr(site_data['NTransectLocations'], site_data['MeanEncounterRate'])
    ax2.set_xlabel('N Transect Locations')
    ax2.set_ylabel('Mean Encounter Rate')
    ax2.set_title(f'Spatial Coverage vs Encounter Rate\nr = {r:.3f}, p = {p:.3f}')
    
    # 3. Partial correlation visualization
    ax3 = axes[1, 0]
    
    # Residualize both variables on N Transect Locations
    slope1, int1, _, _, _ = stats.linregress(site_data['NTransectLocations'], site_data['HabitatRichness'])
    resid_richness = site_data['HabitatRichness'] - (slope1 * site_data['NTransectLocations'] + int1)
    
    slope2, int2, _, _, _ = stats.linregress(site_data['NTransectLocations'], site_data['MeanEncounterRate'])
    resid_encounter = site_data['MeanEncounterRate'] - (slope2 * site_data['NTransectLocations'] + int2)
    
    ax3.scatter(resid_richness, resid_encounter, s=100, alpha=0.7, c='forestgreen', edgecolor='black')
    
    r, p = pearsonr(resid_richness, resid_encounter)
    slope, intercept, _, _, _ = stats.linregress(resid_richness, resid_encounter)
    x_line = np.array([resid_richness.min(), resid_richness.max()])
    ax3.plot(x_line, slope * x_line + intercept, 'r--', linewidth=2)
    
    ax3.set_xlabel('Habitat Richness (residualized)')
    ax3.set_ylabel('Encounter Rate (residualized)')
    ax3.set_title(f'Partial Correlation\n(controlling for spatial coverage)\nr = {r:.3f}, p = {p:.3f}')
    ax3.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax3.axvline(0, color='gray', linestyle=':', alpha=0.5)
    
    # 4. Normalized metric: Habitats per transect location
    ax4 = axes[1, 1]
    ax4.scatter(site_data['HabitatsPerTransect'], site_data['MeanEncounterRate'],
                s=100, alpha=0.7, c='purple', edgecolor='black')
    
    r, p = pearsonr(site_data['HabitatsPerTransect'], site_data['MeanEncounterRate'])
    slope, intercept, _, _, _ = stats.linregress(site_data['HabitatsPerTransect'], site_data['MeanEncounterRate'])
    x_line = np.array([site_data['HabitatsPerTransect'].min(), site_data['HabitatsPerTransect'].max()])
    ax4.plot(x_line, slope * x_line + intercept, 'r--', linewidth=2)
    
    ax4.set_xlabel('Habitats Per Transect Location (Normalized)')
    ax4.set_ylabel('Mean Encounter Rate')
    ax4.set_title(f'Normalized Diversity vs Encounter Rate\nr = {r:.3f}, p = {p:.3f}')
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, 'spatial_coverage_analysis')
    plt.close()


def plot_detection_by_diversity(site_data: pd.DataFrame):
    """
    Visualize detection probability by habitat diversity.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Detection rate by richness
    ax1 = axes[0]
    detection_by_richness = site_data.groupby('HabitatRichness')['HasDetection'].agg(['sum', 'count', 'mean'])
    detection_by_richness.columns = ['Detections', 'Total', 'Rate']
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(detection_by_richness)))
    bars = ax1.bar(detection_by_richness.index.astype(str), detection_by_richness['Rate'] * 100,
                   color=colors, edgecolor='black')
    
    ax1.set_xlabel('Habitat Richness')
    ax1.set_ylabel('Detection Rate (%)')
    ax1.set_title('Pycnopodia Detection by Habitat Richness')
    ax1.set_ylim(0, 100)
    
    for i, (idx, row) in enumerate(detection_by_richness.iterrows()):
        ax1.annotate(f'{int(row["Detections"])}/{int(row["Total"])}', 
                     xy=(i, row['Rate'] * 100 + 3), ha='center', fontsize=10)
    
    # 2. Stacked bar: Detection vs No Detection by site type
    ax2 = axes[1]
    contingency = pd.crosstab(site_data['SiteType'], site_data['HasDetection'])
    contingency.columns = ['No Detection', 'Detection']
    contingency = contingency.reindex(['Single-Habitat', 'Multi-Habitat'])
    
    contingency.plot(kind='bar', stacked=True, ax=ax2, 
                     color=['#95a5a6', '#3498db'], edgecolor='black')  # Gray/Blue for detection status
    
    ax2.set_xlabel('')
    ax2.set_ylabel('Number of Sites')
    ax2.set_title('Detection Status by Site Type')
    ax2.legend(title='')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, 'detection_by_diversity')
    plt.close()


def generate_summary_tables(site_data: pd.DataFrame, results: dict):
    """
    Generate and save summary tables.
    """
    # Site diversity summary
    site_data.to_csv(OUTPUT_DIR / 'site_diversity_data.csv', index=False)
    print(f"\n  Saved: {OUTPUT_DIR / 'site_diversity_data.csv'}")
    
    # Summary by richness
    richness_summary = site_data.groupby('HabitatRichness').agg({
        'MeanEncounterRate': ['mean', 'std', 'count'],
        'HasDetection': ['sum', 'mean'],
        'TotalPycnoCount': 'sum',
        'NTransectLocations': 'mean'
    }).round(4)
    richness_summary.to_csv(OUTPUT_DIR / 'richness_summary.csv')
    print(f"  Saved: {OUTPUT_DIR / 'richness_summary.csv'}")
    
    # Statistical results
    stats_summary = []
    
    if 'correlation' in results:
        for metric, vals in results['correlation'].items():
            if 'pearson_r' in vals:
                stats_summary.append({
                    'Test': f'{metric} vs Encounter Rate',
                    'Statistic': 'Pearson r',
                    'Value': vals['pearson_r'],
                    'p_value': vals['pearson_p']
                })
                stats_summary.append({
                    'Test': f'{metric} vs Encounter Rate',
                    'Statistic': 'Spearman ρ',
                    'Value': vals['spearman_r'],
                    'p_value': vals['spearman_p']
                })
    
    if 'regression' in results:
        stats_summary.append({
            'Test': 'Simple Regression',
            'Statistic': 'R²',
            'Value': results['regression']['simple']['r2'],
            'p_value': results['regression']['simple']['p']
        })
    
    if 'group_comparison' in results:
        gc = results['group_comparison']
        if 't_stat' in gc:
            stats_summary.append({
                'Test': 'Single vs Multi-Habitat',
                'Statistic': 't',
                'Value': gc['t_stat'],
                'p_value': gc['t_pval']
            })
    
    stats_df = pd.DataFrame(stats_summary)
    stats_df.to_csv(OUTPUT_DIR / 'statistical_tests.csv', index=False)
    print(f"  Saved: {OUTPUT_DIR / 'statistical_tests.csv'}")


def main():
    print(f"\n{'='*60}")
    print("HABITAT DIVERSITY & PYCNOPODIA ABUNDANCE ANALYSIS")
    print(f"{'='*60}\n")
    
    set_style()
    
    # Load data
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} survey records from {df['SiteName'].nunique()} sites")
    
    # Prepare site-level diversity data
    print("\nCalculating site-level habitat diversity metrics...")
    site_data = prepare_site_diversity_data(df)
    
    print(f"\n  Sites analyzed: {len(site_data)}")
    print(f"  Habitat richness range: {site_data['HabitatRichness'].min()} - {site_data['HabitatRichness'].max()}")
    print(f"  Single-habitat sites: {(site_data['HabitatRichness'] == 1).sum()}")
    print(f"  Multi-habitat sites: {(site_data['HabitatRichness'] > 1).sum()}")
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Run analyses
    results = {}
    
    results['correlation'] = run_correlation_analysis(site_data)
    results['regression'] = run_regression_analysis(site_data)
    results['group_comparison'] = run_group_comparison(site_data)
    results['richness_category'] = run_richness_category_analysis(site_data)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    print("\nPlotting diversity-encounter relationships...")
    plot_diversity_encounter_relationship(site_data)
    
    print("Plotting spatial coverage analysis...")
    plot_confound_analysis(site_data)
    
    print("Plotting detection by diversity...")
    plot_detection_by_diversity(site_data)
    
    # Save summary tables
    print("\nGenerating summary tables...")
    generate_summary_tables(site_data, results)
    
    # Final summary
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    corr = results['correlation']
    reg = results['regression']
    gc = results['group_comparison']
    spatial = corr.get('spatial_confound', {})
    
    print(f"""
HYPOTHESIS: Sites with more habitat types have more Pycnopodia

═══════════════════════════════════════════════════════════
SPATIAL COVERAGE CONFOUND
═══════════════════════════════════════════════════════════
  • N Transect Locations vs Habitat Richness: r = {spatial.get('transects_habitats_r', 'N/A'):.3f}, p = {spatial.get('transects_habitats_p', 'N/A'):.4f}
    (More transect locations → more habitats detected)
  • This is a SAMPLING DESIGN artifact, not temporal resampling

═══════════════════════════════════════════════════════════
RAW ANALYSIS (NOT controlling for spatial coverage)
═══════════════════════════════════════════════════════════
  • Pearson r = {corr['HabitatRichness']['pearson_r']:.3f}, p = {corr['HabitatRichness']['pearson_p']:.4f}
  • Spearman ρ = {corr['HabitatRichness']['spearman_r']:.3f}, p = {corr['HabitatRichness']['spearman_p']:.4f}
  • R² = {reg['simple']['r2']:.4f} ({reg['simple']['r2']*100:.1f}% variance explained)

═══════════════════════════════════════════════════════════
CONTROLLED ANALYSIS (accounting for spatial coverage)
═══════════════════════════════════════════════════════════
  • Partial r (controlling for N transect locations): {reg['partial_spatial']['r']:.3f}, p = {reg['partial_spatial']['p']:.4f}
  • Normalized metric (Habitats/Transect) vs Encounter Rate:
    r = {corr['HabitatsPerTransect']['pearson_r']:.3f}, p = {corr['HabitatsPerTransect']['pearson_p']:.4f}
  • R² (normalized) = {reg['normalized']['r2']:.4f}

═══════════════════════════════════════════════════════════
GROUP COMPARISON (Single vs Multi-Habitat)
═══════════════════════════════════════════════════════════
  • Single-habitat: mean = {gc.get('mean_single', 0):.4f} Pycno/hr (n={gc.get('n_single', 'N/A')})
  • Multi-habitat:  mean = {gc.get('mean_multi', 0):.4f} Pycno/hr (n={gc.get('n_multi', 'N/A')})
  • Cohen's d = {gc.get('cohens_d', 0):.3f}, p = {gc.get('t_pval', 'N/A'):.4f}
""")
    
    # Check matched comparison results
    if 'matched' in gc:
        print(f"  • MATCHED (similar spatial coverage): p = {gc['matched']['p']:.4f}")
    
    # Interpretation based on controlled analysis
    partial_p = reg['partial_spatial']['p']
    norm_p = corr['HabitatsPerTransect']['pearson_p']
    
    print("\n═══════════════════════════════════════════════════════════")
    print("CONCLUSION")
    print("═══════════════════════════════════════════════════════════")
    
    if partial_p < 0.05 or norm_p < 0.05:
        print("✓ SUPPORTED: After controlling for spatial coverage,")
        print("   habitat diversity still predicts Pycnopodia encounter rate.")
    elif partial_p < 0.1 or norm_p < 0.1:
        print("~ MARGINAL: Trend persists after spatial coverage control,")
        print("   but not statistically significant.")
    else:
        print("✗ NOT SUPPORTED: After accounting for spatial coverage,")
        print("   no significant relationship between habitat diversity")
        print("   and Pycnopodia encounter rate.")
        print("\n   The raw correlation is likely driven by the confound that")
        print("   sites with more transect locations detect more habitats.")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

