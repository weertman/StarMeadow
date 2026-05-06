"""
Eelgrass Site Presence and Pycnopodia Encounter Rate: Basin-Controlled Analysis

Tests whether site-level eelgrass presence predicts Pycnopodia encounter rate
after controlling for geographic basin effects.

Research Question:
Do sites with eelgrass have higher Pycnopodia encounter rates than sites 
without eelgrass, when accounting for variation across basins?

Analytical Approaches:
1. Stratified Analysis: Test eelgrass effect within each basin
2. Two-way ANOVA: Encounter Rate ~ Eelgrass + Basin + Eelgrass×Basin
3. Partial Correlation: Eelgrass vs Encounter Rate, controlling for Basin

Why Control for Basin?
- Pycnopodia abundance varies by basin (oceanographic conditions, recovery status)
- Eelgrass prevalence may vary by basin (depth, substrate)
- Without controlling, eelgrass-abundance correlation could be spurious

Outputs:
- Basin-stratified comparisons
- ANOVA results
- Partial correlation analysis
- Visualizations by basin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, mannwhitneyu, ttest_ind
import warnings

from utils import (
    get_output_dir, load_data, set_style, save_figure
)

OUTPUT_DIR = get_output_dir(__file__)


def prepare_site_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare site-level data with eelgrass indicator and basin information.
    """
    site_data = []
    
    for site_name, site_df in df.groupby('SiteName'):
        # Get basin (should be consistent within site)
        basin = site_df['Basin'].mode().iloc[0] if len(site_df['Basin'].mode()) > 0 else 'Unknown'
        
        # Eelgrass presence at site
        has_eelgrass = (site_df['HabitatType'] == 'Eelgrass').any()
        
        # Encounter rate metrics
        mean_encounter_rate = site_df['Encounter.Rate.Hr'].mean()
        total_pycno_count = site_df['Pycnopodia_count'].sum()
        has_detection = total_pycno_count > 0
        
        # Survey effort
        n_records = len(site_df)
        n_transect_locations = site_df['Transect..'].nunique()
        
        site_data.append({
            'SiteName': site_name,
            'Basin': basin,
            'HasEelgrass': has_eelgrass,
            'EelgrassLabel': 'Eelgrass Site' if has_eelgrass else 'Non-Eelgrass Site',
            'MeanEncounterRate': mean_encounter_rate,
            'TotalPycnoCount': total_pycno_count,
            'HasDetection': has_detection,
            'NRecords': n_records,
            'NTransectLocations': n_transect_locations
        })
    
    return pd.DataFrame(site_data)


def analyze_basin_distribution(site_data: pd.DataFrame):
    """
    Examine how eelgrass sites and basins are distributed.
    """
    print("\n" + "="*60)
    print("BASIN AND EELGRASS DISTRIBUTION")
    print("="*60)
    
    # Cross-tabulation
    cross_tab = pd.crosstab(site_data['Basin'], site_data['HasEelgrass'], 
                            margins=True, margins_name='Total')
    cross_tab.columns = ['Non-Eelgrass', 'Eelgrass', 'Total']
    
    print("\nSites by Basin and Eelgrass Presence:")
    print(cross_tab.to_string())
    
    # Chi-square test for independence
    contingency = pd.crosstab(site_data['Basin'], site_data['HasEelgrass'])
    if contingency.shape[0] > 1 and contingency.shape[1] == 2:
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"\nChi-square test (Basin × Eelgrass independence):")
        print(f"  χ² = {chi2:.3f}, p = {p:.4f}, df = {dof}")
        if p < 0.05:
            print("  ⚠️  Eelgrass prevalence varies significantly by basin")
    
    # Encounter rate by basin
    print("\nMean Encounter Rate by Basin:")
    basin_stats = site_data.groupby('Basin')['MeanEncounterRate'].agg(['mean', 'std', 'count'])
    print(basin_stats.round(4).to_string())
    
    return cross_tab


def run_stratified_analysis(site_data: pd.DataFrame) -> dict:
    """
    Test eelgrass effect on encounter rate within each basin.
    """
    print("\n" + "="*60)
    print("STRATIFIED ANALYSIS: Eelgrass Effect Within Each Basin")
    print("="*60)
    
    results = {}
    
    for basin in sorted(site_data['Basin'].unique()):
        basin_df = site_data[site_data['Basin'] == basin]
        eelgrass = basin_df[basin_df['HasEelgrass'] == True]
        no_eelgrass = basin_df[basin_df['HasEelgrass'] == False]
        
        print(f"\n--- {basin} ---")
        print(f"  Eelgrass sites: n={len(eelgrass)}, Non-eelgrass sites: n={len(no_eelgrass)}")
        
        if len(eelgrass) < 2 or len(no_eelgrass) < 2:
            print("  (Insufficient data for comparison)")
            results[basin] = {'n_eelgrass': len(eelgrass), 'n_no_eelgrass': len(no_eelgrass), 
                             'sufficient_data': False}
            continue
        
        eg_mean = eelgrass['MeanEncounterRate'].mean()
        noeg_mean = no_eelgrass['MeanEncounterRate'].mean()
        
        print(f"  Eelgrass: mean = {eg_mean:.4f} Pycno/hr")
        print(f"  Non-eelgrass: mean = {noeg_mean:.4f} Pycno/hr")
        print(f"  Difference: {eg_mean - noeg_mean:+.4f}")
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pval = mannwhitneyu(eelgrass['MeanEncounterRate'], 
                                       no_eelgrass['MeanEncounterRate'],
                                       alternative='two-sided')
        print(f"  Mann-Whitney U: U = {u_stat:.0f}, p = {u_pval:.4f}")
        
        # T-test
        t_stat, t_pval = ttest_ind(eelgrass['MeanEncounterRate'], 
                                   no_eelgrass['MeanEncounterRate'])
        print(f"  T-test: t = {t_stat:.3f}, p = {t_pval:.4f}")
        
        # Effect size
        n1, n2 = len(eelgrass), len(no_eelgrass)
        pooled_std = np.sqrt(((n1-1)*eelgrass['MeanEncounterRate'].var() + 
                              (n2-1)*no_eelgrass['MeanEncounterRate'].var()) / (n1 + n2 - 2))
        cohens_d = (eg_mean - noeg_mean) / pooled_std if pooled_std > 0 else 0
        print(f"  Cohen's d: {cohens_d:.3f}")
        
        results[basin] = {
            'n_eelgrass': len(eelgrass),
            'n_no_eelgrass': len(no_eelgrass),
            'mean_eelgrass': eg_mean,
            'mean_no_eelgrass': noeg_mean,
            'difference': eg_mean - noeg_mean,
            'u_stat': u_stat,
            'u_pval': u_pval,
            't_stat': t_stat,
            't_pval': t_pval,
            'cohens_d': cohens_d,
            'sufficient_data': True
        }
    
    # Summary across basins
    print("\n" + "-"*60)
    print("STRATIFIED SUMMARY")
    print("-"*60)
    
    basins_with_data = [b for b, r in results.items() if r.get('sufficient_data', False)]
    if basins_with_data:
        sig_basins = [b for b in basins_with_data if results[b]['u_pval'] < 0.05]
        trend_basins = [b for b in basins_with_data if 0.05 <= results[b]['u_pval'] < 0.1]
        
        print(f"\nBasins with sufficient data: {len(basins_with_data)}")
        print(f"Basins with significant eelgrass effect (p<0.05): {len(sig_basins)}")
        if sig_basins:
            print(f"  → {', '.join(sig_basins)}")
        print(f"Basins with trend (0.05<p<0.10): {len(trend_basins)}")
        if trend_basins:
            print(f"  → {', '.join(trend_basins)}")
        
        # Direction consistency
        positive_effect = [b for b in basins_with_data if results[b]['difference'] > 0]
        print(f"\nBasins where eelgrass sites have HIGHER encounter rate: {len(positive_effect)}/{len(basins_with_data)}")
    
    return results


def run_anova_analysis(site_data: pd.DataFrame) -> dict:
    """
    Two-way ANOVA: Encounter Rate ~ Eelgrass + Basin + Eelgrass×Basin
    Using scipy.stats for simplicity (Type I SS).
    """
    print("\n" + "="*60)
    print("TWO-WAY ANALYSIS: Eelgrass + Basin Effects")
    print("="*60)
    
    results = {}
    
    # Filter to basins with both eelgrass and non-eelgrass sites
    basin_counts = site_data.groupby('Basin')['HasEelgrass'].agg(['sum', lambda x: (~x).sum()])
    basin_counts.columns = ['n_eelgrass', 'n_no_eelgrass']
    valid_basins = basin_counts[(basin_counts['n_eelgrass'] >= 1) & 
                                 (basin_counts['n_no_eelgrass'] >= 1)].index
    
    site_data_valid = site_data[site_data['Basin'].isin(valid_basins)].copy()
    
    print(f"\nBasins with both site types: {len(valid_basins)}")
    print(f"Sites included: {len(site_data_valid)}")
    
    if len(valid_basins) < 2:
        print("Insufficient basins for two-way analysis")
        return results
    
    # Overall effect of eelgrass (ignoring basin)
    print("\n--- Main Effect: Eelgrass ---")
    eelgrass_sites = site_data_valid[site_data_valid['HasEelgrass'] == True]['MeanEncounterRate']
    no_eelgrass_sites = site_data_valid[site_data_valid['HasEelgrass'] == False]['MeanEncounterRate']
    
    print(f"Eelgrass sites: mean = {eelgrass_sites.mean():.4f} (n={len(eelgrass_sites)})")
    print(f"Non-eelgrass sites: mean = {no_eelgrass_sites.mean():.4f} (n={len(no_eelgrass_sites)})")
    
    t_main, p_main = ttest_ind(eelgrass_sites, no_eelgrass_sites)
    print(f"T-test (main effect): t = {t_main:.3f}, p = {p_main:.4f}")
    
    results['main_eelgrass'] = {'t': t_main, 'p': p_main}
    
    # Effect of basin
    print("\n--- Main Effect: Basin ---")
    basin_groups = [group['MeanEncounterRate'].values for name, group in site_data_valid.groupby('Basin')]
    if len(basin_groups) >= 2:
        h_stat, kw_p = stats.kruskal(*basin_groups)
        print(f"Kruskal-Wallis (basin effect): H = {h_stat:.3f}, p = {kw_p:.4f}")
        results['main_basin'] = {'H': h_stat, 'p': kw_p}
    
    # Partial correlation: Eelgrass vs Encounter Rate, controlling for Basin
    print("\n--- Partial Correlation (Controlling for Basin) ---")
    
    # Encode basin as dummy variables and residualize
    basin_dummies = pd.get_dummies(site_data_valid['Basin'], drop_first=True)
    
    # Residualize encounter rate on basin
    from scipy.linalg import lstsq
    X_basin = np.column_stack([np.ones(len(site_data_valid)), basin_dummies.values])
    y_encounter = site_data_valid['MeanEncounterRate'].values
    
    # Simple OLS to get residuals
    coef, _, _, _ = lstsq(X_basin, y_encounter)
    y_pred_basin = X_basin @ coef
    resid_encounter = y_encounter - y_pred_basin
    
    # Residualize eelgrass on basin
    y_eelgrass = site_data_valid['HasEelgrass'].astype(int).values
    coef_eg, _, _, _ = lstsq(X_basin, y_eelgrass)
    y_pred_eelgrass = X_basin @ coef_eg
    resid_eelgrass = y_eelgrass - y_pred_eelgrass
    
    # Partial correlation
    r_partial, p_partial = pearsonr(resid_eelgrass, resid_encounter)
    print(f"Partial r (eelgrass → encounter, controlling for basin): {r_partial:.4f}")
    print(f"p-value: {p_partial:.4f}")
    
    results['partial_correlation'] = {'r': r_partial, 'p': p_partial}
    
    # R-squared for eelgrass after controlling for basin
    # Add eelgrass to model
    X_full = np.column_stack([X_basin, y_eelgrass])
    coef_full, _, _, _ = lstsq(X_full, y_encounter)
    y_pred_full = X_full @ coef_full
    
    ss_res_basin = np.sum((y_encounter - y_pred_basin)**2)
    ss_res_full = np.sum((y_encounter - y_pred_full)**2)
    ss_tot = np.sum((y_encounter - y_encounter.mean())**2)
    
    r2_basin = 1 - ss_res_basin / ss_tot
    r2_full = 1 - ss_res_full / ss_tot
    r2_eelgrass_given_basin = r2_full - r2_basin
    
    print(f"\nR² (basin only): {r2_basin:.4f}")
    print(f"R² (basin + eelgrass): {r2_full:.4f}")
    print(f"Δ R² (eelgrass after basin): {r2_eelgrass_given_basin:.4f}")
    
    results['r2_basin'] = r2_basin
    results['r2_full'] = r2_full
    results['r2_eelgrass_given_basin'] = r2_eelgrass_given_basin
    
    return results


def run_detection_analysis(site_data: pd.DataFrame) -> dict:
    """
    Analyze detection probability by eelgrass presence, controlling for basin.
    """
    print("\n" + "="*60)
    print("DETECTION PROBABILITY ANALYSIS")
    print("="*60)
    
    results = {}
    
    # Overall detection rates
    print("\n--- Overall Detection Rates ---")
    eelgrass_detect = site_data[site_data['HasEelgrass'] == True]['HasDetection'].mean()
    no_eelgrass_detect = site_data[site_data['HasEelgrass'] == False]['HasDetection'].mean()
    
    print(f"Eelgrass sites: {eelgrass_detect:.1%} have detections")
    print(f"Non-eelgrass sites: {no_eelgrass_detect:.1%} have detections")
    
    # Chi-square
    contingency = pd.crosstab(site_data['HasEelgrass'], site_data['HasDetection'])
    if contingency.shape == (2, 2):
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        print(f"Chi-square: χ² = {chi2:.3f}, p = {p:.4f}")
        results['overall'] = {'chi2': chi2, 'p': p, 
                             'detect_eelgrass': eelgrass_detect,
                             'detect_no_eelgrass': no_eelgrass_detect}
    
    # Within-basin detection rates
    print("\n--- Detection Rates by Basin ---")
    for basin in sorted(site_data['Basin'].unique()):
        basin_df = site_data[site_data['Basin'] == basin]
        eg = basin_df[basin_df['HasEelgrass'] == True]
        noeg = basin_df[basin_df['HasEelgrass'] == False]
        
        if len(eg) >= 1 and len(noeg) >= 1:
            eg_det = eg['HasDetection'].mean() if len(eg) > 0 else np.nan
            noeg_det = noeg['HasDetection'].mean() if len(noeg) > 0 else np.nan
            print(f"  {basin}: Eelgrass={eg_det:.0%} (n={len(eg)}), Non-eelgrass={noeg_det:.0%} (n={len(noeg)})")
    
    return results


def plot_basin_comparison(site_data: pd.DataFrame):
    """
    Visualize eelgrass effect on encounter rate by basin.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = {'Eelgrass Site': '#2ecc71', 'Non-Eelgrass Site': '#e74c3c'}
    
    # 1. Box plot by basin and eelgrass
    ax1 = axes[0, 0]
    
    # Filter to basins with both types
    basin_counts = site_data.groupby('Basin')['HasEelgrass'].agg(['sum', lambda x: (~x).sum()])
    valid_basins = basin_counts[(basin_counts.iloc[:, 0] >= 1) & (basin_counts.iloc[:, 1] >= 1)].index
    plot_data = site_data[site_data['Basin'].isin(valid_basins)]
    
    if len(plot_data) > 0:
        sns.boxplot(data=plot_data, x='Basin', y='MeanEncounterRate', hue='EelgrassLabel',
                    palette=colors, ax=ax1)
        ax1.set_xlabel('Basin')
        ax1.set_ylabel('Mean Encounter Rate (Pycno/hr)')
        ax1.set_title('Encounter Rate by Basin and Eelgrass Presence')
        ax1.legend(title='', loc='upper right')
        ax1.tick_params(axis='x', rotation=45)
    
    # 2. Effect size by basin
    ax2 = axes[0, 1]
    
    effect_data = []
    for basin in valid_basins:
        basin_df = site_data[site_data['Basin'] == basin]
        eg = basin_df[basin_df['HasEelgrass'] == True]['MeanEncounterRate']
        noeg = basin_df[basin_df['HasEelgrass'] == False]['MeanEncounterRate']
        
        if len(eg) >= 2 and len(noeg) >= 2:
            diff = eg.mean() - noeg.mean()
            pooled_std = np.sqrt(((len(eg)-1)*eg.var() + (len(noeg)-1)*noeg.var()) / 
                                (len(eg) + len(noeg) - 2))
            cohens_d = diff / pooled_std if pooled_std > 0 else 0
            effect_data.append({'Basin': basin, 'Effect': diff, 'CohensD': cohens_d})
    
    if effect_data:
        effect_df = pd.DataFrame(effect_data)
        bar_colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in effect_df['Effect']]
        ax2.barh(effect_df['Basin'], effect_df['Effect'], color=bar_colors, edgecolor='black')
        ax2.axvline(0, color='black', linestyle='-', linewidth=1)
        ax2.set_xlabel('Difference in Encounter Rate\n(Eelgrass - Non-Eelgrass)')
        ax2.set_title('Eelgrass Effect by Basin')
    
    # 3. Overall comparison with basin as hue
    ax3 = axes[1, 0]
    
    summary = site_data.groupby(['Basin', 'EelgrassLabel'])['MeanEncounterRate'].mean().unstack()
    if summary.shape[1] == 2:
        summary.plot(kind='bar', ax=ax3, color=[colors['Non-Eelgrass Site'], colors['Eelgrass Site']],
                     edgecolor='black')
        ax3.set_xlabel('Basin')
        ax3.set_ylabel('Mean Encounter Rate')
        ax3.set_title('Mean Encounter Rate by Basin and Site Type')
        ax3.legend(title='')
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Scatter: Basin mean vs Eelgrass effect
    ax4 = axes[1, 1]
    
    basin_means = site_data.groupby('Basin')['MeanEncounterRate'].mean()
    basin_effects = {}
    for basin in site_data['Basin'].unique():
        basin_df = site_data[site_data['Basin'] == basin]
        eg = basin_df[basin_df['HasEelgrass'] == True]['MeanEncounterRate'].mean()
        noeg = basin_df[basin_df['HasEelgrass'] == False]['MeanEncounterRate'].mean()
        if not np.isnan(eg) and not np.isnan(noeg):
            basin_effects[basin] = eg - noeg
    
    if basin_effects:
        plot_basins = list(basin_effects.keys())
        x = [basin_means[b] for b in plot_basins]
        y = [basin_effects[b] for b in plot_basins]
        
        ax4.scatter(x, y, s=100, c='steelblue', edgecolor='black', alpha=0.7)
        for i, basin in enumerate(plot_basins):
            ax4.annotate(basin[:15], (x[i], y[i]), fontsize=8, alpha=0.8)
        
        ax4.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Basin Mean Encounter Rate')
        ax4.set_ylabel('Eelgrass Effect\n(Eelgrass - Non-Eelgrass)')
        ax4.set_title('Basin Abundance vs Eelgrass Effect')
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, 'basin_eelgrass_comparison')
    plt.close()


def plot_partial_correlation(site_data: pd.DataFrame):
    """
    Visualize partial correlation analysis.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Filter to valid basins
    basin_counts = site_data.groupby('Basin')['HasEelgrass'].agg(['sum', lambda x: (~x).sum()])
    valid_basins = basin_counts[(basin_counts.iloc[:, 0] >= 1) & (basin_counts.iloc[:, 1] >= 1)].index
    site_data_valid = site_data[site_data['Basin'].isin(valid_basins)].copy()
    
    if len(site_data_valid) < 10:
        plt.close()
        return
    
    # 1. Raw relationship
    ax1 = axes[0]
    colors = {'Eelgrass Site': '#2ecc71', 'Non-Eelgrass Site': '#e74c3c'}
    
    for label, color in colors.items():
        subset = site_data_valid[site_data_valid['EelgrassLabel'] == label]
        jitter = np.random.normal(0, 0.1, len(subset))
        x = (1 if label == 'Eelgrass Site' else 0) + jitter
        ax1.scatter(x, subset['MeanEncounterRate'], c=color, s=60, alpha=0.6, 
                   edgecolor='black', label=label)
    
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Non-Eelgrass', 'Eelgrass'])
    ax1.set_ylabel('Mean Encounter Rate')
    ax1.set_title('Raw Relationship')
    ax1.legend()
    
    # 2. Residualized relationship (controlling for basin)
    ax2 = axes[1]
    
    # Residualize on basin
    basin_dummies = pd.get_dummies(site_data_valid['Basin'], drop_first=True)
    from scipy.linalg import lstsq
    X_basin = np.column_stack([np.ones(len(site_data_valid)), basin_dummies.values])
    
    y_encounter = site_data_valid['MeanEncounterRate'].values
    coef, _, _, _ = lstsq(X_basin, y_encounter)
    resid_encounter = y_encounter - X_basin @ coef
    
    y_eelgrass = site_data_valid['HasEelgrass'].astype(int).values
    coef_eg, _, _, _ = lstsq(X_basin, y_eelgrass)
    resid_eelgrass = y_eelgrass - X_basin @ coef_eg
    
    site_data_valid = site_data_valid.copy()
    site_data_valid['ResidEncounter'] = resid_encounter
    site_data_valid['ResidEelgrass'] = resid_eelgrass
    
    for label, color in colors.items():
        subset = site_data_valid[site_data_valid['EelgrassLabel'] == label]
        ax2.scatter(subset['ResidEelgrass'], subset['ResidEncounter'], 
                   c=color, s=60, alpha=0.6, edgecolor='black', label=label)
    
    # Add regression line
    slope, intercept, r, p, se = stats.linregress(resid_eelgrass, resid_encounter)
    x_line = np.array([resid_eelgrass.min(), resid_eelgrass.max()])
    ax2.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2)
    
    ax2.set_xlabel('Eelgrass (residualized on Basin)')
    ax2.set_ylabel('Encounter Rate (residualized on Basin)')
    ax2.set_title(f'Partial Correlation\nr = {r:.3f}, p = {p:.3f}')
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(0, color='gray', linestyle=':', alpha=0.5)
    
    # 3. Detection probability by eelgrass and basin
    ax3 = axes[2]
    
    detection_summary = site_data_valid.groupby(['Basin', 'HasEelgrass'])['HasDetection'].mean().unstack()
    if detection_summary.shape[1] == 2:
        detection_summary.columns = ['Non-Eelgrass', 'Eelgrass']
        detection_summary.plot(kind='bar', ax=ax3, 
                               color=[colors['Non-Eelgrass Site'], colors['Eelgrass Site']],
                               edgecolor='black')
        ax3.set_xlabel('Basin')
        ax3.set_ylabel('Detection Probability')
        ax3.set_title('Detection Rate by Basin and Eelgrass')
        ax3.set_ylim(0, 1)
        ax3.legend(title='')
        ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, 'partial_correlation_analysis')
    plt.close()


def generate_summary_tables(site_data: pd.DataFrame, results: dict):
    """
    Generate summary tables.
    """
    # Site-level data
    site_data.to_csv(OUTPUT_DIR / 'site_data_basin_eelgrass.csv', index=False)
    print(f"\n  Saved: {OUTPUT_DIR / 'site_data_basin_eelgrass.csv'}")
    
    # Summary by basin and eelgrass
    summary = site_data.groupby(['Basin', 'EelgrassLabel']).agg({
        'MeanEncounterRate': ['mean', 'std', 'count'],
        'HasDetection': ['sum', 'mean']
    }).round(4)
    summary.to_csv(OUTPUT_DIR / 'summary_by_basin_eelgrass.csv')
    print(f"  Saved: {OUTPUT_DIR / 'summary_by_basin_eelgrass.csv'}")
    
    # Stratified results
    if 'stratified' in results:
        strat_df = pd.DataFrame(results['stratified']).T
        strat_df.to_csv(OUTPUT_DIR / 'stratified_results.csv')
        print(f"  Saved: {OUTPUT_DIR / 'stratified_results.csv'}")


def main():
    print(f"\n{'='*60}")
    print("EELGRASS & ENCOUNTER RATE: BASIN-CONTROLLED ANALYSIS")
    print(f"{'='*60}\n")
    
    set_style()
    
    # Load data
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} survey records from {df['SiteName'].nunique()} sites")
    print(f"  Basins: {df['Basin'].nunique()}")
    
    # Prepare site-level data
    print("\nPreparing site-level data...")
    site_data = prepare_site_data(df)
    
    n_eelgrass = site_data['HasEelgrass'].sum()
    n_no_eelgrass = (~site_data['HasEelgrass']).sum()
    print(f"  Eelgrass sites: {n_eelgrass}")
    print(f"  Non-eelgrass sites: {n_no_eelgrass}")
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Run analyses
    results = {}
    
    cross_tab = analyze_basin_distribution(site_data)
    results['stratified'] = run_stratified_analysis(site_data)
    results['anova'] = run_anova_analysis(site_data)
    results['detection'] = run_detection_analysis(site_data)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    print("\nPlotting basin comparisons...")
    plot_basin_comparison(site_data)
    
    print("Plotting partial correlation analysis...")
    plot_partial_correlation(site_data)
    
    # Save tables
    print("\nGenerating summary tables...")
    generate_summary_tables(site_data, results)
    
    # Final summary
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    anova = results.get('anova', {})
    strat = results.get('stratified', {})
    
    # Count significant basins
    basins_tested = [b for b, r in strat.items() if r.get('sufficient_data', False)]
    sig_basins = [b for b in basins_tested if strat[b].get('u_pval', 1) < 0.05]
    positive_basins = [b for b in basins_tested if strat[b].get('difference', 0) > 0]
    
    print(f"""
═══════════════════════════════════════════════════════════
STRATIFIED ANALYSIS (Within-Basin Comparisons)
═══════════════════════════════════════════════════════════
  • Basins with sufficient data: {len(basins_tested)}
  • Basins with significant eelgrass effect (p<0.05): {len(sig_basins)}
  • Basins where eelgrass sites have HIGHER rates: {len(positive_basins)}/{len(basins_tested)}

═══════════════════════════════════════════════════════════
PARTIAL CORRELATION (Controlling for Basin)
═══════════════════════════════════════════════════════════
  • Partial r: {anova.get('partial_correlation', {}).get('r', 'N/A'):.4f}
  • p-value: {anova.get('partial_correlation', {}).get('p', 'N/A'):.4f}
  • Δ R² (eelgrass after basin): {anova.get('r2_eelgrass_given_basin', 'N/A'):.4f}

═══════════════════════════════════════════════════════════
CONCLUSION
═══════════════════════════════════════════════════════════
""")
    
    partial_p = anova.get('partial_correlation', {}).get('p', 1)
    partial_r = anova.get('partial_correlation', {}).get('r', 0)
    
    if partial_p < 0.05:
        direction = "HIGHER" if partial_r > 0 else "LOWER"
        print(f"✓ SIGNIFICANT: After controlling for basin, eelgrass sites have")
        print(f"   significantly {direction} Pycnopodia encounter rates")
        print(f"   (partial r = {partial_r:.3f}, p = {partial_p:.4f})")
    elif partial_p < 0.1:
        direction = "higher" if partial_r > 0 else "lower"
        print(f"~ MARGINAL: Trend for eelgrass sites to have {direction} rates")
        print(f"   after controlling for basin (p = {partial_p:.4f})")
    else:
        print(f"✗ NOT SIGNIFICANT: After controlling for basin, eelgrass presence")
        print(f"   does not significantly predict encounter rate (p = {partial_p:.4f})")
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()






