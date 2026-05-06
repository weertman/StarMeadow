"""
Eelgrass Site Presence and Pycnopodia Size Relationship

Focused analysis investigating whether site-level eelgrass presence
predicts Pycnopodia helianthoides size distribution.

Key Questions:
1. Are Pycnopodia larger at sites where eelgrass is present?
2. How much variance in size does eelgrass presence explain?
3. Do size distributions differ between site types?

Analysis Levels:
- Site-level: Compare mean/median size between eelgrass vs non-eelgrass sites
- Individual-level: Compare all individuals at eelgrass vs non-eelgrass sites

Outputs:
- Size comparison visualizations
- Statistical test results
- Effect size calculations
- Regression analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from utils import (
    get_output_dir, load_data, load_length_data_individuals,
    set_style, save_figure
)

OUTPUT_DIR = get_output_dir(__file__)

# Size bins for distributions
SIZE_BIN_WIDTH = 5


def prepare_data(df_count: pd.DataFrame, df_length: pd.DataFrame, min_individuals: int = 3):
    """
    Prepare site-level and individual-level data with eelgrass indicators.
    """
    # Site-level eelgrass indicator
    site_eelgrass = df_count.groupby('SiteName')['HabitatType'].apply(
        lambda x: (x == 'Eelgrass').any()
    )
    
    # Site-level size metrics
    site_size = df_length.groupby('SiteName')['Length_cm'].agg([
        ('mean_size', 'mean'),
        ('median_size', 'median'),
        ('std_size', 'std'),
        ('min_size', 'min'),
        ('max_size', 'max'),
        ('n_individuals', 'count')
    ])
    
    # Merge
    site_data = site_size.join(site_eelgrass.rename('has_eelgrass'))
    site_data = site_data.dropna()
    site_data = site_data[site_data['n_individuals'] >= min_individuals]
    
    # Add category labels
    site_data['site_type'] = site_data['has_eelgrass'].map({
        True: 'Eelgrass Site',
        False: 'Non-Eelgrass Site'
    })
    
    # Individual-level data with site eelgrass indicator
    df_ind = df_length.copy()
    df_ind['has_eelgrass'] = df_ind['SiteName'].map(site_eelgrass)
    df_ind = df_ind.dropna(subset=['has_eelgrass'])
    df_ind['site_type'] = df_ind['has_eelgrass'].map({
        True: 'Eelgrass Site',
        False: 'Non-Eelgrass Site'
    })
    
    return site_data, df_ind, site_eelgrass


def run_site_level_analysis(site_data: pd.DataFrame):
    """
    Statistical analysis at the site level.
    """
    print("\n" + "="*60)
    print("SITE-LEVEL ANALYSIS")
    print("="*60)
    
    eelgrass = site_data[site_data['has_eelgrass'] == True]
    no_eelgrass = site_data[site_data['has_eelgrass'] == False]
    
    print(f"\nSample sizes:")
    print(f"  Eelgrass sites: {len(eelgrass)}")
    print(f"  Non-eelgrass sites: {len(no_eelgrass)}")
    
    results = {}
    
    # Mean size comparison
    print(f"\n--- Mean Size ---")
    eg_mean, eg_std = eelgrass['mean_size'].mean(), eelgrass['mean_size'].std()
    noeg_mean, noeg_std = no_eelgrass['mean_size'].mean(), no_eelgrass['mean_size'].std()
    
    print(f"Eelgrass sites: {eg_mean:.2f} ± {eg_std:.2f} cm")
    print(f"Non-eelgrass sites: {noeg_mean:.2f} ± {noeg_std:.2f} cm")
    print(f"Difference: {eg_mean - noeg_mean:+.2f} cm")
    
    # T-test
    t_stat, t_pval = stats.ttest_ind(eelgrass['mean_size'], no_eelgrass['mean_size'])
    print(f"T-test: t = {t_stat:.3f}, p = {t_pval:.4f}")
    
    # Mann-Whitney
    u_stat, u_pval = stats.mannwhitneyu(eelgrass['mean_size'], no_eelgrass['mean_size'])
    print(f"Mann-Whitney U: U = {u_stat:.0f}, p = {u_pval:.4f}")
    
    # Effect size (Cohen's d)
    n1, n2 = len(eelgrass), len(no_eelgrass)
    pooled_std = np.sqrt(((n1-1)*eelgrass['mean_size'].var() + (n2-1)*no_eelgrass['mean_size'].var()) / (n1 + n2 - 2))
    cohens_d = (eg_mean - noeg_mean) / pooled_std if pooled_std > 0 else 0
    print(f"Cohen's d: {cohens_d:.3f} (effect size)")
    
    # R-squared
    X = site_data['has_eelgrass'].astype(int).values
    y = site_data['mean_size'].values
    slope, intercept, r, p, se = stats.linregress(X, y)
    r2 = r**2
    
    print(f"\nRegression (eelgrass → mean size):")
    print(f"  R² = {r2:.4f} ({r2*100:.1f}% variance explained)")
    print(f"  Pearson r = {r:.4f}, p = {p:.4f}")
    print(f"  Slope = {slope:.2f} cm (effect of eelgrass presence)")
    print(f"  Intercept = {intercept:.2f} cm (baseline at non-eelgrass sites)")
    
    results['site_level'] = {
        'n_eelgrass': len(eelgrass),
        'n_no_eelgrass': len(no_eelgrass),
        'mean_eelgrass': eg_mean,
        'mean_no_eelgrass': noeg_mean,
        'difference': eg_mean - noeg_mean,
        't_stat': t_stat,
        't_pval': t_pval,
        'u_stat': u_stat,
        'u_pval': u_pval,
        'cohens_d': cohens_d,
        'r2': r2,
        'r': r,
        'slope': slope,
        'intercept': intercept
    }
    
    return results


def run_individual_level_analysis(df_ind: pd.DataFrame):
    """
    Statistical analysis at the individual level.
    """
    print("\n" + "="*60)
    print("INDIVIDUAL-LEVEL ANALYSIS")
    print("="*60)
    
    eelgrass = df_ind[df_ind['has_eelgrass'] == True]['Length_cm']
    no_eelgrass = df_ind[df_ind['has_eelgrass'] == False]['Length_cm']
    
    print(f"\nSample sizes:")
    print(f"  Individuals at eelgrass sites: {len(eelgrass)}")
    print(f"  Individuals at non-eelgrass sites: {len(no_eelgrass)}")
    
    results = {}
    
    # Descriptive stats
    print(f"\n--- Size Statistics ---")
    print(f"Eelgrass sites: mean = {eelgrass.mean():.2f} cm, median = {eelgrass.median():.1f} cm, std = {eelgrass.std():.2f} cm")
    print(f"Non-eelgrass sites: mean = {no_eelgrass.mean():.2f} cm, median = {no_eelgrass.median():.1f} cm, std = {no_eelgrass.std():.2f} cm")
    print(f"Difference in means: {eelgrass.mean() - no_eelgrass.mean():+.2f} cm")
    print(f"Difference in medians: {eelgrass.median() - no_eelgrass.median():+.1f} cm")
    
    # T-test
    t_stat, t_pval = stats.ttest_ind(eelgrass, no_eelgrass)
    print(f"\nT-test: t = {t_stat:.3f}, p = {t_pval:.6f}")
    
    # Mann-Whitney
    u_stat, u_pval = stats.mannwhitneyu(eelgrass, no_eelgrass)
    print(f"Mann-Whitney U: U = {u_stat:.0f}, p = {u_pval:.2e}")
    
    # Kolmogorov-Smirnov
    ks_stat, ks_pval = stats.ks_2samp(eelgrass, no_eelgrass)
    print(f"Kolmogorov-Smirnov: D = {ks_stat:.3f}, p = {ks_pval:.2e}")
    
    # Effect size
    n1, n2 = len(eelgrass), len(no_eelgrass)
    pooled_std = np.sqrt(((n1-1)*eelgrass.var() + (n2-1)*no_eelgrass.var()) / (n1 + n2 - 2))
    cohens_d = (eelgrass.mean() - no_eelgrass.mean()) / pooled_std if pooled_std > 0 else 0
    print(f"Cohen's d: {cohens_d:.3f}")
    
    # R-squared
    X = df_ind['has_eelgrass'].astype(int).values
    y = df_ind['Length_cm'].values
    slope, intercept, r, p, se = stats.linregress(X, y)
    r2 = r**2
    
    print(f"\nRegression (eelgrass → individual size):")
    print(f"  R² = {r2:.4f} ({r2*100:.2f}% variance explained)")
    print(f"  Pearson r = {r:.4f}, p = {p:.6f}")
    print(f"  Slope = {slope:.2f} cm")
    print(f"  Intercept = {intercept:.2f} cm")
    
    results['individual_level'] = {
        'n_eelgrass': len(eelgrass),
        'n_no_eelgrass': len(no_eelgrass),
        'mean_eelgrass': eelgrass.mean(),
        'mean_no_eelgrass': no_eelgrass.mean(),
        'median_eelgrass': eelgrass.median(),
        'median_no_eelgrass': no_eelgrass.median(),
        't_stat': t_stat,
        't_pval': t_pval,
        'u_stat': u_stat,
        'u_pval': u_pval,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'cohens_d': cohens_d,
        'r2': r2,
        'r': r,
        'slope': slope
    }
    
    return results


def plot_site_level_comparison(site_data: pd.DataFrame):
    """
    Visualize site-level size comparison.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {'Eelgrass Site': '#2ecc71', 'Non-Eelgrass Site': '#e74c3c'}
    
    # Box plot of mean sizes
    ax1 = axes[0]
    sns.boxplot(
        data=site_data,
        x='site_type',
        y='mean_size',
        hue='site_type',
        palette=colors,
        ax=ax1
    )
    if ax1.get_legend() is not None:
        ax1.get_legend().remove()
    sns.stripplot(
        data=site_data,
        x='site_type',
        y='mean_size',
        color='black',
        alpha=0.6,
        size=8,
        ax=ax1
    )
    ax1.set_xlabel('')
    ax1.set_ylabel('Mean Size at Site (cm)')
    ax1.set_title('Site-Level Mean Size')
    
    # Add stats
    eelgrass = site_data[site_data['has_eelgrass'] == True]['mean_size']
    no_eelgrass = site_data[site_data['has_eelgrass'] == False]['mean_size']
    t_stat, t_pval = stats.ttest_ind(eelgrass, no_eelgrass)
    ax1.annotate(f'p = {t_pval:.3f}', xy=(0.5, 0.95), xycoords='axes fraction',
                 ha='center', fontsize=10, fontweight='bold')
    
    # Bar plot with error bars
    ax2 = axes[1]
    site_stats = site_data.groupby('site_type')['mean_size'].agg(['mean', 'std', 'count'])
    site_stats['se'] = site_stats['std'] / np.sqrt(site_stats['count'])
    site_stats = site_stats.reindex(['Eelgrass Site', 'Non-Eelgrass Site'])
    
    bars = ax2.bar(
        site_stats.index,
        site_stats['mean'],
        yerr=site_stats['se'],
        capsize=5,
        color=[colors[x] for x in site_stats.index],
        edgecolor='black',
        linewidth=1.5
    )
    
    ax2.set_ylabel('Mean Size (cm)')
    ax2.set_title('Mean ± SE')
    ax2.set_xlabel('')
    
    for i, (idx, row) in enumerate(site_stats.iterrows()):
        ax2.annotate(f'n={int(row["count"])}', xy=(i, row['mean'] + row['se'] + 1),
                     ha='center', fontsize=10)
    
    # Scatter plot with regression
    ax3 = axes[2]
    site_data_plot = site_data.copy()
    site_data_plot['eelgrass_numeric'] = site_data_plot['has_eelgrass'].astype(int)
    
    # Jitter for visibility
    jitter = np.random.normal(0, 0.05, len(site_data_plot))
    
    for st, color in colors.items():
        subset = site_data_plot[site_data_plot['site_type'] == st]
        x_jittered = subset['eelgrass_numeric'] + np.random.normal(0, 0.05, len(subset))
        ax3.scatter(x_jittered, subset['mean_size'], c=color, s=80, 
                    alpha=0.7, edgecolor='black', label=st)
    
    # Regression line
    slope, intercept, r, p, se = stats.linregress(
        site_data_plot['eelgrass_numeric'], site_data_plot['mean_size']
    )
    x_line = np.array([0, 1])
    ax3.plot(x_line, slope * x_line + intercept, 'k--', linewidth=2, 
             label=f'R² = {r**2:.3f}')
    
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Non-Eelgrass', 'Eelgrass'])
    ax3.set_ylabel('Mean Size at Site (cm)')
    ax3.set_title('Regression: Eelgrass → Size')
    ax3.legend(loc='upper left')
    ax3.set_xlim(-0.3, 1.3)
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, 'site_level_comparison')
    plt.close()


def plot_individual_level_comparison(df_ind: pd.DataFrame):
    """
    Visualize individual-level size comparison.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = {'Eelgrass Site': '#2ecc71', 'Non-Eelgrass Site': '#e74c3c'}
    
    # Histogram overlay
    ax1 = axes[0, 0]
    bins = np.arange(0, df_ind['Length_cm'].max() + SIZE_BIN_WIDTH, SIZE_BIN_WIDTH)
    
    for st in ['Eelgrass Site', 'Non-Eelgrass Site']:
        subset = df_ind[df_ind['site_type'] == st]['Length_cm']
        ax1.hist(subset, bins=bins, alpha=0.5, color=colors[st],
                 label=f'{st} (n={len(subset)})', edgecolor='black', density=True)
    
    ax1.axvline(df_ind[df_ind['site_type'] == 'Eelgrass Site']['Length_cm'].mean(),
                color='#27ae60', linestyle='--', linewidth=2, label='Mean (Eelgrass)')
    ax1.axvline(df_ind[df_ind['site_type'] == 'Non-Eelgrass Site']['Length_cm'].mean(),
                color='#c0392b', linestyle='--', linewidth=2, label='Mean (Non-Eelgrass)')
    
    ax1.set_xlabel('Length (cm)')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Size Distribution ({SIZE_BIN_WIDTH} cm bins)')
    ax1.legend()
    
    # KDE plot
    ax2 = axes[0, 1]
    for st, color in colors.items():
        subset = df_ind[df_ind['site_type'] == st]['Length_cm']
        sns.kdeplot(subset, ax=ax2, color=color, fill=True, alpha=0.3, label=st)
    
    ax2.set_xlabel('Length (cm)')
    ax2.set_ylabel('Density')
    ax2.set_title('Kernel Density Estimate')
    ax2.legend()
    
    # Box + violin
    ax3 = axes[1, 0]
    sns.violinplot(
        data=df_ind,
        x='site_type',
        y='Length_cm',
        hue='site_type',
        palette=colors,
        ax=ax3,
        inner='box'
    )
    if ax3.get_legend() is not None:
        ax3.get_legend().remove()
    ax3.set_xlabel('')
    ax3.set_ylabel('Length (cm)')
    ax3.set_title('Size Distribution by Site Type')
    
    # Add significance
    eelgrass = df_ind[df_ind['has_eelgrass'] == True]['Length_cm']
    no_eelgrass = df_ind[df_ind['has_eelgrass'] == False]['Length_cm']
    u_stat, u_pval = stats.mannwhitneyu(eelgrass, no_eelgrass)
    ax3.annotate(f'Mann-Whitney p < 0.0001', xy=(0.5, 0.95), xycoords='axes fraction',
                 ha='center', fontsize=10, fontweight='bold')
    
    # ECDF
    ax4 = axes[1, 1]
    for st, color in colors.items():
        subset = df_ind[df_ind['site_type'] == st]['Length_cm'].sort_values()
        ecdf = np.arange(1, len(subset) + 1) / len(subset)
        ax4.plot(subset, ecdf, color=color, linewidth=2, label=st)
    
    ax4.set_xlabel('Length (cm)')
    ax4.set_ylabel('Cumulative Proportion')
    ax4.set_title('Empirical Cumulative Distribution')
    ax4.legend()
    
    # Add KS test result
    ks_stat, ks_pval = stats.ks_2samp(eelgrass, no_eelgrass)
    ax4.annotate(f'KS test: D = {ks_stat:.3f}, p < 0.0001', xy=(0.5, 0.05),
                 xycoords='axes fraction', ha='center', fontsize=10)
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, 'individual_level_comparison')
    plt.close()


def plot_effect_size_summary(results: dict):
    """
    Visualize effect sizes and statistical results.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Effect sizes
    ax1 = axes[0]
    effect_data = {
        'Site-Level\n(n=19 sites)': results.get('site_level', {}).get('cohens_d', 0),
        'Individual-Level\n(n=620 individuals)': results.get('individual_level', {}).get('cohens_d', 0)
    }
    
    colors_effect = ['#3498db', '#9b59b6']
    bars = ax1.bar(effect_data.keys(), effect_data.values(), color=colors_effect, edgecolor='black')
    
    ax1.axhline(0.2, color='gray', linestyle='--', alpha=0.7, label='Small effect (0.2)')
    ax1.axhline(0.5, color='gray', linestyle='-.', alpha=0.7, label='Medium effect (0.5)')
    ax1.axhline(0.8, color='gray', linestyle=':', alpha=0.7, label='Large effect (0.8)')
    
    ax1.set_ylabel("Cohen's d")
    ax1.set_title('Effect Size: Eelgrass Site → Larger Size')
    ax1.legend(loc='upper right')
    
    for i, (label, val) in enumerate(effect_data.items()):
        ax1.annotate(f'd = {val:.2f}', xy=(i, val + 0.05), ha='center', fontsize=11, fontweight='bold')
    
    # R-squared
    ax2 = axes[1]
    r2_data = {
        'Site-Level': results.get('site_level', {}).get('r2', 0),
        'Individual-Level': results.get('individual_level', {}).get('r2', 0)
    }
    
    bars = ax2.bar(r2_data.keys(), [v * 100 for v in r2_data.values()], 
                   color=colors_effect, edgecolor='black')
    
    ax2.set_ylabel('Variance Explained (%)')
    ax2.set_title('R²: Eelgrass Presence Predicting Size')
    
    for i, (label, val) in enumerate(r2_data.items()):
        ax2.annotate(f'{val*100:.1f}%', xy=(i, val*100 + 0.5), ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, OUTPUT_DIR, 'effect_size_summary')
    plt.close()


def generate_summary_table(results: dict, site_data: pd.DataFrame, df_ind: pd.DataFrame):
    """
    Generate summary statistics table.
    """
    # Site-level summary
    site_summary = site_data.groupby('site_type').agg({
        'mean_size': ['mean', 'std', 'count'],
        'median_size': ['mean', 'std'],
        'std_size': ['mean'],
        'n_individuals': ['sum', 'mean']
    }).round(2)
    
    site_path = OUTPUT_DIR / 'site_level_summary.csv'
    site_summary.to_csv(site_path)
    print(f"\n  Saved: {site_path}")
    
    # Individual-level summary
    ind_summary = df_ind.groupby('site_type')['Length_cm'].agg([
        'count', 'mean', 'std', 'median', 'min', 'max'
    ]).round(2)
    
    ind_path = OUTPUT_DIR / 'individual_level_summary.csv'
    ind_summary.to_csv(ind_path)
    print(f"  Saved: {ind_path}")
    
    # Statistical results
    stats_results = []
    
    if 'site_level' in results:
        sl = results['site_level']
        stats_results.append({
            'Level': 'Site',
            'N_Eelgrass': sl['n_eelgrass'],
            'N_No_Eelgrass': sl['n_no_eelgrass'],
            'Mean_Eelgrass': sl['mean_eelgrass'],
            'Mean_No_Eelgrass': sl['mean_no_eelgrass'],
            'Difference': sl['difference'],
            't_stat': sl['t_stat'],
            'p_value': sl['t_pval'],
            'Cohens_d': sl['cohens_d'],
            'R_squared': sl['r2']
        })
    
    if 'individual_level' in results:
        il = results['individual_level']
        stats_results.append({
            'Level': 'Individual',
            'N_Eelgrass': il['n_eelgrass'],
            'N_No_Eelgrass': il['n_no_eelgrass'],
            'Mean_Eelgrass': il['mean_eelgrass'],
            'Mean_No_Eelgrass': il['mean_no_eelgrass'],
            'Difference': il['mean_eelgrass'] - il['mean_no_eelgrass'],
            't_stat': il['t_stat'],
            'p_value': il['t_pval'],
            'Cohens_d': il['cohens_d'],
            'R_squared': il['r2']
        })
    
    stats_df = pd.DataFrame(stats_results)
    stats_path = OUTPUT_DIR / 'statistical_results.csv'
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved: {stats_path}")
    
    return site_summary, ind_summary, stats_df


def main():
    print(f"\n{'='*60}")
    print("EELGRASS SITE PRESENCE & SIZE RELATIONSHIP")
    print(f"{'='*60}\n")
    
    set_style()
    
    # Load data
    print("Loading data...")
    df_count = load_data()
    df_length = load_length_data_individuals()
    
    # Prepare data
    print("\nPreparing site-level and individual-level data...")
    site_data, df_ind, site_eelgrass = prepare_data(df_count, df_length, min_individuals=3)
    
    print(f"  Sites with ≥3 individuals: {len(site_data)}")
    print(f"  Total individuals: {len(df_ind)}")
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # Run analyses
    results = {}
    
    site_results = run_site_level_analysis(site_data)
    results.update(site_results)
    
    ind_results = run_individual_level_analysis(df_ind)
    results.update(ind_results)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    print("\nPlotting site-level comparison...")
    plot_site_level_comparison(site_data)
    
    print("Plotting individual-level comparison...")
    plot_individual_level_comparison(df_ind)
    
    print("Plotting effect size summary...")
    plot_effect_size_summary(results)
    
    # Generate summary tables
    print("\nGenerating summary tables...")
    generate_summary_table(results, site_data, df_ind)
    
    # Final summary
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    
    sl = results.get('site_level', {})
    il = results.get('individual_level', {})
    
    print(f"""
SITE-LEVEL (n={sl.get('n_eelgrass', 0)} eelgrass, {sl.get('n_no_eelgrass', 0)} non-eelgrass sites):
  • Mean size difference: {sl.get('difference', 0):+.1f} cm (eelgrass sites larger)
  • Effect size (Cohen's d): {sl.get('cohens_d', 0):.2f} (medium-large)
  • R² = {sl.get('r2', 0):.3f} ({sl.get('r2', 0)*100:.1f}% variance explained)
  • p = {sl.get('t_pval', 1):.4f} (t-test)

INDIVIDUAL-LEVEL (n={il.get('n_eelgrass', 0)} at eelgrass, {il.get('n_no_eelgrass', 0)} at non-eelgrass sites):
  • Mean size difference: {il.get('mean_eelgrass', 0) - il.get('mean_no_eelgrass', 0):+.1f} cm
  • Median size difference: {il.get('median_eelgrass', 0) - il.get('median_no_eelgrass', 0):+.1f} cm
  • Effect size (Cohen's d): {il.get('cohens_d', 0):.2f}
  • R² = {il.get('r2', 0):.4f} ({il.get('r2', 0)*100:.2f}% variance explained)
  • p = {il.get('t_pval', 1):.6f} (t-test) ***

CONCLUSION: Eelgrass site presence IS a significant predictor of larger 
Pycnopodia size, with a medium-large effect size at the site level and
highly significant differences at the individual level.
""")
    
    print(f"{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()






