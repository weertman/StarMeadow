#!/usr/bin/env python3
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path('/home/weertman/Documents/StarMeadow')
OUT = ROOT / 'outputs' / 'publication_figures'
STATS = OUT / 'stats'
FIGDIR = OUT / 'sources'
SUBDIR = OUT / 'submission'
QCDIR = OUT / 'qc'
for d in (FIGDIR, SUBDIR, QCDIR):
    d.mkdir(parents=True, exist_ok=True)

COUNT_MODELS = STATS / 'diver_count_model_results.csv'
LENGTH_TESTS = STATS / 'diver_length_site_eelgrass_distribution_shape_tests.csv'

BLUE = '#2f6fbb'
GREEN = '#2a9d55'
ORANGE = '#c76f2a'
GRAY = '#777777'
LIGHT_GRAY = '#d6d6d6'
DARK = '#222222'


def clean_model_name(name: str) -> str:
    return {
        'M1_eelgrass_only': 'NB: effort only',
        'M2_eelgrass_plus_habitat': 'NB: + habitat',
        'M3_eelgrass_plus_basin': 'NB: + basin',
        'M4_eelgrass_plus_basin_plus_habitat': 'NB: + basin + habitat',
        'P1_eelgrass_only': 'Poisson: effort only',
        'P1_eelgrass_only_poisson_sensitivity': 'Poisson: effort only',
        'P2_eelgrass_plus_habitat_poisson_sensitivity': 'Poisson: + habitat',
        'P3_eelgrass_plus_basin_poisson_sensitivity': 'Poisson: + basin',
        'P4_eelgrass_plus_basin_plus_habitat_poisson_sensitivity': 'Poisson: + basin + habitat',
    }.get(name, name)


def clean_term(term: str) -> str:
    if term == 'SiteHasEelgrassInt':
        return 'Eelgrass recorded at site'
    m = re.match(r'C\(Basin(?:,.*)?\)\[T\.(.*)\]', term)
    if m:
        return f"Basin: {m.group(1)}"
    m = re.match(r'C\(HabitatType(?:,.*)?\)\[T\.(.*)\]', term)
    if m:
        return f"Habitat: {m.group(1)}"
    return term


def p_label(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return 'p<0.001'
    return f'p={p:.3f}'


def forest(ax, df, label_col, point_col, low_col, high_col, color_col=None, primary_mask=None, xlim=None):
    y = np.arange(len(df))[::-1]
    colors = df[color_col].tolist() if color_col else [BLUE] * len(df)
    for yi, (_, row), color in zip(y, df.iterrows(), colors):
        is_primary = bool(primary_mask.loc[row.name]) if primary_mask is not None else False
        lw = 2.6 if is_primary else 1.8
        ms = 6.5 if is_primary else 5.2
        ax.plot([row[low_col], row[high_col]], [yi, yi], color=color, lw=lw, solid_capstyle='round')
        ax.scatter(row[point_col], yi, color=color, s=ms**2, zorder=3, edgecolor='white', linewidth=0.7)
    ax.axvline(1, color='#444444', lw=1, ls='--', zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(df[label_col].tolist())
    ax.set_xscale('log')
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis='x', color=LIGHT_GRAY, lw=0.6, alpha=0.8)
    ax.tick_params(axis='both', labelsize=8.5)
    ax.spines[['top', 'right']].set_visible(False)


count = pd.read_csv(COUNT_MODELS)
site = count[count['term'].eq('SiteHasEelgrassInt')].copy()
model_order = [
    'M1_eelgrass_only', 'M2_eelgrass_plus_habitat', 'M3_eelgrass_plus_basin', 'M4_eelgrass_plus_basin_plus_habitat',
    'P1_eelgrass_only_poisson_sensitivity', 'P2_eelgrass_plus_habitat_poisson_sensitivity', 'P3_eelgrass_plus_basin_poisson_sensitivity', 'P4_eelgrass_plus_basin_plus_habitat_poisson_sensitivity'
]
site['order'] = site['model_name'].map({m: i for i, m in enumerate(model_order)})
site = site.sort_values('order')
site['label'] = site['model_name'].map(clean_model_name)
site['color'] = np.where(site['family'].str.contains('NegativeBinomial'), BLUE, ORANGE)
site_primary = site['model_name'].eq('M4_eelgrass_plus_basin_plus_habitat')

m4 = count[count['model_name'].eq('M4_eelgrass_plus_basin_plus_habitat') & ~count['term'].isin(['Intercept', 'alpha'])].copy()
m4 = m4[~m4['term'].str.contains('Sponge Garden', na=False)].copy()
m4['label'] = m4['term'].map(clean_term)
m4['group'] = np.select(
    [m4['term'].eq('SiteHasEelgrassInt'), m4['term'].str.contains('C\\(Basin', regex=True), m4['term'].str.contains('C\\(HabitatType', regex=True)],
    ['Focal eelgrass term', 'Basin covariate', 'Habitat covariate'],
    default='Other'
)
m4['group_order'] = m4['group'].map({'Focal eelgrass term': 0, 'Basin covariate': 1, 'Habitat covariate': 2}).fillna(9)
m4 = m4.sort_values(['group_order', 'irr'], ascending=[True, False])
m4['color'] = m4['group'].map({'Focal eelgrass term': GREEN, 'Basin covariate': BLUE, 'Habitat covariate': GRAY}).fillna(GRAY)
m4_primary = m4['term'].eq('SiteHasEelgrassInt')

length = pd.read_csv(LENGTH_TESTS)
length = length.copy()
length['label'] = length['comparison'].replace({
    'Soft Bottom only': 'Soft Bottom: eelgrass vs no eelgrass',
    'Natural Reef only': 'Natural Reef: eelgrass vs no eelgrass',
    'Overall positive length distribution': 'Overall: eelgrass vs no eelgrass',
    'Artificial Reef only': 'Artificial Reef: eelgrass vs no eelgrass',
    'Kelp Forest only': 'Kelp Forest: eelgrass vs no eelgrass',
})
length['support'] = np.where(length['sparse_flag'], 'Sparse/descriptive', np.where(length['ks_site_permutation_fdr_q'] < 0.05, 'Site-aware support', 'Not significant'))
length['color'] = length['support'].map({'Site-aware support': GREEN, 'Not significant': BLUE, 'Sparse/descriptive': GRAY})
length = length.sort_values(['sparse_flag', 'ks_d'], ascending=[True, False])

fig = plt.figure(figsize=(10.2, 11.0), constrained_layout=False)
gs = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.65, 1.0], hspace=0.45)

ax1 = fig.add_subplot(gs[0, 0])
forest(ax1, site, 'label', 'irr', 'irr_ci_low', 'irr_ci_high', 'color', site_primary, xlim=(0.8, 30))
ax1.set_title('A. Eelgrass-at-site association across count models', loc='left', fontsize=12, fontweight='bold')
ax1.set_xlabel('Incidence-rate ratio for eelgrass-recorded sites (log scale)', fontsize=9.5)
for y, (_, row) in zip(np.arange(len(site))[::-1], site.iterrows()):
    txt = f"{row['irr']:.2f} ({row['irr_ci_low']:.2f}–{row['irr_ci_high']:.2f}); {p_label(row['p_value'])}"
    ax1.text(31, y, txt, va='center', ha='left', fontsize=7.7, clip_on=False)
ax1.text(31, len(site)-0.35, 'IRR (95% CI); p', ha='left', va='bottom', fontsize=8, fontweight='bold', clip_on=False)

ax2 = fig.add_subplot(gs[1, 0])
forest(ax2, m4, 'label', 'irr', 'irr_ci_low', 'irr_ci_high', 'color', m4_primary, xlim=(0.03, 180))
ax2.set_title('B. Primary adjusted negative-binomial model terms', loc='left', fontsize=12, fontweight='bold')
ax2.set_xlabel('Incidence-rate ratio (log scale; reference = Central basin / Natural Reef / no eelgrass)', fontsize=9.5)
for y, (_, row) in zip(np.arange(len(m4))[::-1], m4.iterrows()):
    txt = f"{row['irr']:.2f} ({row['irr_ci_low']:.2f}–{row['irr_ci_high']:.2f})"
    ax2.text(190, y, txt, va='center', ha='left', fontsize=7.2, clip_on=False)
ax2.text(190, len(m4)-0.25, 'IRR (95% CI)', ha='left', va='bottom', fontsize=8, fontweight='bold', clip_on=False)

ax3 = fig.add_subplot(gs[2, 0])
y3 = np.arange(len(length))[::-1]
for yi, (_, row) in zip(y3, length.iterrows()):
    ax3.barh(yi, row['ks_d'], color=row['color'], alpha=0.88, edgecolor='white')
    label = f"KS D={row['ks_d']:.3f}; q={row['ks_site_permutation_fdr_q']:.3f}"
    if row['sparse_flag']:
        label += '; sparse'
    ax3.text(row['ks_d'] + 0.015, yi, label, va='center', ha='left', fontsize=7.5)
ax3.set_yticks(y3)
ax3.set_yticklabels(length['label'].tolist())
ax3.set_xlim(0, 0.78)
ax3.set_xlabel('Kolmogorov-Smirnov distance between length distributions', fontsize=9.5)
ax3.set_title('C. Length distribution-shape contrasts by site eelgrass status', loc='left', fontsize=12, fontweight='bold')
ax3.grid(axis='x', color=LIGHT_GRAY, lw=0.6, alpha=0.8)
ax3.spines[['top', 'right']].set_visible(False)
ax3.tick_params(axis='both', labelsize=8.5)

legend = [
    Line2D([0], [0], marker='o', color='none', markerfacecolor=GREEN, markeredgecolor='white', markersize=7, label='Focal eelgrass term / supported length contrast'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=BLUE, markeredgecolor='white', markersize=7, label='Negative-binomial model terms / non-significant length contrast'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=ORANGE, markeredgecolor='white', markersize=7, label='Poisson sensitivity models'),
    Line2D([0], [0], marker='o', color='none', markerfacecolor=GRAY, markeredgecolor='white', markersize=7, label='Habitat covariate / sparse length contrast'),
]
fig.legend(handles=legend, loc='lower center', ncol=2, fontsize=8.2, frameon=False, bbox_to_anchor=(0.5, 0.01))

fig.suptitle('Supplemental statistical summary of survey analyses', fontsize=14, fontweight='bold', y=0.985)
fig.text(0.01, 0.004, 'Dashed forest-plot lines indicate no association (IRR = 1). Primary count model is NB + basin + habitat; sparse Sponge Garden term omitted from Panel B. Length p/q values use site-label permutation tests.', fontsize=7.5, color=DARK)
fig.subplots_adjust(left=0.28, right=0.78, top=0.94, bottom=0.08)

png = FIGDIR / 'S5_survey_statistical_summary.png'
pdf = FIGDIR / 'S5_survey_statistical_summary.pdf'
tif = SUBDIR / 'S5_Fig.tif'
fig.savefig(png, dpi=300, bbox_inches='tight')
fig.savefig(pdf, bbox_inches='tight')
fig.savefig(tif, dpi=300, bbox_inches='tight', pil_kwargs={'compression': 'tiff_lzw'})
plt.close(fig)

# QC outputs
site[['model_name', 'label', 'family', 'irr', 'irr_ci_low', 'irr_ci_high', 'p_value']].to_csv(QCDIR / 'S5_panel_A_count_model_eelgrass_forest.csv', index=False)
m4[['term', 'label', 'group', 'irr', 'irr_ci_low', 'irr_ci_high', 'p_value']].to_csv(QCDIR / 'S5_panel_B_primary_model_terms.csv', index=False)
length[['comparison', 'label', 'ks_d', 'ks_site_permutation_p', 'ks_site_permutation_fdr_q', 'sparse_flag']].to_csv(QCDIR / 'S5_panel_C_length_shape_contrasts.csv', index=False)
print(png)
print(pdf)
print(tif)
