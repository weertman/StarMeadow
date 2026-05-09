#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path('/home/weertman/Documents/StarMeadow')
STATS = ROOT / 'outputs' / 'publication_figures' / 'stats'
SHORE = ROOT / 'outputs' / 'publication_figures' / 'qc' / 'shorezone_report_stats'
OUT = ROOT / 'outputs' / 'publication_figures' / 'supplemental_tables'
OUT.mkdir(parents=True, exist_ok=True)


def fmt_num(x, digits=2):
    if pd.isna(x):
        return ''
    return f"{float(x):.{digits}f}"


def fmt_p(x):
    if pd.isna(x):
        return ''
    x = float(x)
    if x < 0.001:
        return '<0.001'
    return f"{x:.3f}"


def fmt_ci(low, high, digits=2):
    if pd.isna(low) or pd.isna(high):
        return ''
    return f"{float(low):.{digits}f}–{float(high):.{digits}f}"


def md_table(df):
    if df.empty:
        return '_No rows._\n'
    safe = df.fillna('').astype(str)
    headers = list(safe.columns)
    lines = []
    lines.append('| ' + ' | '.join(headers) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    for _, row in safe.iterrows():
        vals = [str(row[h]).replace('|', '\\|') for h in headers]
        lines.append('| ' + ' | '.join(vals) + ' |')
    return '\n'.join(lines)


# Table S1: survey input summary
input_audit = pd.read_csv(STATS / 'diver_stats_input_audit.csv')
count_row = input_audit[input_audit['table'].eq('count_rows')].iloc[0]
length_row = input_audit[input_audit['table'].eq('positive_length_rows')].iloc[0]
table_s1 = pd.DataFrame([
    ['Survey rows', f"{int(count_row['rows']):,}"],
    ['Survey sites', f"{int(count_row['sites']):,}"],
    ['Basins', f"{int(count_row['basins']):,}"],
    ['Survey-level habitat categories', f"{int(count_row['habitats']):,}"],
    ['Total survey effort, hours', fmt_num(count_row['total_survey_hours'], 1)],
    ['Total P. helianthoides observed', f"{int(count_row['total_pycno_count']):,}"],
    ['Rows with zero P. helianthoides', f"{100*float(count_row['zero_fraction']):.1f}%"],
    ['Positive individual length measurements', f"{int(length_row['rows']):,}"],
    ['Sites with positive length measurements', f"{int(length_row['sites']):,}"],
], columns=['Quantity', 'Value'])
table_s1.to_csv(OUT / 'table_s1_survey_analysis_input_summary.csv', index=False)

# Table S2: focal eelgrass model terms across models
models = pd.read_csv(STATS / 'diver_count_model_results.csv')
focal = models[models['term'].eq('SiteHasEelgrassInt')].copy()
model_order = [
    'M1_eelgrass_only', 'M2_eelgrass_plus_habitat', 'M3_eelgrass_plus_basin', 'M4_eelgrass_plus_basin_plus_habitat',
    'P1_eelgrass_only_poisson_sensitivity', 'P2_eelgrass_plus_habitat_poisson_sensitivity',
    'P3_eelgrass_plus_basin_poisson_sensitivity', 'P4_eelgrass_plus_basin_plus_habitat_poisson_sensitivity'
]
model_labels = {
    'M1_eelgrass_only': 'NB: eelgrass only',
    'M2_eelgrass_plus_habitat': 'NB: + habitat',
    'M3_eelgrass_plus_basin': 'NB: + basin',
    'M4_eelgrass_plus_basin_plus_habitat': 'NB: + basin + habitat',
    'P1_eelgrass_only_poisson_sensitivity': 'Poisson: eelgrass only',
    'P2_eelgrass_plus_habitat_poisson_sensitivity': 'Poisson: + habitat',
    'P3_eelgrass_plus_basin_poisson_sensitivity': 'Poisson: + basin',
    'P4_eelgrass_plus_basin_plus_habitat_poisson_sensitivity': 'Poisson: + basin + habitat',
}
focal['sort'] = focal['model_name'].map({m:i for i,m in enumerate(model_order)})
focal = focal.sort_values('sort')
table_s2 = pd.DataFrame({
    'Model': focal['model_name'].map(model_labels),
    'Family': focal['family'].str.replace('NegativeBinomial(alpha=1)', 'Negative binomial', regex=False),
    'Adjustment': focal['model_name'].map({
        'M1_eelgrass_only': 'Survey effort only',
        'M2_eelgrass_plus_habitat': 'Survey effort + habitat',
        'M3_eelgrass_plus_basin': 'Survey effort + basin',
        'M4_eelgrass_plus_basin_plus_habitat': 'Survey effort + basin + habitat',
        'P1_eelgrass_only_poisson_sensitivity': 'Survey effort only',
        'P2_eelgrass_plus_habitat_poisson_sensitivity': 'Survey effort + habitat',
        'P3_eelgrass_plus_basin_poisson_sensitivity': 'Survey effort + basin',
        'P4_eelgrass_plus_basin_plus_habitat_poisson_sensitivity': 'Survey effort + basin + habitat',
    }),
    'IRR for eelgrass-at-site': [fmt_num(v, 2) for v in focal['irr']],
    '95% CI': [fmt_ci(l, h, 2) for l, h in zip(focal['irr_ci_low'], focal['irr_ci_high'])],
    'p': [fmt_p(v) for v in focal['p_value']],
    'Rows': [f"{int(v):,}" for v in focal['n_rows']],
    'Sites': [f"{int(v):,}" for v in focal['n_sites']],
})
table_s2.to_csv(OUT / 'table_s2_count_model_eelgrass_association.csv', index=False)

# Table S3: primary adjusted model coefficients, readable labels
m4 = models[(models['model_name']=='M4_eelgrass_plus_basin_plus_habitat') & (~models['term'].eq('Intercept'))].copy()
def clean_term(t):
    if t == 'SiteHasEelgrassInt':
        return 'Eelgrass recorded at site'
    if '[T.' in t:
        val = t.split('[T.', 1)[1].rstrip(']')
        if 'C(Basin' in t:
            return f'Basin: {val}'
        if 'C(HabitatType' in t:
            return f'Habitat: {val}'
    return t
m4['label'] = m4['term'].map(clean_term)
m4['term_class'] = np.select(
    [m4['term'].eq('SiteHasEelgrassInt'), m4['term'].str.contains('C\\(Basin', regex=True), m4['term'].str.contains('C\\(HabitatType', regex=True)],
    ['Focal eelgrass term', 'Basin covariate', 'Habitat covariate'],
    default='Other'
)
m4['term_class_order'] = m4['term_class'].map({'Focal eelgrass term': 0, 'Basin covariate': 1, 'Habitat covariate': 2}).fillna(9)
m4 = m4.sort_values(['term_class_order', 'irr'], ascending=[True, False])
table_s3 = pd.DataFrame({
    'Primary adjusted model term': m4['label'],
    'Term group': m4['term_class'],
    'IRR': [fmt_num(v, 2) for v in m4['irr']],
    '95% CI': [fmt_ci(l, h, 2) for l, h in zip(m4['irr_ci_low'], m4['irr_ci_high'])],
    'p': [fmt_p(v) for v in m4['p_value']],
})
table_s3.to_csv(OUT / 'table_s3_primary_count_model_terms.csv', index=False)

# Table S4: descriptive cells with sparse flags
hab = pd.read_csv(STATS / 'diver_fig2_habitat_eelgrass_summary.csv')
hab['Grouping'] = 'Habitat × eelgrass'
hab = hab.rename(columns={'HabitatType':'Group'})
basin = pd.read_csv(STATS / 'diver_fig3_basin_eelgrass_summary.csv')
basin['Grouping'] = 'Basin × eelgrass'
basin = basin.rename(columns={'Basin':'Group'})
cell = pd.concat([hab, basin], ignore_index=True, sort=False)
cell['grouping_order'] = cell['Grouping'].map({'Habitat × eelgrass': 0, 'Basin × eelgrass': 1})
cell = cell.sort_values(['grouping_order', 'mean_encounter_rate_hr'], ascending=[True, False])
table_s4 = pd.DataFrame({
    'Grouping': cell['Grouping'],
    'Group': cell['Group'],
    'Eelgrass status': cell['EelgrassStatus'],
    'Rows': [f"{int(v):,}" for v in cell['rows']],
    'Sites': [f"{int(v):,}" for v in cell['sites']],
    'Survey hours': [fmt_num(v, 1) for v in cell['total_survey_hours']],
    'P. helianthoides': [f"{int(v):,}" for v in cell['total_pycno']],
    'Mean encounter rate hr^-1': [fmt_num(v, 2) for v in cell['mean_encounter_rate_hr']],
    'SE': [fmt_num(v, 2) for v in cell['se_encounter_rate_hr']],
    'Sparse descriptive cell': cell['sparse_cell_flag'].map({True:'Yes', False:'No'}),
})
table_s4.to_csv(OUT / 'table_s4_descriptive_cells_and_sparse_flags.csv', index=False)

# Table S5: length distribution-shape tests
length_site = pd.read_csv(STATS / 'diver_length_site_eelgrass_distribution_shape_tests.csv')
length_site = length_site.sort_values(['sparse_flag', 'ks_d'], ascending=[True, False])
length_site['Panel/contrast'] = length_site['comparison']
length_site['Comparison'] = length_site['group_a'] + ' vs ' + length_site['group_b']
length_site['KS 95% CI'] = ''
length_site['Energy 95% CI'] = ''
length_site['KS p/q'] = [f"p={fmt_p(p)}; q={fmt_p(q)}" for p,q in zip(length_site['ks_site_permutation_p'], length_site['ks_site_permutation_fdr_q'])]
length_site['Energy p/q'] = [f"p={fmt_p(p)}; q={fmt_p(q)}" for p,q in zip(length_site['energy_site_permutation_p'], length_site['energy_site_permutation_fdr_q'])]
s2 = pd.read_csv(STATS / 'diver_s2_habitat_distribution_shape_tests.csv')
s2_keep = s2[~s2['sparse_flag']].copy()
s2_keep = s2_keep.sort_values(['eelgrass_status_panel', 'ks_d'], ascending=[True, False])
s2_keep['Panel/contrast'] = 'S2 panel: ' + s2_keep['eelgrass_status_panel']
s2_keep['Comparison'] = s2_keep['group_a'] + ' vs ' + s2_keep['group_b']
s2_keep['KS 95% CI'] = [fmt_ci(l,h,2) for l,h in zip(s2_keep['ks_cluster_bootstrap_ci_low'], s2_keep['ks_cluster_bootstrap_ci_high'])]
s2_keep['Energy 95% CI'] = [fmt_ci(l,h,2) for l,h in zip(s2_keep['energy_cluster_bootstrap_ci_low'], s2_keep['energy_cluster_bootstrap_ci_high'])]
s2_keep['KS p/q'] = ['sensitivity q=' + fmt_p(q) for q in s2_keep['individual_ks_sensitivity_fdr_q']]
s2_keep['Energy p/q'] = ['bootstrap CI only' for _ in range(len(s2_keep))]
length_combined = pd.concat([length_site, s2_keep], ignore_index=True, sort=False)
table_s5 = pd.DataFrame({
    'Panel/contrast': length_combined['Panel/contrast'],
    'Comparison': length_combined['Comparison'],
    'N individuals': [f"{int(a):,} vs {int(b):,}" for a,b in zip(length_combined['n_a_individuals'], length_combined['n_b_individuals'])],
    'N sites': [f"{int(a):,} vs {int(b):,}" for a,b in zip(length_combined['n_a_sites'], length_combined['n_b_sites'])],
    'KS D': [fmt_num(v, 3) for v in length_combined['ks_d']],
    'KS CI or p/q': [ci if ci else pq for ci,pq in zip(length_combined['KS 95% CI'], length_combined['KS p/q'])],
    'Energy distance': [fmt_num(v, 3) for v in length_combined['energy_distance']],
    'Energy CI or p/q': [ci if ci else pq for ci,pq in zip(length_combined['Energy 95% CI'], length_combined['Energy p/q'])],
    'Sparse': length_combined['sparse_flag'].map({True:'Yes', False:'No'}),
})
table_s5.to_csv(OUT / 'table_s5_length_distribution_shape_tests.csv', index=False)

# Table S6: ShoreZone model summary + top features
model_summary = pd.read_csv(SHORE / 'shorezone_model_summary.csv')
ms = model_summary.iloc[0]
table_s6a = pd.DataFrame([
    ['Survey sites with complete ShoreZone features', f"{int(ms['survey_sites_with_shorezone_features']):,}"],
    ['One-hot encoded ShoreZone features', f"{int(ms['model_features']):,}"],
    ['Cross-validated R^2, mean ± SD', f"{fmt_num(ms['cv_r2_mean'],3)} ± {fmt_num(ms['cv_r2_sd'],3)}"],
    ['Prediction vs observed Pearson r', fmt_num(ms['cv_pred_observed_pearson_r'], 3)],
    ['Prediction vs observed p', fmt_p(ms['cv_pred_observed_pearson_p'])],
    ['RMSE, individuals hr^-1', fmt_num(ms['cv_rmse'], 2)],
    ['MAE, individuals hr^-1', fmt_num(ms['cv_mae'], 2)],
], columns=['Metric', 'Value'])
top = pd.read_csv(SHORE / 'shorezone_top_prediction_loss_features.csv').head(10)
table_s6b = pd.DataFrame({
    'Feature': top['FeatureLabel'],
    'Prediction loss': [fmt_num(v,3) for v in top['PredictionLoss']],
    'SE': [fmt_num(v,3) for v in top['PredictionLossSE']],
})
table_s6a.to_csv(OUT / 'table_s6a_shorezone_model_summary.csv', index=False)
table_s6b.to_csv(OUT / 'table_s6b_shorezone_top_prediction_loss_features.csv', index=False)

# Table S7: ShoreZone group summaries/tests/agreement
grp = pd.read_csv(SHORE / 'shorezone_fig4b_group_summary.csv')
grp = grp.sort_values('mean_encounter_rate', ascending=False)
table_s7a = pd.DataFrame({
    'Exposure group': grp['RefugiaStatus'],
    'ShoreZone eelgrass group': grp['ContinuousEelgrassStatus'],
    'Sites': [f"{int(v):,}" for v in grp['sites']],
    'Mean encounter rate hr^-1': [fmt_num(v,2) for v in grp['mean_encounter_rate']],
    'SE': [fmt_num(v,2) for v in grp['se_encounter_rate']],
    'Median encounter rate hr^-1': [fmt_num(v,2) for v in grp['median_encounter_rate']],
    'P. helianthoides': [f"{int(v):,}" for v in grp['total_pycno']],
    'Survey hours': [fmt_num(v,1) for v in grp['total_survey_hours']],
})
tests = pd.read_csv(SHORE / 'shorezone_fig4b_tests.csv')
tests = tests.sort_values('p_value')
table_s7b = pd.DataFrame({
    'Contrast': tests['contrast'],
    'Test': tests['test'],
    'Statistic': [fmt_num(v,2) for v in tests['statistic']],
    'p': [fmt_p(v) for v in tests['p_value']],
})
agree = pd.read_csv(SHORE / 'shorezone_s4_agreement_tests.csv')
agree = agree.sort_values('p_value')
table_s7c = pd.DataFrame({
    'Agreement contrast': agree['contrast'],
    'Test': agree['test'],
    'Statistic': [fmt_num(v,2) for v in agree['statistic']],
    'df': [fmt_num(v,0) for v in agree['df']],
    'p': [fmt_p(v) for v in agree['p_value']],
    'Effect': agree['effect'],
    'Effect value': [fmt_num(v,3) for v in agree['effect_value']],
})
table_s7a.to_csv(OUT / 'table_s7a_shorezone_fig4b_group_summary.csv', index=False)
table_s7b.to_csv(OUT / 'table_s7b_shorezone_fig4b_tests.csv', index=False)
table_s7c.to_csv(OUT / 'table_s7c_shorezone_eelgrass_agreement_tests.csv', index=False)

# Markdown with captions
sections = []
sections.append('# Supplemental statistical tables\n')
sections.append('These tables collect the statistical analyses that support the manuscript figures. They are intended for supplemental materials so the inferential results are visible alongside the descriptive figures. Count models used survey effort offsets and SiteName-clustered standard errors. Length analyses emphasize distribution-shape contrasts because individual measurements are clustered within sites. ShoreZone analyses are exploratory habitat-context analyses because the mapped biotic layers pre-date the biological surveys.\n')
sections.append('## S Table 1. Survey analysis input summary\n\n' + md_table(table_s1) + '\n')
sections.append('## S Table 2. Site-level eelgrass association across count models\n\nIncidence-rate ratios (IRRs) compare surveys at sites where eelgrass was recorded with surveys at sites where eelgrass was not recorded. All models include a survey-effort offset.\n\n' + md_table(table_s2) + '\n')
sections.append('## S Table 3. Primary adjusted count-model terms\n\nPrimary negative-binomial model adjusted for basin and survey-level habitat. Reference categories were Central basin and Natural Reef habitat. The focal eelgrass term is listed first; basin and habitat covariates are then ranked by incidence-rate ratio within term group.\n\n' + md_table(table_s3) + '\n')
sections.append('## S Table 4. Descriptive habitat-by-eelgrass and basin-by-eelgrass cells\n\nSparse descriptive cells had fewer than 3 sites, fewer than 20 survey rows, or zero total P. helianthoides. Sparse flags are descriptive subgroup cautions, not exclusions from the count models. Rows are ranked by mean encounter rate within habitat and basin groupings.\n\n' + md_table(table_s4) + '\n')
sections.append('## S Table 5. Length distribution-shape analyses\n\nKS D is the two-sample Kolmogorov-Smirnov distance. Energy distance is a complementary distribution-shape statistic. Overall and habitat-specific site-eelgrass tests use site-label permutation p-values; S2 within-panel habitat-pair rows show site-cluster bootstrap intervals and individual-level KS q-values only as sensitivity values. Non-sparse contrasts are ranked by KS D within each analysis block.\n\n' + md_table(table_s5) + '\n')
sections.append('## S Table 6. ShoreZone random-forest model and top prediction-loss features\n\n### S Table 6A. Model performance\n\n' + md_table(table_s6a) + '\n\n### S Table 6B. Top ShoreZone prediction-loss features\n\n' + md_table(table_s6b) + '\n')
sections.append('## S Table 7. ShoreZone group contrasts and eelgrass agreement tests\n\n### S Table 7A. Protected-exposure by continuous-eelgrass group summaries\n\n' + md_table(table_s7a) + '\n\n### S Table 7B. Protected-exposure and continuous-eelgrass group tests\n\n' + md_table(table_s7b) + '\n\n### S Table 7C. ShoreZone eelgrass versus survey-recorded eelgrass agreement tests\n\n' + md_table(table_s7c) + '\n')

(OUT / 'supplemental_statistical_tables.md').write_text('\n'.join(sections), encoding='utf-8')
print('Wrote supplemental tables to', OUT)
