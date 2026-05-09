#!/usr/bin/env python3
from pathlib import Path
import html
import pandas as pd

ROOT = Path('/home/weertman/Documents/StarMeadow')
SUPP = ROOT / 'outputs' / 'publication_figures' / 'supplemental_tables'
OUT_HTML = SUPP / 'supplemental_statistical_tables_doc_friendly.html'
OUT_TSV = SUPP / 'supplemental_statistical_tables_doc_friendly.tsv'

TABLES = [
    ('S Table 1. Survey analysis input summary', 'table_s1_survey_analysis_input_summary.csv', 'Summary of survey rows, sites, effort, observations, and length measurements used in the statistical analyses.'),
    ('S Table 2. Site-level eelgrass association across count models', 'table_s2_count_model_eelgrass_association.csv', 'Incidence-rate ratios compare surveys at sites where eelgrass was recorded with surveys at sites where eelgrass was not recorded. All models include a survey-effort offset.'),
    ('S Table 3. Primary adjusted count-model terms', 'table_s3_primary_count_model_terms.csv', 'Primary negative-binomial model adjusted for basin and survey-level habitat. Reference categories were Central basin and Natural Reef habitat.'),
    ('S Table 4. Descriptive habitat-by-eelgrass and basin-by-eelgrass cells', 'table_s4_descriptive_cells_and_sparse_flags.csv', 'Sparse descriptive cells had fewer than 3 sites, fewer than 20 survey rows, or zero total P. helianthoides. Sparse flags are descriptive subgroup cautions, not exclusions from the count models.'),
    ('S Table 5. Length distribution-shape analyses', 'table_s5_length_distribution_shape_tests.csv', 'KS D is the two-sample Kolmogorov-Smirnov distance. Energy distance is a complementary distribution-shape statistic. Overall and habitat-specific site-eelgrass tests use site-label permutation p-values; S2 within-panel habitat-pair rows show site-cluster bootstrap intervals and individual-level KS q-values only as sensitivity values.'),
    ('S Table 6A. ShoreZone model performance', 'table_s6a_shorezone_model_summary.csv', 'Performance summary for the exploratory ShoreZone random-forest model.'),
    ('S Table 6B. Top ShoreZone prediction-loss features', 'table_s6b_shorezone_top_prediction_loss_features.csv', 'Features with the largest mean prediction loss when permuted.'),
    ('S Table 7A. Protected-exposure by continuous-eelgrass group summaries', 'table_s7a_shorezone_fig4b_group_summary.csv', 'Site-level encounter-rate summaries by ShoreZone exposure group and continuous mapped eelgrass status.'),
    ('S Table 7B. Protected-exposure and continuous-eelgrass group tests', 'table_s7b_shorezone_fig4b_tests.csv', 'Non-parametric tests comparing ShoreZone exposure and continuous eelgrass groups.'),
    ('S Table 7C. ShoreZone eelgrass versus survey-recorded eelgrass agreement tests', 'table_s7c_shorezone_eelgrass_agreement_tests.csv', 'Agreement tests comparing ShoreZone eelgrass classes with survey-recorded site-level eelgrass status.'),
]

CSS = """
body { font-family: Arial, sans-serif; font-size: 10.5pt; color: #111; }
h1 { font-size: 16pt; margin-bottom: 0.2in; }
h2 { font-size: 12pt; margin-top: 0.28in; margin-bottom: 0.06in; page-break-after: avoid; }
p.note { margin-top: 0; margin-bottom: 0.08in; font-style: italic; }
table { border-collapse: collapse; margin-bottom: 0.18in; width: 100%; }
th, td { border: 1px solid #777; padding: 3px 5px; vertical-align: top; }
th { background: #eaeaea; font-weight: bold; }
td.num { text-align: right; }
.small { font-size: 9pt; }
"""


def is_numericish(value: object) -> bool:
    s = str(value).strip()
    if not s or s in {'<0.001'}:
        return False
    s2 = s.replace(',', '').replace('%', '').replace('–', '-').replace('±', '')
    if ' vs ' in s2 or 'p=' in s2 or ';' in s2:
        return False
    try:
        float(s2)
        return True
    except ValueError:
        return False


def table_to_html(df: pd.DataFrame) -> str:
    lines = ['<table>']
    lines.append('<thead><tr>' + ''.join(f'<th>{html.escape(str(c))}</th>' for c in df.columns) + '</tr></thead>')
    lines.append('<tbody>')
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            val = '' if pd.isna(row[c]) else str(row[c])
            cls = ' class="num"' if is_numericish(val) else ''
            cells.append(f'<td{cls}>{html.escape(val)}</td>')
        lines.append('<tr>' + ''.join(cells) + '</tr>')
    lines.append('</tbody></table>')
    return '\n'.join(lines)

html_parts = [
    '<!doctype html><html><head><meta charset="utf-8">',
    '<title>Supplemental statistical tables</title>',
    '<style>' + CSS + '</style>',
    '</head><body>',
    '<h1>Supplemental statistical tables</h1>',
    '<p class="note">Doc-friendly version for copying into a manuscript document. Tables are generated from the CSV files in outputs/publication_figures/supplemental_tables/.</p>',
]

tsv_parts = []

for title, filename, note in TABLES:
    df = pd.read_csv(SUPP / filename, dtype=str).fillna('')
    html_parts.append(f'<h2>{html.escape(title)}</h2>')
    html_parts.append(f'<p class="note">{html.escape(note)}</p>')
    html_parts.append(table_to_html(df))

    tsv_parts.append(title)
    tsv_parts.append(note)
    tsv_parts.append('\t'.join(df.columns))
    for _, row in df.iterrows():
        tsv_parts.append('\t'.join(str(row[c]) for c in df.columns))
    tsv_parts.append('')

html_parts.append('</body></html>')
OUT_HTML.write_text('\n'.join(html_parts), encoding='utf-8')
OUT_TSV.write_text('\n'.join(tsv_parts), encoding='utf-8')
print(OUT_HTML)
print(OUT_TSV)
