#!/usr/bin/env python3
from pathlib import Path
import html
import pandas as pd

ROOT = Path('/home/weertman/Documents/StarMeadow')
SUPP = ROOT / 'outputs' / 'publication_figures' / 'supplemental_tables'
OUTDIR = SUPP / 'google_docs_copy_paste_html'
OUTDIR.mkdir(parents=True, exist_ok=True)

TABLES = [
    ('S Table 1. Survey analysis input summary', 'table_s1_survey_analysis_input_summary.csv', 'Summary of survey rows, sites, effort, observations, and length measurements used in the statistical analyses.'),
    ('S Table 2. Site-level eelgrass association across count models', 'table_s2_count_model_eelgrass_association.csv', 'Incidence-rate ratios compare surveys at sites where eelgrass was recorded with surveys at sites where eelgrass was not recorded. All models include a survey-effort offset.'),
    ('S Table 3. Primary adjusted count-model terms', 'table_s3_primary_count_model_terms.csv', 'Primary negative-binomial model adjusted for basin and survey-level habitat. Reference categories were Central basin and Natural Reef habitat. The focal eelgrass term is listed first; basin and habitat covariates are ranked by incidence-rate ratio within term group.'),
    ('S Table 4. Descriptive habitat-by-eelgrass and basin-by-eelgrass cells', 'table_s4_descriptive_cells_and_sparse_flags.csv', 'Sparse descriptive cells had fewer than 3 sites, fewer than 20 survey rows, or zero total P. helianthoides. Sparse flags are descriptive subgroup cautions, not exclusions from the count models. Rows are ranked by mean encounter rate within habitat and basin groupings.'),
    ('S Table 5. Length distribution-shape analyses', 'table_s5_length_distribution_shape_tests.csv', 'KS D is the two-sample Kolmogorov-Smirnov distance. Energy distance is a complementary distribution-shape statistic. Non-sparse contrasts are ranked by KS D within analysis blocks.'),
    ('S Table 6A. ShoreZone model performance', 'table_s6a_shorezone_model_summary.csv', 'Performance summary for the exploratory ShoreZone random-forest model.'),
    ('S Table 6B. Top ShoreZone prediction-loss features', 'table_s6b_shorezone_top_prediction_loss_features.csv', 'Features with the largest mean prediction loss when permuted.'),
    ('S Table 7A. Protected-exposure by continuous-eelgrass group summaries', 'table_s7a_shorezone_fig4b_group_summary.csv', 'Site-level encounter-rate summaries by ShoreZone exposure group and continuous mapped eelgrass status.'),
    ('S Table 7B. Protected-exposure and continuous-eelgrass group tests', 'table_s7b_shorezone_fig4b_tests.csv', 'Non-parametric tests comparing ShoreZone exposure and continuous eelgrass groups.'),
    ('S Table 7C. ShoreZone eelgrass versus survey-recorded eelgrass agreement tests', 'table_s7c_shorezone_eelgrass_agreement_tests.csv', 'Agreement tests comparing ShoreZone eelgrass classes with survey-recorded site-level eelgrass status.'),
]

PAGE_STYLE = 'font-family: Arial, sans-serif; font-size: 10pt; line-height: 1.25; color: #111; background: white;'
TITLE_STYLE = 'font-family: Arial, sans-serif; font-size: 12pt; font-weight: bold; margin: 0 0 6px 0;'
NOTE_STYLE = 'font-family: Arial, sans-serif; font-size: 9.5pt; font-style: italic; margin: 0 0 8px 0;'
TABLE_STYLE = 'border-collapse: collapse; border: 1px solid #666; font-family: Arial, sans-serif; font-size: 9pt;'
TH_STYLE = 'border: 1px solid #666; padding: 4px 6px; background-color: #e6e6e6; font-weight: bold; vertical-align: top; text-align: left;'
TD_STYLE = 'border: 1px solid #777; padding: 3px 6px; vertical-align: top; text-align: left;'
TD_NUM_STYLE = 'border: 1px solid #777; padding: 3px 6px; vertical-align: top; text-align: right;'


def esc(x):
    return html.escape('' if pd.isna(x) else str(x))


def is_num(x):
    s = str(x).strip()
    if not s or any(token in s for token in [' vs ', 'p=', ';', '–', '±']):
        return False
    if s.startswith('<'):
        return True
    try:
        float(s.replace(',', '').replace('%', ''))
        return True
    except ValueError:
        return False


def table_html(df):
    parts = [f'<table style="{TABLE_STYLE}">']
    parts.append('<thead><tr>')
    for c in df.columns:
        parts.append(f'<th style="{TH_STYLE}">{esc(c)}</th>')
    parts.append('</tr></thead><tbody>')
    for _, row in df.iterrows():
        parts.append('<tr>')
        for c in df.columns:
            val = row[c]
            style = TD_NUM_STYLE if is_num(val) else TD_STYLE
            parts.append(f'<td style="{style}">{esc(val)}</td>')
        parts.append('</tr>')
    parts.append('</tbody></table>')
    return ''.join(parts)

index_links = []
all_blocks = []
for i, (title, filename, note) in enumerate(TABLES, 1):
    df = pd.read_csv(SUPP / filename, dtype=str).fillna('')
    block = (
        f'<div style="{PAGE_STYLE}">'
        f'<p style="{TITLE_STYLE}">{esc(title)}</p>'
        f'<p style="{NOTE_STYLE}">{esc(note)}</p>'
        f'{table_html(df)}'
        '</div>'
    )
    all_blocks.append(block)
    safe_name = f'table_{i:02d}.html'
    (OUTDIR / safe_name).write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>' + esc(title) + '</title></head>'
        f'<body style="{PAGE_STYLE}">' + block + '</body></html>',
        encoding='utf-8'
    )
    index_links.append(f'<li><a href="{safe_name}">{esc(title)}</a></li>')

(OUTDIR / 'all_tables_google_docs_copy_paste.html').write_text(
    '<!doctype html><html><head><meta charset="utf-8"><title>Copy-paste supplemental tables</title></head>'
    f'<body style="{PAGE_STYLE}">'
    '<h1 style="font-size:14pt;">Copy-paste supplemental statistical tables</h1>'
    '<p>For best Google Docs results, open an individual table below, select the title/note/table in the browser, copy, then paste into Google Docs.</p>'
    '<ol>' + ''.join(index_links) + '</ol><hr>' + '<br><br>'.join(all_blocks) + '</body></html>',
    encoding='utf-8'
)
print(OUTDIR / 'all_tables_google_docs_copy_paste.html')
