#!/usr/bin/env python3
from pathlib import Path
import zipfile
import pandas as pd
from xml.sax.saxutils import escape

ROOT = Path('/home/weertman/Documents/StarMeadow')
SUPP = ROOT / 'outputs' / 'publication_figures' / 'supplemental_tables'
OUT_DOCX = SUPP / 'supplemental_statistical_tables_doc_friendly.docx'

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

NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

def tx(text):
    return escape(str(text), {'\n': ' '})

def p(text='', style=None, italic=False):
    style_xml = f'<w:pStyle w:val="{style}"/>' if style else ''
    italic_xml = '<w:i/>' if italic else ''
    return f'<w:p><w:pPr>{style_xml}</w:pPr><w:r><w:rPr>{italic_xml}</w:rPr><w:t>{tx(text)}</w:t></w:r></w:p>'

def cell(text, header=False):
    shade = '<w:shd w:fill="EAEAEA"/>' if header else ''
    bold = '<w:b/>' if header else ''
    return (
        '<w:tc><w:tcPr><w:tcW w:w="2400" w:type="dxa"/>' + shade + '</w:tcPr>'
        '<w:p><w:r><w:rPr>' + bold + '</w:rPr><w:t>' + tx(text) + '</w:t></w:r></w:p></w:tc>'
    )

def table(df):
    rows = []
    rows.append('<w:tr>' + ''.join(cell(c, header=True) for c in df.columns) + '</w:tr>')
    for _, r in df.iterrows():
        rows.append('<w:tr>' + ''.join(cell('' if pd.isna(r[c]) else r[c]) for c in df.columns) + '</w:tr>')
    props = (
        '<w:tblPr><w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="777777"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="777777"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="777777"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="777777"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="777777"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="777777"/>'
        '</w:tblBorders></w:tblPr>'
    )
    return '<w:tbl>' + props + ''.join(rows) + '</w:tbl>'

body = []
body.append(p('Supplemental statistical tables', style='Title'))
body.append(p('Doc-friendly version for copying into a manuscript document. Tables are generated from the CSV files in outputs/publication_figures/supplemental_tables/.', italic=True))

for title, filename, note in TABLES:
    df = pd.read_csv(SUPP / filename, dtype=str).fillna('')
    body.append(p(title, style='Heading1'))
    body.append(p(note, italic=True))
    body.append(table(df))
    body.append(p(''))

document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{NS}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <w:body>
    {''.join(body)}
    <w:sectPr><w:pgSz w:w="15840" w:h="12240" w:orient="landscape"/><w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720" w:header="360" w:footer="360" w:gutter="0"/></w:sectPr>
  </w:body>
</w:document>'''

content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>'''
rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''

with zipfile.ZipFile(OUT_DOCX, 'w', zipfile.ZIP_DEFLATED) as z:
    z.writestr('[Content_Types].xml', content_types)
    z.writestr('_rels/.rels', rels)
    z.writestr('word/document.xml', document_xml)
print(OUT_DOCX)
