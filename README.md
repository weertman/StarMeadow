# StarMeadow

Reproducible publication materials for the StarMeadow analysis of sunflower sea star (*Pycnopodia helianthoides*) survey encounters, body-size distributions, eelgrass-associated habitat patterns, and ShoreZone habitat context in the Salish Sea.

This repository is intentionally publication-focused. It is not a complete archive of every exploratory analysis run during project development. The GitHub-facing surface is meant to be sparse, readable, and sharable: it keeps the code, data tables, figures, captions, statistical tables, and environment files that directly support the manuscript.

## Manuscript focus

The analysis asks where *P. helianthoides* were encountered during surveys and whether encounter rates and size distributions were associated with eelgrass-linked, protected shoreline habitats. The current manuscript emphasizes three linked results:

1. *P. helianthoides* encounters were unevenly distributed among surveyed sites and habitats.
2. Sites where eelgrass was recorded had higher expected encounter rates after accounting for survey effort, basin, and survey-level habitat.
3. A Whidbey Basin cluster of protected, continuously mapped eelgrass sites appears to be an especially important remaining/recovering concentration within the study area.

ShoreZone products are used as independent, older habitat-context layers, not as contemporaneous field truth. The ShoreZone source metadata report ground-condition dates from 1994-2000, with the Washington State ShoreZone Inventory published in 2001; these mapped layers pre-date the 2022-2025 surveys by roughly two decades.

## Repository layout

```text
StarMeadow/
├── README.md
├── ENVIRONMENT.md
├── code/
│   ├── figure_*.py                         # main manuscript figure scripts
│   ├── supplemental_figure_*.py            # supplemental figure scripts
│   ├── diver_only_stats_for_report.py      # survey count-model analysis
│   ├── length_distribution_shape_stats_for_report.py
│   ├── 15_shorezone_site_analysis.py       # retained ShoreZone preprocessing script
│   ├── 16_shorezone_recovery_analysis.py   # retained ShoreZone modeling script
│   ├── utils.py
│   └── shorezone_utils.py
├── data/
│   ├── PycnoCountCLean_12_31_2025.csv
│   ├── PycnoLengthCLean_12_31_2025.csv
│   ├── Site_LatLong.csv
│   └── state_DNR_ShoreZone/                # metadata only; full FileGDB not tracked
├── docs/
│   └── DATA_NOTES.md
├── scripts/
│   ├── build_publication_outputs
│   ├── run_pipeline
│   ├── analysis/
│   └── tables/
└── outputs/
    ├── 15_shorezone_site_analysis/         # minimal derived ShoreZone inputs
    ├── 16_shorezone_recovery_analysis/     # minimal derived ShoreZone inputs
    └── publication_figures/                # manuscript figures, captions, stats, tables
```

## Quick start

From the repository root:

```bash
/home/weertman/miniforge3/bin/conda env create -f code/environment.yml
./scripts/build_publication_outputs
```

If the environment already exists:

```bash
/home/weertman/miniforge3/bin/conda env update -n star_meadow -f code/environment.yml --prune
./scripts/build_publication_outputs
```

The build command runs the publication statistical analyses, figure scripts, ShoreZone summary script, and supplemental-table builders. Outputs are written under:

```text
outputs/publication_figures/
```

`./scripts/run_pipeline` is kept as a compatibility wrapper and calls the same publication-only build script.

## Environment

The intended environment is documented in `ENVIRONMENT.md` and specified in:

```text
code/environment.yml
```

On the analysis workstation, the canonical Conda executable is:

```text
/home/weertman/miniforge3/bin/conda
```

The expected environment name is:

```text
star_meadow
```

Do not run the analysis with bare system `python3`; the system Python on the analysis workstation lacks several required packages.

To verify the environment:

```bash
/home/weertman/miniforge3/bin/conda run -n star_meadow python --version
/home/weertman/miniforge3/bin/conda run -n star_meadow python - <<'PY'
import pandas, numpy, matplotlib, seaborn, scipy, sklearn, geopandas, fiona, pyproj, shapely, folium, joblib, statsmodels
print('StarMeadow environment ready')
PY
```

## Data files

Canonical manuscript input tables:

| File | Purpose |
|---|---|
| `data/PycnoCountCLean_12_31_2025.csv` | Survey-row count data used for encounter-rate and count-model analyses. |
| `data/PycnoLengthCLean_12_31_2025.csv` | Individual/row-level length data used for size-distribution analyses. |
| `data/Site_LatLong.csv` | Site coordinates used for mapping and spatial joins. |
| `data/state_DNR_ShoreZone/dnr_shorezone_metadata.xml` | Public ShoreZone metadata. |
| `data/state_DNR_ShoreZone/dnr_shorezone_metadata.html` | Human-readable ShoreZone metadata. |

Additional data notes and known quirks are documented in:

```text
docs/DATA_NOTES.md
```

Important coordinate note: `data/Site_LatLong.csv` contains exact site locations. Confirm sharing permissions before public release if exact locations are considered sensitive.

## ShoreZone data policy

The full Washington DNR ShoreZone FileGDB payload is not tracked in this sparse publication repository. Only metadata and the minimal derived products directly used by the manuscript are tracked.

Tracked derived ShoreZone products:

```text
outputs/15_shorezone_site_analysis/site_shorezone_pycno_summary.csv
outputs/15_shorezone_site_analysis/shorezone_ZOS_UNIT_proportions.csv
outputs/16_shorezone_recovery_analysis/feature_importance_permutation.csv
```

These files support the current ShoreZone figures, tables, and manuscript text. For long-term public reproducibility, the preferred next improvement is to add a concise ShoreZone preprocessing note or script that downloads/locates the public source data and regenerates these minimal derived products.

## Main figure scripts

| Figure | Script | Output examples |
|---|---|---|
| Fig 1 | `code/figure_1_site_map.py` | `outputs/publication_figures/submission/Fig1.tif` |
| Fig 2 | `code/figure_2_habitat_eelgrass_size.py` | `outputs/publication_figures/submission/Fig2.tif` |
| Fig 3 | `code/figure_3_basin_eelgrass_encounter_rate.py` | `outputs/publication_figures/submission/Fig3.tif` |
| Fig 4 | `code/figure_4_shorezone.py` | `outputs/publication_figures/submission/Fig4.tif` |

## Supplemental figure scripts

| Figure | Script | Output examples |
|---|---|---|
| S1 Fig | `code/supplemental_figure_1_effort_basin_map.py` | `outputs/publication_figures/submission/S1_Fig.tif` |
| S2 Fig | `code/supplemental_figure_2_length_by_habitat_eelgrass_status.py` | `outputs/publication_figures/submission/S2_Fig.tif` |
| S3 Fig | `code/supplemental_figure_3_continuous_eelgrass_map.py` | `outputs/publication_figures/submission/S3_Fig.tif` |
| S4 Fig | `code/supplemental_figure_4_shorezone_vs_diver_eelgrass.py` | `outputs/publication_figures/submission/S4_Fig.tif` |
| S5 Fig | `code/supplemental_figure_5_survey_statistical_summary.py` | `outputs/publication_figures/submission/S5_Fig.tif` |

Historical note: one script filename still contains `diver` from earlier drafts. The manuscript and figure language should use `survey`.

## Figures and captions

The manuscript figures are provided as PLOS-style TIFF files in `outputs/publication_figures/submission/`, with PNG/PDF source exports in `outputs/publication_figures/sources/`. Captions are maintained in `outputs/publication_figures/captions.md` so that figure images remain free of embedded manuscript text.

### Fig 1. Survey sites and *Pycnopodia* encounter rates in the Salish Sea

![Fig 1. Survey sites and Pycnopodia encounter rates in the Salish Sea](outputs/publication_figures/sources/Fig1_site_map_draft.png)

Map of survey sites. Circle color indicates whether eelgrass was recorded at the site during surveys; green sites had eelgrass recorded and purple sites had no eelgrass recorded. Circle size encodes the site-level mean *Pycnopodia* encounter rate in individuals per survey hour.

### Fig 2. Encounter rates and size distributions by habitat and eelgrass status

![Fig 2. Encounter rates and size distributions by habitat and eelgrass status](outputs/publication_figures/sources/Fig2_habitat_eelgrass_size_draft.png)

(A) Mean *Pycnopodia* encounter rate in individuals per survey hour across survey-transect habitat categories, ordered by the eelgrass-at-site mean encounter rate from highest to lowest. Non-eelgrass habitat categories are grouped by whether eelgrass was recorded at the site, while eelgrass transects are shown as a single eelgrass-at-site category. Whiskers show standard errors, and numbers in parentheses above bars indicate the number of survey rows underlying each mean. (B) Kernel density distributions of individual *Pycnopodia* lengths by site-level eelgrass status for all surveys split by presence or absence of eelgrass at the sites. Green indicates sites with eelgrass recorded during surveys, and purple indicates sites without eelgrass recorded.

### Fig 3. Encounter rates by basin and eelgrass status

![Fig 3. Encounter rates by basin and eelgrass status](outputs/publication_figures/sources/Fig3_basin_eelgrass_encounter_rate_draft.png)

Mean *Pycnopodia* encounter rate in individuals per survey hour by basin, with bars grouped by whether eelgrass was recorded at the site. Whiskers show standard errors, and numbers in parentheses above bars indicate the number of survey rows underlying each mean. Basins are ordered from left to right by the mean encounter rate for sites with eelgrass recorded, with the highest value shown first. Green indicates sites with eelgrass recorded during surveys, and purple indicates sites without eelgrass recorded.

### Fig 4. ShoreZone predictors and protected-exposure encounter rates

![Fig 4. ShoreZone predictors and protected-exposure encounter rates](outputs/publication_figures/sources/Fig4_shorezone_draft.png)

(A) Prediction loss from dropping individual ShoreZone features from the encounter-rate prediction model. Bars show the mean permutation prediction loss for each dropped feature, with whiskers showing standard errors. Features are ordered by prediction loss, with the largest loss at the top. (B) Mean site-level *Pycnopodia* encounter rate in individuals per survey hour by ShoreZone exposure category and continuous eelgrass status. Protected / very protected sites are those with ShoreZone exposure class P or VP; other exposure includes all remaining ShoreZone exposure classes. Green bars indicate ShoreZone continuous eelgrass, and purple bars indicate other eelgrass states or no mapped continuous eelgrass. Whiskers show standard errors, and numbers in parentheses above bars indicate the number of sites underlying each mean.

### S1 Fig. Survey effort and basin assignments across the Salish Sea study area

![S1 Fig. Survey effort and basin assignments across the Salish Sea study area](outputs/publication_figures/sources/S1_Fig_effort_basin_map_draft.png)

Map of the survey sites. Circle color indicates the geographic basin assigned to each site, and circle size encodes total survey effort in hours at that site.

### S2 Fig. Individual size distributions by survey-level habitat and site-level eelgrass status

![S2 Fig. Individual size distributions by survey-level habitat and site-level eelgrass status](outputs/publication_figures/sources/S2_length_distribution_by_habitat_eelgrass_status_draft.png)

Kernel density distributions of individual *Pycnopodia* lengths by recorded survey-level habitat for (A) sites where eelgrass was recorded and (B) sites where eelgrass was not recorded. Both panels use the same x- and y-axis limits. Numbers in the legend indicate the number of individual length observations contributing to each habitat-specific density curve.

### S3 Fig. ShoreZone eelgrass categories and *Pycnopodia* encounter rates

![S3 Fig. ShoreZone eelgrass categories and Pycnopodia encounter rates](outputs/publication_figures/sources/S3_Fig_continuous_eelgrass_encounter_map_draft.png)

Map of survey sites colored by ShoreZone Zostera category and sized by site-level mean *Pycnopodia* encounter rate in individuals per survey hour. Green indicates continuous mapped ShoreZone eelgrass, gold indicates patchy mapped eelgrass, purple indicates mapped absence of eelgrass, and gray indicates sites without a ShoreZone Zostera match. ShoreZone categories are mapped habitat context rather than contemporaneous survey observations.

### S4 Fig. ShoreZone and survey-recorded eelgrass classifications

![S4 Fig. ShoreZone and survey-recorded eelgrass classifications](outputs/publication_figures/sources/S4_shorezone_vs_diver_eelgrass_draft.png)

Comparison of site-level ShoreZone Zostera categories and survey-recorded eelgrass categories for sites with a ShoreZone match. Survey-recorded eelgrass status indicates whether eelgrass was recorded in any survey row at a site. (A) Heatmap of site counts in each ShoreZone-by-survey category combination. (B) Proportion of sites within each ShoreZone category where eelgrass was or was not recorded during surveys; parenthetical values indicate the number of matched survey sites in each ShoreZone category.

### S5 Fig. Statistical summary of survey analyses

![S5 Fig. Statistical summary of survey analyses](outputs/publication_figures/sources/S5_survey_statistical_summary.png)

(A) Eelgrass-at-site incidence-rate ratios from negative-binomial count models and Poisson sensitivity models. Points show estimates, bars show 95% confidence intervals, and the dashed line indicates no association. (B) Terms from the primary negative-binomial model adjusted for basin and survey-level habitat, with incidence-rate ratios shown relative to no eelgrass recorded at the site, Central basin, and Natural Reef habitat. (C) Length distribution-shape contrasts summarized by Kolmogorov-Smirnov distance. Green indicates the supported Soft Bottom contrast, blue indicates non-significant non-sparse contrasts, and gray indicates sparse descriptive contrasts.

## Supplemental statistical tables

The tables below summarize the statistical analyses that support the manuscript figures and results. The same content is maintained as a standalone supplemental-table file at `outputs/publication_figures/supplemental_tables/supplemental_statistical_tables.md`, with machine-readable CSV versions in the same directory. Count models used survey-effort offsets and `SiteName`-clustered standard errors; length analyses emphasize distribution-shape contrasts because individual measurements are clustered within sites; and ShoreZone analyses are treated as exploratory habitat-context analyses because the mapped biotic layers pre-date the biological surveys.

### S Table 1. Survey analysis input summary

| Quantity | Value |
| --- | --- |
| Survey rows | 3,978 |
| Survey sites | 133 |
| Basins | 9 |
| Survey-level habitat categories | 6 |
| Total survey effort, hours | 848.0 |
| Total P. helianthoides observed | 2,651 |
| Rows with zero P. helianthoides | 85.3% |
| Positive individual length measurements | 1,954 |
| Sites with positive length measurements | 48 |

### S Table 2. Site-level eelgrass association across count models

Incidence-rate ratios (IRRs) compare surveys at sites where eelgrass was recorded with surveys at sites where eelgrass was not recorded. All models include a survey-effort offset.

| Model | Family | Adjustment | IRR for eelgrass-at-site | 95% CI | p | Rows | Sites |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NB: eelgrass only | Negative binomial | Survey effort only | 8.78 | 3.59–21.46 | <0.001 | 3,978 | 133 |
| NB: + habitat | Negative binomial | Survey effort + habitat | 5.15 | 2.45–10.84 | <0.001 | 3,978 | 133 |
| NB: + basin | Negative binomial | Survey effort + basin | 6.06 | 1.67–21.95 | 0.006 | 3,978 | 133 |
| NB: + basin + habitat | Negative binomial | Survey effort + basin + habitat | 4.28 | 1.49–12.27 | 0.007 | 3,978 | 133 |
| Poisson: eelgrass only | Poisson | Survey effort only | 9.03 | 3.80–21.47 | <0.001 | 3,978 | 133 |
| Poisson: + habitat | Poisson | Survey effort + habitat | 5.57 | 2.65–11.69 | <0.001 | 3,978 | 133 |
| Poisson: + basin | Poisson | Survey effort + basin | 6.32 | 2.23–17.90 | <0.001 | 3,978 | 133 |
| Poisson: + basin + habitat | Poisson | Survey effort + basin + habitat | 3.84 | 1.65–8.94 | 0.002 | 3,978 | 133 |

### S Table 3. Primary adjusted count-model terms

Primary negative-binomial model adjusted for basin and survey-level habitat. Reference categories were Central basin and Natural Reef habitat. The focal eelgrass term is listed first; basin and habitat covariates are then ranked by incidence-rate ratio within term group.

| Primary adjusted model term | Term group | IRR | 95% CI | p |
| --- | --- | --- | --- | --- |
| Eelgrass recorded at site | Focal eelgrass term | 4.28 | 1.49–12.27 | 0.007 |
| Basin: Hood Canal | Basin covariate | 36.39 | 9.13–145.14 | <0.001 |
| Basin: Whidbey | Basin covariate | 27.96 | 10.31–75.85 | <0.001 |
| Basin: Strait of Juan de Fuca | Basin covariate | 18.92 | 7.24–49.43 | <0.001 |
| Basin: Strait of Georgia | Basin covariate | 18.00 | 5.56–58.30 | <0.001 |
| Basin: Admiralty Inlet | Basin covariate | 4.13 | 1.15–14.89 | 0.030 |
| Basin: Howe Sound | Basin covariate | 2.07 | 0.37–11.55 | 0.406 |
| Basin: San Juan | Basin covariate | 1.25 | 0.24–6.45 | 0.787 |
| Basin: South | Basin covariate | 0.21 | 0.04–1.14 | 0.070 |
| Habitat: Soft Bottom | Habitat covariate | 3.12 | 1.31–7.42 | 0.010 |
| Habitat: Eelgrass | Habitat covariate | 1.34 | 0.63–2.87 | 0.450 |
| Habitat: Artificial Reef | Habitat covariate | 1.17 | 0.41–3.33 | 0.769 |
| Habitat: Kelp Forest | Habitat covariate | 0.64 | 0.29–1.43 | 0.281 |
| Habitat: Sponge Garden | Habitat covariate | 0.00 | 0.00–0.00 | <0.001 |

### S Table 4. Descriptive habitat-by-eelgrass and basin-by-eelgrass cells

Sparse descriptive cells had fewer than 3 sites, fewer than 20 survey rows, or zero total P. helianthoides. Sparse flags are descriptive subgroup cautions, not exclusions from the count models. Rows are ranked by mean encounter rate within habitat and basin groupings.

| Grouping | Group | Eelgrass status | Rows | Sites | Survey hours | P. helianthoides | Mean encounter rate hr^-1 | SE | Sparse descriptive cell |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Habitat × eelgrass | Soft Bottom | Eelgrass at site | 536 | 32 | 105.0 | 1,484 | 12.97 | 1.05 | No |
| Habitat × eelgrass | Eelgrass | Eelgrass at site | 243 | 34 | 31.2 | 175 | 4.10 | 0.67 | No |
| Habitat × eelgrass | Artificial Reef | Eelgrass at site | 548 | 16 | 131.5 | 523 | 3.14 | 0.44 | No |
| Habitat × eelgrass | Kelp Forest | Eelgrass at site | 48 | 4 | 13.2 | 31 | 1.93 | 0.57 | No |
| Habitat × eelgrass | Soft Bottom | No eelgrass at site | 316 | 41 | 52.0 | 73 | 1.39 | 0.37 | No |
| Habitat × eelgrass | Natural Reef | Eelgrass at site | 368 | 20 | 80.1 | 95 | 1.30 | 0.28 | No |
| Habitat × eelgrass | Natural Reef | No eelgrass at site | 1,377 | 78 | 290.9 | 229 | 0.77 | 0.13 | No |
| Habitat × eelgrass | Artificial Reef | No eelgrass at site | 468 | 17 | 126.4 | 39 | 0.20 | 0.06 | No |
| Habitat × eelgrass | Kelp Forest | No eelgrass at site | 59 | 17 | 13.6 | 2 | 0.03 | 0.03 | No |
| Habitat × eelgrass | Sponge Garden | Eelgrass at site | 3 | 1 | 1.0 | 0 | 0.00 | 0.00 | Yes |
| Habitat × eelgrass | Sponge Garden | No eelgrass at site | 12 | 2 | 3.1 | 0 | 0.00 | 0.00 | Yes |
| Basin × eelgrass | Whidbey | Eelgrass at site | 506 | 6 | 122.5 | 1,940 | 16.88 | 1.13 | No |
| Basin × eelgrass | Strait of Georgia | Eelgrass at site | 4 | 1 | 0.5 | 3 | 5.73 | 3.31 | Yes |
| Basin × eelgrass | Hood Canal | Eelgrass at site | 143 | 4 | 27.9 | 141 | 5.17 | 0.77 | No |
| Basin × eelgrass | Strait of Juan de Fuca | No eelgrass at site | 18 | 5 | 10.1 | 20 | 4.42 | 2.24 | Yes |
| Basin × eelgrass | Strait of Juan de Fuca | Eelgrass at site | 177 | 4 | 40.3 | 200 | 4.36 | 0.78 | No |
| Basin × eelgrass | Hood Canal | No eelgrass at site | 354 | 19 | 66.5 | 242 | 3.39 | 0.51 | No |
| Basin × eelgrass | Admiralty Inlet | Eelgrass at site | 90 | 3 | 15.9 | 7 | 0.57 | 0.28 | No |
| Basin × eelgrass | Admiralty Inlet | No eelgrass at site | 270 | 5 | 85.6 | 33 | 0.53 | 0.27 | No |
| Basin × eelgrass | Central | Eelgrass at site | 501 | 10 | 98.3 | 15 | 0.20 | 0.10 | No |
| Basin × eelgrass | Howe Sound | No eelgrass at site | 158 | 9 | 30.0 | 4 | 0.20 | 0.14 | No |
| Basin × eelgrass | Central | No eelgrass at site | 350 | 15 | 73.5 | 24 | 0.18 | 0.09 | No |
| Basin × eelgrass | San Juan | No eelgrass at site | 656 | 35 | 121.5 | 14 | 0.08 | 0.03 | No |
| Basin × eelgrass | South | Eelgrass at site | 162 | 3 | 29.5 | 2 | 0.07 | 0.05 | No |
| Basin × eelgrass | Whidbey | No eelgrass at site | 363 | 6 | 87.9 | 6 | 0.06 | 0.03 | No |
| Basin × eelgrass | Howe Sound | Eelgrass at site | 21 | 1 | 3.8 | 0 | 0.00 | 0.00 | Yes |
| Basin × eelgrass | Strait of Georgia | No eelgrass at site | 4 | 1 | 0.5 | 0 | 0.00 | 0.00 | Yes |
| Basin × eelgrass | San Juan | Eelgrass at site | 142 | 2 | 23.2 | 0 | 0.00 | 0.00 | Yes |
| Basin × eelgrass | South | No eelgrass at site | 59 | 4 | 10.5 | 0 | 0.00 | 0.00 | Yes |

### S Table 5. Length distribution-shape analyses

KS D is the two-sample Kolmogorov-Smirnov distance. Energy distance is a complementary distribution-shape statistic. Overall and habitat-specific site-eelgrass tests use site-label permutation p-values; S2 within-panel habitat-pair rows show site-cluster bootstrap intervals and individual-level KS q-values only as sensitivity values. Non-sparse contrasts are ranked by KS D within each analysis block.

| Panel/contrast | Comparison | N individuals | N sites | KS D | KS CI or p/q | Energy distance | Energy CI or p/q | Sparse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Soft Bottom only | Eelgrass at site vs No eelgrass at site | 1,052 vs 59 | 11 vs 10 | 0.623 | p=0.001; q=0.006 | 4.009 | p=0.001; q=0.005 | No |
| Natural Reef only | Eelgrass at site vs No eelgrass at site | 98 vs 180 | 9 vs 23 | 0.316 | p=0.370; q=0.616 | 1.953 | p=0.344; q=0.573 | No |
| Overall positive length distribution | Eelgrass at site vs No eelgrass at site | 1,698 vs 256 | 17 vs 31 | 0.259 | p=0.110; q=0.275 | 1.600 | p=0.051; q=0.126 | No |
| Kelp Forest only | Eelgrass at site vs No eelgrass at site | 22 vs 2 | 2 vs 1 | 0.409 | p=1.000; q=1.000 | 1.828 | p=1.000; q=1.000 | Yes |
| Artificial Reef only | Eelgrass at site vs No eelgrass at site | 384 vs 15 | 5 vs 5 | 0.353 | p=0.659; q=0.824 | 1.566 | p=0.643; q=0.803 | Yes |
| S2 panel: Eelgrass at site | Eelgrass vs Artificial Reef | 142 vs 384 | 6 vs 5 | 0.513 | 0.22–0.81 | 2.475 | 0.85–4.77 | No |
| S2 panel: Eelgrass at site | Soft Bottom vs Eelgrass | 1,052 vs 142 | 11 vs 6 | 0.459 | 0.20–0.59 | 2.038 | 0.80–2.72 | No |
| S2 panel: Eelgrass at site | Eelgrass vs Natural Reef | 142 vs 98 | 6 vs 9 | 0.441 | 0.21–0.91 | 3.112 | 1.04–6.21 | No |
| S2 panel: Eelgrass at site | Soft Bottom vs Natural Reef | 1,052 vs 98 | 11 vs 9 | 0.406 | 0.24–0.86 | 2.192 | 1.26–5.12 | No |
| S2 panel: Eelgrass at site | Artificial Reef vs Natural Reef | 384 vs 98 | 5 vs 9 | 0.371 | 0.23–0.81 | 1.970 | 1.20–4.69 | No |
| S2 panel: Eelgrass at site | Soft Bottom vs Artificial Reef | 1,052 vs 384 | 11 vs 5 | 0.123 | 0.05–0.67 | 0.549 | 0.17–3.44 | No |
| S2 panel: No eelgrass at site | Soft Bottom vs Natural Reef | 59 vs 180 | 10 vs 23 | 0.502 | 0.18–0.72 | 3.820 | 0.89–5.37 | No |

### S Table 6. ShoreZone random-forest model and top prediction-loss features

#### S Table 6A. Model performance

| Metric | Value |
| --- | --- |
| Survey sites with complete ShoreZone features | 104 |
| One-hot encoded ShoreZone features | 77 |
| Cross-validated R^2, mean ± SD | 0.104 ± 0.263 |
| Prediction vs observed Pearson r | 0.291 |
| Prediction vs observed p | 0.003 |
| RMSE, individuals hr^-1 | 5.08 |
| MAE, individuals hr^-1 | 2.64 |

#### S Table 6B. Top ShoreZone prediction-loss features

| Feature | Prediction loss | SE |
| --- | --- | --- |
| Eelgrass: continuous | 0.292 | 0.065 |
| BC shoreline type: Sand beach | 0.069 | 0.028 |
| Low-exposure sand/gravel habitat | 0.050 | 0.013 |
| Oyster beds: absent | 0.036 | 0.006 |
| Oyster beds: patchy | 0.009 | 0.004 |
| Fucus band: patchy | 0.009 | 0.002 |
| Eelgrass: absent | 0.005 | 0.002 |
| Supratidal + two intertidal + subtidal components | 0.005 | 0.003 |
| Eelgrass: patchy | 0.004 | 0.003 |
| Ulva green algae: absent | 0.003 | 0.002 |

### S Table 7. ShoreZone group contrasts and eelgrass agreement tests

#### S Table 7A. Protected-exposure by continuous-eelgrass group summaries

| Exposure group | ShoreZone eelgrass group | Sites | Mean encounter rate hr^-1 | SE | Median encounter rate hr^-1 | P. helianthoides | Survey hours |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Protected / very protected | Eelgrass: continuous | 6 | 15.29 | 5.97 | 11.75 | 1,931 | 110.6 |
| Protected / very protected | Other | 41 | 1.80 | 0.45 | 0.10 | 333 | 269.6 |
| Other exposure | Other | 62 | 1.01 | 0.32 | 0.00 | 336 | 429.6 |
| Other exposure | Eelgrass: continuous | 2 | 1.00 | 1.00 | 1.00 | 1 | 1.7 |

#### S Table 7B. Protected-exposure and continuous-eelgrass group tests

| Contrast | Test | Statistic | p |
| --- | --- | --- | --- |
| protected_continuous_eelgrass_vs_all_other | Mann-Whitney U | 532.50 | 0.003 |
| continuous_eelgrass_vs_other_zos | Mann-Whitney U | 631.00 | 0.008 |
| four_fig4b_groups | Kruskal-Wallis | 11.42 | 0.010 |
| protected_or_very_protected_vs_other_exposure | Mann-Whitney U | 1863.00 | 0.023 |

#### S Table 7C. ShoreZone eelgrass versus survey-recorded eelgrass agreement tests

| Agreement contrast | Test | Statistic | df | p | Effect | Effect value |
| --- | --- | --- | --- | --- | --- | --- |
| Three-category ShoreZone eelgrass by survey-recorded eelgrass | chi-square | 12.09 | 2 | 0.002 | Cramers V | 0.341 |
| Any ShoreZone eelgrass by survey-recorded eelgrass | Fisher exact | 3.55 |  | 0.005 | phi | 0.265 |

## Statistical analyses

Survey count-model analysis:

```text
code/diver_only_stats_for_report.py
```

This script generates negative-binomial count models and Poisson sensitivity models, using survey duration as an effort offset and site-clustered standard errors. The main manuscript reports coefficients as incidence-rate ratios (IRRs).

Length distribution-shape analysis:

```text
code/length_distribution_shape_stats_for_report.py
```

This script generates site-aware and descriptive length-distribution contrasts used by the manuscript, S2 Fig, S5 Fig, and supplemental tables.

ShoreZone statistical summary:

```text
scripts/analysis/shorezone_stats_for_report.py
```

This script summarizes the ShoreZone model and group comparisons used in Fig 4, S3 Fig, S4 Fig, S5-related text, and supplemental tables.

Primary statistical outputs are written to:

```text
outputs/publication_figures/stats/
outputs/publication_figures/qc/shorezone_report_stats/   # generated locally; not tracked by default
```

## Supplemental statistical tables

The combined supplemental table file is:

```text
outputs/publication_figures/supplemental_tables/supplemental_statistical_tables.md
```

Tracked table CSVs:

```text
outputs/publication_figures/supplemental_tables/table_s1_survey_analysis_input_summary.csv
outputs/publication_figures/supplemental_tables/table_s2_count_model_eelgrass_association.csv
outputs/publication_figures/supplemental_tables/table_s3_primary_count_model_terms.csv
outputs/publication_figures/supplemental_tables/table_s4_descriptive_cells_and_sparse_flags.csv
outputs/publication_figures/supplemental_tables/table_s5_length_distribution_shape_tests.csv
outputs/publication_figures/supplemental_tables/table_s6a_shorezone_model_summary.csv
outputs/publication_figures/supplemental_tables/table_s6b_shorezone_top_prediction_loss_features.csv
outputs/publication_figures/supplemental_tables/table_s7a_shorezone_fig4b_group_summary.csv
outputs/publication_figures/supplemental_tables/table_s7b_shorezone_fig4b_tests.csv
outputs/publication_figures/supplemental_tables/table_s7c_shorezone_eelgrass_agreement_tests.csv
```

Table-building scripts:

```text
scripts/tables/build_supplemental_statistical_tables.py
scripts/tables/build_doc_friendly_supplemental_tables.py
scripts/tables/build_docx_supplemental_tables.py
scripts/tables/build_google_docs_copy_paste_html.py
```

The Google Docs copy-paste HTML outputs are useful for manuscript drafting but are intentionally not tracked.

## Captions and manuscript-facing outputs

Captions are kept separately from figure images:

```text
outputs/publication_figures/captions.md
```

PLOS-style submission TIFFs are in:

```text
outputs/publication_figures/submission/
```

Editable/source PNG/PDF versions are in:

```text
outputs/publication_figures/sources/
```

## Rebuilding individual components

Run one script at a time with the Conda environment:

```bash
/home/weertman/miniforge3/bin/conda run -n star_meadow python code/figure_1_site_map.py
/home/weertman/miniforge3/bin/conda run -n star_meadow python code/diver_only_stats_for_report.py
/home/weertman/miniforge3/bin/conda run -n star_meadow python scripts/tables/build_supplemental_statistical_tables.py
```

Or rebuild the full publication surface:

```bash
./scripts/build_publication_outputs
```

## What is intentionally excluded from GitHub

The `.gitignore` is strict by design. It keeps local research history available on the analysis machine while preventing older or private material from becoming part of the public publication repository.

Excluded from the GitHub-facing surface:

- older exploratory scripts from the numbered pre-publication pipeline
- older generated outputs under `outputs/01_*` through older analysis folders
- the full ShoreZone FileGDB payload
- internal manuscript drafts and collaborator planning notes
- local QC/check/audit files
- Google Docs copy-paste helper HTML/TSV/DOCX outputs
- caches, logs, archives, local agent files, and operating-system cruft

If a file is needed for the paper and is ignored, move it into one of the publication-facing paths or adjust `.gitignore` intentionally rather than force-adding ad hoc files.

## Reproducibility status

Current status:

- publication figure scripts are tracked
- final publication figure outputs are tracked
- supplemental statistical tables are tracked
- old exploratory pipeline code and outputs are not tracked on the publication branch
- minimal derived ShoreZone inputs are tracked
- full ShoreZone source data are not tracked

Known reproducibility caveat:

The current repository contains derived ShoreZone inputs rather than the full third-party ShoreZone geodatabase. This keeps the repository small and publication-focused, but a fully independent public rerun from raw ShoreZone source data would require restoring or documenting the ShoreZone preprocessing step.

## Citation and license

Citation and license information should be added before public release, after the manuscript target, data-sharing permissions, and repository visibility are finalized.
