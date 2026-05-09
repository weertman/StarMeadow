# Supplemental statistical tables

These tables collect the statistical analyses that support the manuscript figures. They are intended for supplemental materials so the inferential results are visible alongside the descriptive figures. Count models used survey effort offsets and SiteName-clustered standard errors. Length analyses emphasize distribution-shape contrasts because individual measurements are clustered within sites. ShoreZone analyses are exploratory habitat-context analyses because the mapped biotic layers pre-date the biological surveys.

## S Table 1. Survey analysis input summary

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

## S Table 2. Site-level eelgrass association across count models

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

## S Table 3. Primary adjusted count-model terms

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

## S Table 4. Descriptive habitat-by-eelgrass and basin-by-eelgrass cells

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

## S Table 5. Length distribution-shape analyses

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

## S Table 6. ShoreZone random-forest model and top prediction-loss features

### S Table 6A. Model performance

| Metric | Value |
| --- | --- |
| Survey sites with complete ShoreZone features | 104 |
| One-hot encoded ShoreZone features | 77 |
| Cross-validated R^2, mean ± SD | 0.104 ± 0.263 |
| Prediction vs observed Pearson r | 0.291 |
| Prediction vs observed p | 0.003 |
| RMSE, individuals hr^-1 | 5.08 |
| MAE, individuals hr^-1 | 2.64 |

### S Table 6B. Top ShoreZone prediction-loss features

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

## S Table 7. ShoreZone group contrasts and eelgrass agreement tests

### S Table 7A. Protected-exposure by continuous-eelgrass group summaries

| Exposure group | ShoreZone eelgrass group | Sites | Mean encounter rate hr^-1 | SE | Median encounter rate hr^-1 | P. helianthoides | Survey hours |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Protected / very protected | Eelgrass: continuous | 6 | 15.29 | 5.97 | 11.75 | 1,931 | 110.6 |
| Protected / very protected | Other | 41 | 1.80 | 0.45 | 0.10 | 333 | 269.6 |
| Other exposure | Other | 62 | 1.01 | 0.32 | 0.00 | 336 | 429.6 |
| Other exposure | Eelgrass: continuous | 2 | 1.00 | 1.00 | 1.00 | 1 | 1.7 |

### S Table 7B. Protected-exposure and continuous-eelgrass group tests

| Contrast | Test | Statistic | p |
| --- | --- | --- | --- |
| protected_continuous_eelgrass_vs_all_other | Mann-Whitney U | 532.50 | 0.003 |
| continuous_eelgrass_vs_other_zos | Mann-Whitney U | 631.00 | 0.008 |
| four_fig4b_groups | Kruskal-Wallis | 11.42 | 0.010 |
| protected_or_very_protected_vs_other_exposure | Mann-Whitney U | 1863.00 | 0.023 |

### S Table 7C. ShoreZone eelgrass versus survey-recorded eelgrass agreement tests

| Agreement contrast | Test | Statistic | df | p | Effect | Effect value |
| --- | --- | --- | --- | --- | --- | --- |
| shorezone_three_category_by_diver_binary | chi-square | 12.09 | 2 | 0.002 | Cramers_V | 0.341 |
| shorezone_any_eelgrass_by_diver_binary | Fisher exact | 3.55 |  | 0.005 | phi | 0.265 |
