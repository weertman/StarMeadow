# StarMeadow

Publication-oriented analysis repository for *Pycnopodia helianthoides* (sunflower sea star) encounter rates, size distributions, habitat associations, eelgrass-site effects, and ShoreZone-informed recovery hypotheses in the Salish Sea.

This repository is being prepared as the shared code/data/figure backbone for a manuscript with collaborators. It contains the analysis scripts, canonical input data tables, generated output products, environment specification, and documentation needed to reproduce and refine the current results.

## Current status

Last documentation refresh: 2026-05-06.

This repo now reflects the current 2025-named data files in `data/`, which include some records with 2026 dates that still need human/data-curation review.

| Item | Current value |
|---|---:|
| Survey/transect records | 3,998 |
| Unique surveyed sites in count table | 139 |
| Site-coordinate records | 240 |
| Basins in count table | 10 |
| Habitat types in count table | 6 |
| Total *Pycnopodia* counted | 2,654 |
| Total survey effort | 51,060 minutes |
| Transect-level detection rate | 14.7% |
| Count date range after known typo fix | 2020-07-04 to 2026-08-06 |
| Length-table rows | 5,511 |
| Positive-length individuals | 1,957 |
| Positive size range | 1 to 85 cm |
| Mean positive size | 19.37 cm |

Data-quality notes that matter for interpretation:

- `data/PycnoCountCLean_12_31_2025.csv` and `data/PycnoLengthCLean_12_31_2025.csv` contain one unparseable/blank date each after the known `2/20/0204` -> `2/20/2024` fix.
- The length file contains one non-numeric/bad length value.
- The count file includes dates extending into 2026 despite the `12_31_2025` filename; confirm whether these are valid newer entries or data-entry issues.
- Exact site-name joins between survey data and `data/Site_LatLong.csv` are incomplete; some coordinate-only sites may be unsurveyed/priority sites, while some count-data sites need coordinate/name cleanup.
- `Region` values are not normalized; most analyses rely on `Basin` instead.

## Repository layout

```text
StarMeadow/
├── code/                     # Numbered Python analysis scripts and shared utilities
│   ├── run_all.py             # Runs the available analysis scripts sequentially
│   ├── utils.py               # Shared data loading, paths, plotting helpers
│   ├── shorezone_utils.py     # ShoreZone/GIS loading and spatial-join helpers
│   ├── environment.yml        # Conda environment specification
│   ├── 01_*.py ... 16_*.py    # Core, mapping, and ShoreZone/recovery analyses
│   └── 19_probability_density_map.py
├── data/                     # Canonical input data and ShoreZone GIS data
├── outputs/                  # Generated tables, figures, maps, and model artifacts
├── docs/                     # Manuscript/project planning documents
├── scripts/                  # Repo helper scripts
├── ENVIRONMENT.md            # Python/Conda setup and verification notes
└── README.md                 # This file
```

Private collaborator correspondence exported from email is intentionally excluded from git via `.gitignore`.

## Canonical inputs

### Count data

`data/PycnoCountCLean_12_31_2025.csv`

Survey/transect-level table used by `utils.load_data()`.

Important columns:

- `Date`
- `Survey.Time`
- `Region`
- `SiteName`
- `Basin`
- `Water.Temperature..F.`
- `DepthBin`
- `Transect..`
- `HabitatType`
- `Pycno.notes..disease.presence..behavior..etc..`
- `Pycnopodia_count`

The loader computes:

```text
Encounter.Rate.Hr = (Pycnopodia_count / Survey.Time) * 60
```

### Length data

`data/PycnoLengthCLean_12_31_2025.csv`

Individual/observation-level size table used by `utils.load_length_data()`. The loader converts `Length(cm)` to numeric `Length_cm`; positive values are treated as measured individuals, while zero-length records represent no-observation rows and should not be interpreted as zero-size animals.

### Site coordinates

`data/Site_LatLong.csv`

Coordinate table used by mapping and ShoreZone scripts. One positive longitude is currently handled by map code by flipping it negative, but the source table should still be reviewed.

### ShoreZone GIS data

`data/state_DNR_ShoreZone/`

Washington State DNR ShoreZone FileGDB-style data and metadata used by scripts 15, 15b, 16, and 19.

Important interpretation note: older documentation described a 250 m buffer method, but the active current `15_shorezone_site_analysis.py` path uses a 3 km nearest-neighbor join from each site to the nearest ShoreZone segment. Under nearest-neighbor mode, output “diversity” metrics are compatibility placeholders rather than true local habitat-diversity estimates.

## Analysis scripts

| Script | Purpose |
|---|---|
| `01_encounter_rate_analysis.py` | Encounter rates by habitat, basin, depth, and habitat-depth combinations |
| `02_size_structure_analysis.py` | Size distributions by habitat, depth, basin, and season |
| `03_clustering_analysis.py` | Site clustering by habitat composition |
| `04_temporal_analysis.py` | Annual/monthly/temporal effort and observation trends |
| `05_statistical_summary.py` | Statistical tests and publication-style summary tables |
| `06_eelgrass_site_analysis.py` | Site-level eelgrass effects and four-category habitat classification |
| `07_size_clustering_analysis.py` | Size-bin/site clustering and eelgrass-size comparisons |
| `08_combined_clustering_analysis.py` | Combined habitat and size clustering visualizations |
| `09_size_prediction_analysis.py` | Size prediction models from habitat summaries |
| `10_eelgrass_size_relationship.py` | Eelgrass-site relationships with individual and site-level size |
| `11_habitat_diversity_analysis.py` | Habitat richness/diversity vs abundance with spatial-coverage controls |
| `12_eelgrass_basin_analysis.py` | Eelgrass effects while controlling for basin/geography |
| `13_static_maps.py` | Static site/effort/encounter/eelgrass/basin/hotspot maps |
| `14_interactive_maps.py` | Folium interactive site and encounter maps |
| `15_shorezone_site_analysis.py` | ShoreZone nearest-neighbor site join and site-level feature export |
| `15b_site_coverage_map.py` | Interactive coverage/diagnostic map and embedded analysis report |
| `16_shorezone_recovery_analysis.py` | ShoreZone recovery/refugia/model/priority-site analyses |
| `19_probability_density_map.py` | Segment-level predicted encounter-rate map from trained model |

`run_all.py` runs the current analysis pipeline through `19_probability_density_map.py`. The previous `20_publication_figures.py` script and generated `outputs/20_publication_figures/` products were removed because manuscript figures will be rebuilt from scratch.

## Current high-level findings from existing outputs

These values come from the current input tables and existing generated outputs; they should be revalidated after the clean environment rerun.

Encounter-rate summary by habitat from `outputs/01_encounter_rate_analysis/encounter_rate_summary.csv`:

| Habitat | Mean rate (Pycno/hr) | Detection interpretation |
|---|---:|---|
| Soft Bottom | 8.673 | Highest mean encounter rate in current outputs |
| Eelgrass | 4.104 | Second-highest mean encounter rate |
| Artificial Reef | 1.789 | Lower than soft bottom/eelgrass |
| Kelp Forest | 0.911 | Low mean rate |
| Natural Reef | 0.875 | Low mean rate |
| Sponge Garden | 0.000 | Present in current data but no detections in current summary |

Manuscript framing from `docs/Outline.docx` emphasizes:

- Whether habitat type affects *P. helianthoides* recruitment/encounter patterns.
- Whether habitat or geography is more important in driving recruitment/recovery patterns.
- Whether eelgrass/soft-bottom/hard-bottom context can inform conservation, restoration, or critical-habitat thinking.
- A figure set covering surveyed sites, encounter rates by habitat and space, reef habitat with/without eelgrass at site level, length by habitat/space, and settlement-related collaborator work.

## Environment setup

Use the Conda environment documented in `ENVIRONMENT.md`.

Quick start on this machine:

```bash
/home/weertman/miniforge3/bin/conda env create -f code/environment.yml
```

If it already exists:

```bash
/home/weertman/miniforge3/bin/conda env update -n star_meadow -f code/environment.yml --prune
```

Verify:

```bash
/home/weertman/miniforge3/bin/conda run -n star_meadow python --version
/home/weertman/miniforge3/bin/conda run -n star_meadow python -m py_compile code/*.py
```

## Running analyses

Preferred helper:

```bash
./scripts/run_pipeline
```

Direct command:

```bash
cd code
/home/weertman/miniforge3/bin/conda run -n star_meadow python run_all.py
```

Generated outputs are written to `outputs/<script_stem>/`.

## Reproducibility and publication cleanup priorities

Before treating outputs as manuscript-final:

1. Create/verify the Conda environment and rerun the full current pipeline with a timestamped log.
2. Reconcile 2026 dates in the `12_31_2025` files.
3. Fix or explicitly flag the remaining bad date/length values.
4. Normalize/join site names between survey and coordinate tables.
5. Decide whether the ShoreZone analysis should remain nearest-neighbor, return to a 250 m buffer approach, or report both.
6. Treat ShoreZone model/priority-map outputs as exploratory unless validation improves.
7. Build new publication figures from scratch after analysis outputs are revalidated; the old publication-figure script/output set has been removed.

## GitHub/publication sharing notes

This repository is intended for code/data sharing with collaborators and eventually publication support. Before making it public, review:

- Data-sharing permissions for the survey tables.
- Whether all site coordinates can be public.
- Whether generated output files should be committed or regenerated by users.
- Whether manuscript drafts/outlines should remain in the repository or move to a private writing workspace.

Email exports and private correspondence are excluded from git by default.
