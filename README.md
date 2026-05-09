# StarMeadow

Publication repository for the StarMeadow analysis of *Pycnopodia helianthoides* encounter rates, size distributions, eelgrass-associated habitat patterns, and ShoreZone habitat context in the Salish Sea.

This GitHub-facing branch is intentionally sparse. It is meant to contain the materials directly used for the manuscript: publication figure scripts, statistical-analysis scripts, final figure/table outputs, canonical input tables approved for sharing, and environment/setup documentation. Older exploratory analyses remain available locally on the analysis computer but are ignored or untracked so they do not clutter the public repository.

## Repository layout

```text
StarMeadow/
├── code/                         # Publication figure/statistical scripts
├── data/                         # Canonical manuscript input tables and public metadata
├── scripts/                      # Publication build and table/helper scripts
├── outputs/publication_figures/   # Final manuscript figures, captions, and tables
├── docs/                         # Public-facing documentation only
├── ENVIRONMENT.md                # Environment setup notes
└── README.md
```

## Current publication scripts

Main figures:

- `code/figure_1_site_map.py`
- `code/figure_2_habitat_eelgrass_size.py`
- `code/figure_3_basin_eelgrass_encounter_rate.py`
- `code/figure_4_shorezone.py`

Supplemental figures:

- `code/supplemental_figure_1_effort_basin_map.py`
- `code/supplemental_figure_2_length_by_habitat_eelgrass_status.py`
- `code/supplemental_figure_3_continuous_eelgrass_map.py`
- `code/supplemental_figure_4_shorezone_vs_diver_eelgrass.py`
- `code/supplemental_figure_5_survey_statistical_summary.py`

Statistical analyses and tables:

- `code/diver_only_stats_for_report.py`
- `code/length_distribution_shape_stats_for_report.py`
- `scripts/analysis/shorezone_stats_for_report.py`
- `scripts/tables/build_supplemental_statistical_tables.py`
- `scripts/tables/build_doc_friendly_supplemental_tables.py`

Current note: some script names still use historical `diver` terminology. The manuscript text uses `survey` terminology.

## Rebuild publication outputs

Preferred command:

```bash
./scripts/build_publication_outputs
```

The command uses the Conda environment documented in `ENVIRONMENT.md` and writes outputs under:

```text
outputs/publication_figures/
```

## Publication outputs kept in the sparse repo

Final figure files:

- `outputs/publication_figures/submission/`
- `outputs/publication_figures/sources/`

Captions:

- `outputs/publication_figures/captions.md`

Supplemental statistical tables:

- `outputs/publication_figures/supplemental_tables/table_s*.csv`
- `outputs/publication_figures/supplemental_tables/supplemental_statistical_tables.md`
- `outputs/publication_figures/supplemental_tables/supplemental_statistical_tables_doc_friendly.pdf`

Statistical source outputs used by tables/figures:

- `outputs/publication_figures/stats/`

## Data and ShoreZone notes

The survey count, length, and site-coordinate tables are the canonical manuscript inputs. Exact sharing scope for coordinates and source survey tables should be confirmed before making a public release.

The full Washington DNR ShoreZone FileGDB is not included in the sparse GitHub-facing surface by default. Public metadata files are retained, and the current scripts use minimal derived ShoreZone products that were generated locally from the ShoreZone source data. For long-term public reproducibility, the preferred path is to document the public ShoreZone source and/or provide a clean preprocessing script that regenerates the small derived tables used by the manuscript.

## What is intentionally not in the sparse public surface

The following remain local/private or are regenerated as needed:

- older exploratory scripts and outputs from the pre-publication numbered pipeline
- large third-party ShoreZone FileGDB payloads
- internal manuscript drafts and collaborator planning documents
- internal QC/check scripts and audit folders
- copy-paste helper outputs for Google Docs
- logs, caches, archives, and local agent scratch files

This keeps the repository focused on the publication rather than the full exploratory history.
