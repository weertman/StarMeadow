# StarMeadow Python Environment

This file documents the Python environment used to rebuild the publication-facing StarMeadow analyses and figures.

## Summary

- Repository path on the analysis workstation: `/home/weertman/Documents/StarMeadow`
- Environment manager: Miniforge/Conda
- Conda executable on the analysis workstation: `/home/weertman/miniforge3/bin/conda`
- Environment name: `star_meadow`
- Environment specification: `code/environment.yml`
- Preferred build command: `./scripts/build_publication_outputs`

Do not use bare system `python3` for analysis runs on the analysis workstation. The system Python lacks several required packages used by the publication scripts.

## Create the environment

From the repository root:

```bash
/home/weertman/miniforge3/bin/conda env create -f code/environment.yml
```

## Update an existing environment

If the environment already exists and dependencies change:

```bash
/home/weertman/miniforge3/bin/conda env update -n star_meadow -f code/environment.yml --prune
```

## Verify the environment

```bash
/home/weertman/miniforge3/bin/conda run -n star_meadow python --version
/home/weertman/miniforge3/bin/conda run -n star_meadow python - <<'PY'
import pandas, numpy, matplotlib, seaborn, scipy, sklearn, geopandas, fiona, pyproj, shapely, folium, joblib, statsmodels
print('StarMeadow environment ready')
PY
```

The analysis workstation environment has been verified with Python 3.11 and the packages listed in `code/environment.yml`.

## Rebuild publication outputs

Preferred command:

```bash
./scripts/build_publication_outputs
```

Compatibility wrapper:

```bash
./scripts/run_pipeline
```

Both commands run the publication-facing analysis scripts and write outputs under:

```text
outputs/publication_figures/
```

The build includes:

1. survey count-model analysis
2. length distribution-shape analysis
3. ShoreZone statistical summaries
4. main manuscript figures
5. supplemental figures
6. supplemental statistical tables

## Run an individual script

Examples:

```bash
/home/weertman/miniforge3/bin/conda run -n star_meadow python code/figure_1_site_map.py
/home/weertman/miniforge3/bin/conda run -n star_meadow python code/diver_only_stats_for_report.py
/home/weertman/miniforge3/bin/conda run -n star_meadow python scripts/tables/build_supplemental_statistical_tables.py
```

## Notes

- `code/run_all.py` was part of the older exploratory pipeline and is not part of the sparse publication repository.
- Some script names still contain historical `diver` terminology. Manuscript-facing language should use `survey`.
- Generated local QC files, old exploratory outputs, and full ShoreZone FileGDB data are intentionally ignored by Git.
