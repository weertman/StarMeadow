# StarMeadow Python Environment

30-second summary:

- Repo path: `/home/weertman/Documents/StarMeadow`
- Canonical environment manager: Miniforge/Conda
- Canonical Conda executable on this machine: `/home/weertman/miniforge3/bin/conda`
- Intended env name: `star_meadow`
- Environment spec: `code/environment.yml`
- Do not use bare system `python3` for analysis runs; the system Python lacks key dependencies such as seaborn, scipy, scikit-learn, geopandas, fiona, shapely, pyproj, folium, and joblib.

## Bootstrap

From the project root:

```bash
/home/weertman/miniforge3/bin/conda env create -f code/environment.yml
```

If the environment already exists and dependencies change:

```bash
/home/weertman/miniforge3/bin/conda env update -n star_meadow -f code/environment.yml --prune
```

## Verify

```bash
/home/weertman/miniforge3/bin/conda run -n star_meadow python --version
/home/weertman/miniforge3/bin/conda run -n star_meadow python - <<'PY'
import pandas, numpy, matplotlib, seaborn, scipy, sklearn, geopandas, fiona, pyproj, shapely, folium, joblib
print('StarMeadow environment ready')
PY
```

## Run the pipeline

Current helper:

```bash
./scripts/run_pipeline
```

Equivalent direct command:

```bash
cd code
/home/weertman/miniforge3/bin/conda run -n star_meadow python run_all.py
```

Notes:

- `run_all.py` has been updated to use the current available script list: scripts 01-16, 15b, 19, and 20.
- Rerun logs should still be checked carefully because this is a research pipeline with generated outputs rather than a formal test suite.
- Generated figures/tables are written to `outputs/<script_stem>/`.
