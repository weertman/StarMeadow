# StarMeadow Next Steps

Last updated: 2026-05-06.

## Current reproducible baseline

- GitHub repo: `https://github.com/weertman/StarMeadow`
- Visibility at creation: private
- Local repo path: `/home/weertman/Documents/StarMeadow`
- Canonical env: Conda env `star_meadow`
- Environment spec: `code/environment.yml`
- Pipeline helper: `./scripts/run_pipeline`
- Latest successful rerun log: `logs/run_all-20260506-115906.log` (ignored by git)
- Latest pushed commit at time of this note: see `git log --oneline -5` for the current cleanup/refinement state.

## What is now done

1. Local git repository initialized on `main`.
2. Private GitHub repository created and pushed.
3. Email exports/private correspondence excluded via `.gitignore`.
4. Documentation refreshed to current 2025-named data state.
5. Conda environment created and verified.
6. Full current pipeline rerun completed successfully.
7. Pipeline manifest fixed to include existing scripts and exit nonzero on failures.
8. Current compatibility bugs fixed for pandas/seaborn behavior in temporal/recovery scripts.

## Known warnings from the successful rerun

These did not stop the run but should guide cleanup:

- Constant-input correlation warnings occur in ShoreZone diversity/correlation sections. This is expected under the current nearest-neighbor ShoreZone path because several diversity metrics are placeholders or have no variance.
- Seaborn future warnings remain in `16_shorezone_recovery_analysis.py` for some boxplot palette usage. These are not currently fatal but should be cleaned before long-term maintenance.

## Immediate analysis validation priorities

1. Data curation checks
   - Resolve/confirm records dated in 2026 in files named `12_31_2025`.
   - Identify and fix/flag the one unparseable date in each canonical CSV.
   - Identify and fix/flag the one bad/non-numeric length value.
   - Normalize obvious site-name and region spelling inconsistencies.

2. Coordinate/site reconciliation
   - Produce a site-name mismatch report between `PycnoCountCLean_12_31_2025.csv` and `Site_LatLong.csv`.
   - Classify coordinate-only sites as unsurveyed candidates vs. spelling/merge problems.
   - Confirm whether all site coordinates are appropriate for GitHub sharing before public release.

3. ShoreZone method decision
   - Decide whether manuscript analyses should use:
     - current 3 km nearest-neighbor ShoreZone segment attributes,
     - restored 250 m buffer/diversity joins,
     - or both as a sensitivity comparison.
   - Avoid calling current nearest-neighbor outputs true local habitat-diversity products.

4. Model/priority-map validation
   - Treat current RandomForest/priority-site outputs as exploratory.
   - Review validation outputs and confidence interval coverage.
   - Decide whether ShoreZone-only prediction is manuscript-relevant or should be demoted to supplemental/exploratory material.

## New publication figure priorities

The old `code/20_publication_figures.py` implementation and `outputs/20_publication_figures/` products have been removed. Do not use them as a design base. New manuscript figures should be built from scratch using the validated analysis outputs and shared data-loading utilities.

1. Figure 1 candidate: site map by habitat/effort
   - Verify site coverage, basins, and coordinate joins.
   - Decide whether point size should represent survey effort, annual effort, total effort, or detections.

2. Figure 2 candidate: encounter by habitat and basin
   - Confirm whether habitat categories should include Sponge Garden or be collapsed/excluded due to low sample size.
   - Decide whether mean encounter rate is the primary statistic or whether detection/zero-inflation should be shown alongside it.

3. Figure 3 candidate: reef eelgrass effect
   - Central manuscript figure candidate.
   - Validate four-category habitat logic and site-level eelgrass classification.
   - Consider paired basin/site controls and effect-size display.

4. Figure 4 candidate: size by habitat/basin
   - Confirm positive-length filtering and treatment of zero/no-observation rows.
   - Decide whether juvenile/recruitment thresholds should be added if manuscript framing emphasizes recruitment.

## Code cleanup priorities

1. Centralize repeated site-level eelgrass and four-category habitat logic.
2. Add a data-contract validation script under `scripts/` or `code/`.
3. Add a lightweight smoke-test command that checks imports, data columns, script list, and selected output existence without rerunning the full pipeline.
4. Remove or update stale comments/docstrings that still imply a 250 m buffer method when active code uses nearest-neighbor.
5. Clean remaining seaborn future warnings.

## Collaboration/publication repo policy questions

Before making the repo public:

- Are the canonical data CSVs approved for public sharing?
- Are the 240 site coordinates approved for public sharing?
- Should `docs/Outline.docx` remain in the GitHub repo, or should manuscript drafts live elsewhere?
- Should generated outputs be committed for collaborator convenience, or should only code/data be committed with outputs generated on demand?
- Is the ShoreZone FileGDB redistribution permitted as currently committed, or should scripts download/reference the public source instead?
