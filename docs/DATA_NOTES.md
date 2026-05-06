# StarMeadow Data Notes

Last updated: 2026-05-06.

These notes summarize the current canonical data files used by the analysis code. They are intended as a lightweight data contract for collaborators and for future reruns.

## Canonical files

- `data/PycnoCountCLean_12_31_2025.csv`
- `data/PycnoLengthCLean_12_31_2025.csv`
- `data/Site_LatLong.csv`
- `data/state_DNR_ShoreZone/`

## Current count-data summary

From `data/PycnoCountCLean_12_31_2025.csv`:

- Rows: 3,998
- Unique surveyed sites: 139
- Basins: 10
- Habitat types: 6
- Total *Pycnopodia* counted: 2,654
- Total survey effort: 51,060 minutes
- Transect-level detection rate: 14.7%
- Parsed date range after the known `2/20/0204` typo fix: 2020-07-04 to 2026-08-06
- Remaining unparseable/blank dates: 1

Observed basins:

- Admiralty Inlet
- Central
- Hood Canal
- Howe Sound
- Queen Charlotte Strait
- San Juan
- South
- Strait of Georgia
- Strait of Juan de Fuca
- Whidbey

Observed habitat types:

- Artificial Reef
- Eelgrass
- Kelp Forest
- Natural Reef
- Soft Bottom
- Sponge Garden

## Current length-data summary

From `data/PycnoLengthCLean_12_31_2025.csv`:

- Rows: 5,511
- Positive-length individuals: 1,957
- Positive size range: 1 to 85 cm
- Mean positive size: 19.37 cm
- Remaining non-numeric/bad length entries: 1
- Remaining unparseable/blank dates: 1

Zero-length rows should be treated as no-observation records, not zero-sized animals.

## Site coordinates

From `data/Site_LatLong.csv`:

- Rows: 240
- Unique coordinate sites: 240

Known issues:

- One longitude is positive in the source table; mapping code currently flips positive longitudes negative.
- Exact joins between count-data site names and coordinate-site names are incomplete.
- Coordinate-only sites may include unsurveyed future/priority sites and should not automatically be interpreted as errors.

## ShoreZone data

The ShoreZone folder contains Washington State DNR ShoreZone FileGDB-style data and metadata.

Important method note:

- Older documentation described a 250 m buffer spatial join.
- The active current script `code/15_shorezone_site_analysis.py` uses a nearest-neighbor join with `MAX_DISTANCE_M = 3000`.
- Therefore, the current site-level ShoreZone outputs should be interpreted as nearest-segment attributes, not true 250 m local habitat-diversity estimates.

## Curation questions before manuscript-final analysis

1. Are 2026 dates expected in files named `12_31_2025`?
2. What are the one bad/blank date and one bad length entry, and should they be corrected in source data or handled in code?
3. Which site-name mismatches should be corrected vs. treated as unsurveyed coordinate-only candidate sites?
4. Should `Region` be normalized, or should all analyses continue to use `Basin`?
5. Should the ShoreZone analysis report nearest-neighbor, buffer-based, or paired sensitivity results?
