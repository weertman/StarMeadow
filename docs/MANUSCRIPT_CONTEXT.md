# StarMeadow Manuscript Context

Last updated: 2026-05-06.

This note summarizes the project/manuscript context from `docs/Outline.docx` without replacing that working document.

## Central manuscript questions

The working outline frames the manuscript around these questions:

1. Does habitat type affect *Pycnopodia helianthoides* recruitment/encounter patterns, especially juvenile or recovering populations?
2. Is habitat or geography more important in driving observed recruitment/recovery patterns?
3. Can specific features within superficially similar habitats explain differences in encounter rates or size distributions?
4. Can the findings support management or restoration decisions for sunflower stars and subtidal habitats such as eelgrass?

## Current conceptual frame

The outline positions species recovery as spatial and habitat-dependent: recovery depends not only on population status but also on identifying, protecting, and restoring habitats that support key life-history stages.

For this project, the recurring candidate patterns are:

- Higher encounter rates in eelgrass and soft-bottom habitat relative to some hard-bottom categories.
- Site-level eelgrass presence potentially improving encounter rates on nearby reef habitat.
- Strong geographic structure, especially Whidbey and Hood Canal patterns.
- Management relevance if eelgrass/soft-bottom/hard-bottom context can inform recovery planning or critical-habitat thinking.

## Figure/storyboard needs from the outline

The outline calls for figures along these lines:

1. Map of surveyed sites, likely with habitat and/or effort information.
2. Encounter by habitat across space/basin.
3. Reef habitat with vs. without eelgrass at the site level.
4. Length/size by habitat and space.
5. Settlement/collaborator figure material outside the current StarMeadow script set.

Existing script coverage:

- `20_publication_figures.py` currently creates four figure panels: site map, encounter by habitat/basin, reef/eelgrass effect, and size by habitat/basin.
- Additional refinement is expected before manuscript submission.

## Analysis priorities implied by the outline

Before figure polishing, the repo should support:

- Reproducible rerun of all current StarMeadow scripts.
- Data-quality checks around current 2025/2026 data.
- Clear separation between habitat effects and geography/basin effects.
- Explicit treatment of eelgrass as both transect-level habitat and site-level context.
- Cautious interpretation of ShoreZone results because active code currently uses nearest-neighbor shoreline attributes rather than the older documented 250 m buffer-diversity method.

## Privacy/collaboration note

The outline is manuscript-working material. Avoid quoting or publishing full text from `Outline.docx` without collaborator agreement. This summary is meant to guide reproducible analysis and figure planning, not to serve as a manuscript draft.
