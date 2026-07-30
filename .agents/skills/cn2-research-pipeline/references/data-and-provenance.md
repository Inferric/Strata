# Data and provenance

## Source priority

| Tier | Source | Label type | First use | Important constraint |
| --- | --- | --- | --- | --- |
| A | otbench MLO/USNA | Direct surface or path Cn² | First real run and benchmark reproduction | Preserve task-defined splits and acknowledgements |
| A | NCAR MLO_CN2 | Direct/sonic-derived near-surface Cn² | Few-shot and direct-label validation | Short, single high-altitude campaign |
| A | OTProf Zenodo | WRF teacher Cn² profiles paired with ERA5 | Column-model pretraining and reproduction | Synthetic teacher, not independent truth |
| A | ESO Paranal ASM | DIMM/MASS/SLODAR and meteorology | Profile/integrated multitask and long holdouts | Nighttime astronomical selection effects |
| A | TMT site testing | MASS/DIMM/SODAR/weather/sonic | Leave-site and instrument transfer | Login, usage, and publication conditions |
| B | Physics-video turbulence dataset | Video plus path Cn² | Observation encoder proof of concept | Short fixed-scene campaigns |
| B | CASES-99/MATERHORN-X/Perdigão | Turbulence and boundary-layer proxies | Representation pretraining | Derived proxy, not an optical label |
| Covariate | ERA5/ERA5-Land/HRRR | Meteorological fields | Context and forecast inputs | Reanalysis/model bias and resolution |

## Required raw manifest

Record:

- `dataset_id`, source URL/DOI, retrieval time, citation, license/terms, redistribution flag
- original filenames, byte sizes, hashes, transport method, and adapter version
- label provenance: direct, path-averaged, layer-integrated, teacher, derived proxy, or covariate
- instrument, geometry, wavelength, location, time basis, height/pressure coordinates, units, and QC flags
- known gaps, resampling, alignment tolerance, and exclusions

## Canonical observation shape

Each observation must retain:

- identity and provenance
- sample/site/campaign/source IDs
- point, path, profile, or integrated geometry
- instrument and observation operator
- meteorology and surface context
- target value or vector, units, wavelength, uncertainty, and quality flags
- immutable split/group identifiers

Do not coerce unlike observations into one unlabeled scalar table. Use explicit observation operators and masks.

## Acquisition boundaries

- Download only openly accessible files without human acceptance steps.
- For TMT or another gated archive, prepare an import command and stop for the user to acquire or authorize the files.
- Never bypass robots, access controls, rate limits, publication conditions, or export/public-release requirements.
- Check raw and normalized artifacts into DVC metadata, not Git.
