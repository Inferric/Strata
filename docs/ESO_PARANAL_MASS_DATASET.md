# ESO Paranal MASS profile acquisition protocol

Status: **completed 2026-07-31; dataset QC PASS, Column training not
authorized**

## Purpose and boundary

Fusion v2.5 was the strongest custom surface model, but its rolling primary
RMSE margin over MLP was only 1.32% and its tail MAE narrowly missed the frozen
gate. The completed shortcut, loss, scale, cap, and bilinear searches do not
justify another surface-model retune. This bounded data cycle instead asks:

> Can an anonymous, clearly open, directly observed optical-turbulence profile
> source be acquired reproducibly and pass provenance, geometry, unit, and
> checksum checks strongly enough to support a later, separately
> preregistered Strata-OT Column experiment?

This is dataset evidence, not model evidence. It must not release the official
MLO test, the reserved USNA confirmation labels, or any other sealed rows. It
does not authorize Column training, model promotion, or a claim that MASS
layers are pointwise \(C_n^2\).

## Source, access, and terms

- Source: the [ESO Paranal MASS query
  form](https://archive.eso.org/wdb/wdb/asm/mass_paranal/form).
- Instrument and field definitions: the [ESO Paranal ambient-conditions help
  page](https://archive.eso.org/wdb/help/eso/ambient_paranal.html).
- Archive terms: ESO describes its public archive as openly accessible after
  any proprietary period and distributed under a Creative Commons Attribution
  license on the [ESO Science Archive
  page](https://www.eso.org/public/science/archive/).
- Access method: anonymous HTTPS POST to the documented WDB query endpoint.
  No account, login, cookie, click-through acceptance, or credential is
  permitted.
- Citation: acknowledge ESO and cite Kornilov et al. (2007), *MNRAS* 382,
  1268--1278, for the MASS-DIMM instrument.

The exact URLs, request fields, access time, response headers, and checksums
must be recorded. If the service redirects to authentication, changes its
terms, or no longer exposes the documented public fields, acquisition stops.

## Frozen bounded request

The request is defined in
`configs/data/eso_paranal_mass_profile_v1.yaml`:

- UTC interval: `2020-01-01..2020-02-01`;
- maximum rows returned: 50,000;
- output: CSV;
- maximum accepted response: 25,000,000 bytes;
- scientific fields: free-atmosphere seeing, uncertainty, coherence time,
  isoplanatic angle, ground-layer fraction, MASS-DIMM integrated quantities,
  characteristic turbulence speed, and airmass;
- profile fields: the six MASS layer strengths and uncertainties at 0.5, 1,
  2, 4, 8, and 16 km plus the DIMM-derived ground layer at 0 km.

Raw bytes are written once under
`data/raw/eso-paranal-mass/canonical-v1/`. Existing bytes are never
overwritten. A checked-in manifest records their SHA-256 hashes while Git
continues to exclude the raw snapshot.

## Observation semantics

MASS measures stellar scintillation and statistically inverts it into
low-resolution layer strengths. The archive labels these fields as Cn2, with
units `10**(-15) m**(1/3)`. Dimensionally and operationally they are integrated
turbulence strengths \(J_i\), not local point samples in
\(\mathrm{m}^{-2/3}\). The ground layer is a MASS-DIMM difference. Therefore:

- label provenance is direct observation with an inversion operator;
- the seven response layers remain separate from surface/path targets;
- values are stored in archive units without silent conversion;
- no surface and profile headline metric may be pooled;
- all later models must retain the observation operator, height grid,
  wavelength, site, and uncertainty metadata.

## Frozen QC and success conditions

Acquisition succeeds only if all of the following hold:

1. the response is anonymous HTTPS CSV, not HTML or an authentication page;
2. byte count is positive and no larger than 25 MB;
3. the header exactly contains the requested profile fields;
4. timestamps parse as UTC-naive archive times, lie within the frozen request,
   are nondecreasing, and have no duplicates;
5. every populated layer strength and uncertainty is finite and nonnegative;
6. every row reports the expected seven-layer grid when `gridsize` is present;
7. populated heights match 0, 500, 1,000, 2,000, 4,000, 8,000, and 16,000 m;
8. missingness, cadence, temporal coverage, row count, and per-field finite
   counts are recorded rather than silently filtered;
9. request and snapshot bytes are hashed in a schema-valid manifest;
10. immediate manifest re-verification reproduces exact size and SHA-256.

Failure is reported honestly; no synthetic or substituted rows are allowed.

## Column decision gate

Passing acquisition QC only establishes a usable public profile source. Before
Strata-OT Column may train, a new protocol must freeze:

- a chronological, purged, multi-season development/confirmation split;
- response-aware profile metrics and integrated optical metrics;
- train-only preprocessing and calibration;
- direct-profile versus teacher-profile separation;
- persistence/climatology and published MASS-profile controls;
- multi-seed gates, resource limits, and an explicit confirmation release.

This one-month snapshot is deliberately too narrow for a promotion claim. It
can validate the adapter and profile geometry; a longer multi-season snapshot
requires a separate frozen acquisition plan after this QC result is reviewed.

## Materialized result

The frozen anonymous request completed without changing its interval, fields,
limits, or checks. The checked-in manifest is
`data/manifests/eso-paranal-mass-profile-2020-01-v1.json`; raw response bytes
remain ignored by Git and immutable on disk.

- 7,857 unique observations and 38 returned columns span
  `2020-01-01T00:34:31` through `2020-01-31T09:23:57`.
- The CSV contains 1,655,957 bytes with SHA-256
  `737430b9fd764f5732664c02a16ae2601261c10dc9fb1b58edcb3dab631e7afc`.
- Median cadence is 79 seconds, p95 cadence is 98 seconds, and the maximum gap
  is 400,830 seconds; any future model protocol must define a strict gap
  policy.
- All populated layer strengths and uncertainties are finite and
  nonnegative. `MASS-DIMM Tau0` and `MASS-DIMM Turb Velocity` each have 72
  missing rows (0.92%); no other requested field has missing values.
- Immediate checksum verification and the frozen provenance, schema,
  timestamp, geometry, and boundary checks passed.

The visually inspected LaTeX dataset-QC report is
`reports/generated/eso-paranal-mass-qc/eso-paranal-mass-qc-report.pdf`.
This PASS applies only to adapter and geometry validation. The one-month,
single-site, nighttime sample and multi-day cadence gap leave Column training,
confirmation evaluation, and champion promotion explicitly unevaluated.
