# Temp12k Preparation

## Workflow Position

Temp12k preparation is an explicit, receipt-tracked stage:

```text
preprocess -> eof -> train-lim -> calibrate-pages2k -> prepare-temp12k
  -> estimate-borehole-error -> assimilate
```

When preparation is enabled, assimilation reads
`stages/prepare-temp12k/temp12k.pkl`. When disabled, the stage records and
fingerprints the legacy `observations.temp12k.path` instead. Inputs are never
modified in place.

## Scientific Procedure

The implementation combines the four legacy scripts in `paper/`:

1. `build_temp12k_cfr_proxydb.py`: read `Temp12k1_1_0.pkl`, retain records
   whose compilation is exactly `Temp12k` and units are exactly `degC`, and
   convert BP ages to CE with `CE = 1950 - BP`. Year/CE values are retained
   directly. Nonfinite samples are removed, times are sorted, and the first
   value is retained at duplicate timestamps. Native sampling is not
   interpolated.
2. Retain only records with latitude, longitude, at least two samples, and a
   finite positive `paleoData_temperature12kUncertainty`. The direct
   temperature observation variance is `R = uncertainty^2`.
3. Parse reported seasonality using hemisphere-aware warm/cold seasons.
   `fix_temp12k_seasonality.py` is applied in memory: calendar annual
   `[1,...,12]` becomes the December-to-November convention
   `[-12,1,...,11]`, July-only becomes JJA, and January-only becomes DJF.
4. Subtract each Temp12k record's mean over `anomaly_period`, normally
   0-2000 CE. Rounded integer years select the native samples, whose ungrouped
   mean defines the baseline, matching the old script.
   Records without baseline samples remain unchanged by default, reproducing
   the legacy final database; `require_anomaly_baseline: true` drops them.
5. Load raw PAGES2k, apply the same anomaly period when available, and create
   annual comparison series by rounding sample ages to integer years and
   averaging samples assigned to the same year.
6. Test every pair within 50 km. A pair is an operational duplicate only when
   correlation is strictly greater than 0.98 and overlap count is strictly
   greater than 11. All overlapping rounded years are then removed from the
   Temp12k record; nonoverlapping history is retained. Records with fewer than
   two remaining samples are dropped.
7. Repeat the duplicate scan and fail if any operational duplicate remains.

The separate exploratory rule from
`find_crossdb_duplicates_pages2k_temp12k.py` is also retained: pairs with at
least 20 overlaps and absolute correlation at least 0.8 are written to a
diagnostic CSV. It does not control removal.

## Configuration

```yaml
observations:
  temp12k:
    enabled: true
    strict: true
    preparation:
      enabled: true
      source_path: /local/path/Temp12k1_1_0.pkl
      source_format: raw_temp12k
      pages2k_source_name: PAGES2kv2
      pages2k_source_path: null
      anomaly_period: [0, 2000]
      require_anomaly_baseline: false
      comparison_period: [0, 2000]
      maximum_distance_km: 50
      duplicate_correlation_minimum: 0.98
      duplicate_minimum_overlap: 11
      diagnostic_absolute_correlation_minimum: 0.8
      diagnostic_minimum_overlap: 20
      minimum_record_points: 2
```

Use `source_format: cfr_proxydb` to rerun anomaly referencing, seasonality
normalization, and PAGES2k deduplication from an existing pre-dedup CFR pickle.
Set `pages2k_source_path` to a local ProxyDatabase pickle or ReconJob directory
for a network-independent run. Otherwise CFR fetches `pages2k_source_name`.

## Artifacts

- `temp12k.pkl`: final CFR compatibility database consumed by assimilation.
- `build_diagnostics.csv`: raw-record retention and rejection counts.
- `candidate_pairs_before.csv` and `candidate_pairs_after.csv`: every tested
  location candidate and the operational duplicate decision.
- `likely_duplicate_candidates.csv`: exploratory absolute-correlation result.
- `removed_overlap_segments.csv`: points and rounded years removed per record.
- `manifest.json`: sources, periods, strict inequalities, method choices, and
  record counts.

The CFR pickle is a compatibility boundary. Assimilation immediately converts
its records to the standalone `ObservationRecord` contract.
