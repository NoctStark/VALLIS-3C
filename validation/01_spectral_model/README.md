# 01_spectral_model

Compact data supporting the conditional spectral-center benchmark and validation of the persistent local site-response layer.

## Files

### `benchmark_summary.csv`

One row per external benchmark in Table 1.

Fields:
- `domain`: response spectrum or Fourier amplitude spectrum.
- `source_type`: `INSLAB` or `INTERPLATE`.
- `reference`: external regional model.
- `range`: comparison period/frequency interval.
- `N_events`: number of events in the paired comparison population.
- `reference_mean_event_RMSE`: mean event-level logarithmic RMSE of the reference model.
- `VALLIS_mean_event_RMSE`: mean event-level logarithmic RMSE of the VALLIS conditional center.
- `paired_bootstrap_ci95_low`, `paired_bootstrap_ci95_high`: 95% paired event-bootstrap interval for the RMSE difference used in the benchmark interpretation.

The INSLAB benchmark uses all 23 CU earthquakes with valid observed spectra and scenario parameters within the applicability range of Jaimes et al. (2015); 8 September 2017 is excluded because it lies outside that range. The INTERPLATE benchmarks retain the common 15-event CU population valid for Reyes (1999), Leonardo-Suárez et al. (2023), Arroyo et al. (2024), and VALLIS, preserving event-by-event pairing across the three regional comparisons.

### `eventwise_metrics.csv`

Event-level values underlying the benchmark summary.

Fields:
- `domain`
- `source_type`
- `event_id`
- `model`
- `rmse_ln`
- `bias_ln`

Use event-level pairing when recomputing differences between VALLIS and an external reference.

### `external_gmpe_paired_comparisons.csv`

Paired event-bootstrap comparisons derived from `eventwise_metrics.csv`. A negative `mean_delta_RMSE` favors VALLIS. The confidence interval is the 2.5–97.5 percentile interval from 20,000 event resamples using seed 20260923.

### `cu_eventwise_loeo_baselines.csv`

Leave-one-CU-event-out results for 68 events and five frequency bands. It compares the production conditional center with:

- `ridge`: standardized ridge regression using magnitude, logarithmic distance and depth;
- `source_mean`: source-specific mean log spectrum.

The table retains event identifiers and event metadata so residual structure can be audited.

### `cu_loeo_summary.csv`

Event-mean RMSE, MAE and bias with 95% event-bootstrap intervals, separated by source, frequency band and model.

### `cu_loeo_paired_comparisons.csv`

Paired event-level RMSE differences and VALLIS win fractions against both baselines. Negative differences favor VALLIS.

### `cu_loeo_residual_trends.csv`

Spearman correlations of full-band VALLIS bias with magnitude, distance and depth. These are residual diagnostics rather than causal tests.

### `site_amplification_loso.csv`

Leave-one-station-out diagnostics for the persistent local site-response texture.

The retained rows correspond to:
- `FULL`: production formulation using the complete database;
- `OOS2017`: strict 2017 withheld-event variant;
- `OOS2012_2017`: strict variant withholding both the 2012 and 2017 validation events.

The model uses normalized frequency `u=f*Ts`, support 0.35–4.0, full application over 0.50–3.0, XY–ln(Ts) KNN interpolation, taper to the identity at support limits, and no extrapolation outside observed support. All three rows use the same 72-station formulation; the record and event counts reflect each variant's exclusion policy.

## Scope

These files audit the distributed architecture. The CU analysis demonstrates event generalization and clear improvement over a source-mean spectrum. The gain over ridge is strong for intraslab and smaller for interplate; the results therefore support competitiveness rather than universal model dominance.

`manifest.json` and `SHA256SUMS.txt` provide file-integrity checks.
