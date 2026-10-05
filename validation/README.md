# VALLIS-3C scientific validation data

This directory contains the numerical products required to audit and reproduce the validation results reported in the VALLIS-3C 1.6.0 manuscript. The temporal-texture selector was trained on 1,623 eligible records.

`model_training_usage_manifest.csv` is the release-level event-station crosswalk for the spectral calibration, component-ratio calibration, temporal-model training and empirical runtime-support populations.

The package is organized by validation layer. CSV files are plain machine-readable tables without decorative spreadsheet formatting. Every validation subdirectory includes a manifest and SHA-256 checksums. No validation table or manifest contains a machine-local path.

## Manuscript-to-data map

| Manuscript item | Validation data |
|---|---|
| Table 1 — regional spectral benchmark | `01_spectral_model/benchmark_summary.csv`, `01_spectral_model/eventwise_metrics.csv`, `01_spectral_model/external_gmpe_paired_comparisons.csv` |
| CU spectral conditional center — leave-one-event-out validation | `01_spectral_model/cu_eventwise_loeo_baselines.csv`, `01_spectral_model/cu_loeo_summary.csv`, `01_spectral_model/cu_loeo_paired_comparisons.csv`, `01_spectral_model/cu_loeo_residual_trends.csv` |
| Persistent local site-response validation | `01_spectral_model/site_amplification_loso.csv` |
| Section 2.3 — event-grouped duration OOF statistics and independent baseline study | `02_duration_model/duration_oof_summary.csv`, `02_duration_model/duration_baseline_study_recordwise_oof.csv`, `02_duration_model/duration_baseline_study_eventwise_oof.csv`, `02_duration_model/duration_baseline_study_summary.csv`, `02_duration_model/duration_baseline_study_paired_comparisons.csv`, `02_duration_model/duration_baseline_study_residual_trends.csv` |
| Fig. 4 — 2017 network-wide strict OOS validation | `03_oos_2017/network_metrics.csv`, `03_oos_2017/network_metrics_bootstrap_summary.csv` |
| Figs. 5–7 — TCYA, AU46 and GC38 FAS/response spectra | `03_oos_2017/*_spectral_curves.csv`, `03_oos_2017/oos2017_spectral_summary.csv`, `03_oos_2017/oos2017_spectral_all_realizations.csv` |
| Figs. 5–7 — generated 3C histories and duration metadata | `03_oos_2017/oos2017_generated_ground_motions.npz`, `03_oos_2017/oos2017_durations.csv` |
| Figs. 5–7 — Q=3 nonlinear response | `03_oos_2017/station_hysteretic_energy_q3.csv` |
| Supplementary strict OOS check — 2012 network metrics | `04_oos_2012/network_metrics.csv` |
| Supplementary strict OOS check — 2012 station bootstrap | `04_oos_2012/network_metrics_bootstrap_summary.csv` |
| Fig. 8 — SASID/VALLIS response spectra and model overlays | `05_sasid/figure8_response_spectra.csv` |
| Table 2 — duration and within-suite spectral variability | `05_sasid/table2_summary.csv`, `05_sasid/spectral_sigma_lnSa_summary.csv` |
| Fig. 9 — representative histories, FAS and Arias accumulation | `05_sasid/figure9_representative_ground_motions.npz`, `05_sasid/representative_ground_motion_fas.csv`, `05_sasid/figure9_normalized_arias.npz` |
| Fig. 9 — duration/PGA and nested-duration diagnostics | `05_sasid/sasid_vallis_duration_pga.csv`, `05_sasid/duration_detail_sasid_vallis.csv`, `05_sasid/duration_suite_summary.csv`, `05_sasid/nested_duration_summary.csv` |
| Fig. 10 — plotted nonlinear P50 curves | `05_sasid/figure10_nonlinear_response.csv` |
| Nonlinear uncertainty, Q = 2, 3 and 4 | `05_sasid/nonlinear_Umax_EH_percentiles_SASID_Q2_Q3_Q4.csv`, `05_sasid/nonlinear_Umax_EH_percentiles_VALLIS_Q2_Q3_Q4.csv` |
| Period-conditioned VALLIS scaling | `05_sasid/conditioned_SaT_scaling_factors_all_sites_sources_VALLIS.csv` |

## 03_oos_2017

Strict withheld-event validation for the 19 September 2017 Puebla–Morelos earthquake. The event is excluded from all data-dependent stages of the OOS model.

The folder includes 30 generated three-component realizations for TCYA, AU46 and GC38 at `dt = 0.01 s`, compact FAS and 5%-damped response-spectrum data for every realization, observed and simulated summaries, significant durations, network-wide metrics, and Q=3 horizontal-RMS specific hysteretic-energy curves.

The original RAII-UNAM and RACM/CIRES acceleration time histories are not included in this directory. Observed scalar and spectral products used in the manuscript comparisons are included as research outputs. See the subdirectory README and manifest for array keys, units, and provenance.

## 04_oos_2012

Strict withheld-event validation for the 20 March 2012 Ometepec earthquake. Its `OOS2012_2017` model excludes both 20 March 2012 and 19 September 2017 from every fitted spectral and temporal stage and from eligible texture donors.

The 54-station common-support table contains observed values and 30-realization P05/P16/P50/P84/P95 summaries for horizontal and vertical significant duration, Arias intensity and average spectral acceleration, together with Q=3 hysteretic energy. Its station-bootstrap table applies the same residual definitions, 10,000 resamples and seed 916 used for the 2017 network validation.

## 05_sasid

Site-specific comparison for four sites, two source scenarios, and 30 horizontal pairs per method and case. The package includes:

- response spectra used in Fig. 8;
- representative SASID and VALLIS ground motions used in Fig. 9;
- representative-motion FAS;
- normalized Arias-intensity curves;
- component-level duration and PGA data;
- significant-duration suite summaries and nested-duration ratios;
- mean within-suite `sigma_lnSa` over 0.1–5.0 s;
- period-by-period P05/P16/P50/P84/P95 peak displacement and specific hysteretic energy for Q = 2, 3 and 4;
- the effective period-conditioned scale factor applied to each VALLIS horizontal pair. SASID is excluded from that table because the SASID suites are already spectrally matched.

## Reviewer auditability

The serialized runtime artifacts (`.joblib`, `.npz` and `.h5`) are execution assets, not the sole validation evidence. The CSV/JSON products in this directory expose the benchmark metrics, event-grouped or fold-level results, strict withheld-event products, suite percentiles and integrity hashes without requiring a reviewer to deserialize a model.

`01_spectral_model` addresses two questions: competitiveness against regional published models and event generalization of the CU conditional center against low-complexity baselines. The paired bootstrap intervals support competitiveness but do not imply universal dominance over every external model.

`02_duration_model` contains the model OOF summary and an independent baseline study with record-level grouped predictions, event summaries, low-complexity baselines, paired event-bootstrap comparisons and residual-trend diagnostics. The baseline study uses the same population and architecture with an independently constructed deterministic fold assignment; small differences between the two summaries reflect the distinct fold assignment.

`03_oos_2017` and `04_oos_2012` provide strict end-to-end validations of the withheld events. Their bootstrap tables summarize duration, Arias intensity and Q=3 hysteretic energy without duplicating the underlying station results.

## Data conventions

- `P05`, `P16`, `P50`, `P84`, `P95`: empirical percentiles across the stated simulation suite.
- `H_RMS`: quadratic/RMS combination of the two horizontal components used by the corresponding analysis.
- `V`: vertical component.
- `IS` / `INSLAB`: intermediate-depth intraslab earthquake.
- `IP` / `INTERPLATE`: subduction-interface earthquake.
- `Ts`: dominant site period in seconds.
- Missing values are retained where frequency support is unavailable; values are not silently extrapolated.

## Integrity and licensing

`manifest.json` and `SHA256SUMS.txt` in every validation subdirectory provide machine-readable provenance and file-integrity checks. Run `python audit_validation_package.py` to verify hashes, schemas, nonlinear percentile consistency and the absence of local paths. The distributed validation products are fixed research artifacts; no data-reconstruction scripts are required. Generated VALLIS histories and the derived numerical products used in the manuscript are included; the original RAII-UNAM and RACM/CIRES acceleration time histories are not.

Installation/runtime `VALIDATION_REPORT.json` files elsewhere in the repository are software self-test products and are not part of this scientific validation dataset.
