# 05_sasid — reproducibility data for Figures 8–10 and Table 2

The data correspond to the VALLIS simulations and SASID comparison suites used in the manuscript.

## Files
- `figure8_response_spectra.csv`: complete response-spectrum data used in Figure 8.
- `figure9_representative_ground_motions.npz`: the four representative horizontal histories (SASID H1/H2 and VALLIS Major/Intermediate) for each of the four sites and two source types; dt=0.01 s, cm/s².
- `figure9_normalized_arias.npz`: normalized Arias curves for all 30 realizations, components, methods, sites and sources.
- `representative_ground_motion_fas.csv`: compact raw one-sided FAS derived from the representative Figure 9 histories; this auxiliary FAS is not a plotted source for Figures 8–10.
- `figure10_nonlinear_response.csv`: exact period-dependent P50 Umax and hysteretic-energy curves used in Figure 10.
- `nonlinear_Umax_EH_percentiles_SASID_Q2_Q3_Q4.csv` and `nonlinear_Umax_EH_percentiles_VALLIS_Q2_Q3_Q4.csv`: P05/P16/P50/P84/P95 curves calculated directly from the 30 horizontal pairs for both `Umax` and horizontal-RMS specific hysteretic energy (`EH_RMS`), for Q = 2, 3 and 4. These percentiles are not reconstructed from the median.
- `conditioned_SaT_scaling_factors_all_sites_sources_VALLIS.csv`: one row per VALLIS pair and oscillator period. It contains pre-broadband component and pair-RMS Sa plus `effective_scale_factor = broadband_scale_factor * conditional_scale_cT`. Multiplying the pre-broadband ordinates by this factor gives the period-conditioned ordinates. SASID is intentionally absent because those suites are already spectrally matched.
- `sasid_vallis_duration_pga.csv`: duration windows and PGA for all 240 VALLIS horizontal component pairs.
- `table2_summary.csv`, `spectral_sigma_lnSa_summary.csv`, `duration_suite_summary.csv`, and `duration_detail_sasid_vallis.csv`: values supporting Table 2 and the SASID discussion.
- `nested_duration_summary.csv`: case-level nested-duration ratios. The observational reference is restricted to records with both horizontal components marked complete, matched by source type and Ts within ±20% of the target; records are summarized within earthquake first and then across earthquakes so that each earthquake receives equal weight.
- `observed_event_level_nested_duration.csv`: event-level observational ratios underlying the balanced reference.
- `nested_duration_aggregate_summary.csv` and `nested_duration_bootstrap_summary.csv`: medians across the eight cases and 95% bootstrap intervals. Simulated suites are resampled by realization within case; observations are resampled by earthquake within case.
- `nested_duration_complete_event_balanced_report.json`: machine-readable definition and audit of the event-balanced reference.
- `manifest.json` and `SHA256SUMS.txt`: provenance and integrity.

All CSV files are plain data tables without decorative fills, borders, workbook-specific formatting or machine-local paths. Site/source/method labels are logical identifiers rather than filesystem locations.
