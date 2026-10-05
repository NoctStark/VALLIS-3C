# 04_oos_2012 — strict out-of-sample validation for 20 March 2012

This directory contains the station-level validation of the 20 March 2012
Ometepec earthquake.  The simulations use the `OOS2012_2017` VALLIS-3C model,
which excludes both 20 March 2012 and 19 September 2017 from every fitted
spectral and temporal model and from the eligible texture-donor population.
Each station is represented by 30 three-component simulations.

## Files

- `network_metrics.csv`: observed values and simulated P05, P16, P50, P84 and
  P95 for horizontal/vertical significant duration, Arias intensity and
  average spectral acceleration, plus Q=3 hysteretic energy.
- `network_metrics_bootstrap_summary.csv`: station-bootstrap summary for
  horizontal/vertical duration, Arias intensity and Q=3 hysteretic energy.
  It reports RMSE and bias of `ln(observed/P50)`, 95% bootstrap intervals and
  empirical P16–P84/P05–P95 coverage with bootstrap intervals.
- `network_metrics_bootstrap_summary.csv` was computed from `network_metrics.csv`
  using 10,000 station resamples and random seed 916.
- `manifest.json` and `SHA256SUMS.txt`: provenance and integrity records.

The compact table contains 54 stations with complete support across all
metrics. Arias intensity remains in the same `m/s` unit used by the 2017
validation tables. For Q=3 hysteretic
energy, logarithmic residual statistics use the 49 stations for which both the
observed value and simulated P50 are at least 1 cm²/s²; all 54 station rows are
retained in `network_metrics.csv`.

Provider-native acceleration samples are not redistributed.  The distributed
tables contain derived validation quantities only.
