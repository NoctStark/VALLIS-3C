# 03_oos_2017 — reproducibility data for Figures 5–7

The data were generated with VALLIS-3C 1.6.0. Each station contains 30 three-component simulations.

## Files
- `oos2017_generated_ground_motions.npz`: all generated Major/Intermediate/Vertical histories; arrays are padded with zeros and accompanied by per-record `npts`. Units: cm/s²; dt=0.01 s. Provider-native observed acceleration samples are intentionally excluded from the public VALLIS package.
- `TCYA_spectral_curves.csv`, `AU46_spectral_curves.csv`, `GC38_spectral_curves.csv`: observed curve plus P05/P16/P50/P84/P95 of the simulation suite for FAS and response spectra.
- `oos2017_spectral_summary.csv`: the three station summaries in one file.
- `oos2017_spectral_all_realizations.csv`: observed derived curves and every simulated FAS/Sa realization in long form.
- `oos2017_durations.csv`: D2.5–97.5 values for the 30 simulations and observed record used in the figure calculations. Machine-local source paths are intentionally excluded.
- `station_hysteretic_energy_q3.csv`: Q=3 elastoplastic-perfect SDOF hysteretic-energy spectra, 51 log-spaced periods from 0.1 to 5 s, NTC-2004 yield strength and 5% damping.
- `network_metrics.csv`: 59-station network validation for horizontal/vertical duration, Arias intensity, average spectral acceleration and Q=3 hysteretic energy.
- `network_metrics_bootstrap_summary.csv`: station-bootstrap summary for horizontal/vertical duration, Arias intensity and Q=3 hysteretic energy. It reports RMSE and bias of `ln(observed/P50)`, 95% bootstrap intervals, and empirical P16–P84/P05–P95 coverage with bootstrap intervals.
- `network_metrics_bootstrap_summary.csv` was computed from `network_metrics.csv` using 10,000 station resamples and random seed 916.
- `manifest.json` and `SHA256SUMS.txt`: provenance and integrity.

Observed acceleration arrays are not redistributed because the CIRES/RAII data remain subject to provider terms. The observed derived curves are retained for verification.

The aggregate bootstrap results support accurate duration transfer and no conclusive median bias for Arias intensity or hysteretic energy. Hysteretic energy retains substantially larger station-to-station error, and the empirical simulation bands are conservative; these are reported as limitations rather than interpreted as selector validation.
