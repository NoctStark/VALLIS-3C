# 02_duration_model

Compact event-grouped out-of-fold validation of the production horizontal significant-duration model.

## Files

### `duration_oof_summary.csv`

Fields:
- `source_type`
- `n_records`
- `n_events`
- `oof_rmse_ln_DH`
- `event_balanced_mean_bias_ln`

Folds are grouped by earthquake, so the reported errors correspond to predictions for events excluded from model fitting rather than station interpolation within represented events.

The manuscript reports approximately 0.224 RMSE for intraslab events and 0.201 for interplate events, with negligible logarithmic bias.

### `duration_baseline_study_recordwise_oof.csv`

Record-level five-fold predictions grouped by earthquake for the final 1,623-record population. It contains a production-like random forest, ridge regression and source-mean baseline. `residual_pred_minus_obs` is prediction minus observation in logarithmic duration.

### `duration_baseline_study_eventwise_oof.csv`

Event-level RMSE, MAE and bias derived from the recordwise archive.

### `duration_baseline_study_summary.csv`

Event-mean performance and 95% event-bootstrap intervals by source and model.

### `duration_baseline_study_paired_comparisons.csv`

Paired event-bootstrap RMSE differences between the production-like random forest and each baseline. Negative differences favor VALLIS.

### `duration_baseline_study_residual_trends.csv`

Spearman residual diagnostics against magnitude, rupture distance, depth and site period. The associations describe remaining calibration structure and are not causal estimates.

## Scope

This directory documents predictive performance of the duration center. The fitted production model is distributed with the model/software release and is not duplicated here.

The baseline study independently constructs deterministic event-grouped folds using the same population and architecture. It is separate from the OOF values in `duration_oof_summary.csv`, so small numerical differences are expected. The random forest clearly improves over a source-only duration center. Its advantage over ridge is modest, especially for interplate, and a weak interplate magnitude trend remains. These limitations are retained in the diagnostics rather than hidden by aggregate metrics.

`manifest.json` and `SHA256SUMS.txt` provide file-integrity checks.
