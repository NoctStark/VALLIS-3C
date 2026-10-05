"""Audit the public VALLIS-3C scientific-validation package."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUBDIRECTORIES = (
    "01_spectral_model",
    "02_duration_model",
    "03_oos_2017",
    "04_oos_2012",
    "05_sasid",
)
LOCAL_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|Users)/)")
TEXT_SUFFIXES = {".csv", ".json", ".md", ".txt", ".py"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_integrity() -> dict:
    result = {}
    for dirname in SUBDIRECTORIES:
        directory = ROOT / dirname
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        entries = {entry["name"]: entry for entry in manifest["files"]}
        expected_data = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
            and path.relative_to(directory).as_posix()
            not in {"manifest.json", "SHA256SUMS.txt"}
        }
        if set(entries) != expected_data:
            raise AssertionError(f"{dirname}: manifest file set is incomplete")
        for name, entry in entries.items():
            path = directory / Path(name)
            if path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
                raise AssertionError(f"{dirname}: manifest mismatch for {name}")

        checksum_rows = {}
        for line in (directory / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
            digest, name = line.split(maxsplit=1)
            checksum_rows[name] = digest
        expected_checksums = {
            path.relative_to(directory).as_posix()
            for path in directory.rglob("*")
            if path.is_file()
            and path.relative_to(directory).as_posix() != "SHA256SUMS.txt"
        }
        if set(checksum_rows) != expected_checksums:
            raise AssertionError(f"{dirname}: checksum file set is incomplete")
        for name, digest in checksum_rows.items():
            if sha256(directory / Path(name)) != digest:
                raise AssertionError(f"{dirname}: checksum mismatch for {name}")
        result[dirname] = {"status": "PASS", "files_hashed": len(checksum_rows)}
    return result


def audit_local_paths() -> dict:
    hits = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        if LOCAL_PATH.search(text):
            hits.append(path.relative_to(ROOT).as_posix())
    if hits:
        raise AssertionError(f"Machine-local paths found in: {hits}")
    return {"status": "PASS", "machine_local_path_hits": 0}


def audit_csvs() -> dict:
    counts = {}
    for path in ROOT.rglob("*.csv"):
        table = pd.read_csv(path)
        if table.columns.empty:
            raise AssertionError(f"No columns in {path.relative_to(ROOT)}")
        counts[path.relative_to(ROOT).as_posix()] = int(len(table))
    return {"status": "PASS", "files_parsed": len(counts), "row_counts": counts}


def assert_percentiles(table: pd.DataFrame, columns: list[str], label: str) -> None:
    values = table[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all() or np.any(np.diff(values, axis=1) < -1.0e-10):
        raise AssertionError(f"{label}: invalid or non-monotone percentiles")


def audit_other_validations() -> dict:
    spectral_dir = ROOT / "01_spectral_model"
    benchmark = pd.read_csv(spectral_dir / "benchmark_summary.csv")
    eventwise = pd.read_csv(spectral_dir / "eventwise_metrics.csv")
    model_map = {
        "Jaimes et al. (2015)": "JA15",
        "Reyes (1999)": "REY99",
        "Leonardo-Suárez et al. (2023)": "LS23",
        "Arroyo et al. (2024) FAS + Ordaz et al. (2024) RVT": "ARR24_RVT",
        "Arroyo et al. (2024), direct FAS": "ARR24",
    }
    domain_map = {
        "Response spectrum": "Sa_0.2_5.0_s",
        "Fourier amplitude spectrum": "FAS_0.1_10_Hz",
    }
    comparison_map = {
        "JA15": "VALLIS - JA15",
        "REY99": "VALLIS - REY99",
        "LS23": "VALLIS - LS23",
        "ARR24_RVT": "VALLIS - ARR24_RVT",
        "ARR24": "VALLIS - ARR24",
    }
    metric_key = ["domain", "source_type", "event_id", "model"]
    if len(eventwise) != 136 or eventwise.duplicated(metric_key).any():
        raise AssertionError("01_spectral_model: eventwise coverage or uniqueness mismatch")
    if not np.isfinite(eventwise[["rmse_ln", "bias_ln"]].to_numpy(float)).all():
        raise AssertionError("01_spectral_model: non-finite eventwise metric")

    external_paired = pd.read_csv(spectral_dir / "external_gmpe_paired_comparisons.csv")
    if len(external_paired) != len(benchmark):
        raise AssertionError("01_spectral_model: external paired comparison mismatch")
    for row in benchmark.itertuples(index=False):
        domain = domain_map[row.domain]
        reference_code = model_map[row.reference]
        subset = eventwise[
            eventwise["domain"].eq(domain)
            & eventwise["source_type"].eq(row.source_type)
        ]
        reference = subset[subset["model"].eq(reference_code)]
        vallis = subset[subset["model"].eq("VALLIS")]
        if reference["event_id"].nunique() != row.N_events or vallis["event_id"].nunique() != row.N_events:
            raise AssertionError("01_spectral_model: event count mismatch")
        paired = reference[["event_id", "rmse_ln"]].merge(
            vallis[["event_id", "rmse_ln"]],
            on="event_id",
            how="inner",
            validate="one_to_one",
            suffixes=("_reference", "_VALLIS"),
        )
        if len(paired) != int(row.N_events):
            raise AssertionError("01_spectral_model: event pairing mismatch")
        np.testing.assert_allclose(reference["rmse_ln"].mean(), row.reference_mean_event_RMSE)
        np.testing.assert_allclose(vallis["rmse_ln"].mean(), row.VALLIS_mean_event_RMSE)
        delta = paired["rmse_ln_VALLIS"] - paired["rmse_ln_reference"]
        comparison = external_paired[
            external_paired["domain"].eq(domain)
            & external_paired["source_type"].eq(row.source_type)
            & external_paired["comparison"].eq(comparison_map[reference_code])
        ]
        if len(comparison) != 1:
            raise AssertionError("01_spectral_model: paired comparison key mismatch")
        comparison = comparison.iloc[0]
        if int(comparison["n_events"]) != len(paired):
            raise AssertionError("01_spectral_model: paired event count mismatch")
        np.testing.assert_allclose(delta.mean(), comparison["mean_delta_RMSE"])
        np.testing.assert_allclose((delta < 0.0).mean(), comparison["VALLIS_event_win_fraction"])
        np.testing.assert_allclose(
            row.VALLIS_mean_event_RMSE - row.reference_mean_event_RMSE,
            comparison["mean_delta_RMSE"],
        )
        np.testing.assert_allclose(row.paired_bootstrap_ci95_low, -comparison["ci95_high"])
        np.testing.assert_allclose(row.paired_bootstrap_ci95_high, -comparison["ci95_low"])

    site_loso = pd.read_csv(spectral_dir / "site_amplification_loso.csv")
    expected_site_counts = {
        "FULL": (2224, 76, 72),
        "OOS2017": (2165, 75, 72),
        "OOS2012_2017": (2110, 74, 72),
    }
    if set(site_loso["variant"]) != set(expected_site_counts):
        raise AssertionError("01_spectral_model: site-amplification variant mismatch")
    for row in site_loso.itertuples(index=False):
        observed = (int(row.n_records), int(row.n_events), int(row.n_stations))
        if observed != expected_site_counts[row.variant]:
            raise AssertionError("01_spectral_model: site-amplification population mismatch")
    site_metrics = site_loso[["texture_profile_rmse", "peak_corr", "peak_mae"]].to_numpy(float)
    if not np.isfinite(site_metrics).all():
        raise AssertionError("01_spectral_model: non-finite site-amplification metric")

    cu_eventwise = pd.read_csv(spectral_dir / "cu_eventwise_loeo_baselines.csv")
    if len(cu_eventwise) != 1020 or cu_eventwise["event_id"].nunique() != 68:
        raise AssertionError("01_spectral_model: CU LOEO coverage mismatch")
    if set(cu_eventwise["model"]) != {"VALLIS", "ridge", "source_mean"}:
        raise AssertionError("01_spectral_model: CU LOEO baseline mismatch")
    if cu_eventwise["frequency_band_hz"].nunique() != 5:
        raise AssertionError("01_spectral_model: CU LOEO frequency-band mismatch")
    if set(cu_eventwise.groupby("source_type")["event_id"].nunique()) != {24, 44}:
        raise AssertionError("01_spectral_model: CU source event counts mismatch")
    cu_summary = pd.read_csv(spectral_dir / "cu_loeo_summary.csv")
    if len(cu_summary) != 90 or set(cu_summary["n_events"].astype(int)) != {24, 44}:
        raise AssertionError("01_spectral_model: CU summary mismatch")
    cu_paired = pd.read_csv(spectral_dir / "cu_loeo_paired_comparisons.csv")
    if len(cu_paired) != 20 or not (cu_paired["comparison"].str.startswith("VALLIS - ")).all():
        raise AssertionError("01_spectral_model: CU paired comparison mismatch")
    cu_trends = pd.read_csv(spectral_dir / "cu_loeo_residual_trends.csv")
    if len(cu_trends) != 6:
        raise AssertionError("01_spectral_model: CU residual trend mismatch")
    duration = pd.read_csv(ROOT / "02_duration_model" / "duration_oof_summary.csv")
    duration_manifest = json.loads(
        (
            ROOT.parent
            / "models/release/variants/full/temporal/duration_clock_final_hv_manifest.json"
        ).read_text(encoding="utf-8")
    )
    for row in duration.itertuples(index=False):
        source = duration_manifest["sources"][row.source_type]
        if int(row.n_records) != source["records"] or int(row.n_events) != source["events"]:
            raise AssertionError("02_duration_model: training population mismatch")
        np.testing.assert_allclose(row.oof_rmse_ln_DH, source["oof_rmse_ln_H"])
        np.testing.assert_allclose(row.event_balanced_mean_bias_ln, source["oof_bias_ln_H"])

    duration_records = pd.read_csv(ROOT / "02_duration_model" / "duration_baseline_study_recordwise_oof.csv")
    if len(duration_records) != 4869 or duration_records["record_id"].nunique() != 1623:
        raise AssertionError("02_duration_model: recordwise grouped OOF coverage mismatch")
    if set(duration_records["model"]) != {"VALLIS_RF", "ridge", "source_mean"}:
        raise AssertionError("02_duration_model: baseline model mismatch")
    per_model = duration_records.groupby("model")["record_id"].nunique()
    if set(per_model.astype(int)) != {1623}:
        raise AssertionError("02_duration_model: incomplete predictions by model")
    event_fold_counts = duration_records.groupby(["source_type", "model", "event_id"])["fold"].nunique()
    if set(event_fold_counts.astype(int)) != {1}:
        raise AssertionError("02_duration_model: an event crosses grouped folds")
    duration_events = pd.read_csv(ROOT / "02_duration_model" / "duration_baseline_study_eventwise_oof.csv")
    recomputed = (
        duration_records.assign(
            sq=lambda frame: frame["residual_pred_minus_obs"] ** 2,
            ab=lambda frame: frame["residual_pred_minus_obs"].abs(),
        )
        .groupby(["source_type", "model", "event_id"], as_index=False)
        .agg(
            rmse_ln=("sq", lambda values: np.sqrt(np.mean(values))),
            mae_ln=("ab", "mean"),
            bias_ln=("residual_pred_minus_obs", "mean"),
            n_records=("record_id", "size"),
        )
    )
    sort_duration = ["source_type", "model", "event_id"]
    pd.testing.assert_frame_equal(
        duration_events.sort_values(sort_duration).reset_index(drop=True),
        recomputed.sort_values(sort_duration).reset_index(drop=True),
        check_exact=False,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    if len(pd.read_csv(ROOT / "02_duration_model" / "duration_baseline_study_summary.csv")) != 18:
        raise AssertionError("02_duration_model: baseline summary mismatch")
    if len(pd.read_csv(ROOT / "02_duration_model" / "duration_baseline_study_paired_comparisons.csv")) != 4:
        raise AssertionError("02_duration_model: paired comparison mismatch")
    if len(pd.read_csv(ROOT / "02_duration_model" / "duration_baseline_study_residual_trends.csv")) != 8:
        raise AssertionError("02_duration_model: residual trend mismatch")

    oos = ROOT / "03_oos_2017"
    network = pd.read_csv(oos / "network_metrics.csv")
    if len(network) != 413 or network["station_id"].nunique() != 59:
        raise AssertionError("03_oos_2017: network coverage mismatch")
    if set(network.groupby("metric")["station_id"].nunique()) != {59} or set(network["n_sim"]) != {30}:
        raise AssertionError("03_oos_2017: incomplete metric or simulation coverage")
    assert_percentiles(
        network,
        ["sim_p05", "sim_p16", "sim_p50", "sim_p84", "sim_p95"],
        "03_oos_2017/network_metrics.csv",
    )
    bootstrap = pd.read_csv(oos / "network_metrics_bootstrap_summary.csv")
    expected_metrics = {"D2p5_97p5_H", "D2p5_97p5_V", "IA_H", "IA_V", "EH_Q3"}
    if len(bootstrap) != 5 or set(bootstrap["metric"]) != expected_metrics:
        raise AssertionError("03_oos_2017: bootstrap summary metric mismatch")
    if set(bootstrap["n_stations"].astype(int)) != {59}:
        raise AssertionError("03_oos_2017: bootstrap station count mismatch")
    for row in bootstrap.itertuples(index=False):
        group = network[network["metric"].eq(row.metric)]
        residual = group["residual_ln_obs_over_p50"].to_numpy(float)
        np.testing.assert_allclose(row.rmse_ln_obs_over_p50, np.sqrt(np.mean(residual**2)))
        np.testing.assert_allclose(row.bias_ln_obs_over_p50, np.mean(residual))
        np.testing.assert_allclose(
            row.coverage_p16_p84,
            np.mean(group["observed"].between(group["sim_p16"], group["sim_p84"])),
        )
        np.testing.assert_allclose(
            row.coverage_p05_p95,
            np.mean(group["observed"].between(group["sim_p05"], group["sim_p95"])),
        )
        if row.bootstrap_resampling_unit != "station" or int(row.bootstrap_replicates) != 10_000 or int(row.random_seed) != 916:
            raise AssertionError("03_oos_2017: bootstrap protocol mismatch")

    spectral_summary = pd.read_csv(oos / "oos2017_spectral_summary.csv")
    station_tables = pd.concat(
        [pd.read_csv(oos / f"{station}_spectral_curves.csv") for station in ("TCYA", "AU46", "GC38")],
        ignore_index=True,
    )
    sort_key = ["station", "component", "quantity", "x"]
    pd.testing.assert_frame_equal(
        spectral_summary.sort_values(sort_key).reset_index(drop=True),
        station_tables.sort_values(sort_key).reset_index(drop=True),
        check_exact=True,
    )
    assert_percentiles(
        spectral_summary,
        ["p05", "p16", "p50", "p84", "p95"],
        "03_oos_2017/oos2017_spectral_summary.csv",
    )

    durations = pd.read_csv(oos / "oos2017_durations.csv")
    if len(durations) != 93 or set(durations.groupby("station")["record_type"].value_counts().unstack()["simulation"]) != {30}:
        raise AssertionError("03_oos_2017: duration record coverage mismatch")
    for component in ("major", "interm", "vertical"):
        np.testing.assert_allclose(
            durations[f"t97p5_{component}_s"] - durations[f"t2p5_{component}_s"],
            durations[f"Ds_{component}_s"],
            rtol=0.0,
            atol=1.0e-7,
        )
    energy = pd.read_csv(oos / "station_hysteretic_energy_q3.csv")
    if len(energy) != 153 or energy["station_id"].nunique() != 3 or set(energy["n_sim"]) != {30}:
        raise AssertionError("03_oos_2017: hysteretic-energy coverage mismatch")
    assert_percentiles(
        energy,
        ["sim_p05", "sim_p16", "sim_p50", "sim_p84", "sim_p95"],
        "03_oos_2017/station_hysteretic_energy_q3.csv",
    )

    oos2012 = ROOT / "04_oos_2012"
    network2012 = pd.read_csv(oos2012 / "network_metrics.csv")
    if len(network2012) != 378 or network2012["station_id"].nunique() != 54:
        raise AssertionError("04_oos_2012: network coverage mismatch")
    if set(network2012.groupby("metric")["station_id"].nunique()) != {54}:
        raise AssertionError("04_oos_2012: incomplete metric coverage")
    if set(network2012["n_sim"].astype(int)) != {30}:
        raise AssertionError("04_oos_2012: every station metric must summarize 30 simulations")
    assert_percentiles(
        network2012,
        ["sim_p05", "sim_p16", "sim_p50", "sim_p84", "sim_p95"],
        "04_oos_2012/network_metrics.csv",
    )
    bootstrap2012 = pd.read_csv(oos2012 / "network_metrics_bootstrap_summary.csv")
    if len(bootstrap2012) != 5 or set(bootstrap2012["metric"]) != expected_metrics:
        raise AssertionError("04_oos_2012: bootstrap summary metric mismatch")
    expected_counts_2012 = {
        "D2p5_97p5_H": 54,
        "D2p5_97p5_V": 54,
        "IA_H": 54,
        "IA_V": 54,
        "EH_Q3": 49,
    }
    for row in bootstrap2012.itertuples(index=False):
        group = network2012[network2012["metric"].eq(row.metric)].copy()
        group = group[np.isfinite(group["residual_ln_obs_over_p50"])]
        if int(row.n_stations) != expected_counts_2012[row.metric]:
            raise AssertionError("04_oos_2012: bootstrap station count mismatch")
        residual = group["residual_ln_obs_over_p50"].to_numpy(float)
        np.testing.assert_allclose(row.rmse_ln_obs_over_p50, np.sqrt(np.mean(residual**2)))
        np.testing.assert_allclose(row.bias_ln_obs_over_p50, np.mean(residual))
        np.testing.assert_allclose(
            row.coverage_p16_p84,
            np.mean(group["observed"].between(group["sim_p16"], group["sim_p84"])),
        )
        np.testing.assert_allclose(
            row.coverage_p05_p95,
            np.mean(group["observed"].between(group["sim_p05"], group["sim_p95"])),
        )
        if row.bootstrap_resampling_unit != "station" or int(row.bootstrap_replicates) != 10_000 or int(row.random_seed) != 916:
            raise AssertionError("04_oos_2012: bootstrap protocol mismatch")

    sasid = ROOT / "05_sasid"
    detail = pd.read_csv(sasid / "duration_detail_sasid_vallis.csv")
    suite = pd.read_csv(sasid / "duration_suite_summary.csv")
    if len(detail) != 480 or detail[["site", "source", "method"]].drop_duplicates().shape[0] != 16:
        raise AssertionError("05_sasid: duration-detail coverage mismatch")
    if set(detail.groupby(["site", "source", "method"])["pair_id"].nunique()) != {30}:
        raise AssertionError("05_sasid: duration detail must contain 30 pairs per suite")
    if len(suite) != 16 or set(suite["n"].astype(int)) != {30}:
        raise AssertionError("05_sasid: duration-suite summary mismatch")
    if len(pd.read_csv(sasid / "table2_summary.csv")) != 8:
        raise AssertionError("05_sasid: Table 2 must contain eight site/source cases")

    return {
        "status": "PASS",
        "spectral_benchmark_rows_recomputed": int(len(benchmark)),
        "spectral_cu_loeo_events": int(cu_eventwise["event_id"].nunique()),
        "spectral_cu_loeo_models": int(cu_eventwise["model"].nunique()),
        "duration_oof_sources_matched_to_runtime_manifest": int(len(duration)),
        "duration_recordwise_grouped_oof_predictions": int(len(duration_records)),
        "oos2017_network_stations": int(network["station_id"].nunique()),
        "oos2017_network_metrics": int(network["metric"].nunique()),
        "oos2017_bootstrap_metrics": int(len(bootstrap)),
        "oos2012_network_stations": int(network2012["station_id"].nunique()),
        "oos2012_network_metrics": int(network2012["metric"].nunique()),
        "oos2012_bootstrap_metrics": int(len(bootstrap2012)),
        "sasid_site_source_cases": 8,
    }


def audit_npz_metadata() -> dict:
    files = {}
    for path in ROOT.rglob("*.npz"):
        string_values = []
        with np.load(path, allow_pickle=False) as archive:
            for key in archive.files:
                array = archive[key]
                if array.dtype.kind in {"U", "S"}:
                    string_values.extend(str(value) for value in array.reshape(-1))
            files[path.relative_to(ROOT).as_posix()] = len(archive.files)
        if any(LOCAL_PATH.search(value) for value in string_values):
            raise AssertionError(f"Machine-local path found in {path.relative_to(ROOT)}")
    return {"status": "PASS", "files_checked": len(files), "array_counts": files}


def audit_nonlinear_and_scaling() -> dict:
    directory = ROOT / "05_sasid"
    filenames = {
        "SASID": "nonlinear_Umax_EH_percentiles_SASID_Q2_Q3_Q4.csv",
        "VALLIS": "nonlinear_Umax_EH_percentiles_VALLIS_Q2_Q3_Q4.csv",
    }
    tables = {}
    statistic_order = ["P05", "P16", "P50", "P84", "P95"]
    for method, filename in filenames.items():
        table = pd.read_csv(directory / filename)
        if len(table) != 12240:
            raise AssertionError(f"{method}: expected 12,240 percentile rows")
        if set(table["method"]) != {method}:
            raise AssertionError(f"{method}: method label mismatch")
        if set(table["metric"]) != {"Umax", "EH_RMS"}:
            raise AssertionError(f"{method}: missing nonlinear metric")
        if set(table["Q"].astype(int)) != {2, 3, 4}:
            raise AssertionError(f"{method}: missing Q value")
        if set(table["statistic"]) != set(statistic_order):
            raise AssertionError(f"{method}: missing percentile")
        if table[["site", "source"]].drop_duplicates().shape[0] != 8:
            raise AssertionError(f"{method}: incomplete site/source coverage")
        if set(table["n_pairs"].astype(int)) != {30}:
            raise AssertionError(f"{method}: expected 30 pairs")
        wide = table.pivot(
            index=["site", "source", "metric", "Q", "T_s"],
            columns="statistic",
            values="value",
        )[statistic_order]
        if not np.isfinite(wide.to_numpy(float)).all():
            raise AssertionError(f"{method}: non-finite percentile")
        if np.any(np.diff(wide.to_numpy(float), axis=1) < -1.0e-10):
            raise AssertionError(f"{method}: non-monotone percentiles")
        tables[method] = table

    scaling = pd.read_csv(
        directory / "conditioned_SaT_scaling_factors_all_sites_sources_VALLIS.csv"
    )
    key = ["site", "source", "pair_id", "T_s"]
    if len(scaling) != 12240 or scaling.duplicated(key).any():
        raise AssertionError("Scaling table has an unexpected size or duplicate key")
    if scaling[["site", "source"]].drop_duplicates().shape[0] != 8:
        raise AssertionError("Scaling table has incomplete site/source coverage")
    if set(scaling.groupby(["site", "source"])["pair_id"].nunique()) != {30}:
        raise AssertionError("Scaling table must contain 30 pairs per case")
    if set(scaling.groupby(["site", "source", "pair_id"])["T_s"].nunique()) != {51}:
        raise AssertionError("Scaling table must contain 51 periods per pair")
    if not np.isfinite(scaling.select_dtypes(include=[np.number]).to_numpy()).all():
        raise AssertionError("Scaling table contains a non-finite number")
    if (scaling["effective_scale_factor"] <= 0.0).any():
        raise AssertionError("Scaling table contains a non-positive factor")
    rms = np.sqrt(
        (
            scaling["Sa_comp1_pre_broadband_cm_s2"].to_numpy(float) ** 2
            + scaling["Sa_comp2_pre_broadband_cm_s2"].to_numpy(float) ** 2
        )
        / 2.0
    )
    np.testing.assert_allclose(
        rms,
        scaling["Sa_pair_RMS_pre_broadband_cm_s2"].to_numpy(float),
        rtol=2.0e-9,
        atol=1.0e-8,
    )

    figure10 = pd.read_csv(directory / "figure10_nonlinear_response.csv")
    vallis = tables["VALLIS"].copy()
    vallis["method"] = "VALLIS"
    combined = pd.concat([tables["SASID"], vallis], ignore_index=True)
    combined = combined[combined["statistic"].eq("P50")]
    sort_key = ["site", "source", "method", "metric", "Q", "T_s"]
    figure10 = figure10.sort_values(sort_key).reset_index(drop=True)
    combined = combined.sort_values(sort_key).reset_index(drop=True)
    if len(figure10) != len(combined):
        raise AssertionError("Figure 10 and percentile P50 row counts differ")
    if not figure10[["site", "source", "method", "metric", "Q"]].equals(
        combined[["site", "source", "method", "metric", "Q"]]
    ):
        raise AssertionError("Figure 10 and percentile P50 keys differ")
    np.testing.assert_allclose(figure10["T_s"], combined["T_s"], rtol=0.0, atol=6.0e-10)
    np.testing.assert_allclose(figure10["value"], combined["value"], rtol=0.0, atol=6.0e-10)

    return {
        "status": "PASS",
        "percentile_rows_per_method": 12240,
        "statistics": statistic_order,
        "metrics": ["Umax", "EH_RMS"],
        "Q": [2, 3, 4],
        "site_source_cases": 8,
        "pairs_per_case": 30,
        "periods_per_pair": 51,
        "figure10_p50_consistent": True,
    }


def main() -> None:
    report = {
        "status": "PASS",
        "release": "VALLIS-3C 1.6.0",
        "integrity": audit_integrity(),
        "local_paths": audit_local_paths(),
        "npz_metadata": audit_npz_metadata(),
        "csv_tables": audit_csvs(),
        "cross_validation_consistency": audit_other_validations(),
        "nonlinear_and_scaling": audit_nonlinear_and_scaling(),
        "notes": [
            "The strict 2017 station bootstrap supports duration transfer; hysteretic energy retains larger station-level error.",
            "The strict 2012 station bootstrap uses the common 54-station support of the seven distributed metric tables.",
            "Serialized runtime models are execution assets and are not treated as sole validation evidence.",
        ],
    }
    (ROOT / "VALIDATION_AUDIT.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("PASS: validation package is internally consistent")


if __name__ == "__main__":
    main()
