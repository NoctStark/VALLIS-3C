"""Validate the numerical operation of the distributed VALLIS-3C models."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "runtime"), str(ROOT / "spectral_core" / "src")]

from runtime.vallis3c_runtime.adapter import VARIANTS, simulate_vallis3c_sync  # noqa: E402


SCENARIOS = {
    "2017-09-19": {
        "event_id": "2017-09-19",
        "station_id": "SCT",
        "site_id": "SCT",
        "Mw": 7.1,
        "Rrup_km": 115.0,
        "depth_used_km": 46.4,
        "Ts_s": 1.83,
        "source_type": "INSLAB",
        "event_longitude": -98.72,
        "event_latitude": 18.40,
        "x_conditioned_km": -11.341992930923563,
        "y_conditioned_km": -2.0279513813053125,
    },
    "2012-03-20": {
        "event_id": "2012-03-20",
        "station_id": "AE02",
        "site_id": "AE02",
        "Mw": 7.5,
        "Rrup_km": 326.0,
        "depth_used_km": 14.82,
        "Ts_s": 4.64,
        "source_type": "INTERPLATE",
        "event_longitude": -98.531,
        "event_latitude": 16.254,
        "x_conditioned_km": -2.0085332,
        "y_conditioned_km": 1.2753806,
    },
}

CASES = (
    ("full_sct_2017", "full", "2017-09-19"),
    ("oos2017_sct_2017", "oos2017", "2017-09-19"),
    ("oos2012_2017_ae02_2012", "oos2012_2017", "2012-03-20"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside_release(path: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(ROOT.resolve())
        return True
    except ValueError:
        return False


def _catalog_event_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            str(row.get("event_id", "")).strip()
            for row in csv.DictReader(handle)
            if str(row.get("event_id", "")).strip()
        }


def run_case(name: str, variant: str, event_id: str, seed: int) -> dict:
    scenario = dict(SCENARIOS[event_id])
    request = {
        "scenario": scenario,
        "n_realizations": 1,
        "seed": int(seed),
        "vallis_config": {
            "model_variant": variant,
            "architecture": "final_hv",
            "dt_s": 0.01,
            "output_components": "three_component",
            "exclude_same_event": False,
        },
        "fas_options": {"fas_event_mode": "record_only"},
        "fas_input_mode": "model",
    }
    started = time.perf_counter()
    payload = simulate_vallis3c_sync(ROOT / "models", request)
    elapsed = time.perf_counter() - started
    acceleration = np.asarray(payload["accelerations"], dtype=float)
    metadata = dict(payload["metadata"][0])
    audit = dict(payload["model_audit"])

    expected_exclusions = sorted(VARIANTS[variant]["heldout"])
    temporal = ROOT / "models" / VARIANTS[variant]["temporal"] / "duration_clock_final_hv_portable.npz"
    selector = temporal.parent / "selector_portable.npz"
    empirical_fas = ROOT / "models" / "release" / "empirical_fas" / "empirical_fas_library.npz"
    empirical_fas_manifest = json.loads(
        (empirical_fas.parent / "manifest.json").read_text(encoding="utf-8")
    )
    expected_selector_records = {
        "full": 1623,
        "oos2017": 1568,
        "oos2012_2017": 1515,
    }[variant]
    catalog_events = _catalog_event_ids(temporal.parent / "catalog.csv")
    heldout_events_present = sorted(set(expected_exclusions).intersection(catalog_events))
    target_event_present = event_id in catalog_events
    fas_provider = dict(payload.get("fas_provider") or {})
    fas_options = dict(fas_provider.get("options") or {})
    empirical_audits = list(fas_provider.get("realization_audit") or [])
    empirical_candidate_events = sorted({
        str(candidate)
        for item in empirical_audits
        for candidate in (item.get("empirical_fas_candidate_events") or [])
    })
    empirical_heldout_events_present = sorted(
        set(expected_exclusions).intersection(empirical_candidate_events)
    )
    empirical_declared_exclusions = sorted({
        str(value)
        for value in (fas_options.get("excluded_donor_event_ids") or [])
    })
    checks = {
        "shape_is_one_by_three": acceleration.ndim == 3 and acceleration.shape[:2] == (1, 3),
        "finite_nonzero_output": bool(np.isfinite(acceleration).all() and np.max(np.abs(acceleration)) > 0),
        "version_1_6_0": metadata.get("version") == "1.6.0",
        "architecture_final_hv": metadata.get("architecture") == "final_hv",
        "phase_diffusion_carrier": metadata.get("carrier") == "phase_diffusion"
        and metadata.get("carrier_audit", {}).get("carrier") == "phase_diffusion"
        and metadata.get("carrier_audit", {}).get("waveform_donor_reconstructed") is False,
        "requested_variant": metadata.get("model_variant") == variant and audit.get("model_variant") == variant,
        "record_only": audit.get("fas_event_mode") == "record_only",
        "no_fallback": audit.get("fallback_used") is False,
        "release_model_pair": audit.get("release_model_pair") is True,
        "expected_exclusions": sorted(audit.get("strict_oos_events", [])) == expected_exclusions,
        "internal_model_paths": _inside_release(audit.get("models_root", ""))
        and _inside_release(audit.get("conditional_fas_capabilities", {}).get("spectral_model_path", "")),
        "final_hv_clock_present": temporal.is_file(),
        "texture_selector_present": selector.is_file(),
        "empirical_fas_asset_present": empirical_fas.is_file(),
        "empirical_fas_asset_integrity": _sha256(empirical_fas)
        == empirical_fas_manifest.get("sha256"),
        "texture_selector_metadata": metadata.get("texture_selector") == "temporal_texture_selector"
        and audit.get("texture_selector") == "temporal_texture_selector"
        and metadata.get("texture_selector_artifact") == "selector_portable.npz"
        and audit.get("texture_selector_artifact") == "selector_portable.npz"
        and metadata.get("texture_selector_training_records") == expected_selector_records
        and audit.get("texture_selector_training_records") == expected_selector_records
        and metadata.get("texture_selector_min_samples_leaf") == 3
        and audit.get("texture_selector_min_samples_leaf") == 3,
        "heldout_events_absent_from_donor_catalog": not heldout_events_present,
        "empirical_fas_audit_present": len(empirical_audits) == 1,
        "empirical_fas_expected_exclusions": empirical_declared_exclusions == expected_exclusions,
        "heldout_events_absent_from_empirical_fas_donor_pool": not empirical_heldout_events_present,
    }
    if not all(checks.values()):
        failed = [key for key, passed in checks.items() if not passed]
        raise RuntimeError(f"{name} failed: {', '.join(failed)}")
    return {
        "name": name,
        "status": "PASS",
        "variant": variant,
        "event_id": event_id,
        "shape": list(acceleration.shape),
        "dt_s": float(payload["dt_s"]),
        "peak_abs_cm_s2": float(np.max(np.abs(acceleration))),
        "elapsed_s": float(elapsed),
        "strict_oos_events": expected_exclusions,
        "target_event_present_in_donor_catalog": bool(target_event_present),
        "heldout_event_present_in_donor_catalog": bool(heldout_events_present) if expected_exclusions else None,
        "heldout_events_found_in_donor_catalog": heldout_events_present,
        "heldout_event_present_in_empirical_fas_donor_pool": bool(empirical_heldout_events_present) if expected_exclusions else None,
        "heldout_events_found_in_empirical_fas_donor_pool": empirical_heldout_events_present,
        "empirical_fas_candidate_event_count": len(empirical_candidate_events),
        "spectral_model": str(Path(audit["conditional_fas_capabilities"]["spectral_model_path"]).relative_to(ROOT)),
        "spectral_model_sha256": _sha256(Path(audit["conditional_fas_capabilities"]["spectral_model_path"])),
        "duration_clock": str(temporal.relative_to(ROOT)),
        "duration_clock_sha256": _sha256(temporal),
        "texture_selector": "temporal_texture_selector",
        "texture_selector_artifact": str(selector.relative_to(ROOT)),
        "texture_selector_sha256": _sha256(selector),
        "texture_selector_training_records": expected_selector_records,
        "texture_selector_min_samples_leaf": 3,
        "empirical_fas_asset": str(empirical_fas.relative_to(ROOT)),
        "empirical_fas_sha256": _sha256(empirical_fas),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs" / "VALIDATION_REPORT.json")
    parser.add_argument("--seed", type=int, default=24680)
    args = parser.parse_args()
    report = {
        "software": "VALLIS-3C",
        "version": "1.6.0",
        "architecture": "final_hv",
        "validation_scope": "phase_diffusion_production",
        "validated_carrier": "phase_diffusion",
        "texture_selector": "temporal_texture_selector",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RUNNING",
        "cases": [],
    }
    try:
        for index, (name, variant, event_id) in enumerate(CASES):
            print(f"START {name}", flush=True)
            result = run_case(name, variant, event_id, args.seed + index)
            report["cases"].append(result)
            print(f"PASS  {name} shape={result['shape']} elapsed={result['elapsed_s']:.2f}s", flush=True)
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"REPORT {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
