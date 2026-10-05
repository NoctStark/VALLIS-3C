"""Verify the exact runtime dependencies required by VALLIS-3C."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMPORT_NAMES = {
    "PyQt6": "PyQt6",
    "scikit-learn": "sklearn",
}


def expected_versions() -> dict[str, str]:
    expected: dict[str, str] = {}
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            raise ValueError(f"Dependency is not exactly pinned: {line}")
        name, version = line.split("==", 1)
        expected[name.strip()] = version.strip()
    return expected


def validate() -> list[str]:
    errors: list[str] = []
    for distribution, expected in expected_versions().items():
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"missing distribution: {distribution}=={expected}")
            continue
        if installed != expected:
            errors.append(
                f"version mismatch: {distribution} expected {expected}, found {installed}"
            )
            continue
        module = IMPORT_NAMES.get(distribution, distribution.replace("-", "_"))
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"import failed: {module}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        errors = validate()
    except Exception as exc:
        errors = [str(exc)]
    if errors:
        if not args.quiet:
            for error in errors:
                print(f"ERROR: {error}")
        return 1
    if not args.quiet:
        print("The shared private environment is complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
