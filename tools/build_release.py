"""Create the release checksum manifest and, optionally, the distribution ZIP."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKSUMS = ROOT / "SHA256SUMS.txt"
EXCLUDED_TOP_LEVEL = {".git", ".venv", "outputs"}
EXCLUDED_DIRECTORY_NAMES = {"__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}


def _included_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == DEFAULT_CHECKSUMS or path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] in EXCLUDED_TOP_LEVEL:
            continue
        if any(part in EXCLUDED_DIRECTORY_NAMES or part.lower().startswith("backup_") for part in relative.parts):
            continue
        yield path, relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", action="store_true", help="also create a ZIP beside the release directory")
    args = parser.parse_args()
    files = list(_included_files())
    lines = [f"{_sha256(path)}  {relative.as_posix()}" for path, relative in files]
    DEFAULT_CHECKSUMS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"WROTE {DEFAULT_CHECKSUMS} ({len(lines)} files)")
    if args.zip:
        destination = ROOT.parent / "VALLIS-3C_v1.6.0.zip"
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path, relative in files:
                archive.write(path, Path(ROOT.name) / relative)
            archive.write(DEFAULT_CHECKSUMS, Path(ROOT.name) / DEFAULT_CHECKSUMS.name)
        print(f"WROTE {destination}")
        checksum_path = destination.with_suffix(destination.suffix + ".sha256")
        checksum_path.write_text(
            f"{_sha256(destination)}  {destination.name}\n",
            encoding="utf-8",
        )
        print(f"WROTE {checksum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
