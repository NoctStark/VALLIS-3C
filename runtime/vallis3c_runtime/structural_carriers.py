"""Phase-diffusion carrier for VALLIS-3C production synthesis."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd
from scipy.fft import rfft, irfft

from .numerics import BAND_CENTERS, filters

BIN_S = 2.0
TEXTURE_GUARD_S = 11.25
NATIVE_DT_S = 0.02

def _target_power(packet):
    n = int(packet["npts"])
    edges = np.asarray(packet["edges"], int)
    centers = (edges[:-1] + edges[1:] - 1.0) / 2.0
    target = np.asarray(packet["power"], float)
    out = np.empty((3, len(BAND_CENTERS), n), dtype=np.float32)
    sample = np.arange(n, dtype=float)
    for c in range(3):
        for b in range(len(BAND_CENTERS)):
            out[c, b] = np.maximum(
                np.interp(sample, centers, target[c, b], left=target[c, b, 0], right=target[c, b, -1]),
                0.0,
            )
    return out


def phase_diffusion_decode(packet, rng):
    n = int(packet["npts"]); dt = float(packet["dt_s"]); H = filters(n, dt)
    t = np.arange(n, dtype=float) * dt
    P = _target_power(packet)
    D = np.asarray(packet["diffusion_rad2_per_s"], float)
    freq = np.asarray(packet["mean_frequency_hz"], float)
    out = np.zeros((3, n // 2 + 1), dtype=complex)
    for b in range(len(BAND_CENTERS)):
        phase = rng.uniform(-np.pi, np.pi, (3, 1)) + 2.0 * np.pi * freq[:, b, None] * t
        increments = np.sqrt(2.0 * D[:, b, None] * dt) * rng.standard_normal((3, n - 1))
        phase[:, 1:] += np.cumsum(increments, axis=-1)
        y = np.sqrt(P[:, b]) * np.cos(phase)
        out += rfft(y, axis=-1) * H[b]
    out[:, 0] = 0.0; out[:, -1] = 0.0
    return irfft(out, n=n, axis=-1)


class StructuralTextureStore:
    """Generate phase-diffusion carriers from processed time-frequency textures."""

    def __init__(self, model_dir, selector):
        model_dir = Path(model_dir).resolve()
        self.texture_root = model_dir.parents[2] / "textures"
        index_path = self.texture_root / "texture_index.csv"
        manifest_path = self.texture_root / "texture_manifest.json"
        if not index_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError("VALLIS-3C structural texture library is incomplete")
        idx = pd.read_csv(index_path, low_memory=False)
        idx["event_id"] = idx["event_id"].astype(str)
        idx["station_id"] = idx["station_id"].astype(str)
        idx = idx.loc[idx["texture_usable_3c"].astype(bool)].copy()
        self._texture_by_pair = {
            (e, s): str(spec).split("/records/", 1)[1]
            for e, s, spec in zip(idx.event_id, idx.station_id, idx.texture_hdf5)
            if "/records/" in str(spec)
        }
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        self._part_by_key = {}
        for part in manifest.get("parts", []):
            name = str(part["file"])
            for key in part.get("record_ids", []):
                self._part_by_key[str(key)] = name
        self._records = {}
        for _, row in selector.catalog.iterrows():
            rid = str(row.record_id)
            pair = (str(row.event_id), str(row.station_id))
            key = self._texture_by_pair.get(pair)
            if key is not None and key in self._part_by_key:
                self._records[rid] = {
                    "key": key,
                    "part": self._part_by_key[key],
                    "npts_prepared": int(row.get("npts_prepared", 0) or 0),
                    "event_id": pair[0],
                    "station_id": pair[1],
                }
        self._handles = {}
        self.calls = 0

    def close(self):
        self._packet.cache_clear()
        for handle in self._handles.values():
            try: handle.close()
            except Exception: pass
        self._handles.clear()

    def has(self, record_id):
        return str(record_id) in self._records

    def _handle(self, part):
        if part not in self._handles:
            path = self.texture_root / part
            if not path.is_file(): raise FileNotFoundError(path)
            self._handles[part] = h5py.File(path, "r")
        return self._handles[part]

    @staticmethod
    def _native_support_npts(npts_prepared, dt):
        if npts_prepared <= 0:
            raise ValueError("Selected structure has no prepared support length")
        duration = float(npts_prepared) * NATIVE_DT_S
        n = max(256, int(round(duration / float(dt))))
        return n + (n % 2)

    @lru_cache(maxsize=192)
    def _packet(self, record_id, dt_rounded):
        record_id = str(record_id); dt = float(dt_rounded)
        if record_id not in self._records:
            raise KeyError(f"No structural texture is available for selected record {record_id}")
        meta = self._records[record_id]
        n = self._native_support_npts(meta["npts_prepared"], dt)
        width = max(1, int(round(BIN_S / dt)))
        edges = np.r_[np.arange(0, n, width), n].astype(int)
        time_edges = edges.astype(float) * dt
        block_duration = np.maximum(np.diff(time_edges), dt)

        h = self._handle(meta["part"]); g = h["records"][meta["key"]]
        freq36 = np.asarray(g["frequency_hz"][:], float)
        qtime = np.asarray(g["quantile_time_s"][:], float)
        interval_mass = np.asarray(g["interval_probability_mass"][:], float)
        local = np.asarray(g["subquantile_local_probability"][:], float)
        unit_edges = np.asarray(g["subquantile_unit_edges"][:], float)
        band_energy = np.asarray(g["band_energy"][:], float)

        p36 = np.zeros((3, len(freq36), len(edges) - 1), dtype=float)
        ucent = 0.5 * (unit_edges[:-1] + unit_edges[1:])
        for c in range(3):
            for b in range(len(freq36)):
                masses = []
                centers = []
                for j in range(interval_mass.shape[-1]):
                    q0, q1 = qtime[c, b, j], qtime[c, b, j + 1]
                    m = interval_mass[c, b, j]
                    if not (np.isfinite(q0) and np.isfinite(q1) and q1 > q0 and np.isfinite(m) and m > 0):
                        continue
                    lm = np.asarray(local[c, b, j], float)
                    good = np.isfinite(lm) & (lm >= 0)
                    if not np.any(good) or float(np.sum(lm[good])) <= 0:
                        continue
                    lm = np.where(good, lm, 0.0); lm /= np.sum(lm)
                    centers.append(q0 + (q1 - q0) * ucent + TEXTURE_GUARD_S)
                    masses.append(m * lm)
                if not masses:
                    continue
                centers = np.concatenate(centers); masses = np.concatenate(masses)
                hist, _ = np.histogram(centers, bins=time_edges, weights=masses)
                density = hist / block_duration
                energy = max(float(band_energy[c, b]), 0.0)
                p36[c, b] = energy * density

        # The packaged texture library spans 0.16-10 Hz; the synthesis transport
        # uses 42 bands over 0.1-20 Hz.  Interpolate in log-frequency inside the
        # texture support and extend the nearest temporal shape at the two edges.
        logf36 = np.log(freq36); logf42 = np.log(BAND_CENTERS)
        p42 = np.empty((3, len(BAND_CENTERS), p36.shape[-1]), dtype=np.float32)
        for c in range(3):
            for k in range(p36.shape[-1]):
                p42[c, :, k] = np.maximum(
                    np.interp(logf42, logf36, p36[c, :, k], left=p36[c, 0, k], right=p36[c, -1, k]),
                    0.0,
                ).astype(np.float32)

        # Fixed bandwidth-consistent diffusion.  No donor phase/coherence
        # trajectory is stored or inferred.  The rate follows the logarithmic
        # spacing of the native filter bank and therefore introduces no
        # station-specific tuning parameter.
        log_spacing = float(np.median(np.diff(np.log(BAND_CENTERS))))
        diffusion = np.broadcast_to((BAND_CENTERS * log_spacing)[None, :], (3, len(BAND_CENTERS))).copy()
        mean_frequency = np.broadcast_to(BAND_CENTERS[None, :], (3, len(BAND_CENTERS))).copy()
        return {
            "power": p42,
            "edges": edges,
            "npts": np.int64(n),
            "dt_s": np.float64(dt),
            "mean_frequency_hz": mean_frequency,
            "diffusion_rad2_per_s": diffusion,
            "record_id": record_id,
            "structure_key": meta["key"],
            "texture_part": meta["part"],
            "schema": "vallis3c.phase_diffusion.texture_carrier.v1",
        }

    def signal(self, record_id, dt, rng):
        packet = self._packet(str(record_id), round(float(dt), 9))
        self.calls += 1
        y = phase_diffusion_decode(packet, rng)
        if y.shape[0] != 3 or not np.isfinite(y).all() or np.any(np.sum(y * y, axis=-1) <= 0):
            raise ValueError("Invalid phase-diffusion structural carrier")
        audit = {
            "carrier": "phase_diffusion",
            "representation": str(packet["schema"]),
            "record_id": str(record_id),
            "structure_key": str(packet["structure_key"]),
            "texture_part": str(packet["texture_part"]),
            "bin_s": float(BIN_S),
            "waveform_donor_reconstructed": False,
            "phase_trajectory_reused": False,
        }
        return y, audit
