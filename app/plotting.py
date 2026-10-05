"""Responsive plotting utilities for 3C Ground-Motion Simulation."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath

COMP_COLORS = ("#2C5AA0", "#D17A22", "#3A8F5D")
ZONE_COLORS = {"I": "#D6EBDD", "II": "#F3E7BF", "III": "#DCE6F4"}
ZONE_DISPLAY = {"I": "A", "II": "B", "III": "C"}
ZONE_EDGE = "#344047"
BOUNDARY_COLOR = "#4C555C"


def _scale(fig: Figure, reference_w=650.0, reference_h=420.0):
    canvas = getattr(fig, "canvas", None)
    try:
        w = float(canvas.width())
        h = float(canvas.height())
    except Exception:
        w = float(fig.get_figwidth() * fig.dpi)
        h = float(fig.get_figheight() * fig.dpi)
    # Large/maximized canvases receive a materially larger scientific font;
    # compact restored windows retain the original baseline.
    return float(np.clip(min(w / reference_w, h / reference_h), 1.00, 1.34))


def symmetric_limit(acc):
    m = float(np.nanmax(np.abs(acc))) if np.size(acc) else 1.0
    if not np.isfinite(m) or m <= 0:
        return 1.0
    raw = 1.05 * m
    p = 10 ** np.floor(np.log10(raw))
    q = raw / p
    step = 1 if q <= 1 else 2 if q <= 2 else 5 if q <= 5 else 10
    return float(step * p)


def style_axis(ax, scale=1.0):
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10.5 * scale)


def _valid_npts(payload, idx, default):
    vals = payload.get("valid_npts")
    if vals is not None:
        try:
            return int(np.clip(int(vals[idx]), 2, default))
        except Exception:
            pass
    meta = payload.get("metadata") or []
    if idx < len(meta) and isinstance(meta[idx], dict):
        for key in ("conditioned_npts", "display_npts", "unpadded_npts"):
            if key in meta[idx]:
                try:
                    return int(np.clip(int(meta[idx][key]), 2, default))
                except Exception:
                    pass
    return int(default)


def plot_accelerograms(fig: Figure, payload, index, labels):
    """Plot one 3C realization; no Arias-derived trimming or shading is computed."""
    fig.clear()
    a = np.asarray(payload["accelerations"], float)
    dt = float(payload["dt_s"])
    idx = max(0, min(int(index), len(a) - 1))
    n = _valid_npts(payload, idx, a.shape[-1])
    rec = a[idx, :, :n]
    t = np.arange(n, dtype=float) * dt
    ylim = symmetric_limit(rec)
    axes = []
    scale = _scale(fig, 900, 650)
    ncomp = int(rec.shape[0])

    for j in range(ncomp):
        ax = fig.add_subplot(ncomp, 1, j + 1, sharex=axes[0] if axes else None)
        axes.append(ax)
        ax.plot(t, rec[j], lw=max(0.65, 0.78 * scale), color=COMP_COLORS[j], zorder=2)
        ax.set_ylim(-ylim, ylim)
        ax.set_yticks([-ylim, 0.0, ylim])
        ax.set_ylabel(f"{labels[j]}\n(cm/s²)", fontsize=10.8 * scale, labelpad=5.0)
        style_axis(ax, scale)
    axes[-1].set_xlabel("Time (s)", fontsize=11.4 * scale, labelpad=4.0)
    # Time histories must start exactly at 0 s.  Matplotlib otherwise adds a
    # small negative x-margin, which is visually misleading for accelerograms.
    if t.size:
        axes[-1].set_xlim(0.0, max(float(dt), float(t[-1])))
    for ax in axes:
        ax.margins(x=0.0)
    fig.suptitle(f"Realization {idx + 1}", fontsize=11.9 * scale, y=0.985)
    # Stable editorial margins keep all component y-labels and the shared x-label
    # visible when the application is restored to its medium-height layout.
    fig.subplots_adjust(left=0.130, right=0.992, bottom=0.125, top=0.915, hspace=0.22)


def fas_display_floor(family, selected=None, reference_hz=10.0):
    """Return a display-only lower log bound for unsmoothed FAS.

    In percentile mode ``selected`` is ``None`` and the floor is anchored to the
    whole family, so changing the realization navigator cannot move the axes.
    In individual mode the floor follows the selected realization.
    Spectral values themselves are never modified.
    """
    if not family:
        return None
    records = family
    if selected is not None:
        idx = max(0, min(int(selected), len(family) - 1))
        records = [family[idx]]
    refs = []
    for rec in records:
        for x, y in rec:
            x = np.asarray(x, float); y = np.asarray(y, float)
            ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
            if not np.any(ok):
                continue
            xx = x[ok]; yy = y[ok]
            j = int(np.argmin(np.abs(xx - float(reference_hz))))
            refs.append(float(yy[j]))
    if not refs:
        return 1e-3
    ref = float(np.nanmedian(refs))
    if not np.isfinite(ref) or ref <= 0:
        return 1e-3
    return float(max(1e-8, 10.0 ** (np.floor(np.log10(ref)) - 3.0)))


def _curve_on_grid(x_ref, x, y):
    """Interpolate a curve to ``x_ref`` without extrapolating outside support."""
    x_ref = np.asarray(x_ref, float)
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]; y = y[ok]
    if x.size < 2:
        return np.full_like(x_ref, np.nan, dtype=float)
    order = np.argsort(x)
    x = x[order]; y = y[order]
    keep = np.r_[True, np.diff(x) > 0]
    x = x[keep]; y = y[keep]
    if x.size < 2:
        return np.full_like(x_ref, np.nan, dtype=float)
    out = np.full_like(x_ref, np.nan, dtype=float)
    support = np.isfinite(x_ref) & (x_ref >= x[0]) & (x_ref <= x[-1])
    if not np.any(support):
        return out
    if np.all(x > 0) and np.all(x_ref[support] > 0):
        out[support] = np.interp(np.log(x_ref[support]), np.log(x), y)
    else:
        out[support] = np.interp(x_ref[support], x, y)
    return out


def _reference_grid(curves):
    """Choose a stable plotting grid from a collection of (x, y) curves."""
    candidates = []
    for x, _ in curves:
        x = np.asarray(x, float).ravel()
        x = x[np.isfinite(x)]
        if x.size >= 2:
            candidates.append(x)
    if not candidates:
        return np.asarray([], dtype=float)
    # The longest native grid retains the maximum resolution without inventing
    # extra sampling. Percentile interpolation is display-only.
    return np.asarray(max(candidates, key=lambda a: a.size), dtype=float)


def _component_family(family, component):
    curves = []
    for rec in family:
        if component < len(rec):
            curves.append(rec[component])
    if not curves:
        return np.asarray([]), np.empty((0, 0))
    grid = _reference_grid(curves)
    if grid.size == 0:
        return grid, np.empty((0, 0))
    stack = np.vstack([_curve_on_grid(grid, x, y) for x, y in curves])
    return grid, stack


def _horizontal_rms_family(family):
    """Build per-realization horizontal RMS curves before taking percentiles."""
    curves = []
    for rec in family:
        if len(rec) < 2:
            continue
        x1, y1 = rec[0]
        x2, y2 = rec[1]
        grid = _reference_grid([(x1, y1), (x2, y2)])
        if grid.size == 0:
            continue
        yy1 = _curve_on_grid(grid, x1, y1)
        yy2 = _curve_on_grid(grid, x2, y2)
        rms = np.sqrt(0.5 * (yy1 ** 2 + yy2 ** 2))
        curves.append((grid, rms))
    if not curves:
        return np.asarray([]), np.empty((0, 0))
    grid = _reference_grid(curves)
    stack = np.vstack([_curve_on_grid(grid, x, y) for x, y in curves])
    return grid, stack


def _percentile_rows(stack, percentiles=(5, 16, 50, 84, 95)):
    if stack.size == 0:
        return {p: np.asarray([]) for p in percentiles}
    with np.errstate(all="ignore"):
        vals = np.nanpercentile(stack, percentiles, axis=0)
    return {p: vals[i] for i, p in enumerate(percentiles)}


def _plot_line(ax, x, y, *, logy, **kwargs):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if logy:
        ok &= y > 0
    if np.count_nonzero(ok) >= 2:
        return ax.plot(x[ok], y[ok], **kwargs)
    return []


def _fill_band(ax, x, lo, hi, *, logy, **kwargs):
    x = np.asarray(x, float); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
    ok = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi) & (hi >= lo)
    if logy:
        ok &= (lo > 0) & (hi > 0)
    if np.count_nonzero(ok) >= 2:
        ax.fill_between(x, lo, hi, where=ok, interpolate=True, **kwargs)


def _family_y_bounds(family, xlim=None, logy=True, legend_rows=1):
    """Return stable y-limits from the complete spectral family.

    The bounds deliberately do not depend on the realization navigator.  This
    prevents the FAS/response-spectrum axes from shifting horizontally as tick
    labels change from one realization to the next.  Extra headroom is reserved
    for the in-axes legend; compact windows may wrap the legend into more than
    one row, so the headroom scales with ``legend_rows``.
    """
    values = []
    for rec in family or []:
        for x, y in rec:
            x = np.asarray(x, float).ravel()
            y = np.asarray(y, float).ravel()
            ok = np.isfinite(x) & np.isfinite(y)
            if xlim is not None:
                ok &= (x >= float(xlim[0])) & (x <= float(xlim[1]))
            if logy:
                ok &= y > 0
            else:
                ok &= y >= 0
            if np.any(ok):
                values.append(y[ok])
    if not values:
        return None, None
    vals = np.concatenate(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    ymax = float(np.nanmax(vals))
    if not np.isfinite(ymax) or ymax <= 0:
        return None, None
    if logy:
        positive = vals[vals > 0]
        if positive.size == 0:
            return None, None
        ymin = float(np.nanmin(positive))
        # The compact corner legend is away from the main FAS peaks; modest
        # headroom is sufficient and keeps the spectrum visually dense.
        return ymin, ymax * 1.45
    # The response-spectrum legend is at long periods, where the curves have
    # nearly decayed to zero.  A small 12% margin avoids wasting vertical area.
    return 0.0, ymax * 1.12


def _canvas_size_px(fig: Figure) -> tuple[float, float]:
    canvas = getattr(fig, "canvas", None)
    try:
        return float(canvas.width()), float(canvas.height())
    except Exception:
        return (
            float(fig.get_figwidth() * fig.dpi),
            float(fig.get_figheight() * fig.dpi),
        )


def _canvas_width_px(fig: Figure) -> float:
    return _canvas_size_px(fig)[0]


def _spectral_subplot_margins(fig: Figure) -> tuple[float, float, float, float]:
    """Stable editorial margins for the spectral panels.

    The axes geometry is intentionally fixed in figure coordinates. This
    avoids the x/y labels drifting or disappearing when the window changes
    between restored and maximized states. Qt/Matplotlib already resize the
    canvas; the figure should not continually recompute its own margins.
    """
    return 0.265, 0.975, 0.245, 0.925


def _compact_corner_legend_style(fig: Figure, n_items: int, scale: float):
    """Compact in-axes legend for narrow spectral panels.

    Legends use one column. Short labels
    (M, I, V; RMS when applicable) keep the legend fully inside the plot in
    both restored and maximized window states.
    """
    _, height_px = _canvas_size_px(fig)
    height_px = max(1.0, height_px)
    fontsize = float(np.clip(8.5 * float(scale), 7.6, 9.0))
    if height_px < 240.0:
        fontsize = min(fontsize, 8.0)
    return {
        "ncol": 1,
        "fontsize": fontsize,
        "handlelength": 1.10,
        "columnspacing": 0.0,
        "handletextpad": 0.24,
        "borderpad": 0.0,
        "labelspacing": 0.12,
    }


def plot_family(fig: Figure, family, selected, labels, xlabel, ylabel,
                logx=True, logy=True, xlim=None, ymin=None, selected_lw_scale=1.0,
                display_mode="percentiles", show_p50=True, show_p16_p84=True,
                show_p05_p95=True, overlay_selected=True, horizontal_rms=False,
                percentile_component_count=None, p50_lw_scale=1.0):
    """Plot one spectral family as percentiles or one individual realization.

    Percentiles are always computed over the complete family and therefore do
    not change when ``selected`` changes.  When requested, the current
    realization is overlaid on those fixed bands.  Horizontal RMS percentiles
    are calculated realization-by-realization from the first two components,
    then summarized across the family.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    scale = _scale(fig, 430, 360)
    style_axis(ax, scale)
    selected = max(0, min(int(selected), len(family) - 1)) if family else 0
    mode = str(display_mode or "percentiles").lower()
    percentile_mode = mode == "percentiles"
    use_rms = bool(horizontal_rms and percentile_mode and family and len(family[0]) >= 2)

    if family and not percentile_mode:
        # Individual mode is an ensemble/spaghetti view: show every realization
        # for every available component.  The navigator still identifies the
        # current realization, which is drawn slightly stronger on top, but it
        # never hides the rest of the family.
        ncomp = max(len(rec) for rec in family)
        for i, rec in enumerate(family):
            is_selected = i == selected
            for j, (x, y) in enumerate(rec[:ncomp]):
                color = COMP_COLORS[j % len(COMP_COLORS)]
                label = labels[j] if j < len(labels) else f"Component {j + 1}"
                legend_label = ("M", "I", "V")[j] if j < 3 else f"C{j + 1}"
                _plot_line(
                    ax, x, y, logy=logy, color=color,
                    alpha=1.0 if is_selected else 0.11,
                    lw=(max(1.10, 1.55 * scale * float(selected_lw_scale))
                        if is_selected else max(0.42, 0.62 * scale)),
                    # The legend represents the CURRENT realization.  Label
                    # only those solid selected curves, never realization 1
                    # simply because it happened to be drawn first.
                    label=(legend_label if is_selected else "_nolegend_"),
                    zorder=5 if is_selected else 2,
                )

    elif family and use_rms:
        # Horizontal RMS replaces only the two horizontal component families.
        # The vertical component remains visible with its own percentile bands.
        grid, stack = _horizontal_rms_family(family)
        q = _percentile_rows(stack)
        if grid.size:
            if show_p05_p95:
                _fill_band(ax, grid, q[5], q[95], logy=logy,
                           facecolor="#6f767b", alpha=0.13, edgecolor="none", zorder=1)
            if show_p16_p84:
                _fill_band(ax, grid, q[16], q[84], logy=logy,
                           facecolor="#5d6469", alpha=0.28, edgecolor="none", zorder=2)
            if show_p50:
                _plot_line(ax, grid, q[50], logy=logy, color="#111111", alpha=1.0,
                           lw=max(1.20, 2.45 * scale * float(p50_lw_scale)), label="RMS - P50", zorder=4)

        # Preserve the vertical response spectrum in RMS mode.
        if any(len(rec) >= 3 for rec in family):
            vgrid, vstack = _component_family(family, 2)
            vq = _percentile_rows(vstack)
            if vgrid.size:
                vcolor = COMP_COLORS[2]
                vlabel = labels[2] if len(labels) >= 3 else "Vertical"
                if show_p05_p95:
                    _fill_band(ax, vgrid, vq[5], vq[95], logy=logy,
                               facecolor=vcolor, alpha=0.09, edgecolor="none", zorder=1)
                if show_p16_p84:
                    _fill_band(ax, vgrid, vq[16], vq[84], logy=logy,
                               facecolor=vcolor, alpha=0.21, edgecolor="none", zorder=2)
                if show_p50:
                    _plot_line(ax, vgrid, vq[50], logy=logy, color=vcolor, alpha=1.0,
                               lw=max(1.20, 2.20 * scale * float(p50_lw_scale)), label="V - P50", zorder=4)

        if overlay_selected and selected < len(family):
            if len(family[selected]) >= 2:
                x1, y1 = family[selected][0]
                x2, y2 = family[selected][1]
                sgrid = _reference_grid([(x1, y1), (x2, y2)])
                if sgrid.size:
                    srms = np.sqrt(0.5 * (
                        _curve_on_grid(sgrid, x1, y1) ** 2 + _curve_on_grid(sgrid, x2, y2) ** 2
                    ))
                    _plot_line(ax, sgrid, srms, logy=logy, color="#2f3336", alpha=0.86,
                               lw=max(1.05, 1.25 * scale), ls="--",
                               label="_nolegend_", zorder=5)
            if len(family[selected]) >= 3:
                vx, vy = family[selected][2]
                _plot_line(ax, vx, vy, logy=logy, color=COMP_COLORS[2], alpha=0.83,
                           lw=max(0.95, 1.10 * scale), ls="--",
                           label="_nolegend_", zorder=5)

    elif family:
        ncomp = max(len(rec) for rec in family)
        if percentile_component_count is not None:
            ncomp = min(ncomp, max(1, int(percentile_component_count)))
        for j in range(ncomp):
            grid, stack = _component_family(family, j)
            if grid.size == 0:
                continue
            q = _percentile_rows(stack)
            color = COMP_COLORS[j % len(COMP_COLORS)]
            label = labels[j] if j < len(labels) else f"Component {j + 1}"
            if show_p05_p95:
                _fill_band(ax, grid, q[5], q[95], logy=logy,
                           facecolor=color, alpha=0.09, edgecolor="none", zorder=1)
            if show_p16_p84:
                _fill_band(ax, grid, q[16], q[84], logy=logy,
                           facecolor=color, alpha=0.21, edgecolor="none", zorder=2)
            if show_p50:
                legend_label = (("M - P50", "I - P50", "V - P50")[j]
                                if j < 3 else f"C{j + 1} - P50")
                _plot_line(ax, grid, q[50], logy=logy, color=color, alpha=1.0,
                           lw=max(1.20, 2.20 * scale * float(p50_lw_scale)), label=legend_label, zorder=4)

        if overlay_selected:
            selected_rec = family[selected]
            if percentile_component_count is not None:
                selected_rec = selected_rec[:max(1, int(percentile_component_count))]
            for j, (x, y) in enumerate(selected_rec):
                color = COMP_COLORS[j % len(COMP_COLORS)]
                label = labels[j] if j < len(labels) else f"Component {j + 1}"
                _plot_line(
                    ax, x, y, logy=logy, color=color, alpha=0.83,
                    lw=max(0.95, 1.10 * scale), ls="--",
                    label="_nolegend_", zorder=5,
                )

    ax.set_xscale("log" if logx else "linear")
    ax.set_yscale("log" if logy else "linear")
    if xlim is not None:
        ax.set_xlim(*xlim)

    handles, legend_labels = ax.get_legend_handles_labels()
    unique = {}
    for h, lab in zip(handles, legend_labels):
        if lab and lab != "_nolegend_" and not lab.startswith("_"):
            unique.setdefault(lab, h)
    # Keep spectral legends compact and entirely inside the axes.  Short
    # labels make a three-row upper-right legend more robust than a long
    # one-row legend when the application is restored to its medium size.
    left, right, bottom, top = _spectral_subplot_margins(fig)
    legend_style = _compact_corner_legend_style(fig, len(unique), scale)

    # Reserve headroom for the in-axes legend so it does not cover the highest
    # spectral ordinates.  Three rows are used for M/I/V component mode.
    legend_rows = max(1, len(unique))
    auto_ymin, auto_ymax = _family_y_bounds(
        family, xlim=xlim, logy=logy, legend_rows=legend_rows
    )
    if logy:
        lower = float(ymin) if ymin is not None and np.isfinite(ymin) and ymin > 0 else auto_ymin
        if lower is not None and auto_ymax is not None and auto_ymax > lower:
            ax.set_ylim(bottom=lower, top=auto_ymax)
        elif lower is not None:
            ax.set_ylim(bottom=lower)
        elif auto_ymax is not None:
            ax.set_ylim(top=auto_ymax)
    elif auto_ymax is not None:
        ax.set_ylim(bottom=0.0, top=auto_ymax)

    ax.set_xlabel(xlabel, fontsize=11.1 * scale, labelpad=4.0)
    ax.set_ylabel(ylabel, fontsize=11.1 * scale, labelpad=5.0)
    ax.tick_params(axis="x", pad=2.4 * scale)

    if unique:
        response_loglog = bool(logx and logy and str(xlabel).lower().startswith("period"))
        legend_loc = "lower left" if response_loglog else "upper right"
        legend_anchor = (0.015, 0.015) if response_loglog else (0.985, 0.985)
        leg = ax.legend(
            unique.values(), unique.keys(), frameon=False,
            loc=legend_loc, bbox_to_anchor=legend_anchor,
            borderaxespad=0.0, **legend_style,
        )

    # Pixel-aware left margin prevents y-axis titles from being clipped in
    # restored/narrow windows while geometry remains stable across realizations.
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def _rings(geom):
    if not geom:
        return
    typ = geom.get("type")
    coords = geom.get("coordinates", [])
    if typ == "Polygon":
        for ring in coords:
            yield np.asarray(ring, float)
    elif typ == "MultiPolygon":
        for poly in coords:
            for ring in poly:
                yield np.asarray(ring, float)


def _polygons(geom):
    if not geom:
        return
    typ = geom.get("type")
    coords = geom.get("coordinates", [])
    if typ == "Polygon":
        yield [np.asarray(r, float) for r in coords]
    elif typ == "MultiPolygon":
        for poly in coords:
            yield [np.asarray(r, float) for r in poly]


def _zone_code(props):
    raw = ""
    for key in ("Zona", "ZONA", "zona", "Zone", "zone"):
        if key in props and props[key] is not None:
            raw = str(props[key]); break
    token = raw.upper().replace(" ", "").replace("_", "").strip()
    if token in {"III", "3", "ZONAIII"}: return "III"
    if token in {"II", "2", "ZONAII"}: return "II"
    if token in {"I", "1", "ZONAI"}: return "I"
    return ""


def _polygon_contains(poly_rings, lon, lat):
    if not poly_rings:
        return False
    point = (float(lon), float(lat))
    ext = np.asarray(poly_rings[0], float)
    if ext.ndim != 2 or ext.shape[0] < 3 or not MplPath(ext[:, :2]).contains_point(point, radius=1e-12):
        return False
    for hole in poly_rings[1:]:
        hole = np.asarray(hole, float)
        if hole.ndim == 2 and hole.shape[0] >= 3 and MplPath(hole[:, :2]).contains_point(point, radius=1e-12):
            return False
    return True


def zone_at_point(assets: Path, lon: float, lat: float) -> str:
    p = Path(assets) / "ZONAS.geojson"
    if not p.is_file():
        return ""
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        feats = list(raw.get("features", []))
        feats.sort(key=lambda feat: {"I": 0, "II": 1, "III": 2}.get(_zone_code(dict(feat.get("properties") or {})), 9))
        for feat in feats:
            zone = _zone_code(dict(feat.get("properties") or {}))
            for poly in _polygons(feat.get("geometry")):
                if _polygon_contains(poly, lon, lat):
                    return zone
    except Exception:
        return ""
    return ""


def plot_site_map(fig: Figure, assets: Path, stations, lon, lat):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_facecolor("#F7F8FA")
    scale = _scale(fig, 650, 720)

    zone_rings = {"I": [], "II": [], "III": []}
    p = Path(assets) / "ZONAS.geojson"
    if p.is_file():
        raw = json.loads(p.read_text(encoding="utf-8"))
        for feat in raw.get("features", []):
            zone = _zone_code(dict(feat.get("properties") or {}))
            if zone not in zone_rings:
                continue
            for ring in _rings(feat.get("geometry")):
                if ring.ndim == 2 and ring.shape[0] >= 3 and ring.shape[1] >= 2:
                    zone_rings[zone].append(ring)

        # Draw from C to B to A so all A islands have the same final fill.
        for zone in ("III", "II", "I"):
            for ring in zone_rings[zone]:
                ax.fill(
                    ring[:, 0], ring[:, 1],
                    facecolor=ZONE_COLORS[zone],
                    edgecolor=ZONE_EDGE,
                    linewidth=0.60,
                    alpha=1.0,
                    zorder={"III": 1, "II": 2, "I": 3}[zone],
                )

    b = Path(assets) / "CDMX_boundary.geojson"
    if b.is_file():
        raw = json.loads(b.read_text(encoding="utf-8"))
        for feat in raw.get("features", []):
            for ring in _rings(feat.get("geometry")):
                if ring.ndim == 2 and ring.shape[0] >= 3:
                    ax.plot(
                        ring[:, 0], ring[:, 1],
                        ls=(0, (4, 3)), color=BOUNDARY_COLOR,
                        lw=1.0, alpha=0.82, zorder=4,
                    )

    if stations is not None and len(stations):
        ax.scatter(
            stations.longitude, stations.latitude,
            s=24 * scale, marker="s",
            facecolors="white", edgecolors="black",
            linewidths=0.85, alpha=0.98, zorder=5,
        )
    ax.scatter(
        [lon], [lat], s=84 * scale, marker="v",
        facecolors="#D62728", edgecolors="black",
        linewidths=1.0, zorder=7,
    )

    zone_a_rings = zone_rings["I"]
    if zone_a_rings:
        pts = np.vstack([r[:, :2] for r in zone_a_rings])
        za_xmin, za_ymin = np.nanmin(pts, axis=0)
        za_xmax, za_ymax = np.nanmax(pts, axis=0)
        if stations is not None and len(stations):
            sx0, sx1 = float(np.nanmin(stations.longitude)), float(np.nanmax(stations.longitude))
            sy0, sy1 = float(np.nanmin(stations.latitude)), float(np.nanmax(stations.latitude))
            sxpad = 0.035 * max(sx1 - sx0, 1e-6)
            sypad = 0.035 * max(sy1 - sy0, 1e-6)
            xmin = max(za_xmin, sx0 - sxpad)
            xmax = min(za_xmax, sx1 + sxpad)
            ymin = max(za_ymin, sy0 - sypad)
            ymax = min(za_ymax, sy1 + sypad)
            if xmax <= xmin or ymax <= ymin:
                xmin, xmax, ymin, ymax = za_xmin, za_xmax, za_ymin, za_ymax
        else:
            xmin, xmax, ymin, ymax = za_xmin, za_xmax, za_ymin, za_ymax
        dx = max(xmax - xmin, 1e-6)
        dy = max(ymax - ymin, 1e-6)
        ax.set_xlim(xmin - 0.015 * dx, xmax + 0.015 * dx)
        ax.set_ylim(ymin - 0.015 * dy, ymax + 0.015 * dy)

    ax.set_xlabel("Longitude °", fontsize=11.0 * scale, labelpad=4.5)
    ax.set_ylabel("Latitude °", fontsize=11.0 * scale, labelpad=4.5)
    ax.set_title("Mexico City seismic zones · site selection", fontsize=11.8 * scale)
    ax.tick_params(labelsize=9.3 * scale)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(bottom=True, left=True, top=False, right=False)
    ax.grid(False)
    ax.set_aspect("equal", adjustable="box", anchor="C")

    handles = [
        Patch(facecolor=ZONE_COLORS["I"], edgecolor="black", linewidth=0.8, label="A"),
        Patch(facecolor=ZONE_COLORS["II"], edgecolor="black", linewidth=0.8, label="B"),
        Patch(facecolor=ZONE_COLORS["III"], edgecolor="black", linewidth=0.8, label="C"),
        Line2D([0], [0], color=BOUNDARY_COLOR, lw=1.0, ls=(0, (4, 3)), label="CDMX boundary"),
        Line2D([0], [0], marker="s", markerfacecolor="white", markeredgecolor="black",
               lw=0, markersize=5.5 * scale, label="Stations"),
        Line2D([0], [0], marker="v", markerfacecolor="#D62728", markeredgecolor="black",
               lw=0, markersize=6.5 * scale, label="Selected site"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.8 * scale, ncol=2,
              loc="lower left", handletextpad=0.30, columnspacing=0.65, labelspacing=0.22)
    fig.subplots_adjust(left=0.105, right=0.992, bottom=0.105, top=0.945)
