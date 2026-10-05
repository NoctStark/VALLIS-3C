from __future__ import annotations
import copy
from PyQt6.QtCore import QThread, pyqtSignal
from core.generator import generate
from core.spectra import family_spectra
from core.exports import export_family


class GenerationWorker(QThread):
    completed = pyqtSignal(object, object, object, object, object)
    failed = pyqtSignal(str)
    progressed = pyqtSignal(int, int)


    def __init__(self, values, settings, parent=None):
        super().__init__(parent)
        self.values = dict(values)
        self.settings = copy.deepcopy(settings)
        self._cancel = False


    def cancel(self):
        self._cancel = True


    def run(self):
        try:
            def generation_progress(done, total):
                frac = (float(done) / float(total)) if total else 0.0
                self.progressed.emit(int(round(82.0 * frac)), 100)


            payload, scenario, spatial = generate(
                self.values,
                progress=generation_progress,
                cancel=lambda: self._cancel,
            )
            if self._cancel:
                raise RuntimeError("Generation cancelled")
            self.progressed.emit(83, 100)


            def spectra_progress(done, total):
                frac = (float(done) / float(total)) if total else 0.0
                self.progressed.emit(83 + int(round(16.0 * frac)), 100)


            fas, rs = family_spectra(
                payload["accelerations"], float(payload["dt_s"]), self.settings,
                payload.get("valid_npts"), progress=spectra_progress,
            )
            self.progressed.emit(99, 100)
            self.completed.emit(payload, scenario, spatial, fas, rs)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class SpectraWorker(QThread):
    completed = pyqtSignal(object, object, str)
    failed = pyqtSignal(str, str)
    progressed = pyqtSignal(int, int)


    def __init__(self, accelerations, dt_s, settings, valid_npts, context="update", parent=None):
        super().__init__(parent)
        self.accelerations = accelerations
        self.dt_s = float(dt_s)
        self.settings = copy.deepcopy(settings)
        self.valid_npts = valid_npts
        self.context = str(context)


    def run(self):
        try:
            fas, rs = family_spectra(
                self.accelerations, self.dt_s, self.settings, self.valid_npts,
                progress=lambda d, t: self.progressed.emit(int(d), int(t)),
            )
            self.completed.emit(fas, rs, self.context)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}", self.context)


class ExportWorker(QThread):
    completed = pyqtSignal(str, bool)
    failed = pyqtSignal(str, bool)


    def __init__(self, run_dir, payload, fas, rs, settings, scenario, labels, rotation_angles,
                 site_id="Site_001", rotated=False, silent=False, parent=None):
        super().__init__(parent)
        self.run_dir = run_dir
        self.payload = payload
        self.fas = fas
        self.rs = rs
        self.settings = copy.deepcopy(settings)
        self.scenario = dict(scenario)
        self.labels = tuple(labels)
        self.rotation_angles = rotation_angles
        self.site_id = str(site_id or "Site_001")
        self.rotated = bool(rotated)
        self.silent = bool(silent)


    def run(self):
        try:
            export_family(
                self.run_dir, self.payload, self.fas, self.rs, self.settings,
                self.scenario, self.labels, self.rotation_angles,
                site_id=self.site_id, rotated=self.rotated,
            )
            self.completed.emit(str(self.run_dir), self.silent)
        except Exception as exc:
            self.failed.emit(str(exc), self.silent)
