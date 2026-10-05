"""Main academic UI for VALLIS-3C.


Author: Joel D. Cruz-Arguelles
Instituto de Ingeniería, Universidad Nacional Autónoma de México
Coyoacán, 04510 Ciudad de México, México
2026
"""
from __future__ import annotations


from pathlib import Path
import os
from dataclasses import asdict
import datetime
import json
import numpy as np
import pandas as pd


from PyQt6.QtCore import Qt, QTimer, QEvent, QPropertyAnimation, QEasingCurve, QRect, pyqtSignal
from PyQt6.QtGui import QAction, QFont, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QAbstractSpinBox, QButtonGroup, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QComboBox, QDoubleSpinBox, QFrame, QGraphicsOpacityEffect, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QRadioButton, QSizePolicy, QSpinBox, QSplitter, QToolButton, QVBoxLayout, QWidget, QMenu,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


from app.settings import AppSettings
from app.dialogs import SettingsDialog, HelpDialog, DocumentDialog, AboutDialog
from app.worker import GenerationWorker, SpectraWorker, ExportWorker
from app.plotting import plot_site_map, plot_accelerograms, plot_family, zone_at_point, fas_display_floor
from core.rotation import rotate_horizontals
from core.domain import evaluate_calibration_domain


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
DEFAULT_SETTINGS_FILE = ROOT / "config" / "default_settings.json"
_USER_CONFIG_ROOT = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / ".vallis3c"))
SETTINGS_FILE = _USER_CONFIG_ROOT / "VALLIS-3C" / "1.6.0" / "settings.json"
ICON_FILE = ASSETS / "icons" / "VALLIS-3C.ico"
LOGO_FILE = ASSETS / "icons" / "VALLIS-3C_512.png"


class AnimatedLogoLabel(QLabel):
    """Floating toolbar logo with a restrained hover animation."""

    clicked = pyqtSignal()

    def __init__(self, image_path: Path, parent=None):
        super().__init__(parent)
        self._center = None
        self._normal_size = 50
        self._hover_size = 58
        self.setObjectName("brandLogo")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setScaledContents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("About VALLIS-3C")
        if Path(image_path).is_file():
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                self.setPixmap(pixmap)
        self._animation = QPropertyAnimation(self, b"geometry", self)
        self._animation.setDuration(150)
        self._animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _rect_for_size(self, size):
        if self._center is None:
            return self.geometry()
        return QRect(
            int(round(self._center.x() - size / 2)),
            int(round(self._center.y() - size / 2)),
            int(size), int(size),
        )

    def set_anchor_center(self, center):
        self._center = center
        target = self._hover_size if self.underMouse() else self._normal_size
        self.setGeometry(self._rect_for_size(target))
        self.raise_()

    def _animate_to(self, size):
        if self._center is None:
            return
        self._animation.stop()
        self._animation.setStartValue(self.geometry())
        self._animation.setEndValue(self._rect_for_size(size))
        self._animation.start()
        self.raise_()

    def enterEvent(self, event):
        self._animate_to(self._hover_size)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._animate_to(self._normal_size)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class SitePeriodReminder(QFrame):
    """Transient map overlay reminding the user to review the site period."""


    def __init__(self, parent):
        super().__init__(parent)
        self.setObjectName("sitePeriodReminder")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            "QFrame#sitePeriodReminder {"
            " background-color: rgba(255, 255, 255, 222);"
            " border: 1px solid rgba(24, 39, 52, 125);"
            " border-radius: 10px;"
            "}"
        )


        row = QHBoxLayout(self)
        row.setContentsMargins(16, 11, 8, 11)
        row.setSpacing(10)


        self.message = QLabel(
            "Enter a measured site-period value "
            "or the value reported by SASID."
        )
        self.message.setWordWrap(True)
        self.message.setStyleSheet(
            "QLabel { background: transparent; color: #111111; font-weight: 700; }"
        )
        row.addWidget(self.message, 1)


        close_btn = QToolButton(self)
        close_btn.setText("×")
        close_btn.setToolTip("Close")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(
            "QToolButton {"
            " background: transparent; border: none; color: #111111;"
            " font-size: 18px; font-weight: 700; padding: 0 3px;"
            "}"
            "QToolButton:hover { color: #8a1c1c; }"
        )
        close_btn.clicked.connect(self.dismiss)
        row.addWidget(close_btn, 0, Qt.AlignmentFlag.AlignTop)


        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity)


        self._fade = QPropertyAnimation(self._opacity, b"opacity", self)
        self._fade.setDuration(700)
        self._fade.setStartValue(1.0)
        self._fade.setEndValue(0.0)
        self._fade.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._fade.finished.connect(self.hide)


        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._start_fade)
        self.hide()


    def _recenter(self):
        parent = self.parentWidget()
        if parent is None:
            return
        width = min(470, max(300, parent.width() - 60))
        self.setFixedWidth(width)
        self.adjustSize()
        x = max(0, (parent.width() - self.width()) // 2)
        y = max(0, (parent.height() - self.height()) // 2)
        self.move(x, y)


    def show_for(self, total_ms=5000):
        self._timer.stop()
        self._fade.stop()
        self._opacity.setOpacity(1.0)
        self._recenter()
        self.raise_()
        self.show()
        self._timer.start(max(250, int(total_ms) - self._fade.duration()))


    def _start_fade(self):
        if not self.isVisible():
            return
        self._fade.stop()
        self._fade.setStartValue(float(self._opacity.opacity()))
        self._fade.setEndValue(0.0)
        self._fade.start()


    def dismiss(self):
        self._timer.stop()
        self._fade.stop()
        self.hide()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VALLIS-3C V1.6.0")
        if ICON_FILE.is_file():
            self.setWindowIcon(QIcon(str(ICON_FILE)))
        # Restored mode uses a wide scientific-dashboard geometry so the site map remains readable.
        self._restored_target_size = (1880, 930)
        self.resize(*self._restored_target_size)
        self.setMinimumSize(1100, 680)


        self.settings = AppSettings.load(SETTINGS_FILE, defaults_path=DEFAULT_SETTINGS_FILE)
        self.payload = None
        self.raw_accelerations = None
        self.fas = []
        self.rs = []
        self.labels = ("Major", "Intermediate") if int(self.settings.output_components) == 2 else ("Major", "Intermediate", "Vertical")
        self.rotation_angles = np.array([])
        self.worker = None
        self.analysis_worker = None
        self.export_worker = None
        self.last_run_folder = None
        self.rotated_saved_folder = None
        self.rotation_applied = False
        self.selected_station_id = ""
        self.scenario = {}
        self.spatial = {}


        self.stations = self._load_stations()
        self._menus()
        self._build()
        self._update_rs_basis_button()
        self._style()
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._after_resize)
        self._update_theta_controls()
        self._update_rotation_controls()
        self._last_site_reminder_coords = (float(self.site_lon.value()), float(self.site_lat.value()))
        self._site_coordinate_changed()
        self._connect_domain_status()
        self._refresh_calibration_status()
        QTimer.singleShot(0, self._restore_medium_geometry)


    def _load_stations(self):
        try:
            return pd.read_csv(ASSETS / "stations.csv")
        except Exception:
            return pd.DataFrame(columns=["station_id", "longitude", "latitude"])


    def _menus(self):
        mb = self.menuBar()
        # Keep the in-window Qt menu bar so its static labels use the same
        # explicit typography as hover/selected states on every platform.
        mb.setNativeMenuBar(False)
        f = mb.addMenu("File")
        f.addAction("Save project…", self._save_project)
        f.addAction("Load project…", self._load_project)
        f.addSeparator()
        f.addAction("Select output directory…", self._choose_output)
        f.addAction("Export current results", self._export_current)
        f.addSeparator()
        f.addAction("Exit", self.close)


        self.configuration_action = QAction("Configuration", self)
        self.configuration_action.triggered.connect(self._configure)
        mb.addAction(self.configuration_action)


        h = mb.addMenu("Help")
        h.addAction("Technical guide and definitions…", lambda: HelpDialog(self).exec())
        h.addAction("Data provenance…", lambda: DocumentDialog("Data provenance", ROOT / "DATA_PROVENANCE.md", self).exec())
        h.addAction("Scientific disclaimer…", lambda: DocumentDialog("Scientific disclaimer", ROOT / "DISCLAIMER.md", self).exec())
        h.addAction("License…", lambda: DocumentDialog("GNU General Public License v3.0", ROOT / "LICENSE", self).exec())
        h.addSeparator()
        h.addAction("About…", self._about)


    def _spin(self, lo, hi, val, dec=2, step=0.1):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setDecimals(dec)
        s.setSingleStep(step)
        s.setValue(val)
        s.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        return s


    def _compact_group(self, title):
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setContentsMargins(9, 12, 9, 7)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(5)
        return box, form


    def _build(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(8, 6, 8, 8)
        outer.setSpacing(5)


        # Compact global toolbar: branding and the controls that apply across
        # the time-history and response-spectrum panels.
        toolbar_envelope = QWidget()
        toolbar_envelope.setObjectName("toolbarEnvelope")
        toolbar_envelope.setFixedHeight(62)
        envelope_layout = QVBoxLayout(toolbar_envelope)
        envelope_layout.setContentsMargins(0, 9, 0, 9)

        top_bar = QFrame(toolbar_envelope)
        top_bar.setObjectName("topToolbar")
        top_bar.setFixedHeight(44)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(7, 1, 9, 1)
        top_bar_layout.setSpacing(7)

        self.logo_anchor = QWidget()
        self.logo_anchor.setFixedSize(52, 36)
        top_bar_layout.addWidget(self.logo_anchor)
        self.brand_name = QLabel("VALLIS-3C")
        self.brand_name.setObjectName("brandName")
        top_bar_layout.addWidget(self.brand_name)

        separator = QFrame()
        separator.setObjectName("toolbarSeparator")
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Plain)
        top_bar_layout.addWidget(separator)

        top_bar_layout.addStretch(1)

        realization_label = QLabel("Realization")
        realization_label.setObjectName("toolbarLabel")
        top_bar_layout.addWidget(realization_label)
        self.prev_pick = QToolButton()
        self.prev_pick.setObjectName("compactNavButton")
        self.prev_pick.setText("◀")
        self.prev_pick.setToolTip("Previous realization")
        self.prev_pick.clicked.connect(self._previous_realization)
        self.pick = QSpinBox()
        self.pick.setObjectName("realizationPicker")
        self.pick.setRange(1, 1)
        self.pick.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.pick.setKeyboardTracking(False)
        self.pick.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pick.setFixedWidth(58)
        self.pick.valueChanged.connect(self._pick_changed)
        self.next_pick = QToolButton()
        self.next_pick.setObjectName("compactNavButton")
        self.next_pick.setText("▶")
        self.next_pick.setToolTip("Next realization")
        self.next_pick.clicked.connect(self._next_realization)
        top_bar_layout.addWidget(self.prev_pick)
        top_bar_layout.addWidget(self.pick)
        top_bar_layout.addWidget(self.next_pick)

        spectrum_separator = QFrame()
        spectrum_separator.setObjectName("toolbarSeparator")
        spectrum_separator.setFrameShape(QFrame.Shape.VLine)
        spectrum_separator.setFrameShadow(QFrame.Shadow.Plain)
        top_bar_layout.addWidget(spectrum_separator)

        spectrum_label = QLabel("Response spectrum scale")
        spectrum_label.setObjectName("toolbarLabel")
        top_bar_layout.addWidget(spectrum_label)
        self.rs_scale_group = QButtonGroup(self)
        self.rs_log_plot = QRadioButton("Log–Log")
        self.rs_linear_plot = QRadioButton("Lin–Lin")
        self.rs_log_plot.setObjectName("rsScaleRadio")
        self.rs_linear_plot.setObjectName("rsScaleRadio")
        self.rs_scale_group.addButton(self.rs_log_plot)
        self.rs_scale_group.addButton(self.rs_linear_plot)
        self.rs_log_plot.setChecked(not bool(self.settings.rs_plot_linear))
        self.rs_linear_plot.setChecked(bool(self.settings.rs_plot_linear))
        self.rs_linear_plot.toggled.connect(self._toggle_rs_scale)
        top_bar_layout.addWidget(self.rs_log_plot)
        top_bar_layout.addWidget(self.rs_linear_plot)
        self.rs_basis_toggle = QToolButton()
        self.rs_basis_toggle.setObjectName("spectrumBandsButton")
        self.rs_basis_toggle.setCheckable(True)
        self.rs_basis_toggle.setChecked(str(getattr(self.settings, "rs_percentile_basis", "components")) == "rms")
        self.rs_basis_toggle.toggled.connect(self._toggle_rs_percentile_basis)
        top_bar_layout.addWidget(self.rs_basis_toggle)

        self.top_toolbar = top_bar
        self.toolbar_envelope = toolbar_envelope
        envelope_layout.addWidget(top_bar)
        self.brand_logo = AnimatedLogoLabel(LOGO_FILE, toolbar_envelope)
        self.brand_logo.clicked.connect(self._about)
        outer.addWidget(toolbar_envelope, 0)
        QTimer.singleShot(0, self._sync_brand_logo)


        # Three-column scientific layout:
        #   left  -> site map (full available height)
        #   center -> controls + acceleration histories
        #   right -> FAS + response spectrum
        main_split = QSplitter(Qt.Orientation.Horizontal)
        main_split.setChildrenCollapsible(False)
        outer.addWidget(main_split, 1)
        self.main_splitter = main_split


        # LEFT: site inputs above the full-height map column.
        left_col = QWidget()
        left_v = QVBoxLayout(left_col)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(5)
        main_split.addWidget(left_col)


        map_box = QGroupBox("Site map · seismic zonation")
        map_v = QVBoxLayout(map_box)
        map_v.setContentsMargins(6, 10, 6, 6)
        self.map_fig = Figure(figsize=(6.7, 9.0), dpi=100)
        self.map_canvas = FigureCanvas(self.map_fig)
        self.map_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.map_canvas.setMinimumSize(310, 430)
        self.map_canvas.mpl_connect("button_press_event", self._map_click)
        map_v.addWidget(self.map_canvas, 1)
        self.site_period_reminder = SitePeriodReminder(self.map_canvas)
        map_note = QLabel(
            "Click a station to use its current Ts. "
            "Free clicks in Zone A assign Ts = 0.50 s; Zones B/C keep the entered Ts."
        )
        map_note.setWordWrap(True)
        map_note.setObjectName("helper")
        map_v.addWidget(map_note)
        left_v.addWidget(map_box, 1)


        # CENTER + RIGHT region.
        work = QWidget()
        wv = QVBoxLayout(work)
        wv.setContentsMargins(4, 0, 0, 0)
        wv.setSpacing(5)
        main_split.addWidget(work)


        controls = QWidget()
        cg = QGridLayout(controls)
        cg.setContentsMargins(0, 0, 0, 0)
        cg.setHorizontalSpacing(6)
        cg.setVerticalSpacing(3)


        # Scenario and source-to-site path are intentionally grouped as one input block.
        scenario_box = QGroupBox("Scenario")
        sh = QHBoxLayout(scenario_box)
        sh.setContentsMargins(9, 12, 9, 7)
        sh.setSpacing(12)
        sf = QFormLayout()
        sf.setHorizontalSpacing(8); sf.setVerticalSpacing(5)
        pf = QFormLayout()
        pf.setHorizontalSpacing(8); pf.setVerticalSpacing(5)


        self.source = QComboBox()
        self.source.addItem("Intraslab", "INSLAB")
        self.source.addItem("Interplate", "INTERPLATE")
        self.mw = self._spin(4, 9, 7.1)
        self.rrup = self._spin(1, 1000, 110, 1, 5)
        self.depth = self._spin(0, 250, 46.4, 1, 1)
        sf.addRow("Source type", self.source)
        mw_label = QLabel("M<sub>w</sub>")
        mw_label.setTextFormat(Qt.TextFormat.RichText)
        rrup_label = QLabel("R<sub>rup</sub> (km)")
        rrup_label.setTextFormat(Qt.TextFormat.RichText)
        sf.addRow(mw_label, self.mw)
        sf.addRow(rrup_label, self.rrup)
        sf.addRow("Depth (km)", self.depth)


        self.theta_mode = QComboBox()
        self.theta_mode.addItem("Manual", "manual")
        self.theta_mode.addItem("Automatic · CU to source", "automatic")
        self.theta_mode.addItem("Critical", "critical")
        self.theta_mode.addItem("Median", "median")
        self.theta_mode.setCurrentIndex(0)
        self.theta_mode.currentIndexChanged.connect(self._update_theta_controls)
        self.ev_lon = self._spin(-110, -85, -98.72, 4, 0.05)
        self.ev_lat = self._spin(10, 30, 18.40, 4, 0.05)
        self.theta = self._spin(0, 150, 115, 2, 1)
        self.theta_info = QLabel("User-defined path angle.")
        self.theta_info.setObjectName("helper")
        self.theta_info.setWordWrap(True)
        pf.addRow("Path-angle mode", self.theta_mode)
        pf.addRow("Source longitude °", self.ev_lon)
        pf.addRow("Source latitude °", self.ev_lat)
        pf.addRow("Manual θ (deg)", self.theta)
        pf.addRow(self.theta_info)
        sh.addLayout(sf, 1)
        sh.addLayout(pf, 1)
        cg.addWidget(scenario_box, 0, 0, 1, 2)


        site_box, form = self._compact_group("Site")
        self.site_id = QLineEdit("Site_001")
        self.site_lon = self._spin(-101, -97, -99.15, 5, 0.01)
        self.site_lat = self._spin(18, 21, 19.40, 5, 0.01)
        self.ts = self._spin(0.05, 6, 2.0, 3, 0.05)
        form.addRow("Site ID", self.site_id)
        form.addRow("Longitude °", self.site_lon)
        form.addRow("Latitude °", self.site_lat)
        form.addRow("Ts (s)", self.ts)
        self.site_lon.editingFinished.connect(self._manual_site_coordinate_edited)
        self.site_lat.editingFinished.connect(self._manual_site_coordinate_edited)
        self.update_site = QPushButton("Apply site")
        self.update_site.clicked.connect(lambda: self._site_coordinate_changed(show_reminder=True))
        form.addRow(self.update_site)
        self.site_zone_info = QLabel("Zone: unresolved")
        self.site_zone_info.setWordWrap(True)
        self.site_zone_info.setObjectName("helper")
        form.addRow(self.site_zone_info)
        self.site_box = site_box
        left_v.insertWidget(0, site_box, 0)


        family_box = QGroupBox("Generation")
        fv = QGridLayout(family_box)
        fv.setContentsMargins(8, 11, 8, 6)
        fv.setHorizontalSpacing(6)
        fv.setVerticalSpacing(5)
        self.n = QSpinBox()
        self.n.setRange(1, 5000)
        self.n.setValue(30)
        self.n.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(1234)
        self.seed.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.shared_event_seed_label = QLabel("Shared-event seed")
        self.event_seed = QSpinBox()
        self.event_seed.setRange(0, 2147483647)
        self.event_seed.setValue(1234)
        self.event_seed.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.event_seed.setToolTip("Used only in Common event mode to select the one between-event FAS residual shared by the ensemble.")
        self.generate = QPushButton("Generate ensemble")
        self.generate.clicked.connect(self._generate)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status = QLabel("Ready")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.calibration_status = QLabel("Calibration domain: evaluating…")
        self.calibration_status.setObjectName("calibrationStatus")
        self.calibration_status.setWordWrap(True)
        fv.addWidget(QLabel("Realizations"), 0, 0)
        fv.addWidget(self.n, 0, 1)
        fv.addWidget(QLabel("Seed"), 0, 2)
        fv.addWidget(self.seed, 0, 3)
        fv.addWidget(self.shared_event_seed_label, 1, 0)
        fv.addWidget(self.event_seed, 1, 1)
        # Primary generation action spans the full panel width.  The progress
        # bar sits directly below it with the identical span for a cleaner,
        # easier-to-read compact layout.
        fv.addWidget(self.generate, 2, 0, 1, 4)
        fv.addWidget(self.progress, 3, 0, 1, 4)
        fv.addWidget(self.status, 4, 0, 1, 4)
        fv.addWidget(self.calibration_status, 5, 0, 1, 4)
        self._update_event_seed_control()
        cg.addWidget(family_box, 0, 2)


        self.rotation_box, rot = self._compact_group("Horizontal rotation")
        self.rotation_mode = QComboBox()
        self.rotation_mode.addItem("Native axes", "none")
        self.rotation_mode.addItem("Random per realization", "random")
        self.rotation_mode.addItem("Fixed angle", "manual")
        self.rotation_mode.currentIndexChanged.connect(self._rotation_selection_changed)
        self.rotation_angle = self._spin(-1.0e9, 1.0e9, 0.0, 2, 1.0)
        self.rotation_angle.valueChanged.connect(self._rotation_selection_changed)
        self.apply_rotation = QPushButton("Apply rotation")
        self.apply_rotation.clicked.connect(self._apply_rotation)
        self.save_rotated = QPushButton("Save Rotated Motions")
        self.save_rotated.setEnabled(False)
        self.save_rotated.clicked.connect(self._save_rotated_motions)
        self.rotation_summary = QLabel("Available after generation.")
        self.rotation_summary.setObjectName("helper")
        self.rotation_summary.setWordWrap(True)
        rot.addRow("Mode", self.rotation_mode)
        rot.addRow("Angle φ (deg)", self.rotation_angle)
        rot.addRow(self.apply_rotation)
        rot.addRow(self.save_rotated)
        rot.addRow(self.rotation_summary)
        self.rotation_box.setEnabled(False)
        cg.addWidget(self.rotation_box, 0, 3)


        # Desired dashboard proportions (the scientific dashboard layout):
        # Scenario ~41%, Generation ~33.5%, Horizontal rotation ~25.5%.
        # Scenario spans columns 0-1, hence the paired 82/82 weights.
        cg.setColumnStretch(0, 82)
        cg.setColumnStretch(1, 82)
        cg.setColumnStretch(2, 134)
        cg.setColumnStretch(3, 102)
        self.controls_grid = cg
        self.top_controls = controls
        wv.addWidget(controls, 0)


        # Main analysis area: acceleration histories with FAS/response spectrum at right.
        lower = QSplitter(Qt.Orientation.Horizontal)
        lower.setChildrenCollapsible(False)
        wv.addWidget(lower, 1)


        acc_box = QGroupBox("Acceleration time histories")
        av = QVBoxLayout(acc_box)
        av.setContentsMargins(6, 10, 6, 5)
        self.acc_fig = Figure(figsize=(10.0, 8.4), dpi=100)
        self.acc_canvas = FigureCanvas(self.acc_fig)
        self.acc_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.acc_canvas.setMinimumSize(460, 430)
        av.addWidget(self.acc_canvas, 1)
        lower.addWidget(acc_box)


        diagnostics = QWidget()
        dv = QVBoxLayout(diagnostics)
        dv.setContentsMargins(4, 0, 0, 0)
        dv.setSpacing(5)


        fas_box = QGroupBox("Fourier amplitude spectra")
        fas_v = QVBoxLayout(fas_box)
        fas_v.setContentsMargins(5, 10, 5, 5)
        self.fas_fig = Figure(figsize=(5.3, 4.3), dpi=100)
        self.fas_canvas = FigureCanvas(self.fas_fig)
        self.fas_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.fas_canvas.setMinimumSize(260, 220)
        fas_v.addWidget(self.fas_canvas, 1)
        dv.addWidget(fas_box, 1)


        rs_box = QGroupBox("Response Spectrum")
        rs_v = QVBoxLayout(rs_box)
        rs_v.setContentsMargins(5, 10, 5, 5)
        self.rs_fig = Figure(figsize=(5.3, 4.3), dpi=100)
        self.rs_canvas = FigureCanvas(self.rs_fig)
        self.rs_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.rs_canvas.setMinimumSize(260, 220)
        rs_v.addWidget(self.rs_canvas, 1)
        dv.addWidget(rs_box, 1)
        lower.addWidget(diagnostics)


        # Wide dashboard balance: map ~30%, with the remaining workspace split
        # between motion histories and spectral diagnostics.  The slightly
        # wider map is deliberate in restored/medium mode.
        lower.setStretchFactor(0, 735)
        lower.setStretchFactor(1, 265)
        lower.setSizes([987, 357])
        self.lower_splitter = lower
        main_split.setStretchFactor(0, 285)
        main_split.setStretchFactor(1, 715)
        main_split.setSizes([536, 1344])
        QTimer.singleShot(0, self._apply_panel_proportions)


        self._update_realization_nav()


    def _restore_medium_geometry(self):
        """Size and center the restored dashboard on the active screen.

        The requested medium layout is wide rather than tall.  The target is
        capped to the screen's available work area so it remains usable on
        smaller displays, then centered explicitly.
        """
        if self.isMaximized() or self.isMinimized():
            return
        screen = self.screen()
        if screen is None:
            return
        available = screen.availableGeometry()
        target_w, target_h = getattr(self, "_restored_target_size", (1860, 880))
        margin_x = 24
        margin_y = 36
        width = min(int(target_w), max(self.minimumWidth(), available.width() - 2 * margin_x))
        height = min(int(target_h), max(self.minimumHeight(), available.height() - 2 * margin_y))
        self.resize(width, height)
        x = available.x() + max(0, (available.width() - width) // 2)
        y = available.y() + max(0, (available.height() - height) // 2)
        self.move(x, y)
        QTimer.singleShot(0, self._apply_panel_proportions)


    def _apply_panel_proportions(self):
        """Reapply the wide dashboard proportions after window-state changes.

        The restored (medium) window is deliberately wider than it is tall.
        Splitter sizes are expressed as fractions so the map keeps useful
        visual area instead of collapsing when the window is restored.
        """
        if hasattr(self, "main_splitter"):
            total = max(1, int(self.main_splitter.width()))
            left_fraction = 0.35 if (self.isMaximized() or self.isFullScreen()) else 0.31
            left = int(round(total * left_fraction))
            self.main_splitter.setSizes([left, max(1, total - left)])
        if hasattr(self, "lower_splitter"):
            total = max(1, int(self.lower_splitter.width()))
            acc = int(round(total * 0.735))
            self.lower_splitter.setSizes([acc, max(1, total - acc)])


    def _sync_top_panel_heights(self):
        """Keep Site aligned with the upper control row across DPI/font scaling."""
        if not hasattr(self, "site_box") or not hasattr(self, "top_controls"):
            return
        # Release any fixed height before asking Qt for fresh size hints.
        for widget in (self.site_box, self.top_controls):
            widget.setMinimumHeight(0)
            widget.setMaximumHeight(16777215)
        h = max(self.site_box.sizeHint().height(), self.top_controls.sizeHint().height())
        h = max(118, int(h))
        self.site_box.setFixedHeight(h)
        self.top_controls.setFixedHeight(h)


    def _style(self):
        """Apply a stable editorial UI hierarchy in both window states.

        Typography is intentionally fixed in points/pixels instead of being
        recomputed from the current window size. High-DPI scaling is left to
        Qt, which keeps the visual hierarchy consistent between restored and
        maximized layouts.
        """
        large = bool(self.isMaximized() or self.isFullScreen())
        base_pt = 12.0 if large else 10.7
        helper_pt = 11.9 if large else 11.0
        label_pt = 11.9 if large else 10.8
        input_pt = 12.4 if large else 11.2
        button_pt = 12.3 if large else 11.1
        panel_title_pt = 15.8 if large else 14.5
        calibration_pt = 10.2 if large else 9.4
        menu_pt = 12.2 if large else 11.3
        popup_pt = 11.3 if large else 10.5

        mb = self.menuBar()
        mb.setNativeMenuBar(False)
        menu_font = QFont("Segoe UI")
        menu_font.setPointSizeF(menu_pt)
        menu_font.setWeight(QFont.Weight.Medium)
        mb.setFont(menu_font)
        mb.setMinimumHeight(34)

        # Do not assign the large menu-bar font to QAction objects. On Windows
        # that font can leak into popup menus and make the hover/open state look
        # disproportionately large. Top-level labels are controlled by QSS.
        popup_font = QFont("Segoe UI")
        popup_font.setPointSizeF(popup_pt)
        popup_font.setWeight(QFont.Weight.Normal)
        for menu in mb.findChildren(QMenu):
            menu.setFont(popup_font)
            for action in menu.actions():
                action.setFont(popup_font)

        self.setStyleSheet(f"""
        QMainWindow,QWidget{{background:#f5f8fb;color:#1f2b34;font-family:'Segoe UI',Arial;font-size:{base_pt:.2f}pt}}
        QFrame#topToolbar{{background:#ffffff;border:1px solid #d7e1e8;border-radius:6px;}}
        QFrame#toolbarSeparator{{color:#cbd7df;max-width:1px;margin:5px 7px;}}
        QLabel#brandLogo{{background:transparent;border:none;}}
        QLabel#brandName{{background:transparent;color:#173f67;font-size:{panel_title_pt:.2f}pt;font-weight:750;padding:0 5px 0 2px;}}
        QLabel#toolbarLabel{{background:transparent;color:#334f68;font-size:{label_pt:.2f}pt;font-weight:650;padding:0 2px;}}
        QGroupBox{{background:#ffffff;border:1px solid #d7e1e8;border-radius:6px;margin-top:11px;padding:8px 6px 5px 6px;font-size:{panel_title_pt:.2f}pt;font-weight:700}}
        QGroupBox::title{{subcontrol-origin:margin;left:9px;padding:0 4px;color:#17324d;font-size:{panel_title_pt:.2f}pt;font-weight:700}}
        QGroupBox QLabel,QGroupBox QCheckBox,QGroupBox QRadioButton{{background:transparent;font-size:{label_pt:.2f}pt}}
        QLineEdit,QSpinBox,QDoubleSpinBox,QComboBox{{background:#ffffff;color:#18222a;border:1px solid #c9d5de;border-radius:4px;padding:3px 5px;min-height:25px;font-size:{input_pt:.2f}pt}}
        QLineEdit:disabled,QSpinBox:disabled,QDoubleSpinBox:disabled,QComboBox:disabled{{background:#fbfcfd;color:#7d8991;border:1px solid #d3dce3}}
        QPushButton{{background:#1f536d;color:#ffffff;border:0;border-radius:4px;padding:6px 10px;font-size:{button_pt:.2f}pt;font-weight:500}}
        QPushButton:hover{{background:#286a88}}
        QPushButton:disabled{{background:#aeb9c1;color:#ffffff}}
        QPushButton#secondaryButton{{background:#edf3f6;color:#1f536d;border:1px solid #c9d5de;padding:5px 8px}}
        QToolButton{{background:#eef3f6;border:1px solid #c5d0d8;border-radius:4px;padding:4px 8px;min-width:28px;font-size:{label_pt:.2f}pt}}
        QToolButton:hover{{background:#e1eaf0}}
        QToolButton:disabled{{color:#a7b1b8;background:#f6f8f9}}
        QToolButton#compactNavButton{{background:#2C5AA0;color:#ffffff;border:1px solid #23497f;padding:1px;min-width:25px;max-width:25px;min-height:23px;max-height:23px;font-size:8.5pt;font-weight:700;}}
        QToolButton#compactNavButton:hover{{background:#3970bc;border-color:#2c5a98;}}
        QToolButton#compactNavButton:disabled{{background:#a9bfd5;color:#f4f7fa;border-color:#91abc4;}}
        QSpinBox#realizationPicker{{padding:2px 4px;min-height:23px;max-height:23px;}}
        QRadioButton#rsScaleRadio{{background:transparent;color:#2c4355;font-size:{label_pt:.2f}pt;spacing:5px;}}
        QRadioButton#rsScaleRadio::indicator{{width:9px;height:9px;border-radius:5px;border:1px solid #687b88;background:#ffffff;}}
        QRadioButton#rsScaleRadio::indicator:checked{{background:#2C5AA0;border:1px solid #23497f;}}
        QToolButton#spectrumBandsButton{{padding:3px 8px;min-height:23px;font-size:{label_pt:.2f}pt;}}
        QLabel#helper{{color:#40586b;font-size:{helper_pt:.2f}pt;font-weight:500}}
        QLabel#status{{color:#40586b;font-size:{helper_pt:.2f}pt;font-weight:500}}
        QLabel#calibrationStatus{{color:#245d41;font-size:{calibration_pt:.2f}pt;font-weight:700}}
        QProgressBar{{background:#ffffff;border:1px solid #d5e0e7;border-radius:4px;text-align:center;min-height:21px;font-size:9.8pt}}
        QProgressBar::chunk{{background:#8ddfd4}}
        QMenuBar{{background:#ffffff;border-bottom:1px solid #d7e1e8;padding:2px 4px;font-family:'Segoe UI';font-size:{menu_pt:.2f}pt;font-weight:600}}
        QMenuBar::item{{background:transparent;color:#18222a;font-family:'Segoe UI';font-size:{menu_pt:.2f}pt;font-weight:600;padding:6px 13px}}
        QMenuBar::item:selected{{background:#edf3f7;color:#18222a;font-size:{menu_pt:.2f}pt;font-weight:600}}
        QMenu{{background:#ffffff;color:#18222a;border:1px solid #d5dfe6;font-family:'Segoe UI';font-size:{popup_pt:.2f}pt}}
        QMenu::item{{font-family:'Segoe UI';font-size:{popup_pt:.2f}pt;padding:5px 18px 5px 11px}}
        QMenu::item:selected{{background:#edf3f7;font-size:{popup_pt:.2f}pt}}
        """)

        # Apply menu rules directly too; this avoids platform-native static text
        # ignoring the application's general widget font.
        mb.setStyleSheet(
            f"QMenuBar{{background:#ffffff;border-bottom:1px solid #d7e1e8;padding:2px 4px;font-family:'Segoe UI';font-size:{menu_pt:.2f}pt;font-weight:600;}}"
            f"QMenuBar::item{{background:transparent;color:#18222a;font-family:'Segoe UI';font-size:{menu_pt:.2f}pt;font-weight:600;padding:6px 13px;}}"
            f"QMenuBar::item:selected{{background:#edf3f7;color:#18222a;font-size:{menu_pt:.2f}pt;font-weight:600;}}"
        )
        for menu in mb.findChildren(QMenu):
            menu.setStyleSheet(
                f"QMenu{{background:#ffffff;color:#18222a;border:1px solid #d5dfe6;font-family:'Segoe UI';font-size:{popup_pt:.2f}pt;}}"
                f"QMenu::item{{font-family:'Segoe UI';font-size:{popup_pt:.2f}pt;padding:5px 18px 5px 11px;}}"
                f"QMenu::item:selected{{background:#edf3f7;font-size:{popup_pt:.2f}pt;}}"
            )
        QTimer.singleShot(0, self._sync_top_panel_heights)


    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            state = self.windowState()
            if not (state & Qt.WindowState.WindowMaximized) and not (state & Qt.WindowState.WindowMinimized):
                # Returning from maximized/minimized state restores the wide,
                # centered medium dashboard requested for normal-window mode.
                QTimer.singleShot(0, self._restore_medium_geometry)
            if hasattr(self, "_resize_timer"):
                self._resize_timer.start(80)
            QTimer.singleShot(0, self._style)


    def _sync_brand_logo(self):
        if not all(hasattr(self, name) for name in ("brand_logo", "logo_anchor", "toolbar_envelope")):
            return
        center = self.logo_anchor.mapTo(
            self.toolbar_envelope,
            self.logo_anchor.rect().center(),
        )
        self.brand_logo.set_anchor_center(center)


    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_resize_timer"):
            self._resize_timer.start(140)


    def _after_resize(self):
        # Keep resize handling geometric only. Matplotlib canvases resize their
        # existing figures; repeatedly rebuilding every plot on each window
        # resize caused axis labels and plot margins to visibly drift.
        self._apply_panel_proportions()
        self._sync_brand_logo()
        if hasattr(self, "site_period_reminder") and self.site_period_reminder.isVisible():
            self.site_period_reminder._recenter()


    @staticmethod
    def _zone_display(zone):
        return {"I": "A", "II": "B", "III": "C"}.get(str(zone), "")


    def _nearest_station_from_event(self, event, max_pixels=10.0):
        if event.inaxes is None or self.stations is None or not len(self.stations):
            return None
        try:
            pts = np.column_stack([self.stations.longitude.to_numpy(float), self.stations.latitude.to_numpy(float)])
            disp = event.inaxes.transData.transform(pts)
            d = np.hypot(disp[:, 0] - float(event.x), disp[:, 1] - float(event.y))
            k = int(np.nanargmin(d))
            return self.stations.iloc[k] if float(d[k]) <= float(max_pixels) else None
        except Exception:
            return None


    def _map_click(self, event):
        if not event.inaxes or event.xdata is None or event.ydata is None:
            return
        station = self._nearest_station_from_event(event)
        if station is not None:
            self.site_lon.setValue(float(station.longitude))
            self.site_lat.setValue(float(station.latitude))
            if "Ts_current_s" in station.index and np.isfinite(float(station["Ts_current_s"])):
                self.ts.setValue(round(float(station["Ts_current_s"]), 2))
            zone = zone_at_point(ASSETS, self.site_lon.value(), self.site_lat.value())
            zd = self._zone_display(zone)
            sid = str(station.get("station_id", "station"))
            self.selected_station_id = sid
            ev = str(station.get("Ts_current_event_id", "")).strip()
            suffix = f" · current {ev}" if ev and ev.lower() != "nan" else ""
            self.site_zone_info.setText(f"Station {sid} · Zone {zd or 'outside map'} · Ts = {self.ts.value():.2f} s{suffix}.")
            self._refresh_map()
            self._maybe_show_site_period_reminder()
            return
        self.selected_station_id = ""
        self.site_lon.setValue(float(event.xdata))
        self.site_lat.setValue(float(event.ydata))
        self._site_coordinate_changed(show_reminder=True)


    def _manual_site_coordinate_edited(self):
        self.selected_station_id = ""
        self._maybe_show_site_period_reminder()


    def _site_coordinates_changed_for_reminder(self):
        current = (float(self.site_lon.value()), float(self.site_lat.value()))
        previous = getattr(self, "_last_site_reminder_coords", None)
        changed = previous is None or (
            abs(current[0] - previous[0]) > 1e-10 or
            abs(current[1] - previous[1]) > 1e-10
        )
        if changed:
            self._last_site_reminder_coords = current
        return changed


    def _maybe_show_site_period_reminder(self):
        if self._site_coordinates_changed_for_reminder():
            self._show_site_period_reminder()


    def _show_site_period_reminder(self):
        if hasattr(self, "site_period_reminder"):
            self.site_period_reminder.show_for(5000)


    def _site_coordinate_changed(self, *_, show_reminder=False):
        zone = zone_at_point(ASSETS, self.site_lon.value(), self.site_lat.value())
        zd = self._zone_display(zone)
        if zd == "A":
            if abs(self.ts.value() - 0.5) > 1e-12:
                self.ts.setValue(0.5)
            self.site_zone_info.setText("Zone A · Ts set to 0.50 s for free-map selection.")
        elif zd:
            self.site_zone_info.setText(f"Zone {zd} · Ts remains user-specified.")
        else:
            self.site_zone_info.setText("Outside the zoning polygons · Ts remains user-specified.")
        self._refresh_map()
        if show_reminder:
            self._maybe_show_site_period_reminder()


    def _refresh_map(self, *_):
        plot_site_map(
            self.map_fig, ASSETS, self.stations,
            self.site_lon.value(), self.site_lat.value(),
        )
        self.map_canvas.draw_idle()


    def _toggle_rs_scale(self, checked):
        self.settings.rs_plot_linear = bool(checked)
        self.settings.save(SETTINGS_FILE)
        if self.payload is not None:
            self._rerender()

    def _update_rs_basis_button(self):
        if not hasattr(self, "rs_basis_toggle"):
            return
        rms = str(getattr(self.settings, "rs_percentile_basis", "components")) == "rms"
        self.rs_basis_toggle.blockSignals(True)
        self.rs_basis_toggle.setChecked(rms)
        self.rs_basis_toggle.blockSignals(False)
        self.rs_basis_toggle.setText("Bands: Horizontal RMS" if rms else "Bands: Components")
        self.rs_basis_toggle.setToolTip(
            "Show response-spectrum percentile bands for the horizontal RMS ensemble."
            if rms else
            "Show response-spectrum percentile bands separately for each component."
        )
        self.rs_basis_toggle.setEnabled(
            str(getattr(self.settings, "spectral_display_mode", "percentiles")) == "percentiles"
        )

    def _toggle_rs_percentile_basis(self, checked):
        self.settings.rs_percentile_basis = "rms" if checked else "components"
        self._update_rs_basis_button()
        self.settings.save(SETTINGS_FILE)
        if self.payload is not None:
            self._rerender()


    def _pick_changed(self, *_):
        self._update_realization_nav()
        self._rerender()


    def _previous_realization(self):
        if hasattr(self, "pick"):
            self.pick.setValue(max(self.pick.minimum(), self.pick.value() - 1))


    def _next_realization(self):
        if hasattr(self, "pick"):
            self.pick.setValue(min(self.pick.maximum(), self.pick.value() + 1))


    def _update_realization_nav(self):
        if not hasattr(self, "pick"):
            return
        if hasattr(self, "prev_pick"):
            self.prev_pick.setEnabled(self.pick.value() > self.pick.minimum())
        if hasattr(self, "next_pick"):
            self.next_pick.setEnabled(self.pick.value() < self.pick.maximum())


    def _update_theta_controls(self, *_):
        mode = self.theta_mode.currentData() if hasattr(self, "theta_mode") else "manual"
        is_auto = mode == "automatic"
        is_manual = mode == "manual"
        if hasattr(self, "ev_lon"):
            self.ev_lon.setEnabled(is_auto)
            self.ev_lat.setEnabled(is_auto)
            self.theta.setEnabled(is_manual)
        if hasattr(self, "theta_info"):
            self._refresh_resolved_theta()


    def _refresh_resolved_theta(self, *_):
        """Refresh the displayed path angle immediately from the active controls."""
        if not all(hasattr(self, name) for name in ("theta_mode", "theta_info", "source", "rrup")):
            return
        from core.path_angle import resolve_source_path_theta
        mode = str(self.theta_mode.currentData() or "manual")
        scenario = {
            "source_type": self.source.currentData(),
            "Rrup_km": float(self.rrup.value()),
        }
        manual = None
        if mode == "manual":
            manual = float(self.theta.value())
        elif mode == "automatic":
            scenario["event_longitude"] = float(self.ev_lon.value())
            scenario["event_latitude"] = float(self.ev_lat.value())
        try:
            theta, _ = resolve_source_path_theta(scenario, mode=mode, manual=manual, variant="full")
            mode_label = self.theta_mode.currentText().split(" · ")[0]
            self.theta_info.setText(f"Resolved θ = {theta:.1f}° · {mode_label}")
        except Exception as exc:
            self.theta_info.setText(f"Resolved θ unavailable · {exc}")


    def _update_event_seed_control(self):
        shared = str(getattr(self.settings, "fas_event_mode", "independent_realization")) == "shared_batch"
        if hasattr(self, "event_seed"):
            self.event_seed.setEnabled(shared)
        if hasattr(self, "shared_event_seed_label"):
            self.shared_event_seed_label.setEnabled(shared)


    def _rotation_selection_changed(self, *_):
        # Any change after an applied rotation requires the user to apply it again
        # before the rotated ensemble can be saved.
        if getattr(self, "rotation_applied", False):
            self.rotation_applied = False
            if hasattr(self, "save_rotated"):
                self.save_rotated.setEnabled(False)
        self._update_rotation_controls()


    def _update_rotation_controls(self, *_):
        if not hasattr(self, "rotation_mode"):
            return
        manual = self.rotation_mode.currentData() == "manual"
        self.rotation_angle.setEnabled(manual and self.rotation_box.isEnabled())
        if not self.rotation_box.isEnabled():
            self.rotation_summary.setText("Available after generation.")
            return
        mode = self.rotation_mode.currentData()
        if mode == "random":
            self.rotation_summary.setText("Independent full-circle horizontal orientation for each realization.")
        elif mode == "manual":
            self.rotation_summary.setText(f"Current fixed angle: {self.rotation_angle.value():.1f}°.")
        else:
            self.rotation_summary.setText("Native Major/Intermediate principal axes.")


    def _values(self):
        mode = self.theta_mode.currentData()
        return {
            "source_type": self.source.currentData(),
            "Mw": self.mw.value(),
            "Rrup_km": self.rrup.value(),
            "depth_used_km": self.depth.value(),
            "Ts_s": self.ts.value(),
            "site_id": self.site_id.text().strip() or "Site_001",
            "station_id": self.selected_station_id,
            "site_longitude": self.site_lon.value(),
            "site_latitude": self.site_lat.value(),
            "site_zone": zone_at_point(ASSETS, self.site_lon.value(), self.site_lat.value()) or "unresolved",
            "event_longitude": self.ev_lon.value() if mode == "automatic" else None,
            "event_latitude": self.ev_lat.value() if mode == "automatic" else None,
            "theta_mode": mode,
            "theta_manual": self.theta.value() if mode == "manual" else None,
            "n_realizations": self.n.value(),
            "seed": self.seed.value(),
            "output_components": int(self.settings.output_components),
            "model_variant": "full",
            "output_dt_s": self.settings.output_dt_s,
            "pad_start_s": self.settings.pad_start_s,
            "pad_end_s": self.settings.pad_end_s,
            "apply_taper": self.settings.apply_taper,
            "apply_filter": self.settings.apply_filter,
            "filter_low_hz": self.settings.filter_low_hz,
            "filter_high_hz": self.settings.filter_high_hz,
            "filter_order": self.settings.filter_order,
            "baseline_correction_mode": self.settings.baseline_correction_mode,
            "fas_options": {
                "fas_event_mode": self.settings.fas_event_mode,
                "fas_event_seed": (self.event_seed.value() if self.settings.fas_event_mode == "shared_batch" else None),
            },
        }


    def _connect_domain_status(self):
        """Keep the calibration-domain indicator synchronized with the inputs."""
        for widget in (self.mw, self.rrup, self.depth, self.ts, self.site_lon, self.site_lat, self.ev_lon, self.ev_lat, self.theta):
            widget.valueChanged.connect(self._refresh_calibration_status)
        self.source.currentIndexChanged.connect(self._refresh_calibration_status)
        self.theta_mode.currentIndexChanged.connect(self._refresh_calibration_status)
        for widget in (self.rrup, self.ev_lon, self.ev_lat, self.theta):
            widget.valueChanged.connect(self._refresh_resolved_theta)
        self.source.currentIndexChanged.connect(self._refresh_resolved_theta)


    def _refresh_calibration_status(self, *_):
        if not hasattr(self, "calibration_status"):
            return None
        result = evaluate_calibration_domain(self._values())
        self.calibration_domain = result
        if result["in_support"]:
            self.calibration_status.setText("Calibration domain: IN SUPPORT")
            self.calibration_status.setStyleSheet("color:#245d41;font-weight:700;")
        else:
            summary = "; ".join(result["violations"][:2])
            extra = len(result["violations"]) - 2
            if extra > 0:
                summary += f"; +{extra} more"
            self.calibration_status.setText(f"Calibration domain: EXTRAPOLATION — interpret with caution · {summary}")
            self.calibration_status.setStyleSheet("color:#a34221;font-weight:700;")
        return result


    def _generate(self):
        if self.worker and self.worker.isRunning():
            return
        values = self._values()
        domain = evaluate_calibration_domain(values)
        values["calibration_domain"] = domain
        self.calibration_domain = domain
        self._refresh_calibration_status()
        if not domain["in_support"]:
            details = "\n".join(f"• {item}" for item in domain["violations"])
            answer = QMessageBox.warning(
                self,
                "Calibration-domain extrapolation",
                "This scenario is outside the VALLIS-3C calibration domain.\n\n"
                f"{details}\n\nResults require cautious interpretation. Accept extrapolation and continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                self.status.setText("Generation cancelled · extrapolation not accepted")
            return
        self.generate.setEnabled(False)
        self.rotation_box.setEnabled(False)
        self.save_rotated.setEnabled(False)
        self.rotation_applied = False
        self.last_run_folder = None
        self.rotated_saved_folder = None
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText("Generating…")
        self.worker = GenerationWorker(values, self.settings, self)
        self.worker.progressed.connect(self._progress)
        self.worker.completed.connect(self._done)
        self.worker.failed.connect(self._failed)
        self.worker.start()


    def _progress(self, done, total):
        if total > 0:
            pct = int(round(100 * done / total))
            self.progress.setValue(min(99, pct))
            self.status.setText("Computing FAS and response spectra…" if pct >= 83 else f"Generating · {pct}%")


    def _done(self, payload, scenario, spatial, fas, rs):
        try:
            self.payload = dict(payload)
            self.raw_accelerations = np.asarray(payload["accelerations"], float).copy()
            self.scenario = dict(scenario)
            self.spatial = dict(spatial)
            self.pick.setRange(1, len(self.raw_accelerations))
            self.pick.setValue(1)
            self._update_realization_nav()


            self.payload["accelerations"] = self.raw_accelerations.copy()
            names = tuple(payload.get("component_names") or ())
            if len(names) != self.raw_accelerations.shape[1]:
                names = ("Major", "Intermediate") if self.raw_accelerations.shape[1] == 2 else ("Major", "Intermediate", "Vertical")
            self.labels = names
            self.rotation_angles = np.zeros(len(self.raw_accelerations), dtype=float)
            self.rotation_mode.setCurrentIndex(0)
            self.rotation_angle.setValue(0.0)
            self.rotation_box.setEnabled(True)
            self._update_rotation_controls()
            self.fas, self.rs = fas, rs
            self._rerender()


            theta = float(scenario.get("source_path_theta_deg", np.nan))
            if np.isfinite(theta):
                self.theta_info.setText(f"Resolved θ = {theta:.1f}° · {self.theta_mode.currentText().split(' · ')[0]}")
            self.progress.setValue(99)
            self.status.setText("Finalizing display…")
            # Keep Generate disabled until queued canvas paints have had a chance to complete.
            QTimer.singleShot(100, self._mark_generation_ready)
        except Exception as exc:
            self._failed(f"Post-processing: {exc}")
            return


    def _mark_generation_ready(self):
        self.generate.setEnabled(True)
        self.progress.setValue(100)
        nreal = len(self.raw_accelerations)
        self.status.setText(f"Ready · {nreal} realizations")
        self._export_current(silent=True)
        QMessageBox.information(
            self,
            "Generation completed",
            f"Motion ensemble generated successfully.\n{nreal} realizations are ready.",
        )


    def _recompute_response(self, context="update"):
        if self.payload is None:
            return
        if self.analysis_worker is not None and self.analysis_worker.isRunning():
            return
        acc = np.asarray(self.payload["accelerations"], float).copy()
        self.analysis_worker = SpectraWorker(
            acc, float(self.payload["dt_s"]), self.settings, self.payload.get("valid_npts"), context, self
        )
        self.analysis_worker.progressed.connect(self._spectra_progress)
        self.analysis_worker.completed.connect(self._spectra_done)
        self.analysis_worker.failed.connect(self._spectra_failed)
        self.apply_rotation.setEnabled(False)
        self.progress.setValue(0)
        self.status.setText("Computing FAS and response spectra…")
        self.analysis_worker.start()


    def _spectra_progress(self, done, total):
        if total > 0:
            self.progress.setValue(min(99, int(round(100.0 * done / total))))


    def _spectra_done(self, fas, rs, context):
        self.fas, self.rs = fas, rs
        self._rerender()
        self.apply_rotation.setEnabled(self.rotation_box.isEnabled())
        self.progress.setValue(99)
        message = (
            "Horizontal rotation applied · FAS and response spectra recomputed"
            if context == "rotation"
            else "Configuration applied · FAS and response spectra recomputed"
        )
        QTimer.singleShot(80, lambda m=message, c=context: self._finish_secondary_update(m, c))


    def _finish_secondary_update(self, message, context="update"):
        self.progress.setValue(100)
        self.status.setText(message)
        if context == "rotation":
            self.rotation_applied = True
            can_save = self.last_run_folder is not None and not (self.export_worker and self.export_worker.isRunning())
            self.save_rotated.setEnabled(bool(can_save))
            QMessageBox.information(
                self,
                "Rotation completed",
                "Horizontal rotation was applied successfully.\n"
                "FAS and response spectra were recomputed for the rotated motions.",
            )


    def _spectra_failed(self, message, context):
        self.apply_rotation.setEnabled(self.rotation_box.isEnabled())
        self.status.setText("Error")
        QMessageBox.warning(self, "Response calculation", message)


    def _apply_rotation(self):
        if self.payload is None or self.raw_accelerations is None:
            return
        self.rotation_applied = False
        self.save_rotated.setEnabled(False)
        mode = self.rotation_mode.currentData()
        angle = self.rotation_angle.value()
        acc, angles, labels = rotate_horizontals(
            self.raw_accelerations, mode, angle, self.seed.value(),
        )
        self.payload["accelerations"] = acc
        self.labels = labels
        self.rotation_angles = angles
        self.settings.rotation_mode = mode
        self.settings.rotation_angle_deg = angle
        self.settings.save(SETTINGS_FILE)
        if mode == "random":
            self.rotation_summary.setText("Rotation applied · independent full-circle orientation per realization.")
        elif mode == "manual":
            self.rotation_summary.setText(f"Rotation applied · fixed angle {angle:.1f}° (periodic modulo 360°).")
        else:
            self.rotation_summary.setText("Native Major/Intermediate axes restored.")
        self._recompute_response(context="rotation")


    def _save_rotated_motions(self):
        if self.payload is None or not self.rotation_applied:
            QMessageBox.information(
                self, "Save Rotated Motions",
                "Apply a horizontal rotation before saving the rotated motions.",
            )
            return
        if self.last_run_folder is None:
            QMessageBox.information(
                self, "Save Rotated Motions",
                "The original generated ensemble is still being saved. Please try again in a moment.",
            )
            return
        if self.export_worker is not None and self.export_worker.isRunning():
            QMessageBox.information(
                self, "Save Rotated Motions",
                "A save operation is already running in the background.",
            )
            return
        try:
            rotated_dir = Path(self.last_run_folder) / "Rotated Motions"
            self.save_rotated.setEnabled(False)
            self.status.setText("Saving rotated motions…")
            self.export_worker = ExportWorker(
                rotated_dir, dict(self.payload), self.fas, self.rs, self.settings,
                self.scenario, self.labels, self.rotation_angles,
                site_id=self.site_id.text().strip() or "Site_001", rotated=True,
                silent=False, parent=self,
            )
            self.export_worker.completed.connect(self._rotated_export_done)
            self.export_worker.failed.connect(self._rotated_export_failed)
            self.export_worker.start()
        except Exception as exc:
            self.save_rotated.setEnabled(True)
            QMessageBox.warning(self, "Save Rotated Motions", str(exc))


    def _rotated_export_done(self, run, _silent):
        self.rotated_saved_folder = Path(run)
        self.save_rotated.setEnabled(True)
        self.status.setText("Rotated motions saved")
        QMessageBox.information(
            self,
            "Save completed",
            f"Rotated motions and results were saved successfully to:\n{run}",
        )


    def _rotated_export_failed(self, message, _silent):
        self.save_rotated.setEnabled(True)
        self.status.setText("Error saving rotated motions")
        QMessageBox.warning(self, "Save Rotated Motions", message)


    def _failed(self, msg):
        self.generate.setEnabled(True)
        self.rotation_box.setEnabled(self.payload is not None)
        self.status.setText("Error")
        QMessageBox.critical(self, "Simulation", msg)


    def _rerender(self):
        if self.payload is None:
            return
        idx = self.pick.value() - 1
        plot_accelerograms(self.acc_fig, self.payload, idx, self.labels)
        self.acc_canvas.draw_idle()

        mode = str(getattr(self.settings, "spectral_display_mode", "percentiles"))
        show_p50 = bool(getattr(self.settings, "spectral_show_p50", True))
        show_p16_p84 = bool(getattr(self.settings, "spectral_show_p16_p84", True))
        show_p05_p95 = bool(getattr(self.settings, "spectral_show_p05_p95", True))
        overlay_selected = bool(getattr(self.settings, "spectral_overlay_selected", True))

        fas_floor = None
        if not self.settings.fas_smoothing:
            # Keep the FAS y-axis anchored to the full ensemble so changing the
            # realization navigator cannot move the spectral panel.
            fas_floor = fas_display_floor(self.fas, None, reference_hz=10.0)
        plot_family(
            self.fas_fig, self.fas, idx, self.labels, "Frequency (Hz)", "FAS (cm/s)",
            True, True, xlim=(float(self.settings.fas_fmin_hz), float(self.settings.fas_fmax_hz)),
            ymin=fas_floor, selected_lw_scale=0.85, display_mode=mode,
            show_p50=show_p50, show_p16_p84=show_p16_p84, show_p05_p95=show_p05_p95,
            overlay_selected=False, horizontal_rms=False, p50_lw_scale=0.78,
        )
        self.fas_canvas.draw_idle()

        linear = bool(self.rs_linear_plot.isChecked())
        rs_rms = (
            mode == "percentiles"
            and str(getattr(self.settings, "rs_percentile_basis", "components")) == "rms"
        )
        plot_family(
            self.rs_fig, self.rs, idx, self.labels, "Period T (s)", "Sa (cm/s²)",
            not linear, not linear,
            xlim=(0.0, 5.0) if linear else (float(self.settings.rs_tmin_s), float(self.settings.rs_tmax_s)),
            display_mode=mode, show_p50=show_p50, show_p16_p84=show_p16_p84,
            show_p05_p95=show_p05_p95, overlay_selected=overlay_selected,
            horizontal_rms=rs_rms, percentile_component_count=None,
        )
        self.rs_canvas.draw_idle()


    @staticmethod
    def _set_combo_data(combo, value):
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return True
        return False


    def _project_state(self):
        return {
            "project_format": "VALLIS-3C",
            "schema": 3,
            "scenario": {
                "source_type": self.source.currentData(),
                "Mw": self.mw.value(),
                "Rrup_km": self.rrup.value(),
                "depth_km": self.depth.value(),
                "path_angle_mode": self.theta_mode.currentData(),
                "source_longitude": self.ev_lon.value(),
                "source_latitude": self.ev_lat.value(),
                "manual_theta_deg": self.theta.value(),
            },
            "site": {
                "site_id": self.site_id.text().strip() or "Site_001",
                "station_id": self.selected_station_id,
                "longitude": self.site_lon.value(),
                "latitude": self.site_lat.value(),
                "Ts_s": self.ts.value(),
            },
            "generation": {
                "realizations": self.n.value(),
                "seed": self.seed.value(),
                "shared_event_seed": self.event_seed.value(),
            },
            "horizontal_rotation": {
                "mode": self.rotation_mode.currentData(),
                "angle_deg": self.rotation_angle.value(),
            },
            "response_spectrum_view": "linear" if self.rs_linear_plot.isChecked() else "log",
            "configuration": asdict(self.settings),
        }


    def _save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", str(ROOT / "project.json"), "Project JSON (*.json)"
        )
        if not path:
            return
        p = Path(path)
        if p.suffix.lower() != ".json":
            p = p.with_suffix(".json")
        try:
            p.write_text(json.dumps(self._project_state(), indent=2, ensure_ascii=False), encoding="utf-8")
            self.status.setText(f"Project saved · {p.name}")
        except Exception as exc:
            QMessageBox.warning(self, "Save project", str(exc))


    def _load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load project", str(ROOT), "Project JSON (*.json)"
        )
        if not path:
            return
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            if not isinstance(raw, dict):
                raise ValueError("Invalid project file.")


            sc = dict(raw.get("scenario") or {})
            self._set_combo_data(self.source, sc.get("source_type", self.source.currentData()))
            if "Mw" in sc: self.mw.setValue(float(sc["Mw"]))
            if "Rrup_km" in sc: self.rrup.setValue(float(sc["Rrup_km"]))
            if "depth_km" in sc: self.depth.setValue(float(sc["depth_km"]))
            self._set_combo_data(self.theta_mode, sc.get("path_angle_mode", self.theta_mode.currentData()))
            if "source_longitude" in sc: self.ev_lon.setValue(float(sc["source_longitude"]))
            if "source_latitude" in sc: self.ev_lat.setValue(float(sc["source_latitude"]))
            if "manual_theta_deg" in sc: self.theta.setValue(float(sc["manual_theta_deg"]))


            site = dict(raw.get("site") or {})
            self.site_id.setText(str(site.get("site_id", "Site_001")))
            self.selected_station_id = str(site.get("station_id", "")).strip()
            if "longitude" in site: self.site_lon.setValue(float(site["longitude"]))
            if "latitude" in site: self.site_lat.setValue(float(site["latitude"]))
            if "Ts_s" in site: self.ts.setValue(float(site["Ts_s"]))


            gen = dict(raw.get("generation") or {})
            if "realizations" in gen: self.n.setValue(int(gen["realizations"]))
            if "seed" in gen: self.seed.setValue(int(gen["seed"]))
            if "shared_event_seed" in gen: self.event_seed.setValue(int(gen["shared_event_seed"]))


            rot = dict(raw.get("horizontal_rotation") or {})
            self._set_combo_data(self.rotation_mode, rot.get("mode", self.rotation_mode.currentData()))
            if "angle_deg" in rot: self.rotation_angle.setValue(float(rot["angle_deg"]))


            cfg = dict(raw.get("configuration") or {})
            # Normalize persisted export settings to the documented text format.
            if "acceleration_format" not in cfg and "accelerogram_formats" in cfg:
                old_formats = [{"txt_3c":"accel", "accel_3c":"accel", "txt_separate":"accel_separate"}.get(str(x), str(x)) for x in (cfg.get("accelerogram_formats") or [])]
                cfg["acceleration_format"] = next((x for x in old_formats if x in {"accel", "accel_separate", "npz"}), "accel")
            if "fas_format" not in cfg and "fas_formats" in cfg:
                cfg["fas_format"] = next((str(x) for x in (cfg.get("fas_formats") or []) if str(x) in {"csv", "npz"}), "csv")
            if "response_format" not in cfg and "response_formats" in cfg:
                cfg["response_format"] = next((str(x) for x in (cfg.get("response_formats") or []) if str(x) in {"csv", "npz"}), "csv")
            for key, value in cfg.items():
                if key in AppSettings.__dataclass_fields__:
                    setattr(self.settings, key, value)
            self.settings.output_components = 2 if int(getattr(self.settings, "output_components", 3)) == 2 else 3
            self.settings.fas_event_mode = self.settings.fas_event_mode if self.settings.fas_event_mode in {"record_only","shared_batch","independent_realization"} else "independent_realization"
            self.settings.spectral_display_mode = self.settings.spectral_display_mode if self.settings.spectral_display_mode in {"percentiles", "individual"} else "percentiles"
            self.settings.rs_percentile_basis = self.settings.rs_percentile_basis if self.settings.rs_percentile_basis in {"components", "rms"} else "components"
            self.settings.apply_taper = True
            self.settings.baseline_correction_mode = 0
            self.settings.displacement_stabilization = True
            if not (0.001 <= float(self.settings.output_dt_s) < 0.025):
                self.settings.output_dt_s = 0.01
            self.settings.rotation_mode = str(self.rotation_mode.currentData())
            self.settings.rotation_angle_deg = float(self.rotation_angle.value())
            view = str(raw.get("response_spectrum_view", "log")).lower()
            self.settings.rs_plot_linear = (view == "linear")
            self.settings.save(SETTINGS_FILE)


            self.rs_linear_plot.blockSignals(True)
            self.rs_log_plot.blockSignals(True)
            self.rs_linear_plot.setChecked(view == "linear")
            self.rs_log_plot.setChecked(view != "linear")
            self.rs_linear_plot.blockSignals(False)
            self.rs_log_plot.blockSignals(False)
            self._update_rs_basis_button()
            self._update_theta_controls()
            self._update_rotation_controls()
            self._update_event_seed_control()
            zone = zone_at_point(ASSETS, self.site_lon.value(), self.site_lat.value())
            zd = self._zone_display(zone)
            self.site_zone_info.setText(f"Zone {zd or 'outside map'} · Ts = {self.ts.value():.2f} s.")
            self._refresh_map()
            self._last_site_reminder_coords = (float(self.site_lon.value()), float(self.site_lat.value()))
            self.status.setText(f"Project loaded · {Path(path).name} · settings apply to the next generation")
        except Exception as exc:
            QMessageBox.warning(self, "Load project", str(exc))


    def _configure(self):
        dlg = SettingsDialog(self.settings, self)
        if not dlg.exec():
            return
        self.settings = dlg.apply()
        self.settings.rs_plot_linear = bool(self.rs_linear_plot.isChecked())
        self.settings.save(SETTINGS_FILE)
        self._update_event_seed_control()
        self._update_rs_basis_button()
        if self.payload is not None:
            self._rerender()
        self.status.setText("Configuration saved · display settings applied now; simulation settings apply to the next generation")


    def _choose_output(self):
        p = QFileDialog.getExistingDirectory(
            self, "Select output directory",
            self.settings.output_directory or str(ROOT / "outputs"),
        )
        if p:
            self.settings.output_directory = p
            self.settings.save(SETTINGS_FILE)


    def _next_run_folder(self, out: Path):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = out / f"run_{stamp}"
        if not candidate.exists():
            return candidate
        k = 1
        while True:
            c = out / f"run_{stamp}_{k:02d}"
            if not c.exists():
                return c
            k += 1


    def _export_current(self, checked=False, silent=False):
        if self.payload is None:
            if not silent:
                QMessageBox.information(self, "Export", "Generate a motion ensemble before exporting results.")
            return
        if self.export_worker is not None and self.export_worker.isRunning():
            if not silent:
                QMessageBox.information(self, "Export", "An export is already running in the background.")
            return
        try:
            out = Path(self.settings.output_directory)
            out = out if out.is_absolute() else ROOT / out
            run = self._next_run_folder(out)
            self.export_worker = ExportWorker(
                run, dict(self.payload), self.fas, self.rs, self.settings,
                self.scenario, self.labels, self.rotation_angles,
                site_id=self.site_id.text().strip() or "Site_001", rotated=False,
                silent=silent, parent=self,
            )
            self.export_worker.completed.connect(self._export_done)
            self.export_worker.failed.connect(self._export_failed)
            self.export_worker.start()
        except Exception as exc:
            if not silent:
                QMessageBox.warning(self, "Export", str(exc))


    def _export_done(self, run, silent):
        # The silent export launched at generation completion defines the base
        # run folder used later by "Rotated Motions". Manual exports do not
        # replace that reference once it exists.
        if silent or self.last_run_folder is None:
            self.last_run_folder = Path(run)
        if self.rotation_applied and self.last_run_folder is not None:
            self.save_rotated.setEnabled(True)
        if not silent:
            QMessageBox.information(self, "Export", f"Results saved to:\n{run}")


    def _export_failed(self, message, silent):
        if not silent:
            QMessageBox.warning(self, "Export", message)


    def _about(self):
        AboutDialog(self).exec()
