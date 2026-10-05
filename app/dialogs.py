"""Configuration and technical-help dialogs for VALLIS-3C."""
from __future__ import annotations

from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QTextBlockFormat, QTextCursor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QScrollArea, QSpinBox, QTextBrowser, QToolButton, QVBoxLayout, QWidget,
)


ROOT = Path(__file__).resolve().parents[1]
LOGO_FILE = ROOT / "assets" / "icons" / "VALLIS-3C_512.png"


def _set_combo_data(combo: QComboBox, value):
    for i in range(combo.count()):
        if combo.itemData(i) == value:
            combo.setCurrentIndex(i)
            return


class SettingsDialog(QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration")
        self.resize(680, 760)
        self.s = settings
        outer = QVBoxLayout(self)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); root = QVBoxLayout(content)
        scroll.setWidget(content); outer.addWidget(scroll, 1)


        box = QGroupBox("Generation")
        f = QFormLayout(box)
        self.components = QComboBox()
        self.components.addItem("3 components", 3)
        self.components.addItem("2 components", 2)
        _set_combo_data(self.components, 2 if int(getattr(settings, "output_components", 3)) == 2 else 3)
        f.addRow("Components", self.components)
        root.addWidget(box)


        box = QGroupBox("Probabilistic event mode")
        f = QFormLayout(box)
        self.release_event = QComboBox()
        self.release_event.addItem("Predictive total · independent event per realization", "independent_realization")
        self.release_event.addItem("Common event · one event term for the ensemble", "shared_batch")
        self.release_event.addItem("Median event · within-event / record only", "record_only")
        _set_combo_data(self.release_event, settings.fas_event_mode)
        f.addRow("FAS event variability", self.release_event)
        root.addWidget(box)


        box = QGroupBox("Fourier amplitude spectra")
        f = QFormLayout(box)
        self.fmin = QDoubleSpinBox(); self.fmin.setRange(0.001, 100); self.fmin.setDecimals(3); self.fmin.setValue(settings.fas_fmin_hz)
        self.fmax = QDoubleSpinBox(); self.fmax.setRange(0.01, 100); self.fmax.setDecimals(2); self.fmax.setValue(settings.fas_fmax_hz)
        self.smooth = QCheckBox("Display/export smoothed FAS"); self.smooth.setChecked(settings.fas_smoothing)
        self.band = QDoubleSpinBox(); self.band.setRange(5, 100); self.band.setValue(settings.fas_smoothing_bandwidth)
        f.addRow("Minimum frequency (Hz)", self.fmin)
        f.addRow("Maximum frequency (Hz)", self.fmax)
        f.addRow(self.smooth)
        f.addRow("Konno–Ohmachi b", self.band)
        root.addWidget(box)

        box = QGroupBox("Response spectrum")
        f = QFormLayout(box)
        self.damp = QDoubleSpinBox(); self.damp.setRange(0.001, 0.5); self.damp.setSingleStep(0.01); self.damp.setDecimals(3); self.damp.setValue(settings.rs_damping)
        self.tmin = QDoubleSpinBox(); self.tmin.setRange(0.001, 100); self.tmin.setDecimals(3); self.tmin.setValue(settings.rs_tmin_s)
        self.tmax = QDoubleSpinBox(); self.tmax.setRange(0.01, 100); self.tmax.setValue(settings.rs_tmax_s)
        self.np = QSpinBox(); self.np.setRange(10, 5000); self.np.setValue(settings.rs_points)
        self.spacing = QComboBox(); self.spacing.addItem("Logarithmic", "log"); self.spacing.addItem("Linear", "linear")
        self.spacing.setCurrentIndex(0 if settings.rs_spacing == "log" else 1)
        for a, b in [
            ("Damping ratio ξ", self.damp), ("Minimum period (s)", self.tmin),
            ("Maximum period (s)", self.tmax), ("Number of periods", self.np),
            ("Period spacing", self.spacing),
        ]:
            f.addRow(a, b)
        root.addWidget(box)


        box = QGroupBox("FAS and response-spectrum visualization")
        f = QFormLayout(box)
        self.spectral_display_mode = QComboBox()
        self.spectral_display_mode.addItem("Percentile bands", "percentiles")
        self.spectral_display_mode.addItem("Individual realization", "individual")
        _set_combo_data(self.spectral_display_mode, getattr(settings, "spectral_display_mode", "percentiles"))
        self.show_p50 = QCheckBox("Show median (P50)")
        self.show_p50.setChecked(bool(getattr(settings, "spectral_show_p50", True)))
        self.show_p16_p84 = QCheckBox("Show P16–P84 band")
        self.show_p16_p84.setChecked(bool(getattr(settings, "spectral_show_p16_p84", True)))
        self.show_p05_p95 = QCheckBox("Show P05–P95 band")
        self.show_p05_p95.setChecked(bool(getattr(settings, "spectral_show_p05_p95", True)))
        self.overlay_selected = QCheckBox("Overlay selected realization on response-spectrum percentile bands")
        self.overlay_selected.setChecked(bool(getattr(settings, "spectral_overlay_selected", True)))
        f.addRow("Display mode", self.spectral_display_mode)
        f.addRow(self.show_p50)
        f.addRow(self.show_p16_p84)
        f.addRow(self.show_p05_p95)
        f.addRow(self.overlay_selected)
        root.addWidget(box)


        box = QGroupBox("Output conditioning")
        f = QFormLayout(box)
        self.output_dt = QDoubleSpinBox(); self.output_dt.setRange(0.001, 0.024); self.output_dt.setDecimals(3); self.output_dt.setSingleStep(0.001); self.output_dt.setValue(settings.output_dt_s)
        self.pad_start = QDoubleSpinBox(); self.pad_start.setRange(0.0, 120.0); self.pad_start.setDecimals(1); self.pad_start.setSingleStep(1.0); self.pad_start.setValue(settings.pad_start_s)
        self.pad_end = QDoubleSpinBox(); self.pad_end.setRange(0.0, 120.0); self.pad_end.setDecimals(1); self.pad_end.setSingleStep(1.0); self.pad_end.setValue(settings.pad_end_s)
        self.filter_on = QCheckBox("Apply zero-phase Butterworth bandpass"); self.filter_on.setChecked(settings.apply_filter)
        self.filter_low = QDoubleSpinBox(); self.filter_low.setRange(0.001, 50.0); self.filter_low.setDecimals(3); self.filter_low.setValue(settings.filter_low_hz)
        self.filter_high = QDoubleSpinBox(); self.filter_high.setRange(0.01, 100.0); self.filter_high.setDecimals(2); self.filter_high.setValue(settings.filter_high_hz)
        self.filter_order = QSpinBox(); self.filter_order.setRange(1, 10); self.filter_order.setValue(settings.filter_order)
        f.addRow("Output time step dt (s)", self.output_dt)
        f.addRow("Zero padding before active motion (s)", self.pad_start)
        f.addRow("Zero padding after active motion (s)", self.pad_end)
        f.addRow(self.filter_on)
        f.addRow("Bandpass low cutoff (Hz)", self.filter_low)
        f.addRow("Bandpass high cutoff (Hz)", self.filter_high)
        f.addRow("Butterworth order", self.filter_order)
        root.addWidget(box)


        box = QGroupBox("Export")
        f = QFormLayout(box)
        self.save_acc = QCheckBox("Save acceleration time histories"); self.save_acc.setChecked(settings.save_accelerograms)
        self.save_rs = QCheckBox("Save response spectra"); self.save_rs.setChecked(settings.save_response_spectra)
        self.save_fas = QCheckBox("Save FAS"); self.save_fas.setChecked(settings.save_fas)


        self.accfmt = QComboBox()
        self.accfmt.addItem("ACCEL", "accel")
        self.accfmt.addItem("Separate ACCEL", "accel_separate")
        self.accfmt.addItem("NPZ", "npz")
        _set_combo_data(self.accfmt, settings.acceleration_format)


        self.fasfmt = QComboBox()
        self.fasfmt.addItem("CSV", "csv")
        self.fasfmt.addItem("NPZ", "npz")
        _set_combo_data(self.fasfmt, settings.fas_format)


        self.rsfmt = QComboBox()
        self.rsfmt.addItem("CSV", "csv")
        self.rsfmt.addItem("NPZ", "npz")
        _set_combo_data(self.rsfmt, settings.response_format)


        self.out = QLineEdit(settings.output_directory)
        browse = QPushButton("Browse…"); browse.clicked.connect(self._browse)
        hb = QHBoxLayout(); hb.addWidget(self.out); hb.addWidget(browse)
        f.addRow(self.save_acc); f.addRow("Acceleration format", self.accfmt)
        f.addRow(self.save_fas); f.addRow("FAS format", self.fasfmt)
        f.addRow(self.save_rs); f.addRow("Response-spectrum format", self.rsfmt)
        f.addRow("Output directory", hb)
        root.addWidget(box)


        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._validate_accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)


    def _browse(self):
        p = QFileDialog.getExistingDirectory(self, "Select output directory", self.out.text() or ".")
        if p:
            self.out.setText(p)


    def _validate_accept(self):
        if self.fmin.value() >= self.fmax.value():
            QMessageBox.warning(self, "Configuration", "Minimum frequency must be smaller than maximum frequency.")
            return
        if self.tmin.value() >= self.tmax.value():
            QMessageBox.warning(self, "Configuration", "Minimum period must be smaller than maximum period.")
            return
        if self.filter_on.isChecked() and self.filter_low.value() >= self.filter_high.value():
            QMessageBox.warning(self, "Configuration", "Bandpass low cutoff must be smaller than the high cutoff.")
            return
        self.accept()


    def apply(self):
        s = self.s
        s.output_components = int(self.components.currentData())
        s.fas_fmin_hz = self.fmin.value(); s.fas_fmax_hz = self.fmax.value()
        s.fas_smoothing = self.smooth.isChecked(); s.fas_smoothing_bandwidth = self.band.value()
        s.rs_damping = self.damp.value(); s.rs_tmin_s = self.tmin.value(); s.rs_tmax_s = self.tmax.value()
        s.rs_points = self.np.value(); s.rs_spacing = self.spacing.currentData()
        s.fas_event_mode = str(self.release_event.currentData())
        s.spectral_display_mode = str(self.spectral_display_mode.currentData())
        s.spectral_show_p50 = self.show_p50.isChecked()
        s.spectral_show_p16_p84 = self.show_p16_p84.isChecked()
        s.spectral_show_p05_p95 = self.show_p05_p95.isChecked()
        s.spectral_overlay_selected = self.overlay_selected.isChecked()
        s.output_dt_s = self.output_dt.value()
        s.pad_start_s = self.pad_start.value(); s.pad_end_s = self.pad_end.value()
        s.apply_taper = True; s.apply_filter = self.filter_on.isChecked()
        s.filter_low_hz = self.filter_low.value(); s.filter_high_hz = self.filter_high.value(); s.filter_order = self.filter_order.value()
        s.baseline_correction_mode = 0; s.displacement_stabilization = True
        s.save_accelerograms = self.save_acc.isChecked(); s.save_response_spectra = self.save_rs.isChecked(); s.save_fas = self.save_fas.isChecked()
        s.acceleration_format = str(self.accfmt.currentData())
        s.fas_format = str(self.fasfmt.currentData())
        s.response_format = str(self.rsfmt.currentData())
        s.output_directory = self.out.text().strip() or "outputs"
        return s


class AboutDialog(QDialog):
    """Compact publication-facing identity and documentation dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About VALLIS-3C")
        self.resize(600, 420)
        self.setMinimumSize(540, 400)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 22)
        layout.setSpacing(16)

        identity = QHBoxLayout()
        identity.setSpacing(16)
        identity.addStretch(1)
        logo = QLabel()
        logo.setFixedSize(64, 64)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if LOGO_FILE.is_file():
            pixmap = QPixmap(str(LOGO_FILE))
            if not pixmap.isNull():
                logo.setPixmap(pixmap.scaled(
                    64, 64,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
        identity.addWidget(logo)
        title_column = QVBoxLayout()
        title_column.setSpacing(1)
        name = QLabel("VALLIS-3C")
        name.setStyleSheet("font-size:22pt;font-weight:750;color:#173f67;")
        version = QLabel("Version 1.6.0")
        version.setStyleSheet("font-size:11.5pt;font-weight:600;color:#526879;")
        title_column.addWidget(name)
        title_column.addWidget(version)
        identity.addLayout(title_column)
        identity.addStretch(1)
        layout.addLayout(identity)

        purpose = QLabel("Three-component ground-motion simulation at sites in the Basin of Mexico.")
        purpose.setAlignment(Qt.AlignmentFlag.AlignCenter)
        purpose.setWordWrap(True)
        purpose.setStyleSheet("font-size:13pt;font-weight:600;color:#20384b;")
        layout.addWidget(purpose)

        scope = QLabel(
            "<b>Scientific scope</b><br>"
            "<i>Scenario- and site-conditioned stochastic simulation of horizontal and vertical ground motions.</i>"
        )
        scope.setTextFormat(Qt.TextFormat.RichText)
        scope.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scope.setWordWrap(True)
        scope.setStyleSheet("font-size:11.5pt;color:#40586b;")
        layout.addWidget(scope)

        author = QLabel(
            "<b>Joel D. Cruz-Arguelles</b><br>"
            "Instituto de Ingeniería, Universidad Nacional Autónoma de México<br>"
            "<a href='mailto:JCruzAr@iingen.unam.mx' style='color:#174f7a;text-decoration:underline;'>JCruzAr@iingen.unam.mx</a> · "
            "<a href='mailto:joeldan.cruz@gmail.com' style='color:#174f7a;text-decoration:underline;'>joeldan.cruz@gmail.com</a><br><a href='https://orcid.org/0009-0002-6878-0173' style='color:#174f7a;text-decoration:underline;'>ORCID 0009-0002-6878-0173</a>"
        )
        author.setTextFormat(Qt.TextFormat.RichText)
        author.setOpenExternalLinks(True)
        author.setAlignment(Qt.AlignmentFlag.AlignCenter)
        author.setWordWrap(True)
        author.setStyleSheet("font-size:11.5pt;color:#263f52;")
        layout.addWidget(author)
        layout.addSpacing(4)

        license_line = QLabel("GPL-3.0-only · © 2026")
        license_line.setAlignment(Qt.AlignmentFlag.AlignCenter)
        license_line.setStyleSheet("font-size:10.5pt;color:#637786;")
        layout.addWidget(license_line)

        links = QHBoxLayout()
        links.setSpacing(10)
        links.addStretch(1)
        technical = QPushButton("Technical guide")
        provenance = QPushButton("Data provenance")
        license_button = QPushButton("License")
        for button in (technical, provenance, license_button):
            button.setObjectName("aboutLinkButton")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            links.addWidget(button)
        links.addStretch(1)
        layout.addLayout(links)

        use_note = QLabel("Designed for research and methodological evaluation.")
        use_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        use_note.setStyleSheet("font-size:9.5pt;font-style:italic;color:#718390;")
        layout.addWidget(use_note)

        technical.clicked.connect(lambda: HelpDialog(self).exec())
        provenance.clicked.connect(
            lambda: DocumentDialog("Data provenance", ROOT / "DATA_PROVENANCE.md", self).exec()
        )
        license_button.clicked.connect(
            lambda: DocumentDialog("GNU General Public License v3.0", ROOT / "LICENSE", self).exec()
        )

        self.setStyleSheet("""
            QDialog { background:#f7fafc; }
            QLabel { background:transparent; }
            QPushButton#aboutLinkButton {
                background:#edf4f8; color:#1f536d; border:1px solid #c8d8e2;
                border-radius:5px; padding:6px 13px; font-size:10.5pt; font-weight:600;
            }
            QPushButton#aboutLinkButton:hover { background:#deebf2; border-color:#9fbacb; }
        """)


class DocumentDialog(QDialog):
    """Read-only Help-menu viewer for release documents."""

    def __init__(self, title, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"VALLIS-3C · {title}")
        self.resize(1040, 760)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        heading = QLabel(title)
        heading.setStyleSheet("font-size:18pt;font-weight:700;color:#17324d;padding:4px 2px;")
        layout.addWidget(heading)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        browser.setStyleSheet("font-size:11pt;color:#263844;background:#ffffff;padding:8px;")
        text = Path(path).read_text(encoding="utf-8-sig")
        if str(path).lower().endswith(".md"):
            browser.setMarkdown(text)
        else:
            # License texts are conventionally hard-wrapped for distribution.
            # Reflow only whitespace for display; the legal wording is unchanged.
            paragraphs = []
            for block in text.replace("\r\n", "\n").split("\n\n"):
                lines = [line.strip() for line in block.splitlines() if line.strip()]
                if lines:
                    paragraphs.append(" ".join(lines))
            browser.setPlainText("\n\n".join(paragraphs))
            cursor = browser.textCursor()
            cursor.select(QTextCursor.SelectionType.Document)
            block_format = QTextBlockFormat()
            block_format.setAlignment(Qt.AlignmentFlag.AlignJustify)
            cursor.mergeBlockFormat(block_format)
            cursor.clearSelection()
            browser.setTextCursor(cursor)
            browser.moveCursor(QTextCursor.MoveOperation.Start)
        layout.addWidget(browser, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.clicked.connect(self.accept)
        layout.addWidget(buttons)


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VALLIS-3C · Technical guide")
        self.resize(860, 760)
        lay = QVBoxLayout(self)
        heading = QLabel("Technical guide")
        heading.setStyleSheet("font-size:18pt;font-weight:700;color:#17324d;padding:4px 2px;")
        lay.addWidget(heading)
        subtitle = QLabel("Select a section to expand it. Select the same section again to collapse it.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font-size:11pt;color:#40586b;padding:0 2px 8px 2px;")
        lay.addWidget(subtitle)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        guide = QWidget()
        guide_layout = QVBoxLayout(guide)
        guide_layout.setContentsMargins(4, 4, 8, 4)
        guide_layout.setSpacing(5)
        scroll.setWidget(guide)
        lay.addWidget(scroll, 1)


        sections = [
            (
                "Introduction",
                """
                <p><b>VALLIS-3C</b> generates families of three-component ground motions at sites in the Basin of Mexico.</p>
                <p><b>Typical workflow:</b> define the site and source scenario, review the calibration-domain indicator, choose the event-residual interpretation, generate the ensemble, inspect time histories/FAS/response spectra, and export the results with reproducibility metadata.</p>
                """
            ),
            (
                "Scientific model",
                """
                <p>VALLIS-3C separates spectral amplitude from temporal organization and recombines them during synthesis.</p>
                <ol>
                  <li>A source-specific conditional <b>effective amplitude spectrum (EAS)</b> is predicted at the CU reference site from magnitude, rupture distance, depth and path azimuth.</li>
                  <li>Site response adds a broad <b>Ts-conditioned amplification</b> and a persistent local spectral departure represented in normalized frequency <b>u = f Ts</b>.</li>
                  <li>Frequency-dependent <b>Major/Intermediate</b> and <b>Vertical/EAS</b> ratios reconstruct the three component target FAS; empirical residual terms generate realization-to-realization spectral variability.</li>
                  <li>A source-specific multioutput Random Forest predicts horizontal duration and frequency-dependent vertical-to-horizontal duration ratios.</li>
                  <li>An <b>ExtraTrees</b> selector identifies compatible three-component time-frequency textures using scenario/site descriptors and realization-specific Major, Intermediate and Vertical FAS.</li>
                  <li>Fresh stochastic oscillations are generated with the <b>phase-diffusion carrier</b>. The selected texture controls energy/persistence structure; recorded waveforms or phase trajectories are not reconstructed.</li>
                  <li>Two-pass complex transport assigns the ensemble duration distribution and vertical band persistence, followed by smooth FAS correction, zero-lag covariance correction and Major/Intermediate reassignment.</li>
                </ol>
                <p>The production interface uses the FULL model. OOS2017 and OOS2012_2017 use the same architecture with the validation earthquakes removed from all relevant fitted stages and eligible donor populations.</p>
                """
            ),
            (
                "Input variables",
                """
                <ul>
                  <li><b>Mw</b>: moment magnitude.</li>
                  <li><b>Rrup (km)</b>: closest distance from the site to the rupture surface.</li>
                  <li><b>Depth (km)</b>: Ztor, the depth to the top of the rupture.</li>
                  <li><b>Ts (s)</b>: dominant site period.</li>
                  <li><b>Longitude / Latitude</b>: site coordinates in WGS84.</li>
                  <li><b>Path-angle mode</b>: Manual, Automatic, Critical or Median.</li>
                  <li><b>Manual θ</b>: source-to-site path angle entered by the user when Manual mode is active.</li>
                  <li><b>Realizations</b>: number of motions to generate.</li>
                  <li><b>Components</b>: 3 outputs Major/Intermediate/Vertical, or 2 horizontal outputs Major/Intermediate.</li>
                  <li><b>Seed</b>: main random seed for realization-level variability and synthesis.</li>
                  <li><b>Shared-event seed</b>: enabled only in Common event mode; selects the single between-event FAS residual shared by the whole ensemble.</li>
                </ul>
                """
            ),
            (
                "Site zoning",
                """
                <p><b>NTCDS 2023, Section 1.3</b> classifies Mexico City seismic zones using the dominant site period:</p>
                <ul>
                  <li><b>Zone A:</b> Ts ≤ 0.5 s.</li>
                  <li><b>Zone B:</b> 0.5 &lt; Ts ≤ 1.0 s.</li>
                  <li><b>Zone C:</b> Ts &gt; 1.0 s.</li>
                </ul>
                <p>For a free map selection inside Zone A, this application assigns Ts = 0.50 s as a fixed map-selection convention.
                Clicking a station instead uses the current Ts for that station.</p>
                """
            ),
            (
                "Path angle",
                """
                <ul>
                  <li><b>Manual:</b> enter θ directly; source-coordinate fields are locked.</li>
                  <li><b>Automatic:</b> θ is calculated from CU and the source coordinates; Manual θ is locked.</li>
                  <li><b>Critical:</b> the critical path angle is used; source coordinates and Manual θ are locked.</li>
                  <li><b>Median:</b> the median path angle is used; source coordinates and Manual θ are locked.</li>
                </ul>
                """
            ),
            (
                "FAS event variability",
                """
                <p>VALLIS-3C exposes three probabilistic meanings for a generated ensemble:</p>
                <ul>
                  <li><b>Predictive total</b>: each realization draws an independent between-event residual. Use this for the full predictive distribution of future occurrences of the scenario.</li>
                  <li><b>Common event</b>: one between-event residual is shared by every realization in the ensemble. The <b>Shared-event seed</b> selects that common event term; the ordinary <b>Seed</b> continues to control record/local/synthesis variability.</li>
                  <li><b>Median event</b>: no between-event residual is added; only within-event/record and local variability remain.</li>
                </ul>
                """
            ),
            (
                "Components",
                """
                <p>The two horizontal motions use the principal-axis notation of Rezaeian and Der Kiureghian (2012). The horizontal pair is rotated to orthogonal principal axes before model development:</p>
                <ul>
                  <li><b>Major</b>: the horizontal principal component with the larger Arias intensity.</li>
                  <li><b>Intermediate</b>: the other horizontal principal component.</li>
                  <li><b>Vertical</b>: included when 3-component output is selected.</li>
                  <li>The same Major/Intermediate convention is used for the training database and for generated motions. It does not assume that either principal axis points toward the earthquake source.</li>
                  <li>After generation, the horizontal pair may remain in its native Major/Intermediate axes, be rotated by an independent random angle for each realization, or be rotated by a user-specified fixed angle. The vertical component is unchanged.</li>
                  <li>Acceleration histories and pseudo-spectral acceleration are reported in cm/s².</li>
                  <li>Fourier amplitude spectra are reported in cm/s.</li>
                </ul>
                <p><b>Reference:</b> Rezaeian, S., &amp; Der Kiureghian, A. (2012). Simulation of orthogonal horizontal ground motion components for specified earthquake and site characteristics. <i>Earthquake Engineering &amp; Structural Dynamics, 41</i>(2), 335–353. https://doi.org/10.1002/eqe.1132</p>
                """
            ),
            (
                "Response spectrum",
                """
                <p>Elastic pseudo-spectral acceleration is calculated by integrating each single-degree-of-freedom oscillator with the Generalized Single-Step Single-Solve (GSSSS) algorithm at the configured damping ratio.</p>
                <ul>
                  <li><b>Log–Log</b>: logarithmic axes.</li>
                  <li><b>Lin–Lin</b>: linear axes.</li>
                </ul>
                <p><b>Reference:</b> Zhou, X., &amp; Tamma, K. K. (2004). Design, analysis, and synthesis of generalized single step single solve and optimal algorithms for structural dynamics. <i>International Journal for Numerical Methods in Engineering, 59</i>(5), 597–668. https://doi.org/10.1002/nme.873</p>
                """
            ),
            (
                "Spectral visualization",
                """
                <ul>
                  <li><b>Percentile bands</b> is the default display for FAS and response spectra.</li>
                  <li><b>P50</b> is drawn as a continuous median curve; P16–P84 and P05–P95 are shown as progressively lighter uncertainty bands.</li>
                  <li>In FAS percentile mode, only the P50 component curves and their semi-transparent uncertainty bands are shown.</li>
                  <li><b>Overlay selected realization</b> applies only to response spectra; it keeps the percentile bands fixed while the navigator overlays the selected curves.</li>
                  <li>For response spectra, the plot toolbar can switch between separate horizontal-component bands and the horizontal RMS ensemble. Horizontal RMS uses a black median curve with gray percentile bands.</li>
                  <li><b>Individual realization</b> displays only the currently selected realization.</li>
                </ul>
                <p>When FAS smoothing is enabled, VALLIS-3C uses the logarithmic Konno–Ohmachi weighting function with the configured bandwidth coefficient.</p>
                <p><b>Reference:</b> Konno, K., &amp; Ohmachi, T. (1998). Ground-motion characteristics estimated from spectral ratio between horizontal and vertical components of microtremor. <i>Bulletin of the Seismological Society of America, 88</i>(1), 228–241. https://doi.org/10.1785/bssa0880010228</p>
                """
            ),
            (
                "Calibration domain",
                """
                <p><b>Before generating:</b> read the status shown in the Generation panel.</p>
                <ul>
                  <li><b>IN SUPPORT</b>: all checked scenario variables and the site location are inside the observed FULL calibration support.</li>
                  <li><b>EXTRAPOLATION</b>: one or more variables are outside support. The application lists the affected variables and requires explicit acceptance.</li>
                </ul>
                <p>Extrapolated results are allowed for controlled research, but must be interpreted and reported with caution.</p>
                """
            ),
            (
                "Output conditioning",
                """
                <ul>
                  <li>Output time step dt.</li>
                  <li>Configurable zero padding before and after each motion.</li>
                  <li>Optional zero-phase Butterworth bandpass.</li>
                  <li>Standard edge conditioning is applied automatically to minimize end effects and residual drift.</li>
                </ul>
                """
            ),
            (
                "Reproducibility metadata",
                """
                <p><b>After generation:</b> export the ensemble to preserve the complete run record.</p>
                <p>Every export records release/build/hash, model and dataset hashes, scenario/site/path-angle provenance, realization and seed semantics, signal conditioning, spectral settings, rotation, software environment, calibration-domain status, and UTC timestamp.</p>
                """
            ),
        ]


        for index, (title, html) in enumerate(sections):
            header = QToolButton()
            header.setText(title)
            header.setCheckable(True)
            header.setChecked(index == 0)
            header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            header.setArrowType(Qt.ArrowType.DownArrow if index == 0 else Qt.ArrowType.RightArrow)
            header.setStyleSheet(
                "QToolButton{border:0;border-bottom:1px solid #d7e1e8;border-radius:0;"
                "background:#ffffff;color:#17324d;text-align:left;padding:10px 8px;font-size:12pt;font-weight:650;}"
                "QToolButton:hover{background:#edf3f7;}"
            )
            guide_layout.addWidget(header)
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(30, 8, 14, 12)
            lbl = QLabel(html)
            lbl.setWordWrap(True)
            lbl.setTextFormat(Qt.TextFormat.RichText)
            lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            lbl.setStyleSheet("font-size:11pt;color:#263844;line-height:1.35;background:#ffffff;")
            v.addWidget(lbl)
            w.setVisible(index == 0)
            guide_layout.addWidget(w)

            def toggle(checked, section=w, button=header):
                section.setVisible(bool(checked))
                button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)

            header.toggled.connect(toggle)
        guide_layout.addStretch()


        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.clicked.connect(self.accept)
        lay.addWidget(buttons)
