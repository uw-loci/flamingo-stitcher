"""Destripe filter settings dialog.

Exposes the wavelet-FFT stripe filter's tuning knobs (previously reachable only
by hand-editing ``stitching_config.yaml``). Values are returned as a plain dict
matching ``StitchingConfig.destripe_params``; the caller persists them in
QSettings so they survive between sessions.

The knobs, in rough order of how often you'd touch them:

* **Sigma foreground / background** — filter bandwidth in camera pixels; the
  main strength lever. A horizontal feature that runs straight for more than
  (frame size / sigma) times its own thickness is mostly removed, so larger
  sigma also removes shorter real edges. The two bands are filtered separately and blended across
  ``threshold``, so the bright sample and the dark background can be treated
  differently. Setting one to 0 disables filtering for that band.
* **Level** — wavelet decomposition depth. Higher reaches lower-frequency
  (broader) stripes. 0 = auto (maximum for the image size).
* **Wavelet** — mother wavelet. db2/db3 are the usual choices.
* **Threshold** — intensity splitting foreground from background. Auto uses
  Otsu per plane, which is the right default for most data.
* **Crossover** — intensity width (counts) of the blend between the two bands.
  The background band is clipped at ``threshold``, so every sample pixel takes
  a share of a flat image at the threshold: weight on its own value is
  sigmoid((value - threshold) / crossover). A crossover comparable to the
  tissue's brightness above background dims the tissue (run 32, 2026-10-27:
  crossover 900 took ~30% off the signal).
"""

from __future__ import annotations

from typing import Any, Dict

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from flamingo_stitcher.gui._compat import PersistentDialog

# Mirrors the YAML/built-in defaults in pipeline.destripe_volume.
DEFAULTS: Dict[str, Any] = {
    "sigma_foreground": 128.0,
    "sigma_background": 256.0,
    "level": 7,
    "wavelet": "db2",
    "threshold": None,  # None → Otsu (auto)
    "crossover": 10.0,
}

_WAVELETS = ["db1", "db2", "db3", "db4", "db5", "sym2", "sym3", "coif1", "haar"]


class _NoScrollDoubleSpin(QDoubleSpinBox):
    """Ignore wheel events unless focused (stops brushing-past nudges)."""

    def wheelEvent(self, event):  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _NoScrollSpin(QSpinBox):
    def wheelEvent(self, event):  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class _NoScrollCombo(QComboBox):
    def wheelEvent(self, event):  # noqa: N802
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class DestripeParamsForm(QWidget):
    """The destripe parameter widgets, without any dialog chrome.

    Extracted so the settings dialog and the live preview drive the SAME
    widgets: duplicating the form would let the two drift apart, and the
    preview is only trustworthy if it is tuning exactly what the run will use.
    Emits :attr:`changed` on every edit so a preview can re-render live.
    """

    changed = pyqtSignal()

    def __init__(self, params: Dict[str, Any] | None = None, parent=None):
        super().__init__(parent)

        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)

        self._sigma_fg = _NoScrollDoubleSpin()
        self._sigma_fg.setRange(0.0, 4096.0)
        self._sigma_fg.setDecimals(1)
        self._sigma_fg.setSuffix(" px")
        self._sigma_fg.setToolTip(
            "Which horizontal features on the sample count as stripes. Anything\n"
            "that runs straight across for more than (frame size in camera\n"
            "pixels / sigma) times its own thickness is mostly removed: on a\n"
            "2048 px frame, 128 removes features longer than 16x their\n"
            "thickness, 256 those longer than 8x. Raise it if short stripes\n"
            "remain; lower it if real horizontal edges (tissue borders, layers)\n"
            "lose contrast. Default 128; 0 leaves the sample unfiltered.\n"
            "Set on the full-resolution frame; with 'Fast' it is converted\n"
            "for the downsampled frame automatically."
        )
        form.addRow("Sigma — foreground:", self._sigma_fg)

        self._sigma_bg = _NoScrollDoubleSpin()
        self._sigma_bg.setRange(0.0, 4096.0)
        self._sigma_bg.setDecimals(1)
        self._sigma_bg.setSuffix(" px")
        self._sigma_bg.setToolTip(
            "The same as 'Sigma — foreground', for pixels below 'Threshold'.\n"
            "Usually equal or larger, since the background has no detail to\n"
            "lose. Default 256 (removes features longer than 8x their\n"
            "thickness on a 2048 px frame); 0 leaves the background unfiltered."
        )
        form.addRow("Sigma — background:", self._sigma_bg)

        self._level = _NoScrollSpin()
        self._level.setRange(0, 12)
        self._level.setSpecialValueText("Auto (max)")
        self._level.setSuffix(" levels")
        self._level.setToolTip(
            "How thick a stripe the filter reaches: each level doubles it, so\n"
            "7 reaches stripes up to about 128 camera pixels (2^7) thick.\n"
            "Higher also flattens broad horizontal bands of real brightness\n"
            "across the tissue; lower it if that happens. Default 7. 'Auto\n"
            "(max)' uses the deepest level the frame allows; a deeper value is\n"
            "reduced automatically. Set on the full-resolution frame; with\n"
            "'Fast' it is lowered by one per 2x of downsampling."
        )
        form.addRow("Wavelet level:", self._level)

        self._wavelet = _NoScrollCombo()
        self._wavelet.addItems(_WAVELETS)
        self._wavelet.setEditable(True)  # allow any pywt name
        self._wavelet.setToolTip(
            "The wavelet shape the filter uses. Leave it at db2 (default)\n"
            "unless you are experimenting; db3 is the usual alternative.\n"
            "You can also type another wavelet name."
        )
        form.addRow("Wavelet:", self._wavelet)

        self._threshold = _NoScrollDoubleSpin()
        self._threshold.setRange(-1.0, 65535.0)
        self._threshold.setDecimals(0)
        self._threshold.setSuffix(" counts")
        self._threshold.setSpecialValueText("Auto (Otsu)")
        self._threshold.setToolTip(
            "Camera brightness that separates sample from background. Pixels\n"
            "below it are filtered with 'Sigma — background', pixels above it\n"
            "with 'Sigma — foreground'. Set it just above the background:\n"
            "too high and dim tissue is filtered as background, losing more of\n"
            "its horizontal detail. 'Auto (Otsu)' (default) picks it for each\n"
            "plane, so a dim edge tile and a bright one can be filtered\n"
            "differently; type a value to use one level for every plane."
        )
        form.addRow("Threshold:", self._threshold)

        self._crossover = _NoScrollDoubleSpin()
        self._crossover.setRange(0.0, 65535.0)
        self._crossover.setDecimals(1)
        self._crossover.setSuffix(" counts")
        self._crossover.setToolTip(
            "How gradually the filter hands over from background to sample\n"
            "around 'Threshold'. The background version is capped at\n"
            "'Threshold', so every sample pixel is partly replaced by a flat\n"
            "value at 'Threshold': a pixel 1x 'Crossover' above it keeps 73%\n"
            "of its own value, 3x above keeps 95%. Keep it small next to the\n"
            "gap between background and tissue, or the tissue dims and loses\n"
            "contrast: with 'Threshold' 110 and 'Crossover' 900, tissue at\n"
            "400 counts comes out at about 280. Default 10. Raise it slightly\n"
            "only if a visible edge appears where sample meets background."
        )
        form.addRow("Crossover:", self._crossover)

        self.set_params(params or {})

        # Connect AFTER the initial population so set_params() doesn't fire a
        # storm of changed() during construction.
        for w in (self._sigma_fg, self._sigma_bg, self._threshold, self._crossover):
            w.valueChanged.connect(self.changed)
        self._level.valueChanged.connect(self.changed)
        self._wavelet.currentTextChanged.connect(self.changed)

    def set_params(self, params: Dict[str, Any]) -> None:
        """Populate the widgets; missing/None keys fall back to DEFAULTS."""

        def _v(key):
            val = params.get(key)
            return DEFAULTS[key] if val is None else val

        self._sigma_fg.setValue(float(_v("sigma_foreground")))
        self._sigma_bg.setValue(float(_v("sigma_background")))
        self._level.setValue(int(_v("level")))
        idx = self._wavelet.findText(str(_v("wavelet")))
        if idx >= 0:
            self._wavelet.setCurrentIndex(idx)
        else:
            self._wavelet.setEditText(str(_v("wavelet")))
        # threshold: None means Otsu, which the spin shows as its -1 minimum.
        thr = params.get("threshold")
        self._threshold.setValue(-1.0 if thr is None else float(thr))
        self._crossover.setValue(float(_v("crossover")))

    def get_params(self) -> Dict[str, Any]:
        """Current values as a ``StitchingConfig.destripe_params`` dict."""
        thr = self._threshold.value()
        return {
            "sigma_foreground": self._sigma_fg.value(),
            "sigma_background": self._sigma_bg.value(),
            "level": self._level.value(),
            "wavelet": self._wavelet.currentText().strip() or "db2",
            # -1 in the UI = Otsu; store None so the pipeline picks Otsu.
            "threshold": None if thr < 0 else thr,
            "crossover": self._crossover.value(),
        }


class DestripeSettingsDialog(PersistentDialog):
    """Small modal dialog for the destripe filter parameters."""

    def __init__(
        self,
        params: Dict[str, Any] | None = None,
        parent=None,
        source_note: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle("Destripe Settings")
        self.setModal(True)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Tuning for the stripe filter. Hover over a setting to see what it "
            "does to the data. Values are for the full-resolution camera frame."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#555; font-size:11px;")
        layout.addWidget(intro)
        if source_note:
            source = QLabel(source_note)
            source.setWordWrap(True)
            source.setStyleSheet("font-size:11px;")
            layout.addWidget(source)

        self._form = DestripeParamsForm(params)
        layout.addWidget(self._form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok
            | QDialogButtonBox.Cancel
            | QDialogButtonBox.RestoreDefaults
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(
            lambda: self._form.set_params(DEFAULTS)
        )
        layout.addWidget(buttons)

    # -- values ----------------------------------------------------------
    def set_params(self, params: Dict[str, Any]) -> None:
        self._form.set_params(params)

    def get_params(self) -> Dict[str, Any]:
        return self._form.get_params()
