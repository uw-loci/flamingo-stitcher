"""'Optional components' on the Options tab: parts installed once per computer.

Flat-field correction needs its own environment (downloaded once, about 2-4 GB);
deconvolution needs a GPU library this installer does not include; the
background-zeroing preview needs napari. Until 0.14.0, 'Set up flat-field…'
sat beside 'Run All' and 'Cancel'. The user asked for it to move here, with
every control that waits on an install pointing to this box (2026-11-05).

The box shows each part's status. Setting up flat-field runs here, with its
progress shown under the button, and emits ``components_changed`` when done so
the stitching tabs can un-gray 'Flat-field correction'.
"""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
)


class FlatFieldSetupThread(QThread):
    """Builds the pixi flat-field environment off the UI thread.

    Emits ``progress`` lines as pixi downloads Python + basicpy, then ``done``
    with (success, message). The work is ``preprocessing_env.build_env``.
    """

    progress = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def run(self):
        try:
            from flamingo_stitcher import preprocessing_env

            preprocessing_env.build_env(self.progress.emit)
            self.done.emit(True, "")
        except Exception as e:  # shown to the user by the caller
            self.done.emit(False, str(e))


def choose_install_location(parent: QWidget, log=None) -> bool:
    """Confirm or change where flat-field installs. True to go ahead."""
    from flamingo_stitcher import preprocessing_env

    current = preprocessing_env.install_root()
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Question)
    box.setWindowTitle("Flat-field install location")
    box.setText(
        f"The flat-field environment (~2–4 GB incl. package cache) will "
        f"install to:\n\n{current}\n\nInstall here, or choose another "
        f"drive/folder if space is limited?"
    )
    here = box.addButton("Install here", QMessageBox.AcceptRole)
    box.addButton("Choose folder…", QMessageBox.ActionRole)
    cancel = box.addButton("Cancel", QMessageBox.RejectRole)
    box.setDefaultButton(here)
    box.exec_()
    clicked = box.clickedButton()
    if clicked is cancel:
        return False
    if clicked is not here:  # "Choose folder…"
        start = str(current if current.exists() else current.parent)
        chosen = QFileDialog.getExistingDirectory(
            parent, "Choose flat-field install folder", start
        )
        if not chosen:
            return False
        target = Path(chosen) / "FlamingoFlatField"
        preprocessing_env.set_install_root(target)
        if log is not None:
            log(f"Flat-field will install to: {target}")
    return True


def flat_field_status() -> str:
    from flamingo_stitcher.flat_field import is_available

    return "Installed" if is_available() else "Not installed"


def deconvolution_status() -> str:
    try:
        from flamingo_stitcher import deconvolution

        return "Installed" if deconvolution.is_available() else "Not available in this build"
    except Exception:
        return "Not available in this build"


def napari_status() -> str:
    try:
        import importlib.util

        return "Installed" if importlib.util.find_spec("napari") else "Not available in this build"
    except Exception:
        return "Not available in this build"


class OptionalComponentsBox(QGroupBox):
    """Status of each optional part, and the flat-field set-up button."""

    components_changed = pyqtSignal()

    _NOTES = {
        "flat": "Needed for 'Flat-field correction' on the 'Image quality' tab. Downloads "
                "about 2-4 GB, once; you choose the folder. No GPU needed.",
        "deconv": "Needed for 'Deconvolution' on the 'Image quality' tab. It needs a GPU "
                  "library that this installer does not include; see 'For developers / "
                  "Python users' in the README.",
        "napari": "Needed for the live 'Preview…' of 'Background (lossy)'. Without it, "
                  "type each channel's threshold in by hand.",
    }

    def __init__(self, parent: QWidget = None):
        super().__init__("Optional components (installed once on this computer)", parent)
        self._thread = None
        g = QGridLayout(self)
        g.setHorizontalSpacing(14)
        g.setVerticalSpacing(2)
        self._status = {}
        rows = (("flat", "Flat-field correction"), ("deconv", "Deconvolution"),
                ("napari", "Background zeroing preview"))
        for r, (key, name) in enumerate(rows):
            lab = QLabel(name)
            lab.setStyleSheet("font-weight: bold;")
            g.addWidget(lab, 2 * r, 0)
            st = QLabel("")
            self._status[key] = st
            g.addWidget(st, 2 * r, 1)
            note = QLabel(self._NOTES[key])
            note.setWordWrap(True)
            note.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
            g.addWidget(note, 2 * r + 1, 0, 1, 4)
        self.setup_btn = QPushButton("Set up flat-field…")
        self.setup_btn.clicked.connect(self.setup_flat_field)
        g.addWidget(self.setup_btn, 0, 2)
        self._progress = QLabel("")
        self._progress.setStyleSheet("color: #555; font-size: 9pt;")
        self._progress.setWordWrap(True)
        g.addWidget(self._progress, 0, 3)
        g.setColumnStretch(3, 1)
        self.refresh()

    def refresh(self) -> None:
        values = {"flat": flat_field_status(), "deconv": deconvolution_status(),
                  "napari": napari_status()}
        for key, text in values.items():
            ok = text == "Installed"
            self._status[key].setText(text)
            self._status[key].setStyleSheet("color: %s;" % ("#245c2b" if ok else "#a85400"))
        from flamingo_stitcher import preprocessing_env

        built = preprocessing_env.is_built()
        if self._thread is None:
            self.setup_btn.setText("Reinstall flat-field…" if built else "Set up flat-field…")
            self.setup_btn.setToolTip(
                f"Flat-field is installed at:\n{preprocessing_env.env_dir()}\n\n"
                "Click only to repair it or move it to another drive." if built else
                "Installs what 'Flat-field correction' needs, once. Needs internet and "
                "about 2-4 GB of disk; you'll be asked where to install it. No GPU needed."
            )

    def showEvent(self, event):  # noqa: N802
        self.refresh()
        super().showEvent(event)

    def setup_flat_field(self) -> None:
        from flamingo_stitcher import preprocessing_env

        if preprocessing_env.is_built():
            reply = QMessageBox.question(
                self,
                "Flat-field environment exists",
                "The flat-field environment is already installed.\n\n"
                "Reinstall it (e.g. to move it to another drive or repair it)?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        if not choose_install_location(self, self._progress.setText):
            return
        self.setup_btn.setEnabled(False)
        self.setup_btn.setText("Setting up…")
        self._thread = FlatFieldSetupThread(self)
        self._thread.progress.connect(self._progress.setText)
        self._thread.done.connect(self._finished)
        self._thread.start()

    def _finished(self, success: bool, error: str) -> None:
        self._thread = None
        self.setup_btn.setEnabled(True)
        if success:
            from flamingo_stitcher.isolated_service import IsolatedPreprocessingService

            IsolatedPreprocessingService().clear_cache()
            self._progress.setText("Flat-field is ready.")
        else:
            self._progress.setText("Set-up failed.")
            QMessageBox.warning(
                self, "Flat-field setup failed",
                f"Could not build the flat-field environment:\n\n{error}",
            )
        self.refresh()
        self.components_changed.emit()
