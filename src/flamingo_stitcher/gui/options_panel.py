"""The Options tab: per-microscope, per-objective stitching tuning.

These are not preferences. They are statements about an instrument — how much
tile overlap this scope's samples give phase correlation to work with, how much
of a mosaic can be expected to register, how far this stage can plausibly be
wrong. Baked into the code they are wrong for the first rig that differs; kept
as global settings they get retuned for whichever scope was used last and then
silently misapply to the next one. So they live per scope and per objective, and
the values follow the data (see :mod:`flamingo_stitcher.scope_profiles`).

The panel deliberately does NOT read the microscope name from a running
acquisition: a user should be able to set up a new rig before its first stitch,
and read back what an existing one is configured to do without queueing
anything. Picking a scope by name is the whole interaction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from flamingo_stitcher import scope_profiles

logger = logging.getLogger(__name__)

_NEW_SCOPE = "<add a microscope…>"


class OptionsPanel(QWidget):
    """Edit and persist per-microscope/objective stitching options."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # Keyed by field; the widget type follows the tunable's `kind`, so read
        # and write go through _widget_value/_set_widget_value rather than
        # assuming a spin box.
        self._spins: Dict[str, QWidget] = {}
        self._loading = False
        self._last_measure_dir = ""
        self._build()
        self._reload_scopes()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def _build(self) -> None:
        outer = QVBoxLayout(self)

        intro = QLabel(
            "Registration tuning is per microscope and per objective. These "
            "values are applied automatically to any acquisition whose "
            '"Microscope name" matches, so a queue that mixes instruments gets '
            "each one's own settings rather than whichever was last edited."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        picker = QGroupBox("Microscope")
        picker_layout = QHBoxLayout()

        self._scope_combo = QComboBox()
        self._scope_combo.setMinimumWidth(220)
        self._scope_combo.currentIndexChanged.connect(self._on_selection_changed)
        picker_layout.addWidget(QLabel("Name:"))
        picker_layout.addWidget(self._scope_combo, 1)

        self._new_scope_edit = QLineEdit()
        self._new_scope_edit.setPlaceholderText(
            'exactly as "Microscope name" appears in ScopeSettings.txt'
        )
        self._new_scope_edit.setVisible(False)
        self._new_scope_edit.editingFinished.connect(self._on_selection_changed)
        picker_layout.addWidget(self._new_scope_edit, 2)

        self._objective_combo = QComboBox()
        self._objective_combo.setMinimumWidth(150)
        self._objective_combo.currentIndexChanged.connect(self._on_objective_changed)
        picker_layout.addWidget(QLabel("Objective:"))
        picker_layout.addWidget(self._objective_combo, 1)

        self._new_objective_edit = QDoubleSpinBox()
        self._new_objective_edit.setRange(0.1, 200.0)
        self._new_objective_edit.setDecimals(1)
        self._new_objective_edit.setSuffix("x")
        self._new_objective_edit.setValue(17.0)
        self._new_objective_edit.setVisible(False)
        self._new_objective_edit.valueChanged.connect(self._on_objective_changed)
        picker_layout.addWidget(self._new_objective_edit)

        picker.setLayout(picker_layout)
        outer.addWidget(picker)

        self._name_hint = QLabel()
        self._name_hint.setWordWrap(True)
        self._name_hint.setStyleSheet("color: palette(mid);")
        outer.addWidget(self._name_hint)

        # The settings themselves, in a scroll area: the help text under each
        # control is the point of this tab, and truncating it to fit would
        # defeat it.
        body = QWidget()
        form = QFormLayout(body)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        reg = QGroupBox("Registration")
        reg_form = QFormLayout()
        reg_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        for tunable in scope_profiles.TUNABLES:
            reg_form.addRow(*self._control_for(tunable))
        reg.setLayout(reg_form)
        form.addRow(reg)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        buttons = QHBoxLayout()
        self._status = QLabel()
        self._status.setWordWrap(True)
        buttons.addWidget(self._status, 1)

        self._defaults_btn = QPushButton("Reset to defaults")
        self._defaults_btn.setToolTip(
            "Load the built-in defaults into the controls. Nothing is saved "
            "until you press Save."
        )
        self._defaults_btn.clicked.connect(self._on_defaults)
        buttons.addWidget(self._defaults_btn)

        self._delete_btn = QPushButton("Delete profile")
        self._delete_btn.setToolTip(
            "Forget this microscope's saved options. Its acquisitions will use "
            "the defaults again."
        )
        self._delete_btn.clicked.connect(self._on_delete)
        buttons.addWidget(self._delete_btn)

        self._save_btn = QPushButton("Save")
        self._save_btn.setDefault(True)
        self._save_btn.clicked.connect(self._on_save)
        buttons.addWidget(self._save_btn)

        outer.addLayout(buttons)

        # The settings live in a scroll area, so an unguarded spin box under the
        # pointer silently rewrites itself on the way past — the same class of
        # invisible misconfiguration the main dialog guards against.
        from flamingo_stitcher.gui._wheel_guard import install_wheel_guard

        install_wheel_guard(self)

    @staticmethod
    def _widget_value(widget):
        """The control's value, whatever kind of control it is."""
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        return widget.value()

    @staticmethod
    def _set_widget_value(widget, value) -> None:
        if isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(round(float(value))))
        else:
            widget.setValue(float(value))

    def _control_for(self, tunable: scope_profiles.Tunable):
        """(label widget, control+help widget) for one tunable.

        The control follows `kind`. Rendering a bool as a 0.00-1.00 spin box --
        which is what a single QDoubleSpinBox for everything produces -- reads
        as a broken control rather than a choice.
        """
        if tunable.kind == "bool":
            spin = QCheckBox()
        elif tunable.kind == "int":
            spin = QSpinBox()
            spin.setRange(int(tunable.minimum), int(tunable.maximum))
            spin.setSingleStep(max(1, int(tunable.step)))
            if tunable.suffix:
                spin.setSuffix(tunable.suffix)
        else:
            spin = QDoubleSpinBox()
            spin.setRange(tunable.minimum, tunable.maximum)
            spin.setSingleStep(tunable.step)
            spin.setDecimals(tunable.decimals)
            if tunable.suffix:
                spin.setSuffix(tunable.suffix)
        spin.setToolTip(tunable.help)
        # Wheel events reach the scroll area unless the box is focused; see
        # install_wheel_guard at the end of _build.
        spin.setFocusPolicy(Qt.StrongFocus)
        self._spins[tunable.field] = spin

        help_label = QLabel(tunable.help)
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: palette(mid); font-size: 11px;")

        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(0, 0, 0, 8)
        layout.addWidget(spin)
        if tunable.field == "illumination_low_side":
            # This one is not a preference — it is a fact about the instrument,
            # and typing it in backwards is silent (the run keeps the far,
            # blurred half of every sheet and still looks plausible). The
            # frozen build has no Python on PATH, so the command-line probe
            # cannot be run on the microscope PC; this is the way in.
            measure_btn = QPushButton("Measure from data…")
            measure_btn.setToolTip(
                "Pick an acquisition folder and read the answer off the images "
                "instead of typing it. Takes a few seconds — only a subsampled "
                "slice of each raw file is read."
            )
            measure_btn.clicked.connect(self._on_measure_illumination_side)
            layout.addWidget(measure_btn)
        layout.addWidget(help_label)
        return QLabel(tunable.label + ":"), holder

    def _on_measure_illumination_side(self) -> None:
        """Measure which sheet lights the low end, and offer to apply it."""
        from flamingo_stitcher import illumination_geometry

        folder = QFileDialog.getExistingDirectory(
            self, "Pick an acquisition to measure", self._last_measure_dir
        )
        if not folder:
            return
        self._last_measure_dir = folder

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = illumination_geometry.measure_acquisition(folder)
        except Exception as exc:  # noqa: BLE001 - report, never crash the tab
            logger.exception("Illumination geometry measurement failed")
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(
                self, "Could not measure",
                f"Reading {folder} failed:\n\n{exc}",
            )
            return
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()

        if not result.rows:
            QMessageBox.information(
                self, "Nothing to measure",
                f"No two-sided tiles found in:\n{folder}\n\n"
                "Split and Blend need both illumination sides of a tile. A "
                "single-sided acquisition has nothing to choose between.",
            )
            return

        detail = "\n".join(
            f"  {name[:40]}  ch{ch}  "
            f"{'unclear' if side is None else 'side ' + str(side)}  "
            f"(margin {margin:.2f})"
            for name, ch, side, margin, _t in result.rows
        )
        if result.low_side is None:
            QMessageBox.information(
                self, "Measurement inconclusive",
                result.summary() + "\n\nPer tile:\n" + detail,
            )
            return

        spin = self._spins.get("illumination_low_side")
        current = int(spin.value()) if spin is not None else -1
        agrees = current == result.low_side
        box = QMessageBox(self)
        box.setWindowTitle("Illumination geometry")
        box.setIcon(QMessageBox.Information if agrees else QMessageBox.Warning)
        box.setText(result.summary())
        box.setInformativeText(
            f"The setting is already {current}. Nothing to change."
            if agrees else
            f"The setting is currently {current}. Set it to {result.low_side}?\n\n"
            "Running with it backwards keeps the FAR half of each light sheet — "
            "the blurred, attenuated end that Split and Blend exist to discard — "
            "and the output still looks smooth, so nothing downstream catches it."
        )
        box.setDetailedText("Per tile:\n" + detail)
        if agrees:
            box.setStandardButtons(QMessageBox.Ok)
            box.exec_()
            return
        box.setStandardButtons(QMessageBox.Apply | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Apply)
        if box.exec_() == QMessageBox.Apply and spin is not None:
            spin.setValue(int(result.low_side))
            self._status.setText(
                f"Illumination low side set to {result.low_side} from the data — "
                f"Save to keep it for this microscope."
            )

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #

    def showEvent(self, event):  # noqa: N802 - Qt naming
        """Re-read the microscope list whenever this tab comes forward.

        A run can discover a new instrument while this panel is already built,
        and the panel is constructed once for the life of the app. Without this
        the newly-seen microscope would not appear until a restart.
        """
        super().showEvent(event)
        try:
            current = self._current_scope()
            self._reload_scopes(select=current)
        except Exception:  # noqa: BLE001 - a refresh must never break the tab
            pass

    def _reload_scopes(self, select: str = "") -> None:
        self._loading = True
        try:
            # Every microscope we know of: ones with saved settings AND ones
            # merely seen during a run. Before this, a new instrument never
            # appeared here until someone typed its name in by hand — and it
            # has to match the acquisition's own "Microscope name" exactly or
            # the profile silently never applies.
            scopes = scope_profiles.known_microscopes()
            self._scope_combo.clear()
            for scope in scopes:
                self._scope_combo.addItem(scope, scope)
            self._scope_combo.addItem(_NEW_SCOPE, "")
            if select:
                index = self._scope_combo.findData(select)
                if index >= 0:
                    self._scope_combo.setCurrentIndex(index)
        finally:
            self._loading = False
        self._on_selection_changed()

    def _current_scope(self) -> str:
        if self._scope_combo.currentData():
            return str(self._scope_combo.currentData())
        return self._new_scope_edit.text().strip()

    def _current_objective(self) -> Any:
        data = self._objective_combo.currentData()
        if data == "__new__":
            return self._new_objective_edit.value()
        return data

    def _reload_objectives(self, scope: str) -> None:
        self._objective_combo.clear()
        self._objective_combo.addItem(
            "All objectives", scope_profiles.ANY_OBJECTIVE
        )
        seen = set()
        for key in scope_profiles.list_profiles():
            name, _, objective = key.partition("|")
            if name == scope and objective != scope_profiles.ANY_OBJECTIVE:
                if objective not in seen:
                    seen.add(objective)
                    self._objective_combo.addItem(objective, objective)
        self._objective_combo.addItem("Specific objective…", "__new__")

    def _on_selection_changed(self, *_args) -> None:
        """A different microscope: rebuild its objective list, then load."""
        if self._loading:
            return
        self._new_scope_edit.setVisible(not self._scope_combo.currentData())
        self._loading = True
        try:
            self._reload_objectives(self._current_scope())
        finally:
            self._loading = False
        self._on_objective_changed()

    def _on_objective_changed(self, *_args) -> None:
        """A different objective: only toggle the entry box and reload values.

        Deliberately does NOT rebuild the objective list — doing that here would
        reset the selection the user just made.
        """
        if self._loading:
            return
        self._new_objective_edit.setVisible(
            self._objective_combo.currentData() == "__new__"
        )
        self._load_into_controls()

    # ------------------------------------------------------------------ #
    # Values
    # ------------------------------------------------------------------ #

    def _defaults(self) -> Dict[str, Any]:
        from flamingo_stitcher.pipeline import StitchingConfig

        blank = StitchingConfig()
        return {
            field: float(getattr(blank, field))
            for field in scope_profiles.TUNABLE_FIELDS
        }

    def _load_into_controls(self) -> None:
        scope = self._current_scope()
        values, source = (
            scope_profiles.load_profile(scope, self._current_objective())
            if scope
            else ({}, "")
        )
        merged = self._defaults()
        merged.update(values)
        self._loading = True
        try:
            for field, spin in self._spins.items():
                self._set_widget_value(spin, merged.get(field, 0.0))
        finally:
            self._loading = False

        self._delete_btn.setEnabled(bool(values))
        if not scope:
            self._name_hint.setText(
                "Type the microscope name exactly as it appears in the "
                'acquisition\'s ScopeSettings.txt ("Microscope name = …"). '
                "The match ignores case and surrounding spaces, but nothing "
                "else — a name that does not match is a profile that never "
                "applies."
            )
        elif values:
            self._name_hint.setText(f"Showing the saved profile '{source}'.")
        else:
            self._name_hint.setText(
                f"No profile saved for '{scope}' at this objective yet — the "
                "controls show the built-in defaults."
            )
        self._status.clear()

    def _collect(self) -> Dict[str, Any]:
        return {
            field: self._widget_value(spin)
            for field, spin in self._spins.items()
        }

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def _on_defaults(self) -> None:
        self._loading = True
        try:
            for field, value in self._defaults().items():
                self._set_widget_value(self._spins[field], value)
        finally:
            self._loading = False
        self._status.setText("Defaults loaded — press Save to keep them.")

    def _on_save(self) -> None:
        scope = self._current_scope()
        if not scope:
            QMessageBox.warning(
                self,
                "Which microscope?",
                "Enter the microscope name first. It has to match the "
                '"Microscope name" in the acquisition\'s ScopeSettings.txt, '
                "which is how a profile finds its data.",
            )
            return
        objective = self._current_objective()
        if scope_profiles.save_profile(scope, objective, self._collect()):
            key = scope_profiles.profile_key(scope, objective)
            self._status.setText(f"Saved as '{key}'.")
            self._reload_scopes(select=scope.strip().lower())
        else:
            self._status.setText("Could not save — see the log for why.")

    def _on_delete(self) -> None:
        scope = self._current_scope()
        objective = self._current_objective()
        key = scope_profiles.profile_key(scope, objective)
        if not key:
            return
        confirm = QMessageBox.question(
            self,
            "Delete profile?",
            f"Forget the saved options for '{key}'?\n\n"
            "Acquisitions from this microscope will use the defaults again.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        if scope_profiles.delete_profile(scope, objective):
            self._status.setText(f"Deleted '{key}'.")
            self._reload_scopes()
