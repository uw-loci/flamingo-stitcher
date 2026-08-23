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
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
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
        self._spins: Dict[str, QDoubleSpinBox] = {}
        self._loading = False
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

    def _control_for(self, tunable: scope_profiles.Tunable):
        """(label widget, control+help widget) for one tunable."""
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
        layout.addWidget(help_label)
        return QLabel(tunable.label + ":"), holder

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #

    def _reload_scopes(self, select: str = "") -> None:
        self._loading = True
        try:
            profiles = scope_profiles.list_profiles()
            scopes = sorted({key.split("|")[0] for key in profiles})
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
                spin.setValue(float(merged.get(field, 0.0)))
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
        return {field: spin.value() for field, spin in self._spins.items()}

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def _on_defaults(self) -> None:
        self._loading = True
        try:
            for field, value in self._defaults().items():
                self._spins[field].setValue(float(value))
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
