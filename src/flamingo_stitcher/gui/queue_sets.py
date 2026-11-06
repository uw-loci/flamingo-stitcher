"""Settings per queued acquisition, and the settings-set letters (0.14.0).

Until 0.14.0 the whole queue ran with the settings in the window at the moment
'Run All' was pressed, and the queue was locked during a run. Now each queued
acquisition keeps its own settings:

* selecting rows shows their settings; changing a setting changes it for the
  selected rows (pending ones only);
* rows that share settings share a letter in the 'Settings' column, by the
  rules in :mod:`flamingo_stitcher.settings_sets`;
* during a run the queue stays usable: acquisitions can be added, and a
  pending one's settings changed, while the running one's are locked (grayed
  out and folded away while it is selected);
* rows from different microscopes or objectives cannot be edited together.

A row's settings are a ``StitchingConfig`` built by the dialog's own
``_build_config`` plus the 'Channels' text, so a run uses exactly what the
window showed when that row was last edited.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QHeaderView,
    QLineEdit,
    QTableWidgetItem,
)

from flamingo_stitcher import settings_sets as ss

COLUMNS = ["Status", "Directory", "Microscope", "Tiles", "Planes", "Raw size",
           "Est. time", "Settings"]
COL_SETTINGS = 7

SET_TINT = {"A": "#dfe9f7", "B": "#fbe7cf", "C": "#dcf0df", "D": "#efe0f7",
            "E": "#f7f2d0", "F": "#e0f3f3"}

# Values that belong to one acquisition, reset when a new microscope or
# objective starts its own set (run 37: a typed 2048 x 2048 frame would have
# carried onto a second microscope's acquisition).
_IMAGE_SPECIFIC = {"frame_width": None, "frame_height": None, "z_step_um": None,
                   "auto_pixel_size": True, "background_zero_thresholds": {},
                   "background_zero_enabled": False}


def settings_key(config, channels_text: str) -> str:
    """Equal exactly when two rows would run with the same settings."""
    from flamingo_stitcher.pipeline import serialize_stitching_config

    d = serialize_stitching_config(config)
    if getattr(config, "auto_pixel_size", True):
        d.pop("pixel_size_um", None)  # filled per acquisition at run time
    d["_channels"] = (channels_text or "").strip()
    return ss.canonical(d)


class QueueSetsMixin:
    """Per-row settings for the stitching dialogs (see module docstring)."""

    # ---- set-up -----------------------------------------------------------

    def _init_queue_sets(self) -> None:
        t = self._queue_table
        t.setColumnCount(len(COLUMNS))
        t.setHorizontalHeaderLabels(COLUMNS)
        t.setSelectionMode(QAbstractItemView.ExtendedSelection)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        hh = t.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.Stretch)
        for c in range(len(COLUMNS)):
            if c != 1:
                hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        t.horizontalHeaderItem(COL_SETTINGS).setToolTip(
            "Acquisitions with the same letter share one set of settings. The\n"
            "first ones queued are Set A; one from another microscope or\n"
            "objective starts the next letter. Changing the settings of some\n"
            "rows of a set moves them into a new letter."
        )
        t.itemSelectionChanged.connect(self._on_queue_selection_changed)
        self._loading_row_settings = False
        self._template_settings = None  # used while the queue is empty
        self._write_back_timer = QTimer(self)
        self._write_back_timer.setSingleShot(True)
        self._write_back_timer.setInterval(150)
        self._write_back_timer.timeout.connect(self._write_settings_to_selection)
        for w in self._settings_widgets():
            for sig in ("currentIndexChanged", "toggled", "valueChanged", "textChanged"):
                s = getattr(w, sig, None)
                if s is not None:
                    s.connect(self._on_settings_widget_changed)
                    break
        # Queue totals and settings sets, under the table.
        from PyQt5.QtWidgets import QLabel

        self._queue_summary_label = QLabel("")
        self._queue_summary_label.setWordWrap(True)
        self._queue_summary_label.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        lay = t.parentWidget().layout()
        lay.insertWidget(lay.indexOf(t) + 1, self._queue_summary_label)
        self._queue_summary_label.hide()
        self._fit_queue_height()

    def _settings_widgets(self):
        """Every input in Basics, the two tabs and the background panel."""
        roots = [self._config_container, self._settings_tabs]
        seen = set()
        for root in roots:
            for cls in (QComboBox, QCheckBox, QAbstractSpinBox, QLineEdit):
                for w in root.findChildren(cls):
                    if id(w) in seen:
                        continue
                    # A spin box's inner line edit is not a separate input.
                    if isinstance(w, QLineEdit) and isinstance(w.parent(), QAbstractSpinBox):
                        continue
                    seen.add(id(w))
                    yield w

    # ---- rows ---------------------------------------------------------------

    def _row_scope(self, item: Dict[str, Any]) -> str:
        """'n7, 6.2x' from the acquisition's own metadata (cached on the item)."""
        if item.get("scope"):
            return item["scope"]
        try:
            from flamingo_stitcher import scope_profiles

            scope, objective, _values, _source = scope_profiles.resolve_for_acquisition(
                item["path"]
            )
            obj = str(objective or "").strip()
            if obj and not obj.endswith("x"):
                obj += "x"
            text = ", ".join(p for p in (scope or "unknown microscope", obj) if p)
        except Exception:  # noqa: BLE001 - a label must never break the queue
            text = "unknown microscope"
        item["scope"] = text
        return text

    def _rows(self) -> List[ss.Row]:
        return [
            ss.Row(self._row_scope(it), it.get("settings_key", ""), it.get("set", ""),
                   editable=it.get("status") == "pending")
            for it in self._queue
        ]

    def _store_letters(self, rows: List[ss.Row]) -> None:
        for it, r in zip(self._queue, rows):
            it["set"] = r.letter

    def _current_settings(self):
        config = self._build_config()
        text = self._channels_edit.text()
        return config, text, settings_key(config, text)

    def _place_new_row(self, item: Dict[str, Any]) -> None:
        """Give a newly queued acquisition its set and settings."""
        rows = self._rows()[:-1]
        scope = self._row_scope(item)
        selected = self._selected_queue_rows()
        sel_letter = self._queue[selected[0]].get("set") if selected else None
        if not rows and self._template_settings is not None:
            config, text, key = self._template_settings
        else:
            config, text, key = self._current_settings()
        letter, key_out = ss.place_new(rows, scope, key, sel_letter)
        donor = next((it for it in self._queue[:-1] if it.get("set") == letter
                      and self._row_scope(it) == scope), None)
        if donor is not None:
            item["config"] = replace(donor["config"])
            item["channels_text"] = donor.get("channels_text", "")
            item["settings_key"] = donor["settings_key"]
        else:
            if rows:  # a new microscope or objective: drop image-specific values
                config = replace(config, **{k: (dict(v) if isinstance(v, dict) else v)
                                            for k, v in _IMAGE_SPECIFIC.items()
                                            if hasattr(config, k)})
                text = ""
            item["config"] = config
            item["channels_text"] = text
            item["settings_key"] = settings_key(config, text)
        item["set"] = letter

    def _selected_queue_rows(self) -> List[int]:
        return sorted({i.row() for i in self._queue_table.selectedIndexes()})

    # ---- selection ------------------------------------------------------------

    def _on_queue_selection_changed(self) -> None:
        sel = self._selected_queue_rows()
        rows = self._rows()
        state = ss.selection_state(rows, sel)
        if state in ("editable", "locked") and sel:
            self._load_row_settings(self._queue[sel[0]])
        self._apply_selection_state()
        self._update_action_buttons()  # 'Run Selected' follows the selection

    def _load_row_settings(self, item: Dict[str, Any]) -> None:
        """Show a row's settings in the controls without writing them back."""
        config = item.get("config")
        if config is None:
            return
        from flamingo_stitcher.pipeline import serialize_stitching_config

        self._loading_row_settings = True
        try:
            self._apply_stitching_config(serialize_stitching_config(config))
            # Values the shared-configuration loader leaves alone on purpose.
            self._set_combo_by_data(self._frame_size_combo, config.frame_width)
            self._setting_pixel_programmatically = True
            self._pixel_size_spin.setValue(float(config.pixel_size_um))
            self._setting_pixel_programmatically = False
            self._pixel_size_user_set = not getattr(config, "auto_pixel_size", True)
            self._setting_z_step_programmatically = True
            self._z_step_spin.setValue(float(config.z_step_um or 0.0))
            self._setting_z_step_programmatically = False
            self._z_step_user_set = config.z_step_um is not None
            self._scratch_dir_edit.setText(config.scratch_dir or "")
            self._channels_edit.setText(item.get("channels_text", ""))
            self._bg_zero_panel.set_enabled_state(bool(config.background_zero_enabled))
            if config.background_zero_thresholds:
                self._bg_zero_panel.set_thresholds(dict(config.background_zero_thresholds))
        finally:
            self._loading_row_settings = False
        self._write_back_timer.stop()
        # Some controls settle a moment after loading (a list relabels for the
        # new Z step, say). That is not an edit, so it must not split the set.
        self._rebaseline_selection()
        QTimer.singleShot(60, self._rebaseline_selection)

    def _rebaseline_selection(self) -> None:
        """Re-read the shown settings into the selected set without splitting it."""
        sel = self._selected_queue_rows()
        rows = self._rows()
        if ss.selection_state(rows, sel) != "editable":
            return
        keys = {rows[i].key for i in sel}
        if len(keys) != 1:
            return
        old = keys.pop()
        self._loading_row_settings = True
        try:
            config, text, key = self._current_settings()
        finally:
            self._loading_row_settings = False
        if key == old:
            return
        scope = rows[sel[0]].scope
        for it, r in zip(self._queue, rows):
            if r.key == old and r.scope == scope and r.editable:
                it["config"] = replace(config)
                it["channels_text"] = text
                it["settings_key"] = key
        self._write_back_timer.stop()

    def _apply_selection_state(self) -> None:
        """Enable, gray out or fold the settings for what is selected."""
        if not hasattr(self, "_settings_tabs"):
            return
        sel = self._selected_queue_rows()
        rows = self._rows()
        state = ss.selection_state(rows, sel)
        note = ss.edit_note(rows, sel) if len(self._queue) > 1 else ""
        sel_items = [self._queue[i] for i in sel]
        running = any(it.get("status") == "stitching" for it in sel_items)
        if state == "locked":
            note = (
                "Locked: these are the settings the running acquisition is using. "
                "Select a pending acquisition in the queue to change its settings."
                if running else
                "These are the settings this acquisition ran with. 'Re-queue' it to "
                "run it again, with changes if you like."
            )
        editable = state in ("editable", "none")
        self._config_container.setEnabled(editable)
        self._set_tab_pages_enabled(editable)
        self._output_dir_edit.setEnabled(not self._batch_running)
        if running and self._proc_toggle.isChecked():
            self._proc_toggle.setChecked(False)  # fold: the log gets the room
        elif editable and not self._proc_toggle.isChecked() and getattr(
            self, "_folded_for_lock", False
        ):
            self._proc_toggle.setChecked(True)
        self._folded_for_lock = running
        self._selection_note.setText(note)
        self._selection_note.setStyleSheet(
            "font-size: 9pt; color: %s;" % ("#6b5600" if state in ("locked", "mixed")
                                            else "#1d3f6e"))
        self._selection_note.setVisible(bool(note))
        self._fit_queue_height()
        self._run_summary_subtitle = self._selected_set_subtitle()
        self._run_summary_message = (
            ss.edit_note(rows, sel).split(". Settings are")[0] + ". Select acquisitions from one "
            "microscope and objective to see the settings they share."
            if state == "mixed" else ""
        )
        self._refresh_run_summary()

    def _fit_queue_height(self) -> None:
        """2 rows when short, up to 6 before it scrolls; 3 while the running
        acquisition is selected, so the log gets the room."""
        t = self._queue_table
        n = t.rowCount()
        locked_running = any(
            self._queue[i].get("status") == "stitching"
            for i in self._selected_queue_rows() if i < len(self._queue)
        )
        visible = min(max(n, 2), 3 if locked_running else 6)
        rh = t.verticalHeader().defaultSectionSize()
        t.setFixedHeight(t.horizontalHeader().sizeHint().height() + visible * rh + 6)

    # ---- edits --------------------------------------------------------------

    def _on_settings_widget_changed(self, *_args) -> None:
        if self._loading_row_settings:
            return
        self._write_back_timer.start()

    def _write_settings_to_selection(self) -> None:
        """The controls changed: store them on the selected pending rows."""
        if self._loading_row_settings:
            return
        if not self._queue:
            self._template_settings = self._current_settings()
            return
        sel = self._selected_queue_rows()
        rows = self._rows()
        if ss.selection_state(rows, sel) != "editable":
            return
        config, text, key = self._current_settings()
        if all(rows[i].key == key for i in sel):
            return
        ss.apply_edit(rows, sel, key)
        for i in sel:
            it = self._queue[i]
            it["config"] = replace(config)
            it["channels_text"] = text
            it["settings_key"] = key
        self._store_letters(rows)
        self._update_queue_table()

    # ---- table --------------------------------------------------------------

    def _fill_queue_row(self, i: int, item: Dict[str, Any]) -> None:
        """Microscope, Tiles, Planes, Raw size, Est. time and Settings columns."""
        t = self._queue_table

        def put(col, text, tip=None):
            cell = QTableWidgetItem(text)
            if tip:
                cell.setToolTip(tip)
            t.setItem(i, col, cell)
            return cell

        put(2, self._row_scope(item))
        tiles = item.get("tiles")
        if tiles:
            xs = len({round(getattr(tl, "x_mm", 0.0), 3) for tl in tiles})
            ys = len({round(getattr(tl, "y_mm", 0.0), 3) for tl in tiles})
            put(3, f"{len(tiles)} ({xs}x{ys})" if xs * ys == len(tiles) else str(len(tiles)))
            planes = sorted({int(getattr(tl, "n_planes", 0) or 0) for tl in tiles})
            put(4, str(planes[-1]) if len(planes) == 1 else f"{planes[0]}-{planes[-1]}")
            put(5, _raw_size(tiles))
            put(6, self._row_time_text(item))
        else:
            for col in (3, 4, 5, 6):
                put(col, "—", "Press 'Discover Tiles' to read the acquisition.")
        letter = item.get("set", "")
        cell = put(COL_SETTINGS, f"Set {letter}" if letter else "")
        f = cell.font()
        f.setBold(True)
        cell.setFont(f)
        cell.setTextAlignment(Qt.AlignCenter)
        cell.setBackground(QColor(SET_TINT.get(letter[-1:] if letter else "", "#eeeeee")))

    def _row_time_text(self, item: Dict[str, Any]) -> str:
        try:
            from flamingo_stitcher.multi_phase_estimator import _format_duration
            from flamingo_stitcher.pipeline import build_timing_key, rough_run_seconds
            from flamingo_stitcher.timing_cache import StitchingTimingCache

            config = item.get("config") or self._build_config()
            key = build_timing_key(item["tiles"], config, acquisition_dir=item.get("path"),
                                   output_dir=self._output_dir_edit.text().strip() or None)
            s = StitchingTimingCache().get_total_s(key) or rough_run_seconds(item["tiles"], config)
            return "~" + _format_duration(s)
        except Exception:  # noqa: BLE001
            return ""

    def _queue_summary_text(self) -> str:
        if len(self._queue) < 2:
            return ""
        counts: Dict[str, int] = {}
        for it in self._queue:
            counts[it.get("status", "")] = counts.get(it.get("status", ""), 0) + 1
        parts = [f"{len(self._queue)} acquisitions"]
        order = ("done", "stitching", "pending", "error", "cancelled")
        names = {"stitching": "running"}
        parts.append(", ".join(f"{counts[k]} {names.get(k, k)}" for k in order if counts.get(k)))
        sets = ss.summary(self._rows())
        if sets:
            parts.append(sets)
        return "  •  ".join(parts)

    def _selected_set_subtitle(self) -> str:
        sel = self._selected_queue_rows()
        letters = sorted({self._queue[i].get("set", "") for i in sel if i < len(self._queue)})
        if len(letters) == 1 and letters[0]:
            n = sum(1 for it in self._queue if it.get("set") == letters[0])
            return f"Set {letters[0]}" + (f" ({n} acquisitions)" if n > 1 else "")
        return ""


def _raw_size(tiles) -> str:
    total = 0
    for tl in tiles:
        # raw_files: {channel: {light path: file}}
        for by_side in (getattr(tl, "raw_files", None) or {}).values():
            for p in by_side.values():
                try:
                    total += p.stat().st_size
                except Exception:  # noqa: BLE001
                    pass
    if total <= 0:
        return ""
    gb = total / 1024**3
    return f"{gb / 1024:.2f} TB" if gb >= 1024 else f"{gb:.0f} GB"
