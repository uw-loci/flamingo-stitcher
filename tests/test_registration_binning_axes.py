"""XY and Z registration binning are chosen independently, like downsampling.

They used to move together as one three-way preset (Fine/Default/Fast), which
is the wrong coupling. Z binning sets the FLOOR on how precisely a Z shift can
be resolved — a shift lands to about one binned voxel, so at the default z=2
that is one raw plane, and the 3–6 frame tile offsets this registration exists
to correct are only a few times bigger. Halving Z costs one axis of correlation
work; being forced to halve XY alongside it costs four times that and buys
nothing in Z.

The dict stays the single canonical form. XY and Z are how it is *chosen*, not
a second place the number is stored — this package has shipped the same bug
three times by keeping two copies of one calculation.

Run: QT_QPA_PLATFORM=offscreen python -m pytest \\
        tests/test_registration_binning_axes.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestTheDictIsBuiltFromTwoChoices:
    """Pure: exercised through the dialog's own composition, no Qt needed for
    the arithmetic these pin."""

    @classmethod
    def setup_class(cls):
        pytest.importorskip("PyQt5")
        from PyQt5.QtWidgets import QApplication

        from flamingo_stitcher.gui.stitching_dialog import StitchingDialog

        cls._qapp = QApplication.instance() or QApplication([])
        cls._dlg = StitchingDialog()

    @classmethod
    def teardown_class(cls):
        dlg = getattr(cls, "_dlg", None)
        if dlg is not None:
            dlg.deleteLater()
            cls._dlg = None

    def _set(self, xy, z):
        self._dlg._set_registration_binning({"z": z, "y": xy, "x": xy})
        return self._dlg._registration_binning()

    def test_xy_drives_both_lateral_axes(self):
        # A lateral peak is ONE joint correlation, so splitting X from Y would
        # not buy anything and would invite them to disagree.
        assert self._set(xy=8, z=1) == {"z": 1, "y": 8, "x": 8}

    def test_z_is_independent_of_xy(self):
        assert self._set(xy=4, z=1)["z"] == 1
        assert self._set(xy=4, z=8)["z"] == 8
        assert self._set(xy=4, z=8)["y"] == 4

    def test_xy_is_independent_of_z(self):
        assert self._set(xy=1, z=2)["y"] == 1
        assert self._set(xy=8, z=2)["y"] == 8
        assert self._set(xy=8, z=2)["z"] == 2

    def test_the_default_is_what_it_always_was(self):
        # The three-way preset defaulted to (z2 y4 x4). Splitting the control
        # must not quietly change what an untouched dialog runs.
        from flamingo_stitcher.gui.stitching_dialog import StitchingDialog

        fresh = StitchingDialog()
        try:
            assert fresh._registration_binning() == {"z": 2, "y": 4, "x": 4}
        finally:
            fresh.deleteLater()

    def test_every_offered_combination_round_trips(self):
        for xy in (1, 2, 4, 8):
            for z in (1, 2, 4, 8):
                assert self._set(xy, z) == {"z": z, "y": xy, "x": xy}


class TestReadingADictThatCannotBeShownExactly(TestTheDictIsBuiltFromTwoChoices):
    def test_differing_x_and_y_select_the_larger(self):
        # Selecting the smaller would make the run do LESS binning than the
        # config asked for — more work than requested, which costs time rather
        # than correctness, but silently.
        self._dlg._set_registration_binning({"z": 2, "y": 8, "x": 2})
        assert self._dlg._registration_binning()["y"] == 8

    def test_a_missing_x_falls_back_to_y(self):
        self._dlg._set_registration_binning({"z": 1, "y": 2})
        assert self._dlg._registration_binning() == {"z": 1, "y": 2, "x": 2}

    def test_junk_is_ignored_rather_than_applied(self):
        before = self._dlg._registration_binning()
        self._dlg._set_registration_binning({"z": "banana", "y": 4, "x": 4})
        assert self._dlg._registration_binning() == before

    def test_a_non_dict_is_not_an_exception(self):
        before = self._dlg._registration_binning()
        self._dlg._set_registration_binning(None)
        self._dlg._set_registration_binning([1, 2, 3])
        assert self._dlg._registration_binning() == before


class TestSkipRegistrationGreysBothOut:
    @classmethod
    def setup_class(cls):
        pytest.importorskip("PyQt5")
        from PyQt5.QtWidgets import QApplication

        from flamingo_stitcher.gui.stitching_dialog import StitchingDialog

        cls._qapp = QApplication.instance() or QApplication([])
        cls._dlg = StitchingDialog()

    @classmethod
    def teardown_class(cls):
        dlg = getattr(cls, "_dlg", None)
        if dlg is not None:
            dlg.deleteLater()
            cls._dlg = None

    def test_both_combos_follow_the_skip_checkbox(self):
        dlg = self._dlg
        dlg._skip_reg_cb.setChecked(True)
        assert not dlg._reg_binning_xy_combo.isEnabled()
        assert not dlg._reg_binning_z_combo.isEnabled()
        dlg._skip_reg_cb.setChecked(False)
        assert dlg._reg_binning_xy_combo.isEnabled()
        assert dlg._reg_binning_z_combo.isEnabled()

    def test_the_axis_labels_follow_too(self):
        # Otherwise a live "XY"/"Z" label sits beside two dead dropdowns.
        dlg = self._dlg
        dlg._skip_reg_cb.setChecked(True)
        assert not dlg._reg_binning_xy_label.isEnabled()
        assert not dlg._reg_binning_z_label.isEnabled()
        dlg._skip_reg_cb.setChecked(False)


class TestTheSavedSettingCannotBeMisread:
    """The old key held a combo INDEX into a three-way preset.

    Reading it now would turn a stale 0/1/2 into 1x/2x/4x — a binning nobody
    ever chose, applied silently. New keys hold the factors themselves, and the
    old key is left unread so it ages out. Same reasoning as the saved-data
    version gate: a value whose meaning changed must not be reinterpreted.
    """

    @classmethod
    def setup_class(cls):
        pytest.importorskip("PyQt5")
        from PyQt5.QtWidgets import QApplication

        cls._qapp = QApplication.instance() or QApplication([])

    def test_the_old_index_key_is_not_read(self):
        import flamingo_stitcher.gui.stitching_dialog as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert 's.value("reg_binning"' not in source

    def test_the_new_keys_store_factors_not_indices(self):
        import flamingo_stitcher.gui.stitching_dialog as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert 'setValue("reg_binning_xy", self._reg_binning_xy_combo.currentData())' in (
            source
        )
        assert 'setValue("reg_binning_z", self._reg_binning_z_combo.currentData())' in (
            source
        )

    def test_the_restore_defaults_are_the_old_preset(self):
        """These, not the constructor call, are what an untouched dialog runs.

        `_restore_settings` runs at construction and overwrites whatever the
        constructor set, so the defaults passed to `s.value` ARE the default
        binning. Mutating the constructor's own call changes nothing, which is
        how a mutation test first found this line to be dead.
        """
        import flamingo_stitcher.gui.stitching_dialog as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert source.count('s.value("reg_binning_xy", 4, type=int)') == 2
        assert source.count('s.value("reg_binning_z", 2, type=int)') == 2

    def test_every_save_site_has_a_matching_restore(self):
        # NativeStitchingDialog overrides _save_settings/_restore_settings
        # wholesale without calling super(), so these keys live in FOUR places
        # and a miss makes one tab silently forget the setting.
        import flamingo_stitcher.gui.stitching_dialog as mod

        source = Path(mod.__file__).read_text(encoding="utf-8")
        assert source.count('setValue("reg_binning_xy"') == 2
        assert source.count('setValue("reg_binning_z"') == 2
        assert source.count('s.value("reg_binning_xy"') == 2
        assert source.count('s.value("reg_binning_z"') == 2


class TestTheCliOffersTheSameShape:
    """A setting that exists in the GUI and not the CLI is a setting half the
    users cannot reach — and this package's CLI already spent its life
    overriding one dataclass default because a flag was passed unconditionally.
    """

    def _config(self, argv):
        from flamingo_stitcher.__main__ import build_parser

        return build_parser().parse_args(argv)

    def _binning(self, argv):
        """Through the SHIPPED resolver, not a copy of it.

        An earlier version of this test re-implemented the overlay inline and
        passed happily with the real one neutered — a copy of the logic agrees
        with itself no matter what the code does.
        """
        from flamingo_stitcher.__main__ import resolve_registration_binning

        return resolve_registration_binning(self._config(argv))

    def test_neither_flag_means_the_user_said_nothing(self):
        # None, not a default dict: passing a value in unconditionally is how
        # --quality-threshold silently overrode 0.4 with 0.2 on every run.
        assert self._binning(["acq"]) is None

    def test_xy_alone_sets_both_lateral_axes(self):
        assert self._binning(["acq", "--reg-binning-xy", "2"]) == {
            "z": 2,
            "y": 2,
            "x": 2,
        }

    def test_z_alone_leaves_xy_at_the_default(self):
        assert self._binning(["acq", "--reg-binning-z", "1"]) == {
            "z": 1,
            "y": 4,
            "x": 4,
        }

    def test_the_two_compose(self):
        assert self._binning(
            ["acq", "--reg-binning-xy", "8", "--reg-binning-z", "4"]
        ) == {"z": 4, "y": 8, "x": 8}

    def test_the_three_axis_flag_still_works(self):
        # Scripts already pass it; removing it would break them silently.
        assert self._binning(["acq", "--registration-binning", "1", "2", "2"]) == {
            "z": 1,
            "y": 2,
            "x": 2,
        }

    def test_an_axis_flag_overrides_that_axis_of_the_three_axis_flag(self):
        assert self._binning(
            ["acq", "--registration-binning", "4", "8", "8", "--reg-binning-z", "1"]
        ) == {"z": 1, "y": 8, "x": 8}


class TestTheConfigFileSpeaksTheSameLanguage:
    def _load(self, binning, monkeypatch):
        # apply_stitching_yaml_to_config takes only the config and reads the
        # YAML itself, so the YAML is what has to be substituted.
        import flamingo_stitcher.config_loader as loader
        from flamingo_stitcher.pipeline import StitchingConfig

        monkeypatch.setattr(
            loader,
            "get_stitching_defaults",
            lambda: {"registration": {"binning": binning}},
        )
        config = StitchingConfig()
        loader.apply_stitching_yaml_to_config(config)
        return config.registration_binning

    def test_xy_is_accepted(self, monkeypatch):
        # Otherwise the config file would be the one place that spells this
        # differently from the GUI and the CLI.
        assert self._load(monkeypatch=monkeypatch, binning={"xy": 2, "z": 1}) == {"z": 1, "y": 2, "x": 2}

    def test_explicit_y_and_x_still_win(self, monkeypatch):
        # For a file that genuinely needs the lateral axes apart.
        assert self._load(monkeypatch=monkeypatch, binning={"xy": 2, "y": 8, "x": 8, "z": 1}) == {
            "z": 1,
            "y": 8,
            "x": 8,
        }

    def test_the_old_per_axis_form_is_unchanged(self, monkeypatch):
        assert self._load(monkeypatch=monkeypatch, binning={"z": 4, "y": 8, "x": 8}) == {"z": 4, "y": 8, "x": 8}

    def test_omitted_axes_fall_back_to_the_defaults(self, monkeypatch):
        assert self._load(monkeypatch=monkeypatch, binning={"z": 1}) == {"z": 1, "y": 4, "x": 4}
