"""Destripe settings are per-MICROSCOPE, not per GUI session.

Two of the knobs are absolute intensities in raw counts -- ``threshold`` (the
foreground/background split) and ``crossover`` (the blend width) -- so a value
tuned on one instrument can land above everything or below everything on
another. That does not merely detune the filter, it silently changes what it
does. Hence: settings follow the data, and an unrecognised scope warns loudly
rather than quietly inheriting whatever was last used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flamingo_stitcher import destripe_presets  # noqa: E402

LIARA = {
    "sigma_foreground": 128.0,
    "sigma_background": 256.0,
    "level": 7,
    "wavelet": "db2",
    "threshold": 110.0,
    "crossover": 150.0,
}
N7 = {
    "sigma_foreground": 64.0,
    "sigma_background": 128.0,
    "level": 5,
    "wavelet": "db3",
    "threshold": 4200.0,
    "crossover": 300.0,
}


@pytest.fixture(autouse=True)
def isolated_presets(tmp_path, monkeypatch):
    """Never touch the developer's real ~/.flamingo_stitcher presets."""
    monkeypatch.setattr(
        destripe_presets, "_presets_path", lambda: tmp_path / "destripe_presets.json"
    )
    return tmp_path


def _acq_with_scope(tmp_path: Path, name: str, scope: str) -> Path:
    acq = tmp_path / name
    acq.mkdir(parents=True)
    (acq / "ScopeSettings.txt").write_text(
        "<Scope Settings>\n"
        f"  Microscope name = {scope}\n"
        "  Objective lens magnification = 6.205\n"
        "</Scope Settings>\n"
    )
    return acq


class TestRoundTrip:
    def test_saves_and_loads_all_params(self):
        assert destripe_presets.save_destripe_preset("Liara", LIARA, "horizontal")

        got = destripe_presets.load_destripe_preset("Liara")
        assert got["params"] == LIARA  # all six, not a subset
        assert got["direction"] == "horizontal"

    def test_lookup_is_case_and_whitespace_insensitive(self):
        destripe_presets.save_destripe_preset("Liara", LIARA)

        assert destripe_presets.load_destripe_preset("  liARA ") is not None

    def test_scopes_do_not_share_settings(self):
        """The whole point: Liara's threshold must not reach n7."""
        destripe_presets.save_destripe_preset("Liara", LIARA)
        destripe_presets.save_destripe_preset("n7", N7)

        assert destripe_presets.load_destripe_preset("Liara")["params"] == LIARA
        assert destripe_presets.load_destripe_preset("n7")["params"] == N7

    def test_resaving_replaces_that_scope_only(self):
        destripe_presets.save_destripe_preset("Liara", LIARA)
        destripe_presets.save_destripe_preset("n7", N7)
        destripe_presets.save_destripe_preset("Liara", N7)

        assert destripe_presets.load_destripe_preset("Liara")["params"] == N7
        assert destripe_presets.load_destripe_preset("n7")["params"] == N7

    def test_unknown_scope_returns_none(self):
        assert destripe_presets.load_destripe_preset("never-seen") is None

    def test_blank_name_is_not_saved_under_an_empty_key(self):
        assert destripe_presets.save_destripe_preset("", LIARA) is False
        assert destripe_presets.save_destripe_preset(None, LIARA) is False


class TestResolutionFromAcquisition:
    def test_reads_the_scope_from_acquisition_metadata(self, tmp_path):
        acq = _acq_with_scope(tmp_path, "acq_a", "Liara")
        destripe_presets.save_destripe_preset("Liara", LIARA, "horizontal")

        scope, preset = destripe_presets.resolve_for_acquisition(acq)

        assert scope == "Liara"
        assert preset["params"] == LIARA

    def test_two_acquisitions_resolve_to_their_own_scopes(self, tmp_path):
        """A batch mixing instruments must not cross-contaminate."""
        a = _acq_with_scope(tmp_path, "acq_a", "Liara")
        b = _acq_with_scope(tmp_path, "acq_b", "n7")
        destripe_presets.save_destripe_preset("Liara", LIARA)
        destripe_presets.save_destripe_preset("n7", N7)

        assert destripe_presets.resolve_for_acquisition(a)[1]["params"] == LIARA
        assert destripe_presets.resolve_for_acquisition(b)[1]["params"] == N7

    def test_acquisition_with_no_metadata_yields_no_scope(self, tmp_path):
        acq = tmp_path / "bare"
        acq.mkdir()

        scope, preset = destripe_presets.resolve_for_acquisition(acq)

        assert scope is None and preset is None


class TestUserFacingMessage:
    def test_known_scope_reports_applied(self):
        applied, msg = destripe_presets.describe_resolution(
            "Liara", {"params": LIARA, "direction": "auto"}
        )
        assert applied is True
        assert "Liara" in msg

    def test_unknown_scope_warns_and_continues(self):
        """Agreed behaviour: warn in text, keep going with last-used."""
        applied, msg = destripe_presets.describe_resolution("n7", None)

        assert applied is False
        assert "n7" in msg
        assert "last-used" in msg
        # The warning must say WHY carrying settings over is risky.
        assert "absolute intensity" in msg

    def test_unreadable_scope_still_explains_itself(self):
        applied, msg = destripe_presets.describe_resolution(None, None)

        assert applied is False
        assert "last-used" in msg


class TestCorruptStoreIsSurvivable:
    def test_unreadable_file_does_not_raise(self, isolated_presets):
        (isolated_presets / "destripe_presets.json").write_text("{not json")

        assert destripe_presets.load_destripe_preset("Liara") is None

    def test_entry_without_params_is_ignored(self, isolated_presets):
        (isolated_presets / "destripe_presets.json").write_text(
            json.dumps({"liara": {"direction": "horizontal"}})
        )

        assert destripe_presets.load_destripe_preset("Liara") is None

    def test_saving_over_a_corrupt_file_recovers(self, isolated_presets):
        (isolated_presets / "destripe_presets.json").write_text("{not json")

        assert destripe_presets.save_destripe_preset("Liara", LIARA)
        assert destripe_presets.load_destripe_preset("Liara")["params"] == LIARA


class TestMicroscopeNameParsing:
    """A blank "Microscope name =" must not resolve to the next line.

    ``\\s*`` around the ``=`` used to swallow the newline, so an empty field
    captured ``</Scope Settings>``. That string then became the preset key --
    and EVERY scope with a blank name would have shared it, for tile
    orientation as well as destripe settings.
    """

    def _write(self, tmp_path: Path, body: str) -> Path:
        acq = tmp_path / "acq"
        acq.mkdir(exist_ok=True)
        (acq / "ScopeSettings.txt").write_text(body)
        return acq

    def test_blank_name_yields_no_scope(self, tmp_path):
        acq = self._write(
            tmp_path,
            "<Scope Settings>\n  Microscope name = \n</Scope Settings>\n",
        )

        assert destripe_presets.microscope_for_acquisition(acq) is None

    def test_normal_name_still_reads(self, tmp_path):
        acq = self._write(
            tmp_path,
            "<Scope Settings>\n  Microscope name = Liara\n</Scope Settings>\n",
        )

        assert destripe_presets.microscope_for_acquisition(acq) == "Liara"

    def test_name_with_spaces_is_preserved(self, tmp_path):
        acq = self._write(
            tmp_path,
            "<Scope Settings>\n  Microscope name = LOCI n7 scope\n</Scope>\n",
        )

        assert destripe_presets.microscope_for_acquisition(acq) == "LOCI n7 scope"

    def test_crlf_line_endings(self, tmp_path):
        acq = self._write(
            tmp_path,
            "<Scope Settings>\r\n  Microscope name = Liara\r\n</Scope Settings>\r\n",
        )

        assert destripe_presets.microscope_for_acquisition(acq) == "Liara"
