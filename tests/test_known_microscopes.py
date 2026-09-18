"""The Options tab should offer microscopes it has actually seen.

Before this, the microscope dropdown was built only from SAVED profiles, so a
new instrument never appeared until someone typed its name in by hand — and
that name has to match the acquisition's own "Microscope name" exactly, or the
profile silently never applies to anything. Typing it is the step most likely
to go wrong, and the app already knows the answer.

Stored as a set: the same instrument cannot appear twice under different case
or spacing, because names are normalised the same way lookups normalise them.

Run: python -m pytest tests/test_known_microscopes.py -q
"""

from __future__ import annotations

import json

import pytest

from flamingo_stitcher import scope_profiles


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Keep the developer's real ~/.flamingo_stitcher out of this."""
    monkeypatch.setattr(
        scope_profiles, "seen_path", lambda: tmp_path / "seen_microscopes.json"
    )
    monkeypatch.setattr(
        scope_profiles, "profiles_path", lambda: tmp_path / "scope_profiles.json"
    )
    return tmp_path


class TestItIsASet:
    def test_a_new_name_is_remembered(self):
        assert scope_profiles.remember_microscope("n7") is True
        assert "n7" in scope_profiles.known_microscopes()

    def test_the_same_name_is_not_added_twice(self):
        assert scope_profiles.remember_microscope("n7") is True
        assert scope_profiles.remember_microscope("n7") is False
        assert scope_profiles.known_microscopes().count("n7") == 1

    @pytest.mark.parametrize("variant", ["N7", " n7 ", "N7  ", "  N7"])
    def test_case_and_spacing_do_not_create_duplicates(self, variant):
        # Lookups normalise the same way, so two spellings would be one
        # instrument at resolve time but two rows in the dropdown.
        scope_profiles.remember_microscope("n7")
        assert scope_profiles.remember_microscope(variant) is False
        assert scope_profiles.known_microscopes() == ["n7"]

    def test_several_instruments_all_survive(self):
        for name in ("n7", "liara", "ctlsm1"):
            scope_profiles.remember_microscope(name)
        assert scope_profiles.known_microscopes() == ["ctlsm1", "liara", "n7"]

    def test_the_list_is_sorted(self):
        for name in ("zeta", "alpha", "mu"):
            scope_profiles.remember_microscope(name)
        names = scope_profiles.known_microscopes()
        assert names == sorted(names)


class TestItIgnoresNonsense:
    @pytest.mark.parametrize("bad", [None, "", "   "])
    def test_an_empty_name_is_not_stored(self, bad):
        assert scope_profiles.remember_microscope(bad) is False
        assert scope_profiles.known_microscopes() == []

    def test_a_corrupt_file_does_not_raise(self, isolated_home):
        (isolated_home / "seen_microscopes.json").write_text("{ not json")
        assert scope_profiles.known_microscopes() == []
        assert scope_profiles.remember_microscope("n7") is True

    def test_an_unexpected_shape_is_tolerated(self, isolated_home):
        (isolated_home / "seen_microscopes.json").write_text(json.dumps(12345))
        assert scope_profiles.known_microscopes() == []

    def test_a_dict_wrapper_is_read(self, isolated_home):
        (isolated_home / "seen_microscopes.json").write_text(
            json.dumps({"microscopes": ["n7", "liara"]})
        )
        assert scope_profiles.known_microscopes() == ["liara", "n7"]


class TestItMergesWithSavedProfiles:
    def test_a_saved_profile_appears_without_being_seen(self, isolated_home):
        (isolated_home / "scope_profiles.json").write_text(
            json.dumps({"liara|17.0x": {"quality_threshold": 0.5}})
        )
        assert "liara" in scope_profiles.known_microscopes()

    def test_saved_and_seen_are_unioned_without_duplicates(self, isolated_home):
        (isolated_home / "scope_profiles.json").write_text(
            json.dumps({"n7|6.2x": {"quality_threshold": 0.5}})
        )
        scope_profiles.remember_microscope("n7")
        scope_profiles.remember_microscope("liara")
        assert scope_profiles.known_microscopes() == ["liara", "n7"]


class TestTheTwoFilesStaySeparate:
    def test_remembering_does_not_write_a_profile(self, isolated_home):
        # "Seen" is not "configured". Merging them would make an untuned
        # instrument look like it has saved settings.
        scope_profiles.remember_microscope("n7")
        assert not (isolated_home / "scope_profiles.json").exists()
        assert scope_profiles.list_profiles() == {}

    def test_the_seen_file_holds_a_plain_sorted_list(self, isolated_home):
        scope_profiles.remember_microscope("zeta")
        scope_profiles.remember_microscope("alpha")
        data = json.loads((isolated_home / "seen_microscopes.json").read_text())
        assert data == ["alpha", "zeta"]
