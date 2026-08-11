"""A CLI default must never quietly outrank the config it is passed into.

``main()`` builds a ``StitchingConfig`` by handing it ``args.<flag>`` values. Any
flag with a literal argparse default is therefore applied on *every* run,
whether or not the user typed it — which makes that literal, not the dataclass
field and not ``stitching_config.yaml``, the real default.

``--quality-threshold`` spent its life doing exactly that: argparse said 0.2,
the dataclass says 0.4, and the dataclass comment explains that 0.2 was
abandoned because "low-content tiles clear a 0.2 correlation with a garbage
shift". Every CLI stitch ran at the value the code calls out as wrong.

So: value flags default to None, and are applied as an overlay only when given.

Run: python3 -m pytest tests/test_cli_defaults.py -q
"""

import argparse

import pytest

from flamingo_stitcher.__main__ import build_parser
from flamingo_stitcher.pipeline import StitchingConfig


def _action(parser: argparse.ArgumentParser, option: str) -> argparse.Action:
    for action in parser._actions:
        if option in action.option_strings:
            return action
    raise AssertionError(f"{option} is not a flag on this parser")


class TestQualityThreshold:
    def test_the_flag_does_not_carry_its_own_default(self):
        assert _action(build_parser(), "--quality-threshold").default is None

    def test_an_unspecified_flag_leaves_the_config_default_alone(self):
        args = build_parser().parse_args(["/nonexistent"])
        assert args.quality_threshold is None
        assert StitchingConfig().quality_threshold == pytest.approx(0.4)

    def test_an_explicit_flag_is_still_honoured(self):
        args = build_parser().parse_args(["/nonexistent", "--quality-threshold", "0.6"])
        assert args.quality_threshold == pytest.approx(0.6)

    def test_the_help_text_names_the_value_that_actually_applies(self):
        # A help string advertising a default the code does not use is the same
        # defect one layer up — it sends the reader to the wrong number.
        assert "0.4" in (_action(build_parser(), "--quality-threshold").help or "")


class TestNoOtherFlagShadowsItsConfigField:
    """The same defect, swept for across the whole parser.

    For every flag whose destination matches a StitchingConfig field, a literal
    argparse default must agree with the dataclass — otherwise typing nothing
    silently changes behaviour. Flags that deliberately have no config twin
    (paths, --verbose, per-acquisition geometry) are not in the intersection and
    are skipped automatically.
    """

    # dest -> why the mismatch is intentional.
    _EXEMPT = {
        # --pixel-size-um is a manual override; its absence is signalled
        # separately via auto_pixel_size, so the literal fallback is deliberate.
        "pixel_size_um",
    }

    def test_every_literal_default_matches_its_dataclass_field(self):
        config_defaults = vars(StitchingConfig())
        mismatched = []
        for action in build_parser()._actions:
            dest = action.dest
            if dest in self._EXEMPT or dest not in config_defaults:
                continue
            if action.default is None or isinstance(action, argparse._StoreTrueAction):
                continue
            if action.default != config_defaults[dest]:
                mismatched.append(
                    f"--{dest.replace('_', '-')}: argparse={action.default!r} "
                    f"config={config_defaults[dest]!r}"
                )
        assert not mismatched, (
            "these flags override the config default on every run:\n  "
            + "\n  ".join(mismatched)
        )
