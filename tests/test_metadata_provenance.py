"""stitch_metadata.json must record which stitcher produced the output.

The ``"version": 2`` field is the metadata *schema* version — it does NOT
identify the software. These tests assert the software provenance
(``stitcher_version`` etc.) is present both in the helper and in a real run's
metadata, so a stitched output can always be traced back to a stitcher build.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import multiview_stitcher  # noqa: F401
    import tifffile  # noqa: F401

    from _synth_acq import write_synth_acquisition
    from flamingo_stitcher import __version__ as FS_VERSION
    from flamingo_stitcher.pipeline import (
        StitchingConfig,
        StitchingPipeline,
        stitcher_provenance,
    )

    _HAVE = True
except Exception:
    _HAVE = False


@unittest.skipUnless(_HAVE, "needs multiview_stitcher + tifffile")
class TestProvenanceHelper(unittest.TestCase):
    def test_helper_records_version_and_build(self):
        prov = stitcher_provenance()
        self.assertEqual(prov["stitcher_version"], FS_VERSION)
        self.assertIn(prov["stitcher_build"], ("frozen", "source"))


@unittest.skipUnless(_HAVE, "needs multiview_stitcher + tifffile")
class TestMetadataProvenanceEndToEnd(unittest.TestCase):
    def test_run_writes_stitcher_version_into_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            acq = write_synth_acquisition(
                Path(d) / "acq",
                grid=(2, 2),
                n_planes=6,
                channels=(1,),
                frame_size=(32, 32),
                overlap=0.2,
            )
            out = Path(d) / "out"
            cfg = StitchingConfig.with_yaml_defaults()
            cfg.skip_registration = True
            cfg.streaming_mode = True
            cfg.output_format = "ome-tiff"
            cfg.resource_guard_enabled = False
            StitchingPipeline(cfg).run(acq, out)

            meta = json.loads((out / "stitch_metadata.json").read_text())
            # Schema version and software version are distinct facts.
            self.assertEqual(meta["version"], 2)
            self.assertEqual(meta["stitcher_version"], FS_VERSION)
            self.assertIn(meta["stitcher_build"], ("frozen", "source"))


if __name__ == "__main__":
    unittest.main()
