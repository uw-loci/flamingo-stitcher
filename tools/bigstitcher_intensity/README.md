# Checking the intensity adjustment against BigStitcher

`src/flamingo_stitcher/intensity_adjust.py` is a port of BigStitcher's
**Intensity Adjustment** (multiview-reconstruction 9.1.1:
`Intensity_Adjustment` -> `IntensityAdjustmentTools`). This folder runs
BigStitcher's own code on the same input as the port and compares the two.
The results are saved in `tests/data/bigstitcher/`, and
`tests/test_intensity_adjust.py` checks the port against them, so CI needs no
Java.

## What is compared

| Stage | BigStitcher | Compared by |
|---|---|---|
| Bounding box, scaled grid | `BoundingBoxMaximal`, `TransformVirtual` (real classes) | exact |
| Sampling (linear, -1 outside) | `TransformView.transformView` (real class) | per-pair match counts exact; sums of kept samples to 1e-6 |
| Pair order | `HashMap<ValuePair>` iteration (real classes) | exact |
| Thinning to `max_matches - 1` | `Random(344)` + `ArrayList.remove` (copied verbatim) | exact survivors |
| Solve + min-offset shift | `IntensityAdjustmentTools.runGlobal` + mpicbg 1.6.6 (real) | to 1e-5 in scale, 0.01 counts in offset |

`IntensityHarness.java` copies only the loop that collects matches, because
upstream it is tied to SpimData. Everything else it calls is BigStitcher's own
class. BigStitcher's solver shuffles the tile order on every sweep, so its runs
differ slightly; the port solves for the configuration those sweeps converge to.

## Rerunning it

Needs a JDK (17+) and these jars, the versions BigStitcher 9.1.1 builds
against (from `maven.scijava.org`, imglib2 from Maven Central):
`multiview-reconstruction-9.1.1`, `mpicbg-1.6.6`, `imglib2-8.0.0`,
`imglib2-realtransform-4.0.5`, `spim_data-2.3.5`, `ij-1.54p`,
`bigdataviewer-core-10.6.12`.

```
CP=<colon-separated jar paths>
javac -cp "$CP" -d build IntensityHarness.java JavaBits.java
python3 make_case.py /tmp/case --downsampling 2 --max-matches 300
for r in 1 2 3; do java -cp "build:$CP" IntensityHarness /tmp/case/case.txt /tmp/case/java_$r.txt; done
PYTHONPATH=../../src python3 compare.py /tmp/case /tmp/case/java_{1,2,3}.txt
java -cp "build:$CP" JavaBits > ../../tests/data/bigstitcher/java_bits.json
```

`compare.py` prints one JSON verdict. Every "match" item must be true.
