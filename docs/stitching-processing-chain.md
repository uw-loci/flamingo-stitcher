# Stitching processing chain — order of operations

**Status:** current as of flamingo-stitcher v0.10.1 (2026-08-11).
**Companion diagram:** [`stitching-processing-chain.svg`](stitching-processing-chain.svg)
This document covers **ordering only**: what runs when, and why the order is what it is.

## Why this document exists

Every option in the stitching dialog operates on the same voxels, and several of them are
*not* commutative. Destriping a downsampled tile is not the same as downsampling a destriped
one. Deconvolving after illumination fusion is not the same as deconvolving before. Flat-field
applied post-fusion corrects the wrong thing entirely.

As more restoration steps land (deconvolution now, Leonardo FUSE next), "which order?" stops
being an implementation detail and becomes a correctness question a user has to be able to
answer. This is that answer.

---

## The one chain

There is exactly **one** per-tile preprocessing function —
`StitchingPipeline._preprocess_single_tile()` — and both execution modes call it. That is
deliberate and load-bearing: the two modes once had separate copies and drifted, with the
streaming copy dropping the frame-size arguments and loading cropped tiles at the wrong
2048² default. Scrambled data, blank Imaris output. They must never diverge again.

### Per-tile chain (native resolution unless noted)

| # | Step | Function | Gated by | Frame |
|---|------|----------|----------|-------|
| 1 | Load one volume per illumination side | `load_tile_volume` → `load_raw_volume` | always | raw camera |
| 2 | **Destripe (quality)** | `destripe_volume` | `destripe and not destripe_fast` | raw camera |
| 3 | Fuse illumination sides | `fuse_illumination_sides` | >1 side and not `split_illumination` | raw camera |
| 4 | Flat-field correction | `_apply_flatfield_volume` | `flat_field_correction` | raw camera |
| 5 | Depth attenuation | `correct_depth_attenuation` | `depth_attenuation` *(no GUI — see below)* | raw camera |
| 6 | **Deconvolution (quality)** | `_deconvolve_tile` | `deconvolution_enabled and not deconvolution_fast` | raw camera |
| 7 | Downsample | `downsample_volume` | `downsample_xy > 1 or downsample_z > 1` | raw camera |
| 8 | **Destripe (fast)** | `destripe_volume` | `destripe and destripe_fast` | raw camera, downsampled |
| 9 | **Deconvolution (fast)** | `_deconvolve_tile` | `deconvolution_enabled and deconvolution_fast` | raw camera, downsampled |
| 10 | Per-tile orientation | `MosaicOrientation.apply_volume_xy` | always | **→ stage frame** |

### Mosaic chain

| # | Step | Function | Gated by |
|---|------|----------|----------|
| 11 | Overlap gate | `_registration_overlap_gate` | always (refuses registration below `min_registration_overlap_frac`) |
| 11a | Registration, or stage positions | `_register_tiles` / metadata affine | `not skip_registration` |
| 11b | Per-axis shift clamp, relative to the mosaic | `_clamp_registration_shifts` + `_consensus_reference_shift` | always, after registration |
| 11c | Z-refinement pass | `_refine_z_shifts` | `registration_z_refine` |
| 12 | Multi-view rotation placement | `_tile_metadata_affine` → `_rotation_affine_zyx` | `multiview_fusion` |
| 13 | Border QC (diagnostic) | `_run_border_qc_streaming` | `border_qc_enabled` |
| 13a | Intensity adjustment (per-tile `a * v + b`, BigStitcher's method) | `_intensity_adjust_sims` → `intensity_adjust.py` | `intensity_adjustment` = `measure` (log only) or `apply` |
| 14 | Tile-overlap fusion | `_fuse_channel` (multiview-stitcher) | `tile_overlap_fusion`, `content_based_fusion` |
| 15 | Background zeroing | in the dask graph | `background_zero_enabled` |
| 16 | Write + pyramid | format writers | `output_format` |

---

## Why the order is what it is

**Destripe runs in the raw camera frame (step 2), before orientation (step 10).**
The underlying `filter_streaks` is axis-fixed — it removes *horizontal* stripes only. The
per-tile rot/flip in step 10 can swap which image axis the stripes run along, so destriping
after orientation would filter the wrong axis and remove nothing. The direction is *derived*
from the tile orientation, not detected from image content — see
`_resolve_destripe_direction`. This was the v0.9.5 silent-no-op bug.

**Destripe runs per illumination side, before fusion (step 2 before step 3).**
Stripe artifacts originate independently in each light-sheet path. Fusing first and
destriping the combined tile mixes two different stripe patterns, and neither gets removed
cleanly.

**Flat-field runs after fusion, before downsample (step 4).**
It corrects an illumination profile of the *fused* tile at native resolution, per plane.
Applying it post-downsample would correct a profile that no longer matches the optics.

**Downsample sits between the "quality" and "fast" variants (step 7).**
This is the whole reason both variants exist:

- *Quality* (steps 2, 6) run at native resolution: correct, expensive.
- *Fast* (steps 8, 9) run after downsample: far cheaper — fewer voxels, and for
  deconvolution a smaller PSF — but lower quality, because the blur being removed was
  introduced at full resolution and the downsample has already mixed it.

Choosing "fast" is a deliberate speed/quality trade. It is not a different implementation.

**Orientation is per tile, not per mosaic (step 10).**
Each tile's pixels must be aligned to the stage axes *before* placement. Rotating the
finished mosaic cannot fix tiles that don't connect. This was the v0.8.13 finding — the
whole-mosaic approach was wrong.

**Tile-overlap fusion is not illumination fusion (step 14 vs step 3).**
Two distinct controls that are easy to confuse:
- `illumination_fusion` (`max` / `mean` / `split` / `blend` / `content` / `leonardo`) combines the **left/right light-sheet
  paths of one tile**.
- `tile_overlap_fusion` (`max` / `blend` / `brightest`) combines **adjacent tiles** where
  they overlap.

---

## The three stitching modes

All three share the entire chain above. They differ only where noted.

### 1. Multi-Acquisition — `StitchingDialog`

The default. Folder-per-tile discovery (`discover_tiles`), and a batch queue so several
acquisitions run back to back. Everything above applies unchanged.

### 2. Single Workflow — `NativeStitchingDialog`

Subclass of the above. **Only** `_discover_tiles_for_path` changes — it uses
`discover_flat_tiles`, which scans a flat directory of `.raw` files with integer tile
indices (`X000_Y000`) instead of a subfolder per tile. The processing chain is identical.

### 3. Multi-View — `MultiViewStitchingDialog`

Subclass of the base dialog with the same folder-per-tile discovery. Forces
`multiview_fusion = True` and exposes the rotation controls. The only chain difference is
**step 12**: tiles carrying a non-zero rotation-stage angle get a rotation affine about the
vertical Y axis baked into their placement transform, so several angles fuse into one frame.

> ⚠ The rotation sign and centre conventions are validated on synthetic data but **not
> confirmed on the instrument**. Verify with a two-angle test acquisition and flip
> `rotation_sign` if the views come out mirrored.

---

## Execution mode — orthogonal to all of the above

**In-memory** vs **streaming** is chosen by `estimate_memory_usage` (or forced by
`streaming_mode`). It changes *where tiles live*, never what is done to them:

- **In-memory** — all preprocessed tiles held in RAM, then fused.
- **Streaming** — each tile preprocessed once and spilled to a per-tile memmap under
  `.stitch_tmp/`; fusion reads chunks from those files.

`split_illumination` forces streaming, because the split path is only implemented in the
streaming fuse loop.

### Preprocess passes

A run makes **one pass per output unit**, plus possibly one for registration or QC. Watch
this — it is the single biggest driver of runtime:

| Configuration | Passes over every tile |
|---|---|
| Registration on, sides fused | 1 (registration spill is reused for fusion) |
| Registration skipped, QC off, sides fused | 1 |
| Registration skipped, QC on, sides fused | 1 (QC spill reused) |
| Registration skipped, QC on, **sides separate** | **2** (QC spill = first side, reused) |
| *(before v0.9.15)* same, sides separate | 3 — QC's fused spill matched no unit and was discarded |

---

## Performance characteristics worth knowing

**Destriping dominates.** On a 98-tile, 1600-plane, 1024² run it was ~97% of per-tile
preprocessing time (~740 s of ~762 s).

**Destriping does not parallelise.** Measured ~1.5× on 1024² planes, flat past two workers,
with processes no better than threads — it is allocation/memory bound, not CPU bound. So
running *several tiles at once* actively hurts: one tile on 24 threads managed 25.6 planes/s
while four tiles on 6 threads each managed 8.4 in aggregate. Since v0.9.13,
`_pick_preprocess_workers` returns 1 whenever destriping is on. Compare the **aggregate**
throughput line, never the per-tile rate.

---

## Option status

| Option | Config field | Status |
|---|---|---|
| Illumination fusion (max/mean/split/blend/content) | `illumination_fusion` | Shipped |
| Split illumination | `split_illumination` | Shipped (forces streaming) |
| Flat-field (BaSiCPy) | `flat_field_correction` | Shipped — runs in an isolated env |
| Destripe (pystripe) | `destripe`, `destripe_fast` | Shipped — vendored `_pystripe_core` |
| Downsample | `downsample_xy`, `downsample_z` | Shipped |
| Per-tile orientation | `tile_orientation` | Shipped — per microscope, remembered |
| Skip registration | `skip_registration` | Shipped. **Check this first when tiles do not line up** — with it on, placement is stage metadata alone and no correction is attempted |
| Registration report | `registration_report_enabled` | Shipped, ON by default — per-tile and per-seam CSVs + text next to `stitch_metadata.json` |
| Registration map | `registration_map_enabled` | ON by default — `<name>_registration_map.png` beside the report: the microscope's per-tile `*_MP.tif` projections placed exactly as fused, each tile tinted by its Z correction (blue at the run's most negative dz, white at 0, orange at its most positive; opacity grows with |dz|) and labelled with its grid name, dz and how it was placed (carried, clamped), each overlap filled by registration's verdict (agrees / dropped by the solve / below quality / empty) with its quality and, for links the solve did not use, the largest disagreement (measured minus placed, e.g. `Δy -52`). A red X marks every link the solve did not use (dropped, below quality, beyond the bound). A dashed line runs along any row or column join held by one link (red) or by under half of its measured links (amber); the legend lists every join with links kept / measured and its disagreement range. Tile size is set from the label text so it stays legible. CLI `--no-registration-map` |
| Min-overlap registration gate | `min_registration_overlap_frac` | Shipped (5%). Below it, registration is refused rather than attempted: phase correlation on a sliver returns a confident wrong shift |
| Axial shift bound | `max_registration_shift_z_um` | Shipped. Separate from the lateral bound — an XY mosaic has no Z overlap to derive one from |
| Shift bound reference | — | Shipped. Bounds are applied to a tile's disagreement with the mosaic's median correction, not to its displacement from the stage position: a shared offset opens no seam |
| Seam shift choice | `registration_seam_choice`, `registration_min_cell_r` | Shipped 0.13.15, default `cells`: of the shifts one correlation peak stands for (sign flips, overlap-width wraps), keep the one whose band-passed (2-8 µm XY, 5-15 µm Z) overlaps correlate best, instead of multiview-stitcher's SSIM. Seams whose best `cells r` is under 0.1 are dropped before the solve. Falls back to SSIM when the registration voxel is coarser than 4 µm XY. Measured ~10x faster per seam, 100 vs 107 B per compared voxel. YAML `registration.seam_choice` / `min_cell_r`; no GUI control yet |
| Phase-correlation upsampling | `registration_upsample_factor` | Shipped; GUI **Peak refinement**. 0 = leave multiview-stitcher's default, which is **2 for 3-D** (~half a binned voxel): at a 10 µm Z spacing every seam's Z moved in 5 µm steps. 10 resolves 1 µm; measured 1.07-1.18x registration time and no extra memory |
| Global-solve tolerance | `global_opt_abs_tol` | Shipped; GUI **Solve tolerance** (registration px; the log and the dialog give µm). Seams disagreeing with the placement by more are pruned. Higher keeps more seams and averages their disagreement |
| Global-solve model | `registration_solve_transform` | Shipped, **experimental**, off: GUI **Solve model** = Shift + rotation (rigid). Rotations are inferred from where neighbours' overlaps sit; clamp, carry, Z snap and the report work on each tile's CENTRE shift and the rotation goes back on afterwards. Clamped tiles stay shift-only. From 0.13.16 carried tiles follow their neighbours' rotation (each neighbour's whole transform applied to the tile's stage position, averaged); before, they were shift-only at the neighbours' mean shift, 20-40 µm out on Ket_Trial1's rotated rows. Not yet validated across datasets |
| Registration Z spacing | `registration_z_spacing_um` | Shipped (20 µm; 0 = every plane). A physical spacing: planes are averaged in groups of round(spacing / the acquisition's Z voxel), so it means the same at any Z step. Sets the Z precision floor (about half the spacing per seam). Time and RAM scale with the planes compared, and the dialog's cost line, memory estimate and queue time include it. Older configs' `registration_binning.z` is read as N x 10 µm |
| Z refinement pass | `registration_z_refine` | Shipped, opt-in — a second pairwise pass on every plane (XY binned 4x) that keeps only its Z result; costed in the estimates |
| Tile-overlap fusion | `tile_overlap_fusion` | Shipped |
| Content-based blending | `content_based_fusion` | Shipped |
| Blend width | `blending_width_overlap_fraction` | Default 1.0: Blend cross-fades over the whole tile overlap MEASURED from the stage positions in X/Y (0.5 = its middle half, 0 = the fixed `blending_widths`); Z keeps `blending_widths["z"]`. GUI: the fade list beside Tile overlap; CLI `--blend-width-overlap`. Logged as `Blend width: X ... µm (1 x the measured ... µm overlap)` |
| Intensity adjustment | `intensity_adjustment` + `intensity_adjust_*` | Shipped, off by default. A port of BigStitcher's Intensity Adjustment, checked stage by stage against BigStitcher's own code (`tools/bigstitcher_intensity/`). Fits `output = a * input + b` per tile from the intensities overlapping tiles share, after registration and before tile-overlap fusion. **Measure only** logs every tile's `a` and `b` and fuses unchanged. Parameters and defaults are BigStitcher's, set per microscope in the Options tab. Like BigStitcher it uses every shared voxel, background included, and works on the tiles after the two illumination sides were merged, so it does not match one side against the other. From 0.13.15 **Keep the input brightness range** (on by default) divides out the scale and offset every tile shares after the fit: BigStitcher's fit scales all tiles down together on noisy overlaps (a = 0.3-0.5 on n7, 2026-10-22, turning raw 100-170 into ~40-80), which BigStitcher hides by rescaling at export and we would write straight to uint16. Seams match exactly as fitted; the median tile stays as acquired |
| Background zeroing | `background_zero_enabled` | Shipped |
| Border QC | `border_qc_enabled` | Shipped (diagnostic only). Its seam-alignment search is bounded by `border_qc_max_shift_px` (default 8); a shift AT the limit is a floor, not a measurement, and the report says so since v0.9.17 |
| Deconvolution | `deconvolution_enabled` | Shipped, GPU Richardson-Lucy |
| Multi-view rotation | `multiview_fusion` | Shipped, **rig validation pending** |
| Depth attenuation | `depth_attenuation` | **Backend + CLI only — no GUI.** Removed 2026-04-23 as geometry-unaware (wrong axis for TSPIM). Decide: delete or redesign. |
| Leonardo FUSE | `illumination_fusion = "leonardo"` | **Partial** — accepted as a value; the isolated-env integration is not finished |
| Interest-point registration | — | **Planned** — design only. `registration_seams.csv` is the measurement that would justify starting it: many `below_quality` seams on textured data is the signal |
| Global tile-position optimisation | — | **Planned** |

---

## Where to add a new processing step

Decide these three things, in this order:

1. **Which frame does it need?** Raw camera (before step 10) or stage-aligned (after)?
   Anything axis-sensitive — like destriping — must state this explicitly, and derive its
   axis rather than detect it.
2. **Before or after downsample?** If it removes an artefact introduced at full resolution,
   it belongs before (and probably wants a `_fast` variant for people who accept the
   trade).
3. **Per tile or per mosaic?** Per-tile steps go in `_preprocess_single_tile` and cost RAM
   × concurrent tiles. Per-mosaic steps go in the fuse loop and cost RAM × concurrent
   chunks.

Then add it to `build_timing_key` — a step that changes runtime but not the ETA cache key
makes the estimator average incompatible configurations together, which is exactly how the
ETA went 2× wrong in v0.9.12.
