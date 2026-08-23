# Flamingo Stitcher — Run-Log Analysis Guide (for LLM assistants)

**Purpose.** This document gives a language model everything it needs to read a
Flamingo Stitcher run log and advise the user, with no prior context. The
intended use is:

> Paste this guide into a fresh chat, then paste the run log below it and ask
> "what happened / what should I change?"

Be concrete and cite the specific log lines you're reasoning from. When the log
doesn't contain something you need (GPU, total RAM, drive types, whether the
sample fills the field of view), **ask** rather than guess — see
[§9 What to ask](#9-what-to-ask-the-user-when-the-log-is-not-enough).

---

## 1. What Flamingo Stitcher is

A standalone tool that turns a grid of overlapping light-sheet microscope tiles
(a raw Flamingo acquisition) into one large stitched 3-D volume (OME-Zarr,
OME-TIFF, or Imaris). It runs a fixed pipeline and logs each phase. It can run
**in-memory** (fast, needs RAM ≳ output size) or **streaming** (spills tiles and
the fused output to an on-disk scratch folder, bounded RAM, slower). It auto-
selects streaming when the in-memory estimate exceeds ~60% of system RAM.

## 2. The pipeline phases (and the log lines that mark them)

| Phase | What it does | Marker lines |
|---|---|---|
| **Discover** | Find tiles, read the stage grid, frame size, pixel size, channels | `Step 1: … tiles in ~AxB grid`, `Frame size (AOI)`, `Objective (ScopeSettings.txt)`, `Effective XY pixel size: … µm/px (the value stitching will use)` |
| **Preprocess / materialize** | Per tile: fuse L/R illumination sides, optional destripe/flat-field/deconv/downsample; in streaming mode each tile is written once to a scratch memmap | `Materializing N tiles for channel … → …\.stitch_tmp\chNN`, `Ch3: fusing 2 illumination sides (max)`, `Preprocessed N tiles in …s` |
| **Register** | Align tiles. **Often skipped** (stage positions only) | `Step 3: Registering …` **or** `Step 3: Skipping registration — using stage positions only`; then `Registration shift clamp: bounds z=… xy=…`, `Registration SKIPPED: measured tile overlap is only …`, `Z refinement: adjusted N/M tiles`, and the whole `REGISTRATION SHIFT REPORT` block |
| **Border QC** (optional) | Diagnostic pass that flags sharp seams; does not change output | `Running tile-border artifact QC …`, `Border QC: N/M seams flagged` |
| **Fuse** | Combine overlapping tiles into the output volume; in streaming mode fused into an on-disk `fused.dat` memmap in super-block regions | `Step 4: Fusing channel …`, `Auto super-block: fusing in K×K×K-chunk regions`, `Channel …: shape=… — R super-block region(s)` |
| **Write** | Stream the fused volume into the chosen output format (+ pyramid) | `Step 6: Writing multi-channel output …`, `Writing pyramidal OME-TIFF: …`, `.ims write complete`, `Releasing Imaris converter (Destroy)…` / `Imaris converter released.`, `Wrote …\stitch_metadata.json` |
| **Done** | Summary + per-phase time breakdown | `=== Pipeline complete …`, `=== Time breakdown ===` |

## 3. Reading the configuration echo (top of every run)

The header echoes the settings and provenance. Lines worth checking first:

- `Downsample: XY=?x Z=?x` — 1×/1× is full resolution (biggest, slowest).
- `Illumination fusion: max|mean|leonardo` — combines the two light-sheet sides
  **within** a tile (not tile-to-tile).
- `Tile overlap fusion: max|blend` — combines **adjacent tiles**. See [§7](#7-tile-overlap-fusion-max-vs-blend-and-why-seams-look-the-way-they-do).
- `Flat-field correction: True|False` — BaSiC shading + baseline correction.
- `Output format`, `Frame size (AOI)`, `Objective … → effective pixel ~X µm`.
- `Memory estimate: in-memory ~A GB, streaming ~B GB, output ~C GB → <mode>`.
- `Input data on disk: ~X GB across N raw files`.

`Flamingo Stitcher: <version>` and `Key deps:` tell you the code version — some
behaviours below changed across versions; note it.

## 4. The memory estimate and the watchdog — **read this carefully**

This is the most common place to misdiagnose. Two different numbers exist and
they measure different things:

- **The projected number** (`Memory estimate: … streaming ~B GB`) in streaming
  mode is essentially the **preprocess-phase** working set:
  `≈ preprocess_workers × per-tile working set`. A tile's working set is roughly
  `native_tile_uint16 × 1.5`, plus one extra tile-sized buffer each for
  flat-field, non-fast destripe, and (bigger) deconvolution / Z-downsample.
  For a 1024×1024×1662 tile with flat-field and 4 workers that's ~32 GB — and
  that is what the badge/log shows.

- **The watchdog** samples live process memory and warns once if it exceeds
  `projected × 1.5`. It reports the **phase** it fired in.

**What the watchdog measures (changed 2026-08-13):** *private commit* — memory
this process allocated. Memory-mapped FILES are excluded, so the tile spill
(`.stitch_tmp\chNN`) and the fused output (`fused.dat`) no longer inflate it.
That makes the two numbers comparable for the first time: the projection
excludes those memmaps by construction, so the measurement has to as well.

Its message now reads:

```
[memory watchdog] private allocation 6.1 GB exceeded projected 4.0 GB × 1.5
(threshold 6.0 GB) during phase 'fuse' (streaming). ... Separately, 127.7 GB of
memory-mapped scratch (tile spill + fused output) is resident; that is
disk-backed and reclaimable, and is not part of this threshold.
```

**If you are reading an OLDER log** (before 2026-08-13) the watchdog sampled
USS, which on Windows counts modified file-mapped pages — a memmap owned by one
process is not "shared". A 97-tile run reported **127.7 GB against a 9.4 GB
projection** and completed comfortably in 13h15m: that figure was the resident
pages of a 407 GB `fused.dat` plus a 675 GB spill being read back, not
allocation. Flushing per region does **not** fix it — `flush()` writes pages
back and makes them CLEAN, it does not evict them, so they stay resident and
stay counted.

Signs it is the benign mapped-file case (in an old log, or in the "Separately…"
clause of a new one):

- The warning phase is **`fuse` (streaming)** and the run **kept going /
  finished** (a real allocation OOM aborts with `MemoryError` / `Unable to
  allocate …`).
- Output ≫ RAM and a large `fused.dat` / spill is in play.
- The figure is a few× the projection but well under total RAM.

Real allocation pressure to take seriously instead:

- **In-memory mode** watchdog fires — that IS heap; lower resolution or switch
  to streaming.
- The run actually **crashes** with `MemoryError` / `Unable to allocate N GiB`.
- Watchdog fires in **preprocess/register** (not backed by the big memmaps).
- The mapped figure is huge AND the machine becomes unresponsive: that is
  write-back backing up on a slow scratch disk, which is the documented way this
  pipeline freezes a box. Move scratch to fast local NVMe.

## 5. The time breakdown

`=== Time breakdown ===` lists per-phase wall time and share. Use it to find the
bottleneck. Known quirks:

- **`Write output 0s (0.0%)` is usually wrong.** In streaming mode the final
  OME-TIFF/Zarr write reads back the `fused.dat` memmap; that time is often
  attributed to **Fuse** instead of Write. If the log shows `Writing pyramidal …`
  at time T1 and `Wrote …stitch_metadata.json` at T2, the real write took T2−T1
  even if the breakdown says 0s.
- **Fuse dominating (>70%)** on a full-resolution job is expected but points at
  I/O — see [§6](#6-long-run--long-fuse--io-bound).

## 6. Long run / long fuse = I/O-bound

A multi-hour fuse at full resolution is almost always disk-bound, not
compute-bound. In streaming mode the fuse **reads** the whole per-tile spill and
**writes** the whole `fused.dat`, and the final step **re-reads** `fused.dat` to
write the output — a large double-write. Aggravators to look for:

- `Downsample: XY=1x Z=1x` (full res → hundreds of GB output).
- Scratch and output on the **same** or a **slow** drive (`Scratch drive (F:\)`,
  output on `G:\` — check whether they're the same physical disk).
- `Tile overlap fusion: blend` (+ content-based) is heavier than `max`.

Levers (most effective first): put `.stitch_tmp` scratch on a **fast local NVMe**
separate from the output drive; raise **XY/Z downsample** one step (the only lever
that reduces resolution — everything else keeps it); prefer **`max`** overlap
fusion for sparse samples; disable content-based blending. Adding fuse workers
does **not** help an I/O bottleneck.

## 7. Tile-overlap fusion (`max` vs `blend`) and why seams look the way they do

- **`max`** = per-pixel maximum of overlapping tiles → the overlap keeps the
  brighter tile. Good for sparse / sub-field-of-view samples (blend would dilute
  signal against a neighbour's background). Side effect: because `max` of two
  noisy backgrounds is biased upward, empty overlaps can read slightly **bright**.
- **`blend`** = cosine-weighted average, and the weights are **normalized to sum
  to 1** across overlapping tiles. So on truly identical tiles blend is
  mathematically flat (no seam). A **dim band with blend therefore means the two
  tiles genuinely disagree in the overlap** — a per-tile brightness/offset
  difference — which blend renders as a graded dip and `max` renders as a ridge.

**Key diagnostic: seams visible in signal-free background, hard-edged, the same
regardless of overlap %.** That is **not** illumination fall-off and **not** a
blend-geometry artifact — it is a **per-tile intensity offset** (each tile a
slightly different background/gain). No overlap mode fixes it; they only decide
how the mismatch is drawn. Fix hierarchy:

1. **Flat-field correction with darkfield** — removes shading + additive pedestal
   so tiles agree. (Note: a single shared flat-field/darkfield does **not** remove
   a *per-tile* offset that varies tile-to-tile, e.g. from bleaching/laser drift.)
2. **Registration on** — if the mismatch is really structure misaligned at the
   seam (see §8), not brightness.
3. **Per-tile intensity equalization from the overlaps** — the proper fix for a
   residual per-tile offset. *(Not built into the tool as of v0.7.1; if the user
   needs it, say so rather than implying a setting exists.)*

## 8. Reading the Border-QC report

If enabled, the QC prints flagged seams worst-first. Each line looks like:

```
 1. <tileA>  <->  <tileB>   [Y-seam]
       border length: 198 px (208 um)   median step: 44 counts   overlap: 103 px
       aligned shift: (dz=0, ds=8)
```

Interpretation:

- **`median step` (counts)** — the intensity jump across the seam. Small
  (tens of counts) on **well-aligned** seams = the per-tile **intensity offset**
  from §7. Large (hundreds) usually means structure is misaligned.
- **`aligned shift: (dz, ds)`** — the lateral offset the detector had to apply to
  line the two tiles up. **A recurring non-zero `ds` means systematic
  misregistration.** The search is capped (roughly `±max(8, 6µm/pixel)` px); if
  many seams report `ds` exactly at that cap (e.g. `ds=8` when the cap is 8), the
  true offset is **≥ that value, railed at the limit** — a real placement error.
- **Directional skew** — if flagged seams are overwhelmingly **Y-seams** (or
  X-seams), suspect a systematic axis problem (stage pitch or camera angle on
  that axis), not random noise.
- **Registration context** — if the header shows registration was **skipped**,
  those `ds` offsets are **uncorrected**; turning registration on may remove the
  worst seams. If registration was on and offsets persist, it's a harder
  geometry/rotation issue. **Do not guess which** — read
  `registration_report.txt` in the output folder, which says outright whether
  registration ran and what it did (§7b).
- **Sensitivity caveat** — heavy `downsample_xy` softens a true 1-native-pixel
  step; QC is most sensitive at `downsample_xy ≤ 2`. Few flags at `xy=4` doesn't
  prove clean seams.

So a typical mixed report reads as: the **big-step, `ds`-railed** seams are a
**placement** problem (fix with registration / stage-pitch / camera-angle), while
the **small-step, aligned** seams are the **per-tile intensity offset** (fix with
flat-field / intensity equalization).

## 8b. Reading the registration report

Written into the **output folder** (beside `stitch_metadata.json`), not next to
the log, because it describes that store and travels with it. Also echoed into
the run log in full, so an old log still has it.

- **`registration_report.txt`** — the summary. Read the header first: it has
  **three** states, not two. *DID NOT RUN* (registration was never attempted),
  *RAN but its result was NOT APPLIED* (it was measured and refused — the seam
  table is the evidence for why), and a normal applied run. `tiles_registered`
  says how many tiles actually entered the graph, which is **not** the tile
  count when the content gate held some out.

  **Brightness is not content.** The gate scores texture relative to noise
  precisely because a bright, featureless volume — agarose, mounting medium —
  is exactly what an intensity threshold mistakes for sample, and it gives
  phase correlation nothing to lock onto. A straight edge (an FEP tube wall)
  is the opposite trap: it scores as structure and is still a poor alignment
  target, because it slides along its own length without changing the
  correlation. That one is caught by the shift bound, not by the content gate.
- **`registration_report.csv`** — one row per tile: `dz/dy/dx` in µm and in
  frames/pixels, per-axis `clamped_*` flags, and `*_before_clamp`.
- **`registration_seams.csv`** — one row per **expected** neighbour pair from
  the stage grid, so a pair registration rejected appears as a row rather than
  as an absence. `status` is one of `registered`, `pruned` (survived quality,
  dropped by global-optimization edge pruning), `below_quality`, `dropped` (no
  edge, and multiview-stitcher does not report why), `not_run`.

**Read it in this order:**

1. **Did it run at all?** `DID NOT RUN` means placement was stage metadata
   alone. Any tile-to-tile offset in that output is stage placement error and
   nothing tried to correct it. This is the first thing to check when tiles do
   not line up.
2. **How many tiles were clamped?** A clamped axis was **not measured** — the
   tile kept its stage position and the true shift is unknown and larger. The
   summary statistics exclude clamped tiles for exactly this reason. A large
   clamped fraction is itself the diagnosis: read `*_before_clamp` and check
   whether the rejected shifts **scale with distance** (pixel-size error), are
   **constant** (overlap error), or **alternate by row** (tile ordering). None
   of those are registration problems.
3. **How many seams were used?** Many `below_quality` on a visibly textured
   sample means the overlap does not correlate — sparse data, or too little
   shared content.
4. **Empty cells are unknown, never zero.** A `quality` of 0.0 means the
   overlap did not correlate; an empty cell means nobody could recover the
   number. They are not the same finding.

## 9. What to ask the user when the log is not enough

The log rarely contains these; ask before committing to a diagnosis:

- Is the sample **dense (fills the field)** or **sparse**? (Decides `max` vs
  `blend`, and whether background bands are dilution vs offset.)
- Was **flat-field** on, and does a **single raw tile** look flat, or does it have
  a bright-centre/dim-edge shading or a nonzero background pedestal?
- **Total system RAM**, and whether **scratch and output are the same physical
  drive** (for I/O diagnosis).
- Was **registration** intentionally skipped?
- **GPU** present? (Relevant only for deconvolution / Leonardo fusion.)
- What does the **artifact actually look like** (a screenshot beats prose):
  bright vs dim seams, in background vs only under signal, one axis vs both.

## 10. Quick symptom → likely cause → advice

| Symptom in the log / image | Most likely cause | Advice |
|---|---|---|
| **Tiles render spaced apart like pips on dice** (not a connected mosaic) | Wrong XY pixel size — tiles are placed by stage µm but drawn at the wrong scale | Read the `Effective XY pixel size: …` line. If it's wrong, the objective wasn't read: **select the folder that contains `ScopeSettings.txt`** (or its parent — discovery descends into dated subfolders), or set a per-microscope `objective_magnification`. A ⚠ "could not read objective" line confirms the fallback was used. |
| **`⚠ Y tiles do not overlap …` / `X tiles do not overlap …`** (full-width or full-height bands with regular blank gaps between rows/columns) | The acquisition stepped that axis farther than one frame covers — usually a **non-square AOI** whose tile step was computed from a square (width-only) field of view during acquisition (the message says "stepped as if the frame were square"). | The gaps are **missing acquired data**, not a stitching error — no setting closes them. If the message says the step ≈ the other axis's coverage, the frame was stepped transposed: try a **90°/transpose tile orientation** (does the tissue go continuous?). If not, re-acquire with a square AOI, or with explicit per-tile geometry. Pixel size in the same log is unrelated here. |
| **Log ends at `.ims write complete` / `Releasing Imaris converter…` and the run never finishes** (GUI keeps spinning) | PyImarisWriter `Destroy()` blocked while closing the file | The `.ims` is already complete and usable. Fixed in v0.9.1+ (Destroy runs under a 20s watchdog, then the run continues) — update; on older builds, the written `.ims` can be opened despite the hang. |
| Watchdog fires in **fuse (streaming)**, run **finishes** | Dirty/resident pages of `fused.dat` + tile spill (write-back lag), not allocation | Usually benign; move scratch to fast local NVMe; recent builds flush per region. Not a resolution problem. |
| Watchdog fires in **in-memory** mode, or run **crashes** with `MemoryError` | Real allocation over RAM | Switch to streaming (or it already is) and/or raise XY/Z downsample; reduce workers; disable content-based. |
| **Fuse = most of the runtime**, hours | I/O-bound full-res double-write | Fast separate scratch disk; downsample a step; `max` over `blend`; don't add fuse workers. |
| **`Write output 0s`** but obvious write time between `Writing …` and `Wrote …` | Phase-timing attributes write to fuse | Cosmetic; compute real write time from the two timestamps. |
| **Preprocess/materialize appears twice** (`qc_chNN` then `chNN`) with border QC on + registration skipped | Border QC materialized a throwaway spill (fixed after v0.7.1) | Update; or disable border QC / enable registration to avoid the double pass on old builds. |
| Border QC: many seams, big `median step`, `ds` railed at the cap, one axis | Systematic misregistration (stage pitch / camera angle), uncorrected because registration skipped | Turn registration on; check that axis's pitch/camera-angle calibration. |
| `Registration SKIPPED: measured tile overlap is only …%` | Tiles overlap less than `min_registration_overlap_frac` (5%). Registration was refused rather than attempted | Not a bug — phase correlation on a sliver returns a confident wrong shift. Re-acquire with ~10% tile overlap. Lower the gate only to diagnose. |
| `REGISTRATION SHIFT REPORT` says **RAN but its result was NOT APPLIED** | Too few seams registered to trust the result, or the registered seams did not connect the mosaic. multiview-stitcher solves each connected group of tiles **independently**, so applying a partly-registered mosaic slides whole blocks past the tiles that stayed at their stage position — output visibly worse than not registering. | Not a bug, and not something to force. Read `registration_seams.csv`: if whole rows/columns are `below_quality` the sample only covers part of the grid, so restrict the run to the tiles that contain it, or lower **Minimum share of seams that must register** for this microscope in the GUI's Options tab (`--min-registered-seams`). |
| `Registration: N tiles had no registered seam and were moved with the mosaic` | Those tiles registered against nothing. Rather than leave them at their stage position while their neighbours moved — which opens a seam by the full size of the correction — they were given the mosaic's consensus shift. | Informational. Their own placement is **not measured**; a shared shift simply opens no seam. If N is large, the sample covers little of the grid. |
| `REGISTRATION SHIFT REPORT` says **DID NOT RUN** | `Skip registration` is on, or the run fell back | Uncheck it. This is the first thing to check when tiles do not line up: placement was stage metadata alone. |
| Many seams `below_quality` in `registration_seams.csv` | The overlap does not correlate: sparse/dim sample, or too little shared content | Lower `quality_threshold` (0.4 default; 0.2 was abandoned as too permissive), or accept stage placement. Many below-quality seams on visibly textured data is the case that would justify interest-point registration. |
| `Tile content: N of M tiles have nothing to register against` / seams with status `no_content` | Those tiles scored below `min_tile_structure` and were held out of the registration graph, then placed with the mosaic. Their seams are excluded from the trust threshold, because there is no visible content there to be discontinuous. | Normal on a sparse mosaic. Check `tile_structure_range` in the report: featureless material scores ~0.09 whatever its brightness, so a range like 0.08–0.85 is a clean separation. If **dim real sample** is being held out, lower **Minimum tile structure** for this microscope (`--min-tile-structure`); 0 registers every tile. |
| Sample is present but `tiles_without_structure` counts it as empty | The score is texture relative to noise, and it falls to the featureless floor at a contrast-to-noise ratio near 1 — at which point there is genuinely nothing for phase correlation to find either. | Not a threshold problem; the tile is too dim to register on. Fix the acquisition (exposure/laser) rather than the threshold, or accept stage placement for those tiles. |
| `below_quality` seams cluster by **grid position** (whole outer columns/rows near 0.0 while the middle scores 0.3+) | Not a registration fault — the sample only covers part of the mosaic and the outer tiles are background. This is the shape that produces a disconnected registration graph. | Restrict the run to the tiles containing sample. Averaging seam quality per tile onto the grid makes this obvious in one glance. |
| Seams with status `implausible_shift`, and `N seams passed the quality threshold but proposed a shift the geometry forbids` | The seam correlated well and proposed a shift larger than the tiles' overlap — at which point they do not overlap at all, so it cannot be a measurement of their relative position. Dropped **before** the global solve so it could not move every tile in its component. | Usually nothing: the guard did its job. A high count means the overlaps are correlating on something repetitive or elongated. Note the fix is **not** lowering `quality_threshold` — a confident wrong peak scores *well*, so that makes it worse. Tighten **Maximum lateral correction** for this microscope (Options tab / `--max-reg-shift`) instead; the automatic bound is one whole overlap. |
| Many tiles `clamped_z`/`clamped_x` in `registration_report.csv` | Corrections exceeded the plausible bound and were reverted to the mosaic's consensus placement — those tiles are **NOT measured** (they are not at their stage position either; a clamped tile keeps the shift the whole mosaic agreed on, which opens no seam) | Read `*_before_clamp`: shifts scaling with distance ⇒ pixel-size error; constant ⇒ overlap error; alternating by row ⇒ tile ordering. None of those are fixed by registration. |
| Z disagreement in `registration_seams.csv` while X/Y is clean | The first pass registers at Z binning 2 and multiview-stitcher upsamples 3-D by only 2 | Set `registration_binning.z` to 1 first (`--reg-binning-z 1`; independent of XY, so it doubles rather than quadruples the work). If that is not enough, turn on `registration_z_refine` (~2-3x registration time). |
| `Z refinement: N corrections came back at the ±… µm search limit` | Those are floors, not measurements; they were rejected | Widen `--z-refine-range-um`, or check tile order — a large fraction usually means a placement error, not a Z error. |
| Border QC: small `median step` on aligned seams | Per-tile intensity offset | Flat-field (with darkfield); consider intensity equalization (not built in). |
| **Hard seams in signal-free background**, same at any overlap %, `blend`=dim / `max`=bright | Per-tile intensity offset (§7) | Flat-field first; no overlap mode fixes it. |
| Blend gives a **dark dip** at seams on a **sparse** sample | Averaging signal against a neighbour's background | Use `max` (default) for sparse / sub-FOV samples. |

## 11. Key knobs (and what they trade)

| Knob | Effect | When to change |
|---|---|---|
| `Downsample XY / Z` | Only lever that lowers resolution → far less RAM, disk, time | When I/O/RAM-bound and full res isn't required |
| `Tile overlap fusion` (`max`/`blend`) | `max` for sparse (avoids dilution); `blend` for dense, well-corrected | Match to sample density |
| `Flat-field correction` | Removes per-tile shading + baseline (BaSiC) | Seams/shading; leave on for quantitative work |
| `Content-based fusion` | Local-variance overlap weighting; 5–10× slower, more RAM (halo) | Only if seams persist after flat-field and you've confirmed it helps |
| `Fuse workers` / `Preprocess workers` | Parallelism vs RAM; capped auto | Lower on tight RAM; raising won't fix I/O |
| `Fusion super-block chunks` | Region size that bounds streaming fuse memory | Rarely; auto-sized to ~4 GB/region |
| Scratch dir (`.stitch_tmp`) | Where spill + `fused.dat` live | Put on a fast local NVMe, separate from output |

---

*Companion docs shipped alongside this one:*
`stitching_hardware_troubleshooting.md` (RAM/disk requirements, stuck-run
diagnostics, symptom→fix index). This guide is versioned with the tool; some
behaviours noted above changed across releases, so always note the
`Flamingo Stitcher: <version>` line in the log.
