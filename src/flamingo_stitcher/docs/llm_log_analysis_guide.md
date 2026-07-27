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
| **Register** | Align tiles. **Often skipped** (stage positions only) | `Step 3: Registering …` **or** `Step 3: Skipping registration — using stage positions only` |
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

**Critical nuance for streaming mode:** the fused output (`fused.dat`, can be
hundreds of GB) and the per-tile spill (`.stitch_tmp\chNN`, can be hundreds of
GB) are **memory-mapped files on disk** — deliberately "off-RAM." But the OS
keeps their **dirty/resident pages in the process working set** until write-back.
On Windows, `psutil` USS counts modified file-mapped pages. So during **fuse**,
the watchdog can report a large number (e.g. "127 GB") that is mostly
**write-back lag on the scratch disk, not algorithm allocation.**

Signs it's the benign memmap-working-set case, not a real allocation problem:

- The warning phase is **`fuse` (streaming)** and the run **kept going / finished**
  (a real allocation OOM aborts with `MemoryError` / `Unable to allocate …`).
- Output ≫ RAM and a large `fused.dat` / spill is in play.
- The reported figure is a few× the projection but well under total RAM.

Real allocation pressure to take seriously instead:

- **In-memory mode** watchdog fires — that IS heap; lower resolution or switch to
  streaming.
- The run actually **crashes** with `MemoryError` / `Unable to allocate N GiB`.
- Watchdog fires in **preprocess/register** (not backed by the big memmaps).

> Recent versions flush the fused memmap after each super-block region to bound
> those dirty pages, which quiets the spurious fuse-phase warning. If you see the
> old large fuse-phase figure on an older build and the run completed, treat it as
> I/O/write-back, not an allocation bug.

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
  geometry/rotation issue.
- **Sensitivity caveat** — heavy `downsample_xy` softens a true 1-native-pixel
  step; QC is most sensitive at `downsample_xy ≤ 2`. Few flags at `xy=4` doesn't
  prove clean seams.

So a typical mixed report reads as: the **big-step, `ds`-railed** seams are a
**placement** problem (fix with registration / stage-pitch / camera-angle), while
the **small-step, aligned** seams are the **per-tile intensity offset** (fix with
flat-field / intensity equalization).

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
| **Log ends at `.ims write complete` / `Releasing Imaris converter…` and the run never finishes** (GUI keeps spinning) | PyImarisWriter `Destroy()` blocked while closing the file | The `.ims` is already complete and usable. Fixed in v0.9.1+ (Destroy runs under a 20s watchdog, then the run continues) — update; on older builds, the written `.ims` can be opened despite the hang. |
| Watchdog fires in **fuse (streaming)**, run **finishes** | Dirty/resident pages of `fused.dat` + tile spill (write-back lag), not allocation | Usually benign; move scratch to fast local NVMe; recent builds flush per region. Not a resolution problem. |
| Watchdog fires in **in-memory** mode, or run **crashes** with `MemoryError` | Real allocation over RAM | Switch to streaming (or it already is) and/or raise XY/Z downsample; reduce workers; disable content-based. |
| **Fuse = most of the runtime**, hours | I/O-bound full-res double-write | Fast separate scratch disk; downsample a step; `max` over `blend`; don't add fuse workers. |
| **`Write output 0s`** but obvious write time between `Writing …` and `Wrote …` | Phase-timing attributes write to fuse | Cosmetic; compute real write time from the two timestamps. |
| **Preprocess/materialize appears twice** (`qc_chNN` then `chNN`) with border QC on + registration skipped | Border QC materialized a throwaway spill (fixed after v0.7.1) | Update; or disable border QC / enable registration to avoid the double pass on old builds. |
| Border QC: many seams, big `median step`, `ds` railed at the cap, one axis | Systematic misregistration (stage pitch / camera angle), uncorrected because registration skipped | Turn registration on; check that axis's pitch/camera-angle calibration. |
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
