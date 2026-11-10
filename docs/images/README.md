# Documentation pictures

## Made by the script

These are cropped from the real window by `tools/doc_screenshots.py`, on three
small synthetic acquisitions (two from microscope n7, one from liara), with your
own settings kept out of it. Rerun it after any change to the layout and commit
the PNGs it rewrites:

```bash
QT_QPA_PLATFORM=offscreen python tools/doc_screenshots.py
```

| File | Shows | Appears in |
|---|---|---|
| `tabs.png` | The five tabs along the top of the window | README, "Stitch your first dataset" |
| `options-tab.png` | The Options tab with n7 selected, down to the first setting | README, "The other two tabs" |
| `acquisitions.png` | The queue after Discover Tiles, with Settings sets A and B | README, "Understanding the settings" (the queue) |
| `basics.png` | The Basics box | README, "Understanding the settings" (Basics) |
| `image-quality-tab.png` | The Image quality tab; Registration and overlap glows (not opened yet) | README, "Understanding the settings" (the two tabs) |
| `registration-tab.png` | The Registration and overlap tab | README, same fold |
| `this-run.png` | The This run tags | README, "Understanding the settings" |
| `optional-components.png` | Optional components on the Options tab | README, "Understanding the settings" |

They use the Qt "Fusion" style at 9 pt, so they differ a little from the Windows
app in look, not in layout or labels. They are rendered at twice the pixel
density and shown at a set width, so text stays sharp. The synthetic data is
tiny, so 'Raw size' reads 0 GB.

## Still to capture by hand

Each has a matching `<!-- SCREENSHOT NEEDED -> ... -->` block in the README.
Drop the file in with the exact name, then delete the two comment markers around
the `![...]` line so the image renders.

| File | Capture | Appears in |
|---|---|---|
| `layout-single-workflow.png` | Explorer on a Single Workflow acquisition — image files loose in one folder, names like `S000_t000000_V000_R0000_X000_Y000_C03_I0_D0_P00643.raw` | README, "Which tab do I use?" |
| `layout-multi-acquisition.png` | Explorer on a Multi-Acquisition dataset — one subfolder per tile, named `X4.00_Y12.00`, inside a date folder | README, "Which tab do I use?" |
| `registration-report.png` | `registration_report.txt` in a text editor, with the `Seams:` summary line visible | README, "Limitations" |

Frame the two layout shots so the difference is obvious at a glance: loose
files in one, per-tile folders in the other. Those two answer the question the
tab names cannot.

Crop out anything identifying: sample names, patient or collaborator
identifiers, full user paths. A window cropped to the file list is better than a
full desktop.

## Alt text

Every picture carries alt text describing what it shows, so screen readers get
the same information. If you reframe a shot substantially, update the alt text
to match.
