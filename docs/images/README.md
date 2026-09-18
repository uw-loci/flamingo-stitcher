# Screenshots the documentation is waiting for

Each entry below has a matching `<!-- SCREENSHOT NEEDED -> ... -->` block in the
doc that wants it. Drop the file in with the exact name, then delete the two
comment markers around the `![...]` line so the image renders.

| File | Capture | Appears in |
|---|---|---|
| `tabs.png` | The app window with all five tabs readable along the top | README, "Stitch your first dataset" |
| `layout-single-workflow.png` | Explorer on a Single Workflow acquisition — image files loose in one folder, names like `S000_t000000_V000_R0000_X000_Y000_C03_I0_D0_P00643.raw` | README, "Which tab do I use?" |
| `layout-multi-acquisition.png` | Explorer on a Multi-Acquisition dataset — one subfolder per tile, named `X4.00_Y12.00`, inside a date folder | README, "Which tab do I use?" |
| `options-tab.png` | The Options tab with a microscope selected and its settings visible | README, "The other two tabs" |
| `registration-report.png` | `registration_report.txt` in a text editor, with the `Seams:` summary line visible | README, "Limitations" |

Two things worth doing when you capture these:

Frame the two layout shots so the difference is obvious at a glance: loose
files in one, per-tile folders in the other. Those two answer the question the
tab names cannot.

Crop out anything identifying: sample names, patient or collaborator
identifiers, full user paths. A window cropped to the file list is better than a
full desktop.

## Alt text

Each commented `![...]` line already carries alt text describing what the
picture shows, so screen readers get the same information. If you reframe a shot
substantially, update the alt text to match.
