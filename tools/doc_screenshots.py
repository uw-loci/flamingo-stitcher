"""Make the README screenshots from the real window, one picture per section.

    QT_QPA_PLATFORM=offscreen python tools/doc_screenshots.py [--out docs/images]

What it does:
  1. Points HOME, APPDATA, XDG_CONFIG_HOME and QSettings at a temporary folder,
     so your own saved settings, microscope profiles and window positions are
     neither shown nor changed.
  2. Writes three small synthetic acquisitions (tests/_synth_acq.py): two from
     microscope "n7" with a 6.2x objective and one from "liara" with a different
     objective, so the queue shows Settings sets A and B.
  3. Builds the main window with gui.app.build_main_window() (the same code the
     app runs at launch, without its online update check), queues the three
     acquisitions on the Multi-Acquisition tab (the synthetic data has the
     one-folder-per-tile layout) and clicks 'Discover Tiles'.
  4. Grabs each section as its own PNG.

The pictures use the Qt "Fusion" style and this machine's default font at
9 pt, the size of Windows' Segoe UI, so they look slightly different from the
Windows app; the layout, labels and states are the real ones. The test data
goes in a fixed temporary folder ("FlamingoData") so the paths in the queue
read sensibly; it and the settings folder are deleted at the end. Rerun it after any change to the layout
and commit the PNGs it writes. It prints the Markdown for each picture.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Pictures are rendered at twice the pixel density and shown at half width, so
# text stays sharp on high-resolution screens.
SCALE = 2


def _isolate(tmp: Path) -> None:
    """Send every per-user file the app reads or writes to ``tmp``."""
    for var in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME"):
        os.environ[var] = str(tmp)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ["QT_SCALE_FACTOR"] = str(SCALE)


def _synthetic_acquisitions(root: Path):
    sys.path.insert(0, str(REPO / "tests"))
    from _synth_acq import write_synth_acquisition

    made = []
    sensor_um = 6.5  # the synthetic camera pixel; objective = sensor / pixel
    for name, scope, objective in (
        ("2026-11-03_Sample1_brain", "n7", 6.2),
        ("2026-11-03_Sample2_brain", "n7", 6.2),
        ("2026-11-07_Sample3_liver", "liara", 25.48),
    ):
        acq = write_synth_acquisition(
            root / name, grid=(2, 3), n_planes=24, channels=(1, 3),
            illum_sides=(0, 1), frame_size=(64, 64),
            pixel_size_um=sensor_um / objective, sensor_pixel_size_um=sensor_um,
        )
        settings = acq / "ScopeSettings.txt"
        text = settings.read_text()
        settings.write_text(text.replace(
            "<Scope Settings>\n", f"<Scope Settings>\n  Microscope name = {scope}\n", 1))
        made.append(acq)
    return made


def _quiet_message_boxes(log):
    """Answer any message box instead of waiting for a click that never comes."""
    from PyQt5.QtWidgets import QMessageBox

    def answer(kind):
        def _f(*args, **kwargs):
            title = args[1] if len(args) > 1 else kwargs.get("title", "")
            log(f"  (answered a {kind} box: {title!r})")
            return QMessageBox.Ok if kind != "question" else QMessageBox.Yes
        return _f

    for kind in ("information", "warning", "critical", "question"):
        setattr(QMessageBox, kind, staticmethod(answer(kind)))


def _wait(app, seconds):
    """Run the event loop, including widget deletions processEvents() skips.

    Panels rebuild by deleteLater()-ing their old widgets; the app's event loop
    deletes them before anything is drawn, but a bare processEvents() loop does
    not, and the stale widgets would show through in the pictures.
    """
    from PyQt5.QtCore import QCoreApplication, QEvent

    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        time.sleep(0.01)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(REPO / "docs" / "images"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="flamingo-doc-shots-"))
    _isolate(tmp)
    sys.path.insert(0, str(REPO / "src"))

    from PyQt5.QtCore import QSettings
    from PyQt5.QtWidgets import QApplication

    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(tmp / "qsettings"))
    QApplication.setOrganizationName("Flamingo")
    QApplication.setApplicationName("FlamingoStitcher")
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    font = app.font()
    font.setPointSize(9)  # Windows' default UI font size
    app.setFont(font)

    print(f"Settings and test data in {tmp}")
    _quiet_message_boxes(print)
    data = Path(tempfile.gettempdir()) / "FlamingoData"
    shutil.rmtree(data, ignore_errors=True)
    acqs = _synthetic_acquisitions(data)

    from flamingo_stitcher.gui.app import build_main_window

    window, tabs, _update_panel = build_main_window()
    window.resize(1100, 1500)  # launch width; tall so nothing is scrolled away
    window.show()
    _wait(app, 0.3)

    multi = next(tabs.widget(i) for i in range(tabs.count())
                 if tabs.tabText(i) == "Multi-Acquisition")
    tabs.setCurrentWidget(multi)
    for acq in acqs:
        multi._add_path_to_queue(acq)
    multi._output_dir_edit.setText(str(tmp / "stitched"))
    multi._on_discover()
    multi._queue_table.selectRow(0)
    _wait(app, 0.5)

    shots = []

    def save(widget, name, alt, crop_height=None, top=0):
        img = widget.grab()
        if crop_height is not None:
            img = img.copy(0, int(top * SCALE), img.width(),
                           int((crop_height - top) * SCALE))
        path = out / f"{name}.png"
        img.save(str(path))
        shots.append((path, alt, img.width() // SCALE))

    # The tab strip of the whole window.
    save(window, "tabs",
         "The five tabs across the top of Flamingo Stitcher: Single Workflow, "
         "Multi-Acquisition, Multi-View, Options, Updates",
         crop_height=tabs.tabBar().mapTo(window, tabs.tabBar().rect().bottomLeft()).y() + 6)
    save(multi._queue_table.parentWidget(), "acquisitions",
         "The Acquisitions queue after Discover Tiles: two acquisitions from n7 "
         "share Settings set A, one from liara has set B")
    save(multi._config_container, "basics",
         "Basics: Downsample with its impact line, About the image, Saving and "
         "Memory management")

    # Image quality open; Registration not opened yet, so it glows. Wait for
    # the top of the pulse so the picture shows it.
    st = multi._settings_tabs
    st.setCurrentIndex(0)
    multi._tab_glow.arm()
    multi._tab_glow.acknowledge(0)
    t0 = time.time()
    while multi._tab_glow.strength() < 0.95 and time.time() - t0 < 5:
        app.processEvents()
        time.sleep(0.005)
    multi._tab_glow._overlay.repaint()
    save(st, "image-quality-tab",
         "The Image quality tab; the Registration and overlap tab glows light "
         "blue because it has not been opened since Discover Tiles")
    st.setCurrentIndex(1)
    _wait(app, 0.2)
    save(st, "registration-tab", "The Registration and overlap tab")
    save(multi._run_summary, "this-run",
         "This run: the selected acquisition's settings as tags")

    options = next(tabs.widget(i) for i in range(tabs.count())
                   if tabs.tabText(i) == "Options")
    from flamingo_stitcher import scope_profiles

    for scope in ("n7", "liara"):  # as a run would (temporary HOME)
        scope_profiles.remember_microscope(scope)
    tabs.setCurrentWidget(options)
    _wait(app, 0.3)
    options._reload_scopes(select="n7")
    _wait(app, 0.3)
    save(options.components_box, "optional-components",
         "Optional components on the Options tab: flat-field correction, "
         "deconvolution and the background-zeroing preview")
    # From under 'Optional components' to the end of the first setting.
    first_help = options.components_box.mapTo(
        options, options.components_box.rect().bottomLeft()).y() + 4
    picker = options._scope_combo.parentWidget()
    end = picker.mapTo(options, picker.rect().bottomLeft()).y() + 260
    save(options, "options-tab",
         "The Options tab with microscope n7 selected: registration settings "
         "saved per microscope and objective", crop_height=end, top=first_help)

    window.close()
    shutil.rmtree(data, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)

    print("\nMarkdown:")
    for path, alt, width in shots:
        rel = path.relative_to(REPO) if path.is_relative_to(REPO) else path
        print(f'<img src="{rel.as_posix()}" width="{width}" alt="{alt}">')
    return 0


if __name__ == "__main__":
    sys.exit(main())
