"""Compare our intensity-adjustment port with BigStitcher's on one case.

    python3 tools/bigstitcher_intensity/compare.py CASE_DIR JAVA_OUT [JAVA_OUT ...]

Each JAVA_OUT is the text ``IntensityHarness`` wrote for CASE_DIR/case.txt
(several runs show BigStitcher's own run-to-run spread: its solver shuffles).
Prints a JSON verdict: every item must match for the port to be faithful.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from flamingo_stitcher import intensity_adjust as ia


def read_case(case_dir: Path):
    kv, views = {}, []
    for line in (case_dir / "case.txt").read_text().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] == "view":
            x, y, z = int(p[2]), int(p[3]), int(p[4])
            data = np.fromfile(p[5], dtype="<f4").reshape(z, y, x)
            aff = np.array([float(v) for v in p[6:18]]).reshape(3, 4)
            views.append(ia.View(data, aff))
        else:
            kv[p[0]] = p[1]
    params = ia.IntensityAdjustParams(
        downsampling=float(kv["downsampling"]),
        max_matches=int(kv["maxMatches"]),
        affine=int(kv["affine"]) != 0,
        offset_regularization=float(kv["regTrans"]),
        identity_regularization=float(kv["regIdentity"]),
    )
    return views, params


def read_java(path: Path):
    out = {"pairs": {}, "models": {}}
    for line in path.read_text().splitlines():
        p = line.split()
        if not p:
            continue
        if p[0] in ("bb", "scaledbb"):
            v = [int(x) for x in p[1:]]
            out[p[0]] = (v[:3], v[3:])
        elif p[0] == "maporder":
            out["maporder"] = [tuple(int(t) for t in s.split()) for s in line[9:].split(",")]
        elif p[0] == "pair":
            i, j = int(p[1]), int(p[2])
            out["pairs"][(i, j)] = dict(
                before=int(p[3]), after=int(p[4]),
                sums=[float(x) for x in p[5:9]],
            )
        elif p[0] == "model":
            out["models"][int(p[1])] = (float(p[2]), float(p[3]))
    return out


def main():
    case_dir = Path(sys.argv[1])
    javas = [read_java(Path(p)) for p in sys.argv[2:]]
    views, params = read_case(case_dir)
    m = len(views)
    raw, bbox, grid = ia.collect_matches(views, params)
    thinned = ia.thin_matches(raw, m, params.max_matches)
    a, b, connected, _ = ia.solve(thinned, m, params)
    j0 = javas[0]
    verdict = {}
    verdict["bbox"] = [list(map(int, bbox[0])), list(map(int, bbox[1]))] == [list(j0["bb"][0]), list(j0["bb"][1])]
    ds = params.downsampling
    if not (ds == 1.0):
        verdict["scaled_grid"] = [list(map(int, grid[0])), list(map(int, grid[1]))] == [list(j0["scaledbb"][0]), list(j0["scaledbb"][1])]
    verdict["hashmap_order"] = ia.java_hashmap_order(m) == j0["maporder"]
    worst_rel = 0.0
    count_mismatch = []
    for key, jp in j0["pairs"].items():
        ours = thinned[key]
        if (len(raw[key]), len(ours)) != (jp["before"], jp["after"]):
            count_mismatch.append((key, (len(raw[key]), len(ours)), (jp["before"], jp["after"])))
            continue
        if len(ours):
            s = [ours[:, 0].sum(), ours[:, 1].sum(), (ours[:, 0] ** 2).sum(), (ours[:, 0] * ours[:, 1]).sum()]
            for x, y in zip(s, jp["sums"]):
                worst_rel = max(worst_rel, abs(x - y) / max(abs(y), 1e-9))
    verdict["pair_counts_match"] = not count_mismatch
    verdict["count_mismatches"] = count_mismatch[:5]
    verdict["survivor_sums_max_rel_diff"] = worst_rel
    ja = np.array([[jv["models"][i][0] for i in range(m)] for jv in javas])
    jb = np.array([[jv["models"][i][1] for i in range(m)] for jv in javas])
    verdict["java_runs"] = len(javas)
    verdict["java_spread_scale"] = float((ja.max(0) - ja.min(0)).max())
    verdict["java_spread_offset"] = float((jb.max(0) - jb.min(0)).max())
    verdict["max_abs_diff_scale_vs_java_mean"] = float(np.abs(a - ja.mean(0)).max())
    verdict["max_abs_diff_offset_vs_java_mean"] = float(np.abs(b - jb.mean(0)).max())
    verdict["ours"] = {"scale": a.round(6).tolist(), "offset": b.round(4).tolist()}
    verdict["java_mean"] = {"scale": ja.mean(0).round(6).tolist(), "offset": jb.mean(0).round(4).tolist()}
    print(json.dumps(verdict, indent=1))


if __name__ == "__main__":
    main()
