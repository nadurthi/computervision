#!/usr/bin/env python3
"""Aggregate all jsonl results into results/RESULTS.md, results/summary.json and figures in results/figures/.

  python scripts/make_report.py --results results/ [--data-root /data]   (data root only for qualitative panels)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mvsbench.metrics import aggregate  # noqa: E402

STEREO_MAIN = {"middeval3": "bad2", "eth3d_two_view": "bad1", "kitti2015": "d1", "middlebury2014": "bad2"}
MVS_MAIN = {"dtu": "within2", "eth3d": "within0.05", "blendedmvs": "inlier1.03"}
MVS_REGIONS = ["all", "vis_partner", "occ_partner", "oov_partner", "rescued", "unseen"]


def read_jsonl(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        for ln in open(p):
            try:
                rows.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return rows


def fmt(v, nd=2):
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def md_table(header, rows):
    out = "| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n"
    for r in rows:
        out += "| " + " | ".join(str(x) for x in r) + " |\n"
    return out


def stereo_section(rows):
    md, summary = "## Two-view stereo\n\n", {}
    agg = aggregate(rows, ["dataset", "method", "region"])
    times = aggregate([dict(r, region="t") for r in rows if r["region"] == "all"], ["dataset", "method"])
    for ds in sorted({r["dataset"] for r in agg}):
        key = STEREO_MAIN.get(ds, "bad2")
        methods = sorted({r["method"] for r in agg if r["dataset"] == ds})
        tab = []
        for m in methods:
            g = {r["region"]: r for r in agg if r["dataset"] == ds and r["method"] == m}
            t = next((r.get("time_s") for r in times if r["dataset"] == ds and r["method"] == m), None)
            tab.append([m] + [fmt(g.get(reg, {}).get(key)) for reg in ("all", "noc", "occ")] +
                       [fmt(g.get(reg, {}).get("epe")) for reg in ("noc", "occ")] +
                       [fmt(g.get("all", {}).get("density", float("nan")) * 100, 1), fmt(t, 3)])
            summary.setdefault(ds, {})[m] = {reg: g.get(reg, {}) for reg in ("all", "noc", "occ")}
        tab.sort(key=lambda r: float(r[1]) if r[1] != "–" else 1e9)
        md += f"### {ds} ({key} %, lower is better; missing predictions count as errors)\n\n"
        md += md_table(["method", f"{key} all", f"{key} non-occ", f"{key} occluded", "EPE non-occ", "EPE occ",
                        "density %", "time s"], tab) + "\n"
    return md, summary


def mvs_section(rows):
    md, summary = "## Multi-view depth\n\n", {}
    agg = aggregate(rows, ["dataset", "method", "nviews", "region"])
    for ds in sorted({r["dataset"] for r in agg}):
        key = MVS_MAIN.get(ds, "inlier1.03")
        sub = [r for r in agg if r["dataset"] == ds]
        ns = sorted({r["nviews"] for r in sub})
        n_main = 5 if 5 in ns else ns[-1]
        methods = sorted({r["method"] for r in sub})
        md += f"### {ds}: {key} (% of GT pixels, higher is better) at N={n_main} (stereo methods: N=2)\n\n"
        tab = []
        for m in methods:
            n_use = 2 if m.startswith("stereo_") else n_main
            g = {r["region"]: r for r in sub if r["method"] == m and r["nviews"] == n_use}
            if not g:
                continue
            tab.append([m, n_use] + [fmt(g.get(reg, {}).get(key), 1) for reg in MVS_REGIONS] +
                       [fmt(g.get("all", {}).get("abs_rel")), fmt(g.get("all", {}).get("time_s"), 2)])
        tab.sort(key=lambda r: -float(r[2]) if r[2] != "–" else 1e9)
        md += md_table(["method", "N", "all", "visible in partner", "occluded in partner", "outside partner",
                        "rescued by other views", "unseen by all", "AbsRel % (all)", "time s"], tab) + "\n"
        md += f"#### {ds}: {key} on pixels the stereo partner cannot see (occluded or outside), vs number of views\n\n"
        tab2 = []
        for m in methods:
            vals = []
            for n in ns:
                ro = [r for r in sub if r["method"] == m and r["nviews"] == n and r["region"] in ("occ_partner", "oov_partner")]
                if ro:
                    tot = sum(r["n"] for r in ro)
                    vals.append(fmt(sum(r[key] * r["n"] for r in ro if key in r) / max(tot, 1), 1))
                else:
                    vals.append("–")
            tab2.append([m] + vals)
            summary.setdefault(ds, {})[m] = {str(n): {r["region"]: r for r in sub if r["method"] == m and r["nviews"] == n}
                                             for n in ns}
        md += md_table(["method"] + [f"N={n}" for n in ns], tab2) + "\n"
    return md, summary


def pointcloud_section(rows):
    if not rows:
        return "", {}
    md = "## DTU point clouds (N=5 per view, same fusion for all; mm, lower is better)\n\n"
    tab, summary = [], {}
    for m in sorted({r["method"] for r in rows}):
        rs = [r for r in rows if r["method"] == m]
        acc, comp = np.mean([r["acc_mm"] for r in rs]), np.mean([r["comp_mm"] for r in rs])
        tab.append([m, len(rs), fmt(acc, 3), fmt(comp, 3), fmt((acc + comp) / 2, 3)])
        summary[m] = dict(scans=len(rs), acc=acc, comp=comp, overall=(acc + comp) / 2)
    tab.sort(key=lambda r: float(r[4]))
    return md + md_table(["method", "scans", "accuracy", "completeness", "overall"], tab) + "\n", summary


def latency_section(rows):
    """Median latency per method / dataset / N from scripts/run_latency.py (warm, CUDA-synchronised)."""
    if not rows:
        return "", {}
    gpus = sorted({r.get("gpu", "?") for r in rows})
    reps = sorted({r.get("repeats", "?") for r in rows})
    md = (f"## Latency to estimate one depth map ({', '.join(gpus)}; 1 warm-up + {'/'.join(map(str, reps))} timed runs per input; "
          "median ms, p90 in parentheses; peak = GPU memory)\n\n")
    summary = {}
    for kind in ("mvs", "stereo"):
        sub = [r for r in rows if r["kind"] == kind]
        for ds in sorted({r["dataset"] for r in sub}):
            rr = [r for r in sub if r["dataset"] == ds]
            ns = sorted({r["nviews"] for r in rr})
            tab = []
            for m in sorted({r["method"] for r in rr}):
                mr = [r for r in rr if r["method"] == m]
                cells = []
                for n in ns:
                    x = [r for r in mr if r["nviews"] == n]
                    if x:
                        med = float(np.median([r["median_s"] for r in x])) * 1000
                        p90 = float(np.median([r["p90_s"] for r in x])) * 1000
                        cells.append(f"{med:.0f} ({p90:.0f})")
                        summary.setdefault(kind, {}).setdefault(ds, {}).setdefault(m, {})[str(n)] = dict(
                            median_ms=med, p90_ms=p90)
                    else:
                        cells.append("–")
                peaks = [r["peak_mb"] for r in mr if r.get("peak_mb") is not None]
                loads = [r["load_s"] for r in mr if r.get("load_s") is not None]
                tab.append([m, fmt(mr[0]["megapixels"], 2)] + cells +
                           [fmt(max(peaks), 0) if peaks else "–", fmt(loads[0], 1) if loads else "–"])
            first = 2
            tab.sort(key=lambda r: float(r[first].split()[0]) if r[first] != "–" else 1e12)
            head = ["method", "MP"] + ([f"N={n} ms" for n in ns] if kind == "mvs" else ["ms"]) + ["peak MB", "load s"]
            md += f"### {kind} · {ds}\n\n" + md_table(head, tab) + "\n"
    return md, summary


def figures(mvs_rows, stereo_rows, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(fig_dir, exist_ok=True)
    agg = aggregate(mvs_rows, ["dataset", "method", "nviews", "region"])
    for ds in sorted({r["dataset"] for r in agg}):
        key = MVS_MAIN.get(ds, "inlier1.03")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, regs, title in ((axes[0], ("all",), "all GT pixels"),
                                (axes[1], ("occ_partner", "oov_partner"), "pixels the stereo partner cannot see")):
            for m in sorted({r["method"] for r in agg if r["dataset"] == ds}):
                pts = []
                for n in sorted({r["nviews"] for r in agg if r["dataset"] == ds}):
                    ro = [r for r in agg if r["dataset"] == ds and r["method"] == m and r["nviews"] == n and r["region"] in regs]
                    if ro and all(key in r for r in ro):
                        pts.append((n, sum(r[key] * r["n"] for r in ro) / sum(r["n"] for r in ro)))
                if not pts:
                    continue
                xs, ys = zip(*pts)
                style = "--" if m.startswith("stereo_") else ("-" if m in ("planesweep_mean", "planesweep_bestk", "planesweep_bestk_sgm", "patchmatch", "acmmp") else "-.")
                ax.plot(xs, ys, style, marker="o", label=m)
            ax.set_xticks(sorted({r["nviews"] for r in agg if r["dataset"] == ds}))
            ax.set_title(f"{ds}: {title}")
            ax.set_xlabel("views N (reference + N-1 sources)")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel(f"{key} (%)")
        axes[1].legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"nview_sweep_{ds}.png"), dpi=130)
        plt.close(fig)
    sagg = aggregate(stereo_rows, ["dataset", "method", "region"])
    for ds in sorted({r["dataset"] for r in sagg}):
        key = STEREO_MAIN.get(ds, "bad2")
        ms = sorted({r["method"] for r in sagg if r["dataset"] == ds})
        noc = [next((r[key] for r in sagg if r["dataset"] == ds and r["method"] == m and r["region"] == "noc"), np.nan) for m in ms]
        occ = [next((r[key] for r in sagg if r["dataset"] == ds and r["method"] == m and r["region"] == "occ"), np.nan) for m in ms]
        x = np.arange(len(ms))
        fig, ax = plt.subplots(figsize=(max(6, len(ms) * 0.9), 4))
        ax.bar(x - 0.2, noc, 0.4, label="non-occluded")
        ax.bar(x + 0.2, occ, 0.4, label="occluded")
        ax.set_xticks(x, ms, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel(f"{key} % (lower is better)")
        ax.set_title(f"{ds}: stereo error inside vs outside occlusions")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"stereo_occlusion_{ds}.png"), dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    s_rows = read_jsonl(os.path.join(a.results, "stereo", "*.jsonl"))
    m_rows = read_jsonl(os.path.join(a.results, "mvs", "*.jsonl"))
    p_rows = read_jsonl(os.path.join(a.results, "pointcloud", "*.jsonl"))
    l_rows = read_jsonl(os.path.join(a.results, "latency", "*.jsonl"))
    md = "# Results\n\nGenerated by `scripts/make_report.py`.\n\n"
    s_md, s_sum = stereo_section(s_rows) if s_rows else ("", {})
    m_md, m_sum = mvs_section(m_rows) if m_rows else ("", {})
    p_md, p_sum = pointcloud_section(p_rows)
    l_md, l_sum = latency_section(l_rows)
    md += m_md + p_md + s_md + l_md
    open(os.path.join(a.results, "RESULTS.md"), "w").write(md)
    json.dump({"stereo": s_sum, "mvs": m_sum, "pointcloud": p_sum, "latency": l_sum}, open(os.path.join(a.results, "summary.json"), "w"),
              indent=1, default=float)
    figures(m_rows, s_rows, os.path.join(a.results, "figures"))
    print(md)


if __name__ == "__main__":
    main()
