"""Turn results/*.json into Markdown and HTML tables (results/RESULTS.md, results/tables.html)."""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import NAMES, ORDER

COLS = [("F1@0.5", "F1@0.5", "{:.2f}"), ("F1@0.75", "F1@0.75", "{:.2f}"), ("PQ", "PQ", "{:.2f}"),
        ("normal_err_deg_median", "normal err (deg)", "{:.2f}"), ("center_err_cm_median", "centre err (cm)", "{:.1f}"),
        ("size_err_cm_median", "size err (cm)", "{:.1f}"), ("n_images", "walls", "{:d}")]


def rows(path):
    M = json.load(open(path))
    out = []
    for m in ORDER:
        if m not in M:
            continue
        r = M[m]
        cells = []
        for k, _, fmt in COLS:
            v = r.get(k)
            cells.append("–" if v is None else fmt.format(v))
        out.append((NAMES[m][0], cells))
    return out


def md(title, path):
    s = f"### {title}\n\n| Method | " + " | ".join(c[1] for c in COLS) + " |\n|---|" + "---:|" * len(COLS) + "\n"
    for name, cells in rows(path):
        s += f"| {name} | " + " | ".join(cells) + " |\n"
    return s + "\n"


def html(title, path):
    s = f"<h4>{title}</h4>\n<div style=\"overflow-x:auto\"><table style=\"border-collapse:collapse;font-size:0.9em;width:100%\">\n<tr>"
    s += "".join(f"<th style=\"text-align:{'left' if i == 0 else 'right'};border-bottom:2px solid #d4d4d4;padding:4px 8px\">{h}</th>"
                 for i, h in enumerate(["Method"] + [c[1] for c in COLS]))
    s += "</tr>\n"
    for name, cells in rows(path):
        s += f"<tr><td style=\"padding:4px 8px;border-bottom:1px solid #eee\">{name}</td>" + "".join(
            f"<td style=\"text-align:right;padding:4px 8px;border-bottom:1px solid #eee;font-variant-numeric:tabular-nums\">{c}</td>" for c in cells) + "</tr>\n"
    return s + "</table></div>\n"


def lidar(fmt):
    p = "results/lidar_metrics_test.json"
    if not os.path.exists(p):
        return ""
    M = json.load(open(p))
    names = {"lidar_range_image": "Lidar only: range-image cuts", "lidar_ransac_dbscan": "Lidar only: RANSAC planes + DBSCAN",
             "cam2lidar_seamnet_rgbd": "Camera SeamNet RGB-D labels → lidar", "cam2lidar_rgbd_seams": "Camera classical RGB-D labels → lidar"}
    if fmt == "md":
        s = "### Lidar (point-level instances inside the camera view, 100 scans)\n\n| Method | precision | recall | F1@0.5 | PQ | s/scan |\n|---|---:|---:|---:|---:|---:|\n"
        for k, r in M.items():
            s += f"| {names.get(k, k)} | {r['precision']:.2f} | {r['recall']:.2f} | {r['F1@0.5']:.2f} | {r['PQ']:.2f} | {r['sec_per_frame']:.2f} |\n"
        return s + "\n"
    s = "<h4>Lidar (point-level instances inside the camera view, 100 scans)</h4>\n<div style=\"overflow-x:auto\"><table style=\"border-collapse:collapse;font-size:0.9em;width:100%\">\n<tr>"
    s += "".join(f"<th style=\"text-align:{'left' if i == 0 else 'right'};border-bottom:2px solid #d4d4d4;padding:4px 8px\">{h}</th>"
                 for i, h in enumerate(["Method", "precision", "recall", "F1@0.5", "PQ", "s/scan"]))
    s += "</tr>\n"
    for k, r in M.items():
        cells = [f"{r['precision']:.2f}", f"{r['recall']:.2f}", f"{r['F1@0.5']:.2f}", f"{r['PQ']:.2f}", f"{r['sec_per_frame']:.2f}"]
        s += f"<tr><td style=\"padding:4px 8px;border-bottom:1px solid #eee\">{names.get(k, k)}</td>" + "".join(
            f"<td style=\"text-align:right;padding:4px 8px;border-bottom:1px solid #eee;font-variant-numeric:tabular-nums\">{c}</td>" for c in cells) + "</tr>\n"
    return s + "</table></div>\n"


if __name__ == "__main__":
    t1 = ("Standard walls (test, 100 scenes)", "results/metrics_test.json")
    t2 = ("Hard walls (test_hard, 100 scenes: gaps ~half as wide, more yaw and tilt)", "results/metrics_test_hard.json")
    open("results/RESULTS.md", "w").write(md(*t1) + md(*t2) + lidar("md"))
    open("results/tables.html", "w").write(html(*t1) + html(*t2) + lidar("html"))
    print(open("results/RESULTS.md").read())
