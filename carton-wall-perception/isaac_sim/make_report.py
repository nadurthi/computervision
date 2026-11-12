"""Build report.html for one run folder written by run_all.sh (no AWS, no Isaac Sim needed).

    python3 make_report.py ./isaac_runs/run-20261107T012222Z      # writes <run>/report.html

Needs numpy and Pillow; matplotlib is used for the depth colour map and the lidar plots when installed.
Images are embedded as JPEG thumbnails, so the HTML file is self-contained; full-resolution files stay in the
run folder and are linked.
"""
import base64
import html
import io
import json
import os
import re
import sys
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional
    plt = None

RUN = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
TAGS = re.compile(r"\[(smoke|lidar-test|box-wall|cell|trailer|unload)\]")
ERRS = re.compile(r"Traceback|Error: |Segmentation fault|There was an error running python|ErrorException")


# ------------------------------------------------------------------------------------------- helpers
def rel(p):
    return os.path.relpath(p, RUN)


def img_tag(im, width=640, alt="", link=None, quality=85):
    im = im.convert("RGB")
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    src = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    tag = f'<img src="{src}" alt="{html.escape(alt)}" width="{im.width}">'
    return f'<a href="{html.escape(link)}">{tag}</a>' if link else tag


def fig_tag(fig, width=640):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return img_tag(Image.open(buf), width)


def colorize_ids(ids):
    ids = ids.astype(np.int64)
    rng = np.random.default_rng(7)
    lut = rng.integers(40, 255, size=(int(ids.max()) + 1, 3)).astype(np.uint8)
    lut[0] = 0
    return Image.fromarray(lut[ids])


def overlay(rgb, mask_rgb, a=0.55):
    return Image.blend(rgb.convert("RGB"), mask_rgb.convert("RGB").resize(rgb.size), a)


def depth_image(d):
    d = np.asarray(d, dtype=np.float32)
    ok = np.isfinite(d) & (d > 0)
    if not ok.any():
        return None
    lo, hi = np.percentile(d[ok], [1, 99])
    n = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    n[~ok] = 1
    if plt is not None:
        rgb = (plt.get_cmap("turbo")(1 - n)[..., :3] * 255).astype(np.uint8)
    else:
        g = ((1 - n) * 255).astype(np.uint8)
        rgb = np.stack([g, g, g], -1)
    rgb[~ok] = 0
    return Image.fromarray(rgb), (float(lo), float(hi))


def draw_boxes(rgb, bbox_json):
    try:
        boxes = json.load(open(bbox_json)).get("boxes", [])
    except Exception:
        return rgb, 0
    im = rgb.convert("RGB").copy()
    dr = ImageDraw.Draw(im)
    n = 0
    for b in boxes:
        # bounding_box_2d_tight rows: (semanticId, x_min, y_min, x_max, y_max, occlusionRatio)
        if isinstance(b, (list, tuple)) and len(b) >= 5:
            x0, y0, x1, y1 = (float(v) for v in b[1:5])
            dr.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=3)
            n += 1
    return im, n


def lidar_plots(pts, title):
    if plt is None or len(pts) == 0:
        return ""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sub = pts[:: max(1, len(pts) // 40000)]
    ax[0].scatter(sub[:, 0], sub[:, 1], s=0.3, c=sub[:, 2], cmap="viridis")
    ax[0].set_title("top view (x forward, y left)")
    ax[0].set_aspect("equal")
    ax[1].scatter(sub[:, 1], sub[:, 2], s=0.3, c=sub[:, 0], cmap="viridis")
    ax[1].set_title("front view (y, z)")
    ax[1].set_aspect("equal")
    fig.suptitle(title)
    return fig_tag(fig, 900)


def log_summary(name):
    p = os.path.join(RUN, f"{name}.log")
    if not os.path.exists(p):
        return None
    lines = open(p, errors="replace").read().splitlines()
    tagged = [l for l in lines if TAGS.search(l)]
    errs = []
    for i, l in enumerate(lines):
        if ERRS.search(l) and "deprecat" not in l:
            errs.extend(lines[i:i + 12] if "Traceback" in l else [l])
    failed = any("There was an error running python" in l or "Segmentation fault" in l for l in lines)
    return {"tagged": tagged, "errors": errs[:60], "failed": failed, "n": len(lines)}


def steps():
    out = {}
    p = os.path.join(RUN, "steps.txt")
    if os.path.exists(p):
        for l in open(p):
            m = re.match(r"(\S+) exit=(\d+) seconds=(\d+)", l)
            if m:
                out[m.group(1)] = (int(m.group(2)), int(m.group(3)))
    return out


# ------------------------------------------------------------------------------------------- sections
def section_overview():
    rows = []
    st = steps()
    for name in ("smoke", "smoke_ids", "lidar_test", "box_wall", "unload_cell", "trailer_wall", "trailer_unload"):
        s = log_summary(name)
        if s is None:
            continue
        rc, sec = st.get(name, (None, None))
        ok = (rc == 0) if rc is not None else not s["failed"]
        last = s["tagged"][-1] if s["tagged"] else ""
        rows.append(f"<tr><td>{name}</td><td class='{'ok' if ok else 'bad'}'>{'ok' if ok else 'failed'}</td>"
                    f"<td>{'' if sec is None else f'{sec // 60} min {sec % 60} s'}</td>"
                    f"<td><code>{html.escape(last[-160:])}</code></td><td><a href='{name}.log'>log</a></td></tr>")
    gpu = ""
    p = os.path.join(RUN, "nvidia-smi.txt")
    if os.path.exists(p):
        m = re.search(r"Driver Version:\s*([\d.]+)", open(p).read())
        g = re.search(r"\|\s+\d+\s+(NVIDIA [^|]+?)\s{2,}", open(p).read())
        gpu = f"<p>GPU: {html.escape(g.group(1).strip()) if g else '?'} · driver {m.group(1) if m else '?'}</p>"
    return ("<h2>Steps</h2>" + gpu +
            "<table><tr><th>step</th><th>result</th><th>time</th><th>last message</th><th></th></tr>"
            + "".join(rows) + "</table>")


def section_smoke():
    parts = []
    for name in ("smoke", "smoke_ids"):
        p = os.path.join(RUN, name, "smoke.png")
        if os.path.exists(p):
            parts.append(f"<figure>{img_tag(Image.open(p), 320, name, rel(p))}<figcaption>{name}</figcaption></figure>")
    lt = os.path.join(RUN, "lidar_test", "lidar_test.txt")
    txt = f"<p>Lidar test: <code>{html.escape(open(lt).read().strip())}</code></p>" if os.path.exists(lt) else ""
    return ("<h2>Environment checks</h2><div class='row'>" + "".join(parts) + "</div>" + txt) if parts or txt else ""


def section_box_wall():
    d = os.path.join(RUN, "box_wall")
    if not os.path.isdir(d):
        return ""
    frames = sorted(f[4:8] for f in os.listdir(d) if re.match(r"rgb_\d{4}\.png$", f))
    out = ["<h2>Box wall in the warehouse</h2>"]
    cam = os.path.join(d, "camera.json")
    if os.path.exists(cam):
        out.append(f"<p>Camera: <code>{html.escape(open(cam).read().strip()[:300])}</code></p>")
    boxes = {}
    if os.path.exists(os.path.join(d, "boxes.json")):
        boxes = json.load(open(os.path.join(d, "boxes.json")))
    for t in frames:
        rgb_p = os.path.join(d, f"rgb_{t}.png")
        rgb = Image.open(rgb_p).convert("RGB")
        cells = [f"<figure>{img_tag(rgb, 420, 'rgb', rel(rgb_p))}<figcaption>RGB</figcaption></figure>"]
        ip = os.path.join(d, f"instance_{t}.png")
        if os.path.exists(ip):
            ids = np.array(Image.open(ip))
            cells.append(f"<figure>{img_tag(overlay(rgb, colorize_ids(ids)), 420, 'instances', rel(ip))}"
                         f"<figcaption>instances ({len(np.unique(ids)) - 1})</figcaption></figure>")
        sp = os.path.join(d, f"semantic_{t}.png")
        if os.path.exists(sp):
            sem = np.array(Image.open(sp))
            labels = ""
            sj = os.path.join(d, f"semantic_{t}.json")
            if os.path.exists(sj):
                try:
                    info = json.load(open(sj))
                    lab = info.get("idToLabels", info)
                    labels = ", ".join(sorted({str(v.get("class", v)) if isinstance(v, dict) else str(v)
                                               for v in lab.values()}))[:120]
                except Exception:
                    pass
            cells.append(f"<figure>{img_tag(overlay(rgb, colorize_ids(sem)), 420, 'semantic', rel(sp))}"
                         f"<figcaption>semantic: {html.escape(labels)}</figcaption></figure>")
        dp = os.path.join(d, f"depth_{t}.npy")
        if os.path.exists(dp):
            r = depth_image(np.load(dp))
            if r:
                cells.append(f"<figure>{img_tag(r[0], 420, 'depth')}<figcaption>depth {r[1][0]:.2f}–{r[1][1]:.2f} m"
                             f"</figcaption></figure>")
        bp = os.path.join(d, f"bbox2d_{t}.json")
        if os.path.exists(bp):
            im, n = draw_boxes(rgb, bp)
            cells.append(f"<figure>{img_tag(im, 420, 'bbox')}<figcaption>2D boxes ({n})</figcaption></figure>")
        nb = len(boxes.get(t, []))
        out.append(f"<h3>Frame {t} · {nb} cartons placed</h3><div class='row'>" + "".join(cells) + "</div>")
        lp = os.path.join(d, f"lidar_{t}.npy")
        if os.path.exists(lp):
            pts = np.load(lp).reshape(-1, 3)
            out.append(f"<p>Lidar: {len(pts):,} points</p>" + lidar_plots(pts, f"lidar frame {t}"))
    return "".join(out)


def section_cell():
    d = os.path.join(RUN, "unload_cell")
    out = ["<h2>Unloading cell (UR10 + suction + conveyor)</h2>"]
    s = log_summary("unload_cell")
    if s:
        picks = [l for l in s["tagged"] if "pick" in l]
        out.append(f"<p>{len(picks)} picks logged.</p>")
    if os.path.isdir(d):
        frames = sorted(f[4:7] for f in os.listdir(d) if re.match(r"rgb_\d{3}\.png$", f))
        cells = []
        for k in frames:
            p = os.path.join(d, f"rgb_{k}.png")
            cap = f"before pick {int(k) + 1}"
            lp = os.path.join(d, f"lidar_{k}.npy")
            if os.path.exists(lp):
                cap += f" · lidar {np.load(lp).reshape(-1, 3).shape[0]:,} pts"
            cells.append(f"<figure>{img_tag(Image.open(p), 420, cap, rel(p))}<figcaption>{cap}</figcaption></figure>")
        out.append("<div class='row'>" + "".join(cells) + "</div>")
    return "".join(out) if (s or os.path.isdir(d)) else ""


def section_trailer():
    d = os.path.join(RUN, "trailer_wall")
    p = os.path.join(d, "rgb.png")
    if not os.path.exists(p):
        return ""
    rgb = np.asarray(Image.open(p).convert("RGB"))
    figs = [f"<figure>{img_tag(Image.fromarray(rgb), 900, 'trailer wall', rel(p))}<figcaption>RGB</figcaption></figure>"]
    ip = os.path.join(d, "instance.png")
    if os.path.exists(ip):
        ids = np.asarray(Image.open(ip))
        figs.append(f"<figure>{img_tag(overlay(Image.fromarray(rgb), colorize_ids(ids)), 900, 'instances')}"
                    f"<figcaption>instance masks</figcaption></figure>")
    info = ""
    bp = os.path.join(d, "boxes.json")
    if os.path.exists(bp):
        import json
        b = json.load(open(bp))
        info = f"<p>{len(b)} boxes; models: {', '.join(sorted({x.get('model', '?') for x in b}))[:300]}</p>"
    return "<h2>Trailer wall</h2>" + info + "<div class='row'>" + "".join(figs) + "</div>"


def section_unload():
    d = os.path.join(RUN, "trailer_unload")
    if not os.path.isdir(d):
        return ""
    import base64, json
    figs = []
    for name in ("wall_start.png", "wall_end.png"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            figs.append(f"<figure>{img_tag(Image.open(p).convert('RGB'), 640, name, rel(p))}"
                        f"<figcaption>{name}</figcaption></figure>")
    g = os.path.join(d, "unload.gif")
    if os.path.exists(g) and os.path.getsize(g) < 25e6:
        b64 = base64.b64encode(open(g, "rb").read()).decode()
        figs.append(f"<figure><img src='data:image/gif;base64,{b64}' width=640><figcaption>unload.gif</figcaption></figure>")
    ev = ""
    e = os.path.join(d, "events.json")
    if os.path.exists(e):
        E = json.load(open(e))
        rows = "".join(f"<tr><td>{p['box'].split('/')[-1]}</td><td>{p['mass_kg']}</td><td>{p['placed']}</td>"
                       f"<td>{', '.join(f'{k.split(chr(47))[-1]} {c} cm' for k, c in p['neighbours_dragged_cm'])}</td></tr>"
                       for p in E.get("picks", []))
        col = "".join(f"<li>{c['link'].split('/')[-1]} touched {c['hit']} at {c['t_s']} s: stopped</li>"
                      for c in E.get("collisions", []))
        ev = (f"<table><tr><th>box</th><th>kg</th><th>placed</th><th>neighbours dragged</th></tr>{rows}</table>"
              f"<p>Collisions:</p><ul>{col or '<li>none</li>'}</ul><p>Skipped (out of reach): "
              f"{len(E.get('skipped', []))}</p>")
    return "<h2>Trailer unloading (FANUC CRX-10iA/L)</h2>" + ev + "<div class='row'>" + "".join(figs) + "</div>"


def section_logs():
    out = ["<h2>Log excerpts</h2>"]
    for name in ("smoke", "smoke_ids", "lidar_test", "box_wall", "unload_cell", "trailer_wall", "trailer_unload"):
        s = log_summary(name)
        if s is None:
            continue
        body = "\n".join(s["tagged"][-40:])
        if s["errors"]:
            body += "\n\n--- errors ---\n" + "\n".join(l[-300:] for l in s["errors"])
        out.append(f"<details{' open' if s['errors'] else ''}><summary>{name} ({s['n']} lines)</summary>"
                   f"<pre>{html.escape(body)}</pre></details>")
        crash = os.path.join(RUN, f"crash_{name}")
        if os.path.isdir(crash):
            for f in sorted(os.listdir(crash)):
                if f.endswith(".py.txt"):
                    out.append(f"<details><summary>crash Python stack: {f}</summary><pre>"
                               f"{html.escape(open(os.path.join(crash, f), errors='replace').read()[:6000])}</pre></details>")
    return "".join(out)


CSS = """body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;color:#1f2937;max-width:1400px}
h1{margin-bottom:4px} h2{border-bottom:1px solid #e5e7eb;padding-bottom:4px;margin-top:32px}
table{border-collapse:collapse} td,th{border:1px solid #e5e7eb;padding:4px 8px;font-size:14px;text-align:left}
.ok{color:#15803d;font-weight:600} .bad{color:#b91c1c;font-weight:600}
.row{display:flex;flex-wrap:wrap;gap:10px} figure{margin:0} figcaption{font-size:13px;color:#4b5563}
img{max-width:100%;height:auto;border:1px solid #e5e7eb} pre{background:#f9fafb;padding:8px;overflow:auto;font-size:12px}
code{font-size:12px}"""


def main():
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = [f"<h1>Isaac Sim box-wall run</h1><p>{html.escape(os.path.basename(RUN))} · report built {when}</p>",
            section_overview(), section_smoke(), section_box_wall(), section_cell(), section_trailer(), section_unload(), section_logs()]
    out = os.path.join(RUN, "report.html")
    with open(out, "w") as f:
        f.write(f"<!doctype html><html><head><meta charset='utf-8'><title>Isaac box-wall run</title>"
                f"<meta name='viewport' content='width=device-width,initial-scale=1'><style>{CSS}</style></head>"
                f"<body>{''.join(b for b in body if b)}</body></html>")
    print("report:", out)


if __name__ == "__main__":
    main()
