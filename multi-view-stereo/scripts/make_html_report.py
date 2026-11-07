#!/usr/bin/env python3
"""Build a single self-contained results/report.html from RESULTS.md and every figure in results/figures/
(images are inlined as base64, so the file can be opened or e-mailed on its own).

  python scripts/make_html_report.py --results results/quick [--title "..."]
"""
import argparse
import base64
import datetime
import glob
import html
import json
import os
import re

CSS = """
:root{--bg:#fff;--fg:#1f2328;--muted:#59636e;--line:#d1d9e0;--head:#f6f8fa;--accent:#0969da}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--line:#3d444d;--head:#151b23;--accent:#4493f8}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.7em;margin:.2em 0}h2{margin-top:2em;border-bottom:1px solid var(--line);padding-bottom:.3em}
h3{margin-top:1.6em}h4{margin-top:1.2em;color:var(--muted)}
.meta{color:var(--muted);font-size:.9em}
.tw{overflow-x:auto;margin:.6em 0 1.2em}
table{border-collapse:collapse;font-size:.88em;white-space:nowrap}
th,td{border:1px solid var(--line);padding:4px 9px;text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{background:var(--head);position:sticky;top:0}
tbody tr:hover{background:var(--head)}
td.best{font-weight:700;color:var(--accent)}
code{background:var(--head);padding:1px 4px;border-radius:4px}
figure{margin:1.2em 0}figure img{max-width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}
figcaption{color:var(--muted);font-size:.85em}
nav a{margin-right:12px;color:var(--accent);text-decoration:none}
"""


def inline(md_text):
    t = html.escape(md_text)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", t)


def table_html(lines, higher_better):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    head, body = rows[0], rows[2:]
    # highlight the best value per numeric column (direction from the heading of the section)
    best = {}
    for j in range(1, len(head)):
        vals = []
        for r in body:
            try:
                vals.append(float(r[j].split()[0]))
            except (ValueError, IndexError):
                vals.append(None)
        ok = [v for v in vals if v is not None]
        h = head[j]
        if len(ok) < 2 or re.search(r"\b(N|MP|scans|time s|load s|peak MB|density %)$", h) or h in ("N", "MP"):
            continue
        if re.search(r"AbsRel|EPE|bad|D1|ms\b|accuracy|completeness|overall", h):
            col_higher = False                       # errors and times: lower is better whatever the section says
        else:
            col_higher = higher_better
        if col_higher is not None:
            best[j] = max(ok) if col_higher else min(ok)
    out = ['<div class="tw"><table><thead><tr>' + "".join(f"<th>{inline(h)}</th>" for h in head) + "</tr></thead><tbody>"]
    for r in body:
        cells = []
        for j, c in enumerate(r):
            cls = ""
            if j in best:
                try:
                    if float(c.split()[0]) == best[j]:
                        cls = ' class="best"'
                except (ValueError, IndexError):
                    pass
            cells.append(f"<td{cls}>{inline(c)}</td>")
        out.append("<tr>" + "".join(cells) + "</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def md_to_html(md):
    out, i, lines = [], 0, md.splitlines()
    direction = None
    toc = []
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"^(#{1,4}) (.*)", ln)
        if m:
            level, text = len(m.group(1)), m.group(2)
            if "lower is better" in text:
                direction = False
            elif "higher is better" in text:
                direction = True
            elif level <= 3 and "Latency" in text:
                direction = False
            anchor = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
            if level == 2:
                toc.append((anchor, text.split("(")[0].strip()))
            out.append(f'<h{level} id="{anchor}">{inline(text)}</h{level}>')
            i += 1
        elif ln.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(table_html(block, direction))
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
            i += 1
        else:
            i += 1
    return "\n".join(out), toc


CAPTIONS = {
    "nview_sweep": "Accuracy vs number of views N: all GT pixels (left) and pixels the stereo partner cannot see (right). "
                   "Solid = classical MVS, dash-dot = learned MVS, dashed = two-view stereo.",
    "stereo_occlusion": "Two-view stereo error on non-occluded vs occluded pixels.",
    "qualitative_stereo": "Disparity maps of every stereo method on the first pair; occluded GT pixels in red.",
    "qualitative": "Reference image, GT depth, occlusion labels for the stereo partner, and per-method error maps "
                   "at N=2 and N=5 (dark = small error, grey overlay = no prediction).",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--title", default="Multi-view stereo vs stereo: benchmark results")
    a = ap.parse_args()
    md_path = os.path.join(a.results, "RESULTS.md")
    md = open(md_path).read() if os.path.exists(md_path) else "# Results\n\nNo results yet.\n"
    md = re.sub(r"^# Results\n+Generated by `scripts/make_report.py`\.\n*", "", md)
    body, toc = md_to_html(md)
    figs = []
    for p in sorted(glob.glob(os.path.join(a.results, "figures", "*.png"))):
        name = os.path.basename(p)[:-4]
        cap = next((v for k, v in sorted(CAPTIONS.items(), key=lambda kv: -len(kv[0])) if name.startswith(k)), "")
        b64 = base64.b64encode(open(p, "rb").read()).decode()
        figs.append(f'<figure><img alt="{html.escape(name)}" src="data:image/png;base64,{b64}">'
                    f"<figcaption><b>{html.escape(name)}</b>. {html.escape(cap)}</figcaption></figure>")
    gpus = set()
    for p in glob.glob(os.path.join(a.results, "*", "*.jsonl")):
        with open(p) as f:
            first = f.readline()
        try:
            gpus.add(json.loads(first).get("gpu"))
        except Exception:
            pass
    gpus.discard(None)
    meta = (f"Generated {datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M UTC} from <code>{html.escape(os.path.abspath(a.results))}</code>"
            + (f" · GPU: {html.escape(', '.join(sorted(gpus)))}" if gpus else ""))
    nav = "".join(f'<a href="#{k}">{html.escape(t)}</a>' for k, t in toc) + ('<a href="#figures">Figures</a>' if figs else "")
    page = (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(a.title)}</title><style>{CSS}</style></head><body><main>"
            f"<h1>{html.escape(a.title)}</h1><p class=meta>{meta}</p><nav>{nav}</nav>{body}"
            + (f'<h2 id="figures">Figures</h2>' + "".join(figs) if figs else "")
            + "<p class=meta>Best value per column in bold. Missing predictions count as errors; "
              "see the repository README for the protocol.</p></main></body></html>")
    out = os.path.join(a.results, "report.html")
    open(out, "w").write(page)
    print(f"[report] {out} ({len(page) / 1e6:.1f} MB, {len(figs)} figures)")


if __name__ == "__main__":
    main()
