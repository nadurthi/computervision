import json, os
H = os.path.dirname(os.path.abspath(__file__))
IMG = ["fig_polhodes.png", "fig_three_axes.png", "fig_dissipation.png", "fig_cost_vs_gain.png", "fig_value.png",
       "fig_detumble.png", "fig_bdot.png", "anim_dzhanibekov.mp4", "anim_detumble.mp4"]
U = "https://perceptionstack.dev/wp-content/uploads/2026/11/"
src = open(os.path.join(H, "body.html")).read()
body = src
for k in IMG:
    body = body.replace("IMG/" + k, U + "spin-" + k.replace("_", "-"))
assert "IMG/" not in body
body = body.replace("CODE_LINK", '<a href="https://github.com/nadurthi/computervision/tree/spin-stabilization/spin_stabilization">nadurthi/computervision · spin_stabilization/</a>')
desc = ("Spin stabilization from first principles: Euler's equations, polhodes, the intermediate-axis (Dzhanibekov) flip, "
        "the major-axis rule, optimal detumbling and magnetic B-dot, with simulations and animations.")
hdr = "<!-- gdl-static " + json.dumps({"v": 1, "source": "gnc/spin-stabilization.html", "page": "gnc-spin-stabilization",
                                        "title": "Spin stabilization and detumbling", "description": desc,
                                        "root": "/tutorials/", "scripts": ["bootstrap", "nav"]}, ensure_ascii=False) + " -->"
out = "<!-- wp:html -->\n" + hdr + "\n" + body.strip() + "\n<!-- /wp:html -->\n"
open(os.path.join(H, "page_wp.html"), "w").write(out)
prev = src.replace("IMG/", "../outputs/").replace("CODE_LINK", "this folder")
open(os.path.join(H, "page_preview.html"), "w").write(
    '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Spin stabilization (preview)</title>'
    '<script>window.MathJax={tex:{inlineMath:[["$","$"]],displayMath:[["$$","$$"]]}};</script>'
    '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>'
    '<style>body{font-family:system-ui,sans-serif;max-width:980px;margin:2em auto;padding:0 16px;line-height:1.6;color:#1f2937;background:#fff}'
    '.callout{border-left:4px solid #2563eb;background:#eff6ff;padding:.8em 1em;margin:1.2em 0}.callout.warn{border-color:#ea580c;background:#fff7ed}'
    'table{border-collapse:collapse}td,th{border:1px solid #e5e7eb;padding:4px 8px}td.num{text-align:right}pre{background:#f3f4f6;padding:1em;overflow:auto}'
    '.footnote{color:#6b7280;font-size:.92em}.lead{font-size:1.08em}</style></head><body>' + prev + "</body></html>")
print(len(out), "chars;", out.count("wp-content/uploads"), "media refs")
