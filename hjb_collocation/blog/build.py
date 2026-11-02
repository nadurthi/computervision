import json, sys, os
H = os.path.dirname(os.path.abspath(__file__))
IMG = {"fig_points.png": "hjb-fig-points.png", "paper_fig1_cut_axes.png": "hjb-paper-fig1-cut-axes.png",
       "fig_interp.png": "hjb-fig-interp.png", "fig_spin.png": "hjb-fig-spin.png", "fig_spin_coeffs.png": "hjb-fig-spin-coeffs.png",
       "paper_fig5_vdp.png": "hjb-paper-fig5-vdp.png", "fig_vdp_main.png": "hjb-fig-vdp-main.png", "fig_vdp_ratio.png": "hjb-fig-vdp-ratio.png",
       "fig_vdp_traj.png": "hjb-fig-vdp-traj.png", "fig_vdp_sensitivity.png": "hjb-fig-vdp-sensitivity.png",
       "fig_att_flows.png": "hjb-fig-att-flows.png", "fig_att_traj.png": "hjb-fig-att-traj.png"}
U = "https://perceptionstack.dev/wp-content/uploads/2026/11/"
att = sys.argv[1] if len(sys.argv) > 1 else "att_placeholder.html"
summ = sys.argv[2] if len(sys.argv) > 2 else "results pending (rerun in progress)."
body = "".join(open(os.path.join(H, f)).read() + "\n" for f in ["head.html", "theory.html", "results.html", att, "tail.html"])
for k, v in IMG.items():
    body = body.replace("IMG/" + k, U + v)
assert "IMG/" not in body, [l for l in body.split("\n") if "IMG/" in l]
body = body.replace("CODE_LINK", '<a href="https://github.com/nadurthi/computervision/tree/hjb-collocation/hjb_collocation">nadurthi/computervision · hjb_collocation/</a>')
body = body.replace("ATT_SUMMARY", summ)
body = body.replace("\\(", "$").replace("\\)", "$")          # site convention: $…$ inline, $$…$$ display
desc = ("Solving the Hamilton–Jacobi–Bellman equation by policy iteration, CUT collocation points and iterative l1 basis selection: "
        "derivation and a verified Python rerun of Adurthi, Singla & Majji (JGCD 2017).")
hdr = "<!-- gdl-static " + json.dumps({"v": 1, "source": "algorithms/hjb-sparse-collocation.html", "page": "algorithms-hjb-sparse-collocation",
                                        "title": "Sparse collocation for the HJB equation", "description": desc,
                                        "root": "/tutorials/", "scripts": ["bootstrap", "nav"]}, ensure_ascii=False) + " -->"
out = "<!-- wp:html -->\n" + hdr + "\n" + body.strip() + "\n<!-- /wp:html -->\n"
open(os.path.join(H, "page_wp.html"), "w").write(out)
open(os.path.join(H, "desc.txt"), "w").write(desc)
# local preview with local images
prev = "".join(open(os.path.join(H, f)).read() + "\n" for f in ["head.html", "theory.html", "results.html", att, "tail.html"])
for k in IMG:
    prev = prev.replace("IMG/" + k, ("../paper_figs/" if k.startswith("paper_") else "../outputs/figs/") + k)
prev = prev.replace("CODE_LINK", "this folder").replace("ATT_SUMMARY", summ)
open(os.path.join(H, "page_preview.html"), "w").write(
    '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>HJB sparse collocation (preview)</title>'
    '<script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]],displayMath:[["$$","$$"]]}};</script>'
    '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js"></script>'
    '<style>body{font-family:system-ui,sans-serif;max-width:980px;margin:2em auto;padding:0 16px;line-height:1.6;color:#1f2937;background:#fff}'
    '.callout{border-left:4px solid #2563eb;background:#eff6ff;padding:.8em 1em;margin:1.2em 0}.callout.warn{border-color:#ea580c;background:#fff7ed}'
    'table{border-collapse:collapse}td,th{border:1px solid #e5e7eb;padding:4px 8px}td.num{text-align:right}pre{background:#f3f4f6;padding:1em;overflow:auto}'
    '.footnote{color:#6b7280;font-size:.92em}.lead{font-size:1.08em}</style></head><body>' + prev + "</body></html>")
print(len(out), "chars;", out.count("wp-content/uploads"), "media refs;", out.count("$$") // 2, "display equations")
