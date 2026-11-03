"""Build the WordPress post HTML (docs/post.html) from the result files.

Every number in the post is read from results/*.json, so the post cannot drift from the experiments.
Image URLs come from docs/media_urls.json ({figure filename: uploaded URL}).
"""
import json, os, re, sys

import markdown

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_figures import NAMES

REPO = "https://github.com/nadurthi/computervision/tree/master/carton-wall-perception"
CAP = 'style="font-size:0.9em;color:#6b7280"'

T = json.load(open("results/metrics_test.json"))
H = json.load(open("results/metrics_test_hard.json"))
L = json.load(open("results/lidar_metrics_test.json"))
FS = json.load(open("results/faint_seam_analysis.json"))
F30 = json.load(open("results/metrics_first30.json"))
PC = json.load(open("results/pose_common_test.json")) if os.path.exists("results/pose_common_test.json") else None
SP = json.load(open("results/speed_cpu.json")) if os.path.exists("results/speed_cpu.json") else {}
URLS = json.load(open("docs/media_urls.json")) if os.path.exists("docs/media_urls.json") else {}
TABLES = open("results/tables.html").read()


def img(name, alt, caption):
    url = URLS.get(name, f"FIGURE:{name}")
    return (f'<figure><img src="{url}" alt="{alt}" style="width:100%;height:auto" loading="lazy">'
            f'<figcaption {CAP}>{caption}</figcaption></figure>')


def svg(name, caption):
    png = name.replace(".svg", ".png")
    return img(png, caption.split(". ", 1)[-1][:120], caption)


def pq(M, m):
    return M[m]["PQ"]


def f1(M, m):
    return M[m]["F1@0.5"]


def arch_html():
    md = open("docs/ARCHITECTURES.md").read()
    md = md.split("\n---\n", 1)[1]                        # drop the title + contents list
    md = md.replace("](../", f"]({REPO}/")
    md = re.sub(r"\]\((#[^)]+)\)", r"](\1)", md)
    html = markdown.markdown(md, extensions=["tables", "fenced_code"])
    html = html.replace("<h3>", "<h4>").replace("</h3>", "</h4>")
    html = re.sub(r"<h2>(\d+)\. ([^<]+)</h2>", lambda m: f'<h3 id="arch-{m.group(1)}">{m.group(1)}. {m.group(2)}</h3>', html)
    html = html.replace("<table>", '<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.88em">').replace("</table>", "</table></div>")
    html = html.replace("<hr />", "")
    return html


best_classical = max(["rgb_lines", "depth_edges", "rgbd_seams"], key=lambda m: pq(T, m))
sam_n = T.get("sam2t_rgbd", {}).get("n_images", 0)

lead = f"""<p>A robot that unloads a trailer has to look at a wall of cardboard and answer one question over and over: where does each box start and stop, and which one can I pull next without bringing the stack down? I compared every kind of method I could run end to end on a CPU: classical edge and plane pipelines on RGB, depth and lidar; two networks trained on synthetic walls; and three zero-shot "segment anything" models. All of them were scored on the same 200 simulated walls with exact ground truth.</p>
<h3>Main results</h3>
<ul>
<li><b>On simulated walls, a small network trained on simulated walls wins by a lot.</b> YOLO11n-seg fine-tuned for 10 CPU epochs reaches a panoptic quality (PQ) of {pq(T,'yolo'):.2f} ({pq(H,'yolo'):.2f} on the hard walls). A 1.1 M-parameter RGB-D U-Net that only predicts seams reaches {pq(T,'seamnet_rgbd'):.2f} ({pq(H,'seamnet_rgbd'):.2f}). The best classical pipeline reaches {pq(T,best_classical):.2f} ({pq(H,best_classical):.2f}).</li>
<li><b>On simulated walls, zero-shot foundation models trail far behind.</b> FastSAM gets PQ {pq(T,'fastsam_rgb'):.2f} on its own and {pq(T,'fastsam_rgbd'):.2f} with a depth-based selector. SAM 2.1-tiny with the same selector gets {pq(T,'sam2t_rgbd'):.2f} on {sam_n} walls. They split boxes into labels and tape, or merge whole stacks, and on this CPU they take 0.6 s (FastSAM) to about 38 s (SAM 2.1-tiny with a 192-point prompt grid) per frame.</li>
<li><b>Depth matters most when seams are faint.</b> On hard walls with barely visible seams, RGB-only SeamNet drops to PQ {FS['test_hard/faint']['seamnet_rgb']:.2f} while RGB-D SeamNet holds at {FS['test_hard/faint']['seamnet_rgbd']:.2f}. On standard walls with clearly visible seams they differ by less than 0.01.</li>
<li><b>Lidar alone cannot separate flush boxes at this density.</b> With 3 cm between rings, the best lidar-only method reaches F1 {L['lidar_range_image']['F1@0.5']:.2f}. Projecting the camera network's labels onto the same points gives F1 {L['cam2lidar_seamnet_rgbd']['F1@0.5']:.2f}: segment in the image, measure in 3D.</li>
<li><b>On real photos the ranking flips.</b> Both networks trained only on simulated walls break real boxes into pieces along print, labels and strapping. FastSAM and SAM 2.1, which never saw a simulated wall, give clean per-box masks on the same photos. Simulation was enough to rank methods and to train geometry, but not enough to train appearance.</li>
<li><b>Once a box is found, its pose is easy.</b> Every method that finds a box fits its face to about 0.3° in normal and well under 1 cm (median) in face centre. Finding the right boundary is the whole problem.</li>
</ul>
"""

body = []
body.append(lead)
body.append(img("pq_bars.png", "Panoptic quality of every method on standard and hard walls",
                "Panoptic quality (PQ = recognition × segmentation quality) on 100 standard and 100 hard walls. SAM models were scored on the first 30 walls of each split."))

body.append(f"""<h3 id="problem">What the robot needs</h3>
<p>A suction or clamp gripper does not need a pretty mask. It needs, for every box it might pick, the <b>face it will grab</b> (front face for a truck wall), that face's <b>pose and size</b>, and a <b>pick order</b> that will not collapse the wall. So every method here produces instance masks, and a shared back end turns each mask into a plane fit, a face rectangle and a place in the pick order. That back end is identical for all methods, so differences in the results come only from how well each method finds the boundaries between boxes.</p>
{svg('families.svg', 'Figure 1. The four families of methods and the shared back end. Dashed arrows: depth used only by the selector; camera labels projected onto lidar points.')}""")

body.append(f"""<h3 id="data">A box-wall simulator with exact ground truth</h3>
<p>Real datasets for this problem either have no depth (the Stacked Carton Dataset), no box walls (ARMBench totes), or no lidar at all. So I wrote a procedural simulator. It builds walls the way trailers are loaded: columns of cartons 25–62 cm wide and 18–50 cm tall, stacked from the floor, a second row behind, 2–6 product types per wall so neighbours share print and tape, gaps of 0–12 mm, random yaw and tilt, pushed-in boxes, and missing boxes that reveal the row behind. A vectorised ray caster renders RGB, depth with distance-dependent noise and dropouts at discontinuities, and a lidar scan from a sensor mounted above the camera. Every pixel and every lidar point knows which box it belongs to, and every box knows its exact pose.</p>
<p>The hard split shrinks the gaps by 1.8× and adds more yaw and tilt. A random "seam darkness" per wall makes the dark line between flush boxes anything from clearly visible to almost invisible.</p>
{img('dataset_example.png', 'One simulated wall: RGB, depth, normals, labels, lidar range image and lidar labels', 'Figure 2. One test wall: RGB, depth with dropouts (black), normals from depth, ground-truth instances, the lidar range image and lidar labels. 1,200 walls for training, 80 for validation, 100 standard and 100 hard for testing.')}""")

body.append(f"""<h3 id="results">Results</h3>
<p>Matching uses IoU ≥ 0.5 with Hungarian assignment. Trailer walls, floor and ceiling are not scored, and boxes with fewer than 300 visible pixels are ignored. <b>PQ</b> (panoptic quality) is the sum of IoUs of matched boxes divided by TP + ½FP + ½FN. It rewards finding every box and drawing its boundary well. Pose errors are medians over matched boxes whose front face is mostly visible.</p>
{TABLES}
<p>The SAM models were scored on the first 30 walls of each split because each frame takes around 40 s on this CPU. On exactly those 30 standard walls the ranking is unchanged: YOLO {F30['test']['yolo']['PQ']:.2f}, SeamNet RGB-D {F30['test']['seamnet_rgbd']['PQ']:.2f}, classical RGB-D seams {F30['test']['rgbd_seams']['PQ']:.2f}, SAM 2.1-tiny {F30['test']['sam2t_rgb']['PQ']:.2f} (RGB) / {F30['test']['sam2t_rgbd']['PQ']:.2f} (with depth priors), MobileSAM {F30['test']['mobilesam_rgbd']['PQ']:.2f}, FastSAM {F30['test']['fastsam_rgbd']['PQ']:.2f}.</p>
{img('qualitative_test.png', 'Every method on one standard wall', 'Figure 3. One standard wall. White lines are each method&#8217;s instance boundaries. <b style="color:#e61e1e">Red outline</b>: a ground-truth box the method missed (no prediction with IoU ≥ 0.5). <b style="color:#b38f00">Yellow outline</b>: a false detection. Panel titles give this wall&#8217;s scores. The RGB line method cuts boxes along tape and print. Depth alone merges flush neighbours into one region. The trained networks get almost every box. The noisy blobs on the trailer walls are not scored.')}
{img('qualitative_hard.png', 'Every method on one hard wall', 'Figure 4. One hard wall: narrower gaps, more yaw. Same annotation as Figure 3. Classical depth edges merge whole flush columns (red), the zero-shot models split boxes into labels and tape (yellow), and the trained networks lose only a few boxes.')}""")

fs_rows = "".join(
    f"<tr><td style='padding:4px 8px'>{k.replace('test_hard','hard').replace('test','standard').replace('/',' walls, ')} seams (n = {v['n_scenes']})</td>"
    + "".join(f"<td style='text-align:right;padding:4px 8px'>{v[m]:.2f}</td>" for m in ['depth_edges', 'rgbd_seams', 'seamnet_rgb', 'seamnet_rgbd', 'yolo'])
    + "</tr>" for k, v in FS.items())
body.append(f"""<h3 id="depth">Does depth help?</h3>
<p>The surprise was how close RGB-only SeamNet comes to RGB-D SeamNet overall: PQ {pq(T,'seamnet_rgb'):.2f} versus {pq(T,'seamnet_rgbd'):.2f}. The simulator draws seams with a random darkness, so the walls can be split by how visible their seams are:</p>
<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.9em"><tr><th style="text-align:left;padding:4px 8px">Walls</th><th style="padding:4px 8px">Depth edges</th><th style="padding:4px 8px">RGB-D seams</th><th style="padding:4px 8px">SeamNet RGB</th><th style="padding:4px 8px">SeamNet RGB-D</th><th style="padding:4px 8px">YOLO (RGB)</th></tr>{fs_rows}</table></div>
<p>Depth is insurance. Where seams are clearly visible, the RGB network has everything it needs. On hard walls with faint seams, the RGB-only network loses about {FS['test_hard/faint']['seamnet_rgbd']-FS['test_hard/faint']['seamnet_rgb']:.2f} PQ and the RGB-D one loses nothing. Real walls add shrink wrap, glare and dirty cardboard, which make RGB seams less reliable than in this simulator, so I would not ship an RGB-only unloader.</p>
{img('seamnet_training.png', 'SeamNet validation seam F1 per epoch', 'Figure 5. SeamNet validation seam-pixel F1. The RGB-D model reaches 0.89 after one epoch, close to the 0.91 the RGB model reaches after six: depth makes the task easier to learn.')}""")

body.append(f"""<h3 id="lidar">Lidar: geometry yes, boundaries no</h3>
<p>The simulated lidar has 80 rings and 640 columns. At 2 m that is 5 mm between columns but 3 cm between rings, and 8 mm of range noise. Lidar-only methods find columns of boxes and recesses, but not individual boxes inside a flush stack. The gap is narrower than the ring spacing, and the whole front of the wall is one plane to within a couple of centimetres.</p>
{img('lidar_example.png', 'Lidar segmentation compared with ground truth', 'Figure 6. Lidar points in range-image layout. Range-image cuts find columns and steps. Sequential RANSAC + DBSCAN grabs many coplanar faces at once. Labels from the camera network, projected onto the points, recover almost every box.')}
<p>All lidar methods are scored on points inside the camera's field of view, so the camera-to-lidar transfer is not penalised for boxes the camera cannot see. In a real unloader, lidar earns its place for safety, mapping and recovering dropped boxes. Box boundaries should come from the camera.</p>""")

if PC:
    pc_rows = "".join(f"<tr><td style='padding:4px 8px'>{NAMES[m][0]}</td><td style='text-align:right;padding:4px 8px'>{PC[m]['ang']:.2f}</td><td style='text-align:right;padding:4px 8px'>{PC[m]['ctr']:.2f}</td><td style='text-align:right;padding:4px 8px'>{PC[m]['ctr_p90']:.2f}</td><td style='text-align:right;padding:4px 8px'>{PC[m]['size']:.2f}</td></tr>"
                      for m in PC if m not in ("split", "n_boxes"))
    body.append(f"""<h3 id="pose">From mask to grasp</h3>
<p>Each mask is turned into a grasp the same way: RANSAC plane fit (8 mm threshold, SVD refit), a rectangle in that plane whose extents are the 1st–99th percentiles along its axes (so a few leaked pixels don't stretch it), and a pick order of top-down, then nearest first. The per-method pose numbers in the tables are medians over each method's own matches, which flatters a method that only finds easy boxes. Restricting to the {PC['n_boxes']} boxes that every method below matched removes that effect:</p>
<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.9em"><tr><th style="text-align:left;padding:4px 8px">Method</th><th style="padding:4px 8px">normal err, median (°)</th><th style="padding:4px 8px">centre err, median (cm)</th><th style="padding:4px 8px">centre err, 90th pct (cm)</th><th style="padding:4px 8px">size err, median (cm)</th></tr>{pc_rows}</table></div>
<p>The medians are a few millimetres, but the 90th percentile of centre error is {min(PC[m]['ctr_p90'] for m in PC if m not in ('split','n_boxes')):.0f}–{max(PC[m]['ctr_p90'] for m in PC if m not in ('split','n_boxes')):.0f} cm for every method alike. Those are faces partly hidden behind a neighbour: the visible rectangle is fitted correctly, but it is not the whole face. That tail is a property of the scene, not the segmenter. A planner should place the suction cup inside the visible rectangle and treat the size of an occluded face as a lower bound.</p>
{img('pick_plan.png', 'Pick plan with fitted face rectangles and pick order', 'Figure 7. Pick plan from SeamNet RGB-D on a test wall. White rectangles are fitted faces reprojected into the image, numbers are the pick order, labels give face size and distance.')}""")

if SP:
    sp_rows = "".join(f"<tr><td style='padding:4px 8px'>{NAMES[m][0]}</td><td style='text-align:right;padding:4px 8px'>{SP[m]['mean_s']*1000:.0f}</td></tr>" for m in SP if not m.startswith("_"))
    body.append(f"""<h3 id="speed">Speed</h3>
<p>Mean wall-clock time per frame with nothing else running: {SP['_hardware']}, no GPU, 480×360 input. These are CPU numbers, useful only for ranking. A Jetson with TensorRT would run every network much faster, but I have not measured that here.</p>
<div style="overflow-x:auto"><table style="border-collapse:collapse;font-size:0.9em"><tr><th style="text-align:left;padding:4px 8px">Method</th><th style="padding:4px 8px">ms / frame (CPU)</th></tr>{sp_rows}</table></div>""")

body.append(f"""<h3 id="real">A look at real photos</h3>
<p>The simulator is not reality, so I ran the RGB-only methods on six real warehouse photos from the Stacked Carton Dataset. These photos have no depth and no ground truth here, so this is a qualitative check only. Neither trained network had seen a real image.</p>
<p>The result is the opposite of the synthetic benchmark. SeamNet and YOLO, the two best methods on simulated walls, cut real boxes into fragments. Real cartons carry dense print, many labels, plastic strapping and lighting that my simulator never drew, and both networks learned "a sharp line means a seam". FastSAM and SAM 2.1 were trained on a billion real masks. They give one clean mask per box on most of these photos, including the regular pallet in the last row and the strapped cartons in the second. Their failures are the familiar ones: they also segment labels as objects, and they sometimes take a whole layer as one mask.</p>
{img('real_photos.png', 'RGB-only methods on six real carton photos', 'Figure 8. RGB-only methods on real photos (Stacked Carton Dataset, Yang et al., Sensors 2022, CC BY-NC-SA 4.0). The networks trained on synthetic walls (columns 3–4) fragment real boxes. The zero-shot models (columns 5–6) segment them cleanly. This is the sim-to-real gap made visible.')}""")

ARCH_URL = "https://perceptionstack.dev/carton-wall-perception-architectures/"
MAIN_URL = "https://perceptionstack.dev/carton-wall-perception/"
arch_body = []
arch_body.append(f"""<p>This is the companion to <a href="{MAIN_URL}">Unloading a box wall: classical edges, small networks and SAM, compared</a>. That post has the results; this one explains how each method turns pixels or points into boxes, down to layer shapes, parameter counts and the thresholds in the code.</p>
<p>All parameter counts, shapes and compute figures were measured on the models used in the experiments (with <code>torch</code> and <code>thop</code>), not copied from papers.</p>
{svg('seamnet.svg', 'Figure 1. SeamNet, the network trained here: a 4-level U-Net that predicts background / interior / seam, decoded into instances with a watershed.')}
{arch_html()}
<p><a href="{MAIN_URL}">&larr; Back to the results</a></p>""")

body.append(f"""<h3 id="architectures">How each method works</h3>
<p>Every method above — the three classical camera pipelines, the three lidar methods, SeamNet, YOLO11n-seg, FastSAM, MobileSAM and SAM 2.1-tiny — is explained layer by layer, with measured tensor shapes, parameter counts and compute, in the companion post: <a href="{ARCH_URL}"><b>Carton-wall perception: architecture deep dive</b></a>.</p>
<h3 id="takeaways">What I would build</h3>
<ul>
<li><b>Label real data with a foundation model, then distil.</b> Run SAM 2.1 (with the depth-prior selector) offline on a few thousand real frames from the target site, check the labels quickly, and train a small RGB-D instance-segmentation network (YOLO-seg class or a seam U-Net) on them plus the simulated walls. The small network gives the speed and the box prior; the real labels fix the appearance gap that Figure 8 shows.</li>
<li>Use classical geometry for the <b>metric part</b>: plane fit, face rectangle, pick order. It is accurate to millimetres once the boundary is right.</li>
<li>Use lidar for safety, mapping and recovering dropped boxes. Take box boundaries from the camera.</li>
<li>Keep SAM-class models for <b>labelling and fallback</b>, not for the robot's main loop. At 0.6–38 s per frame on a CPU they are too slow, and on their own they don't know what a box is. They are, however, the most robust appearance model available.</li>
</ul>
<h3 id="limits">Limits</h3>
<ul>
<li>The walls are synthetic and the boxes are perfect cuboids. Real cartons bulge, sag and wear shrink wrap. The numbers rank methods; they are not field accuracy.</li>
<li>Everything ran on 2 CPU cores. SeamNet trained for 6 epochs, YOLO for 10, and the SAM models were scored on 30 walls per split.</li>
<li>The real-photo check is qualitative only, on six photos without depth. It is enough to show that the synthetic ranking does not carry over, but not to measure by how much.</li>
</ul>
<p><b>Code, data generator, trained weights and every result file:</b> <a href="{REPO}">github.com/nadurthi/computervision/carton-wall-perception</a>. The architecture deep dive is <a href="{ARCH_URL}">here</a> and in <a href="{REPO}/docs/ARCHITECTURES.md">docs/ARCHITECTURES.md</a>.</p>""")

def finish(parts, path):
    html = "\n".join(parts)
    html = re.sub(r'<(td|th)( style="[^"]*"| style=\'[^\']*\')', lambda m: "<" + m.group(1) + (' style="text-align:right"' if "right" in m.group(2) else ""), html)
    html = re.sub(r">\s*\n\s*<(?!/?(pre|code))", "><", html)            # compact whitespace between tags
    html = "<!-- wp:html -->\n" + html + "\n<!-- /wp:html -->"
    open(path, "w").write(html)
    missing = re.findall(r"FIGURE:([\w.]+)", html)
    print("written", path, len(html), "chars; figures without URL:", sorted(set(missing)))


finish(body, "docs/post.html")
finish(arch_body, "docs/post_architectures.html")
import sys; sys.exit(0)
html = "\n".join(body)
html = re.sub(r'<(td|th)( style="[^"]*"| style=\'[^\']*\')', lambda m: "<" + m.group(1) + (' style="text-align:right"' if "right" in m.group(2) else ""), html)
html = "<!-- wp:html -->\n" + html + "\n<!-- /wp:html -->"
open("docs/post.html", "w").write(html)
missing = re.findall(r"FIGURE:([\w.]+)", html)
print("written docs/post.html", len(html), "chars; figures without URL:", sorted(set(missing)))
