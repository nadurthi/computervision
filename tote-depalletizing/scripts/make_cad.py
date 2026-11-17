"""Write the stand-in tote CAD (OBJ + STL) and the pallet mesh, and print the front-face definition.

    python scripts/make_cad.py --out assets
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from totes.cad import ToteSpec, make_tote, front_face, make_pallet

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="assets")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
s = ToteSpec()
t = make_tote(s)
t.export(os.path.join(a.out, "tote_600x400x300.obj"))
t.export(os.path.join(a.out, "tote_600x400x300.stl"))
make_pallet().export(os.path.join(a.out, "eur_pallet.obj"))
c, n = front_face(s)
info = {"tote": "tote_600x400x300.obj", "units": "m", "frame": "origin base centre, x 600 side, y 400 side, z up; front = -y",
        "front_face_center": c.round(5).tolist(), "front_face_normal": n.round(5).tolist(),
        "watertight": bool(t.is_watertight), "triangles": int(len(t.faces)), "bounds": t.bounds.round(4).tolist()}
json.dump(info, open(os.path.join(a.out, "tote_cad.json"), "w"), indent=1)
print(json.dumps(info, indent=1))
