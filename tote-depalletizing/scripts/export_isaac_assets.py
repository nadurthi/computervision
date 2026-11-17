"""Export what the Isaac Sim tote scene needs (tote + pallet OBJ, stack layout with poses and pick order) into
carton-wall-perception/isaac_sim/assets/totes/, so the Isaac script (which only has numpy) reads one JSON.

    python scripts/export_isaac_assets.py [--cad my_tote.obj]
"""
import argparse, json, os, shutil, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from totes.cad import ToteSpec, front_face
from totes.scene import make_stack, pick_order

here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(here, "..", "..", "carton-wall-perception", "isaac_sim", "assets", "totes"))
ap.add_argument("--cad", default=os.path.join(here, "..", "assets", "tote_600x400x300.obj"))
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
shutil.copy(a.cad, os.path.join(a.out, "tote.obj"))
shutil.copy(os.path.join(here, "..", "assets", "eur_pallet.obj"), os.path.join(a.out, "pallet.obj"))
spec = ToteSpec(); totes = make_stack(seed=a.seed, spec=spec); c, n = front_face(spec)
json.dump(dict(pallet_x=2.0, sensor=[0.1, 0.0, 1.2], camera=[0.1, -0.15, 1.2], front_face_center=c.tolist(),
               front_face_normal=n.tolist(), tote_bounds=[[-0.3, -0.204, 0.0], [0.3, 0.204, 0.3]],
               pick_order=pick_order(totes),
               totes=[dict(id=t["id"], layer=t["layer"], row=t["row"], col=t["col"], T=np.asarray(t["T"]).tolist()) for t in totes]),
          open(os.path.join(a.out, "stack_layout.json"), "w"), indent=1)
print("wrote", os.path.abspath(a.out))
