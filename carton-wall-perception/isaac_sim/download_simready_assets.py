"""Download the NVIDIA SimReady assets used by unload_cell.py and trailer_wall.py (run with any Python, outside Isaac Sim).

    pip install huggingface_hub
    python download_simready_assets.py --out ~/simready
    python download_simready_assets.py --out ~/simready --all-cartons    # + ~70 closed cube-like cartons

Source: https://huggingface.co/datasets/nvidia/simready-assets (CC-BY-4.0). Each asset is a folder
<name>/simready_usd/sm_<name>_01.usd with its payloads/ and Textures/ next to it, so the whole folder is fetched.
"""
import argparse
import os

from huggingface_hub import snapshot_download

BOXES = [
    "box_corrugated_brown_b04", "box_corrugated_brown_b07", "box_corrugated_brown_b08", "box_corrugated_brown_b09",
    "box_corrugated_brown_b10", "box_corrugated_brown_b12", "box_corrugated_brown_b13", "box_corrugated_brown_b17",
    "box_corrugated_white_a08", "box_corrugated_white_a09", "box_corrugated_white_b04", "box_corrugated_white_b09",
    "box_cardboard_a01", "box_cardboard_a02", "box_cardboard_b01", "box_cardboard_c01", "box_cardboard_d01",
    "box_cube_brown_b04", "box_printer_brown_b01", "box_printer_brown_b03",
]
# Closed, roughly cube-shaped shipping cartons (trailer_wall.py): every box_cube_*, the corrugated brown/white
# cartons, the plain and large cardboard boxes and the closed printer-paper boxes. Open, folded, flat, long and
# office boxes are left out.
CUBE_BOXES = (
    [f"box_cube_a{i:02d}" for i in (1, 3, 6, 7, 8, 9, 10)] + ["box_cube_brown_b04", "box_cube_white_b04", "box_cube_white_b05"]
    + [f"box_corrugated_brown_b{i:02d}" for i in (4, 7, 8, 9, 10, 12, 13, 17, 19, 20, 21, 22, 26, 28, 33, 35)]
    + [f"box_corrugated_white_a{i:02d}" for i in (8, 9, 10, 11, 12, 13, 18, 19, 20, 30, 34, 36)]
    + [f"box_corrugated_white_b{i:02d}" for i in (4, 9, 10, 19, 20, 21, 22, 25, 26, 28, 33)]
    + [f"box_cardboard_{c}{i:02d}" for c in "abcd" for i in (1, 2, 3)]
    + ["box_cardboard_large_a02", "box_cardboard_large_a03", "box_cardboard_large_b01", "box_cardboard_large_b02"]
    + [f"box_printer_brown_b{i:02d}" for i in (1, 2, 3, 4, 5, 6, 7, 9, 10)]
    + [f"box_printer_white_b{i:02d}" for i in (1, 2, 4, 5, 7)]
)
CONVEYORS = ["conveyor_belt_a08", "conveyor_belt_a02", "warehouse_conveyorBelt_a01"]
PALLETS = ["pallet_a01"]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/simready"))
ap.add_argument("--boxes-only", action="store_true")
ap.add_argument("--all-cartons", action="store_true",
                help="also fetch the full set of closed cube-like cartons (~70 models) used by trailer_wall.py")
a = ap.parse_args()

names = BOXES + ([] if a.boxes_only else CONVEYORS + PALLETS)
if a.all_cartons:
    names += [n for n in CUBE_BOXES if n not in names]
snapshot_download(repo_id="nvidia/simready-assets", repo_type="dataset", local_dir=a.out,
                  allow_patterns=[f"{n}/simready_usd/**" for n in names])
print("downloaded to", a.out)
for n in names:
    p = os.path.join(a.out, n, "simready_usd", f"sm_{n}_01.usd")
    alt = os.path.isdir(os.path.join(a.out, n, "simready_usd"))
    print(("ok   " if os.path.exists(p) else ("other" if alt else "MISS ")) + " " + p)
