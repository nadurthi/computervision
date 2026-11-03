"""Download the NVIDIA SimReady assets used by unload_cell.py (run with any Python, outside Isaac Sim).

    pip install huggingface_hub
    python download_simready_assets.py --out ~/simready

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
CONVEYORS = ["conveyor_belt_a08", "conveyor_belt_a02", "warehouse_conveyorBelt_a01"]
PALLETS = ["pallet_a01"]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.expanduser("~/simready"))
ap.add_argument("--boxes-only", action="store_true")
a = ap.parse_args()

names = BOXES + ([] if a.boxes_only else CONVEYORS + PALLETS)
snapshot_download(repo_id="nvidia/simready-assets", repo_type="dataset", local_dir=a.out,
                  allow_patterns=[f"{n}/simready_usd/**" for n in names])
print("downloaded to", a.out)
for n in names:
    p = os.path.join(a.out, n, "simready_usd", f"sm_{n}_01.usd")
    print(("ok   " if os.path.exists(p) else "MISS ") + p)
