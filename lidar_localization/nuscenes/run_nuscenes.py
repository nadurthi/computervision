"""Run the localizer on nuScenes scenes and save results + figures.

usage: python run_nuscenes.py <nuscenes_root> <out_dir> [scene ...]

For each scene the map is built from OTHER drives (different logs) through the same area when they exist
("cross-drive map"); otherwise from the same drive's sweeps offset in time ("same-drive map", flagged as optimistic).
"""
import json, os, sys, time
import numpy as np
import nusc_data as nd
import localize, plots

root, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
ns = nd.NuScenes(root, "v1.0-mini")
scenes = sys.argv[3:] or sorted(ns.scenes)
summary = {}
for sc in scenes:
    loc = ns.location(sc)
    frames = nd.sweep_frames(ns, sc, every=2)                     # ~10 Hz, never in the map
    Xmap, minfo = nd.static_keyframe_map(ns, sc)
    kind, map_scenes = "static keyframe map (same drive, annotated objects removed)", [sc]
    print(f"\n=== {sc} ({loc}) {len(frames)} frames, map: {kind} from {map_scenes}, {len(Xmap):,} pts", flush=True)
    res = {}
    for mode, cfg in [("tracking", dict(Npf=1000, init="tracking", s2m_every=5, Lmax=10.0, thmax_deg=15.0)),
                      ("tracking_paper", dict(Npf=1000, init="tracking", s2m_every=5, Lmax=10.0, thmax_deg=15.0, sig0=0.5, dmax=10.0)),
                      ("global", dict(Npf=2000, init="global", s2m_every=5, Lmax=10.0, thmax_deg=15.0, adaptive=True, recheck_every=20))]:
        if os.environ.get("MODES") and mode not in os.environ["MODES"].split(","):
            res[mode] = json.load(open(os.path.join(out, "summary_all.json")))[sc][mode] if os.path.exists(os.path.join(out, "summary_all.json")) else None
            continue
        st = time.time()
        recs, T = localize.run(frames, Xmap, cfg, log=lambda m: print("  ", m, flush=True))
        S = localize.summarize(recs); S["timing_per_frame_s"] = T; S["wall_s"] = time.time() - st
        res[mode] = dict(summary=S, cfg=cfg)
        json.dump(recs, open(os.path.join(out, f"{sc}_{mode}.json"), "w"))
        plots.trajectory(recs, Xmap, os.path.join(out, f"{sc}_{mode}_traj.png"), f"{sc} ({loc}) — {mode}, {kind} map")
        print(f"   {mode}: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in S.items() if k != 'timing_per_frame_s'})}", flush=True)
    plots.errors([(m, json.load(open(os.path.join(out, f"{sc}_{m}.json"))), c) for m, c in [("tracking", "#2563eb"), ("tracking_paper", "#9ca3af"), ("global", "#dc2626")]],
                 os.path.join(out, f"{sc}_errors.png"), f"{sc} ({loc}), {kind} map")
    summary[sc] = dict(location=loc, map=kind, map_info=minfo, map_scenes=map_scenes, frames=len(frames), map_points=len(Xmap), **res)
    json.dump(summary, open(os.path.join(out, f"summary_{'_'.join(scenes)}.json"), "w"), indent=1)
