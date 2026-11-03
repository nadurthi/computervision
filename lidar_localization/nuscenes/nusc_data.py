"""Minimal nuScenes reader (no devkit): LIDAR_TOP sweeps + ground-truth lidar poses for one scene.

Expects the standard layout:  <root>/<version>/{scene,sample,sample_data,ego_pose,calibrated_sensor,sensor,log}.json
and  <root>/samples/LIDAR_TOP/*.pcd.bin,  <root>/sweeps/LIDAR_TOP/*.pcd.bin  (float32 x,y,z,intensity,ring).
"""
import json, os
import numpy as np
from scipy.spatial.transform import Rotation


def _H(translation, rotation_wxyz):
    H = np.eye(4)
    w, x, y, z = rotation_wxyz
    H[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    H[:3, 3] = translation
    return H


class NuScenes:
    def __init__(self, root, version="v1.0-mini"):
        self.root = root
        tab = lambda n: json.load(open(os.path.join(root, version, n + ".json")))
        self.scenes = {s["name"]: s for s in tab("scene")}
        self.samples = {s["token"]: s for s in tab("sample")}
        self.sd = tab("sample_data")
        self.ego = {e["token"]: e for e in tab("ego_pose")}
        self.cal = {c["token"]: c for c in tab("calibrated_sensor")}
        sensor = {s["token"]: s["channel"] for s in tab("sensor")}
        self.logs = {l["token"]: l for l in tab("log")}
        self.lidar_cal = {t for t, c in self.cal.items() if sensor[c["sensor_token"]] == "LIDAR_TOP"}

    def location(self, scene_name):
        return self.logs[self.scenes[scene_name]["log_token"]]["location"]

    def lidar_records(self, scene_name, keyframes_only=False):
        """All LIDAR_TOP sample_data of a scene, time ordered, with the 4x4 lidar->global pose."""
        tokens = set()
        tok = self.scenes[scene_name]["first_sample_token"]
        while tok:
            tokens.add(tok); tok = self.samples[tok]["next"]
        recs = [r for r in self.sd if r["sample_token"] in tokens and r["calibrated_sensor_token"] in self.lidar_cal
                and (r["is_key_frame"] or not keyframes_only)]
        recs.sort(key=lambda r: r["timestamp"])
        out = []
        for r in recs:
            e, c = self.ego[r["ego_pose_token"]], self.cal[r["calibrated_sensor_token"]]
            # Track the EGO frame (x forward, as the CT model assumes). nuScenes' LIDAR_TOP frame has y forward,
            # so points are moved into the ego frame with the lidar extrinsics and H is the ego->global pose.
            out.append(dict(t=r["timestamp"] * 1e-6, file=r["filename"], H=_H(e["translation"], e["rotation"]),
                            Hcal=_H(c["translation"], c["rotation"]), key=r["is_key_frame"]))
        return out

    def load_points(self, filename, Hcal=None, min_range=2.5):
        """Points in the ego frame (if Hcal given) with returns from the car itself removed."""
        X = np.fromfile(os.path.join(self.root, filename), dtype=np.float32).reshape(-1, 5)[:, :3].astype(np.float64)
        X = X[np.linalg.norm(X[:, :2], axis=1) > min_range]
        return X if Hcal is None else X @ Hcal[:3, :3].T + Hcal[:3, 3]


def frames_for(ns, scene, every=2):
    """Sweeps at 20 Hz -> every=2 gives 10 Hz like KITTI."""
    recs = ns.lidar_records(scene)[::every]
    return [dict(t=r["t"], X=ns.load_points(r["file"], r["Hcal"]), H=r["H"]) for r in recs]


def build_map(ns, scenes, every=4, voxel=0.2):
    """Global point map from the sweeps of `scenes` placed with their ground-truth poses."""
    from lidarloc import voxel_down
    P = []
    for s in scenes:
        for r in ns.lidar_records(s)[::every]:
            X = ns.load_points(r["file"], r["Hcal"])
            P.append(voxel_down(X @ r["H"][:3, :3].T + r["H"][:3, 3], voxel))
    return voxel_down(np.vstack(P), voxel)


def overlapping_scenes(ns, scene, radius=30.0, min_shared=0.5):
    """Other scenes (different logs, same location) whose trajectories cover >= min_shared of this scene's path."""
    loc = ns.location(scene)
    path = np.array([r["H"][:2, 3] for r in ns.lidar_records(scene, keyframes_only=True)])
    out = []
    for name, s in ns.scenes.items():
        if name == scene or ns.location(name) != loc or s["log_token"] == ns.scenes[scene]["log_token"]:
            continue
        other = np.array([r["H"][:2, 3] for r in ns.lidar_records(name, keyframes_only=True)])
        d = np.min(np.linalg.norm(path[:, None] - other[None], axis=2), axis=1)
        frac = float(np.mean(d < radius))
        if frac >= min_shared:
            out.append((name, frac))
    return sorted(out, key=lambda x: -x[1])


def static_keyframe_map(ns, scene, inflate=0.5, voxel=0.15):
    """Map from the 2 Hz keyframe scans of `scene` with every annotated object (vehicles, pedestrians, ...) cut out
    using the ground-truth 3-D boxes. The 10 Hz sweeps used for localization are never in the map."""
    from lidarloc import voxel_down
    ann = json.load(open(os.path.join(ns.root, "v1.0-mini", "sample_annotation.json")))
    by_sample = {}
    for a in ann:
        by_sample.setdefault(a["sample_token"], []).append(a)
    P, removed, total = [], 0, 0
    for r in [x for x in ns.sd if x["is_key_frame"] and x["calibrated_sensor_token"] in ns.lidar_cal]:
        smp = ns.samples[r["sample_token"]]
        if smp["scene_token"] != ns.scenes[scene]["token"]:
            continue
        e, c = ns.ego[r["ego_pose_token"]], ns.cal[r["calibrated_sensor_token"]]
        H = _H(e["translation"], e["rotation"]) @ _H(c["translation"], c["rotation"])
        X = ns.load_points(r["filename"]) @ H[:3, :3].T + H[:3, 3]
        keep = np.ones(len(X), bool)
        for a in by_sample.get(r["sample_token"], []):
            w, l, h = a["size"]
            Rb = _H(a["translation"], a["rotation"])
            Y = (X - Rb[:3, 3]) @ Rb[:3, :3]                     # into box frame (x along length)
            inside = (np.abs(Y[:, 0]) < l / 2 + inflate) & (np.abs(Y[:, 1]) < w / 2 + inflate) & (np.abs(Y[:, 2]) < h / 2 + inflate)
            keep &= ~inside
        removed += (~keep).sum(); total += len(X)
        P.append(X[keep])
    return voxel_down(np.vstack(P), voxel), dict(keyframes=len(P), removed_frac=removed / max(total, 1))


def sweep_frames(ns, scene, every=2):
    """Non-keyframe sweeps only (never in the keyframe map), subsampled to ~10 Hz."""
    recs = [r for r in ns.lidar_records(scene) if not r["key"]][::every]
    return [dict(t=r["t"], X=ns.load_points(r["file"], r["Hcal"]), H=r["H"]) for r in recs]
