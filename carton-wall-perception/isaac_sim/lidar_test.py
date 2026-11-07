"""RTX lidar check: a floor, a wall and a box in front of an Example_Rotary lidar; prints how many points each
way of reading the lidar returns. Used to pick the reading method for box_wall_capture.py / unload_cell.py.

    ./python.sh lidar_test.py --out /out/lidar_test [--config Example_Rotary]
"""
import argparse
import os
import sys
import traceback

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="lidar_test_out")
ap.add_argument("--config", default="Example_Rotary")
args, _ = ap.parse_known_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
ok = False
try:
    import numpy as np
    import omni.kit.app
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    for name, c, s in (("Floor", (0, 0, -0.05), (20, 20, 0.1)), ("Wall", (3.0, 0, 1.5), (0.2, 8, 3)),
                       ("Box", (2.0, 0.3, 0.25), (0.5, 0.5, 0.5))):
        cube = UsdGeom.Cube.Define(stage, f"/World/{name}")
        cube.CreateSizeAttr(1.0)
        UsdGeom.Xformable(cube).AddTranslateOp().Set(Gf.Vec3d(*c))
        UsdGeom.Xformable(cube).AddScaleOp().Set(Gf.Vec3f(*s))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(1000.0)
    upd = omni.kit.app.get_app().update
    results = {}

    # A) LidarRtx class + IsaacExtractRTXSensorPointCloudNoAccumulator (the documented standalone path)
    try:
        from isaacsim.sensors.rtx import LidarRtx
        lidar = LidarRtx(prim_path="/World/LidarA", translation=np.array([0.0, 0.0, 1.0]),
                         orientation=np.array([1.0, 0.0, 0.0, 0.0]), config_file_name=args.config)
        lidar.initialize()
        lidar.attach_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")
        results["A_ready"] = True
    except Exception as e:
        results["A_error"] = repr(e)
        lidar = None

    # B) command + replicator ScanBuffer annotator (what the box-wall scripts use)
    try:
        _, sensor = omni.kit.commands.execute("IsaacSensorCreateRtxLidar", path="/World/LidarB", parent=None,
                                              config=args.config, translation=Gf.Vec3d(0, 0, 1.0),
                                              orientation=Gf.Quatd(1, 0, 0, 0))
        rp = rep.create.render_product(sensor.GetPath(), [1, 1], name="LidarB_RP")
        ann_b = rep.AnnotatorRegistry.get_annotator("IsaacCreateRTXLidarScanBuffer")
        ann_b.attach([rp.path])
        ann_c = rep.AnnotatorRegistry.get_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")
        ann_c.attach([rp.path])
    except Exception as e:
        results["B_error"] = repr(e)
        ann_b = ann_c = None

    omni.timeline.get_timeline_interface().play()
    for i in range(150):
        upd()

    def n_points(d):
        if d is None:
            return None
        if isinstance(d, dict):
            arr = d.get("data")
            if arr is None:
                return {k: getattr(v, "shape", type(v).__name__) for k, v in d.items()}
            return int(np.asarray(arr).reshape(-1, 3).shape[0]) if np.size(arr) else 0
        return getattr(d, "shape", type(d).__name__)

    if lidar is not None:
        try:
            dA = lidar.get_annotator_data("IsaacExtractRTXSensorPointCloudNoAccumulator")
        except Exception:
            dA = lidar.get_current_frame().get("IsaacExtractRTXSensorPointCloudNoAccumulator")
        results["A_points"] = n_points(dA)
    if ann_b is not None:
        results["B_scanbuffer_points"] = n_points(ann_b.get_data())
        results["B_extract_points"] = n_points(ann_c.get_data())
    print("[lidar-test]", results, flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "lidar_test.txt"), "w") as f:
        f.write(repr(results) + "\n")
    ok = True
except Exception:
    traceback.print_exc()
finally:
    sys.stdout.flush()
    app.close()
    os._exit(0 if ok else 1)
