"""Shared pieces of the trailer scenes (trailer_unload.py): trailer interior, lights, SimReady carton models and
the packed box-wall layout. Import it after SimulationApp has started (it imports pxr)."""
import glob
import math
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

# 53 ft dry-van interior; x = 0 at the door, back (nose) wall at x = L, floor at z = 0
L, W, H = 16.0, 2.46, 2.70


# ------------------------------------------------------------------------------------------ USD helpers
def add_label(prim, label):
    try:
        from isaacsim.core.utils.semantics import add_labels
        add_labels(prim, labels=[label], instance_name="class")
    except Exception:
        try:
            from isaacsim.core.utils.semantics import add_update_semantics
            add_update_semantics(prim, label)
        except Exception:
            pass


def material(stage, path, rgb, rough=0.8, metal=0.0):
    m = UsdShade.Material.Define(stage, path)
    s = UsdShade.Shader.Define(stage, path + "/S")
    s.CreateIdAttr("UsdPreviewSurface")
    s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
    return m


def physics_material(stage, path, static_friction, dynamic_friction, restitution=0.0):
    m = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
    api.CreateStaticFrictionAttr(static_friction)
    api.CreateDynamicFrictionAttr(dynamic_friction)
    api.CreateRestitutionAttr(restitution)
    return m


def bind_physics_material(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def cube(stage, path, center, size, mat=None, label=None, collide=False, phys_mat=None):
    c = UsdGeom.Cube.Define(stage, path)
    c.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(c)
    xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, center)))
    xf.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    if mat is not None:
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)
    if label:
        add_label(c.GetPrim(), label)
    if collide:
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        if phys_mat is not None:
            bind_physics_material(c.GetPrim(), phys_mat)
    return c


def bounds(prim):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    return cache.ComputeWorldBound(prim).ComputeAlignedRange()


def look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)):
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = Gf.Matrix4d(1.0)
    m.SetRow(0, Gf.Vec4d(*r, 0)); m.SetRow(1, Gf.Vec4d(*u, 0)); m.SetRow(2, Gf.Vec4d(*(-f), 0))
    m.SetRow(3, Gf.Vec4d(*eye, 1))
    return m


def strip_physics(root, keep_root=True):
    """Remove rigid bodies, colliders and articulation roots that referenced assets ship with (instanced
    sub-prims are made editable first, otherwise their physics survives)."""
    for _ in range(4):
        inst = [q for q in Usd.PrimRange(root) if q.IsInstance()]
        if not inst:
            break
        for q in inst:
            q.SetInstanceable(False)
    joints = [q for q in Usd.PrimRange(root) if q.IsA(UsdPhysics.Joint)]
    for q in joints:                                   # joints between stripped bodies only raise errors
        q.SetActive(False)
    for q in Usd.PrimRange(root):
        if keep_root and q == root:
            continue
        for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI,
                    UsdPhysics.ArticulationRootAPI, UsdPhysics.MassAPI):
            if q.HasAPI(api):
                q.RemoveAPI(api)


# ------------------------------------------------------------------------------------------ trailer
def build_trailer(stage, collide=True, phys_mat=None, dock=True):
    """Trailer interior from collision slabs. Posts and rails are siblings of the walls (children of a Cube
    gprim are not rendered). With dock=True a dock floor continues outside the door (x < 0)."""
    steel = material(stage, "/World/Looks/Steel", (0.80, 0.81, 0.82), 0.40, 0.7)
    wood = material(stage, "/World/Looks/Floor", (0.40, 0.28, 0.18), 0.85)
    liner = material(stage, "/World/Looks/Liner", (0.86, 0.86, 0.84), 0.6)
    concrete = material(stage, "/World/Looks/Concrete", (0.55, 0.55, 0.53), 0.9)
    b = "/World/Trailer"
    UsdGeom.Xform.Define(stage, b)
    kw = dict(collide=collide, phys_mat=phys_mat)
    cube(stage, b + "/Floor", (L / 2, 0, -0.025), (L, W + 0.1, 0.05), wood, "trailer", **kw)
    cube(stage, b + "/Roof", (L / 2, 0, H + 0.03), (L, W + 0.1, 0.06), liner, "trailer", **kw)
    cube(stage, b + "/Nose", (L + 0.03, 0, H / 2), (0.06, W + 0.1, H), liner, "trailer", **kw)
    for side, y in (("L", W / 2 + 0.03), ("R", -W / 2 - 0.03)):
        cube(stage, f"{b}/Wall{side}", (L / 2, y, H / 2), (L, 0.06, H), liner, "trailer", **kw)
        UsdGeom.Xform.Define(stage, f"{b}/Trim{side}")
        for i, x in enumerate(np.arange(0.4, L, 0.6)):                 # vertical logistics posts
            cube(stage, f"{b}/Trim{side}/Post{i:02d}", (x, y - np.sign(y) * 0.04, H / 2), (0.05, 0.02, H),
                 steel, "trailer", **kw)
        for k, z in enumerate((0.9, 1.7)):                               # logistics track rails
            cube(stage, f"{b}/Trim{side}/Rail{k}", (L / 2, y - np.sign(y) * 0.045, z), (L, 0.012, 0.12),
                 steel, "trailer", **kw)
    if dock:
        cube(stage, "/World/Dock/Floor", (-3.0, 0, -0.025), (6.0, 6.0, 0.05), concrete, "dock", **kw)


def build_lights(stage, camera_x):
    UsdLux.DomeLight.Define(stage, "/World/Lights/Dome").CreateIntensityAttr(150.0)
    for k, x in enumerate(np.arange(L - 1.0, L - 9.0, -2.0)):           # ceiling lights inside the box
        rl = UsdLux.RectLight.Define(stage, f"/World/Lights/Ceiling{k}")
        rl.CreateIntensityAttr(9000.0); rl.CreateWidthAttr(0.6); rl.CreateHeightAttr(0.25)
        UsdGeom.Xformable(rl).AddTranslateOp().Set(Gf.Vec3d(float(x), 0.0, H - 0.02))
    dl = UsdLux.SphereLight.Define(stage, "/World/Lights/Dock")         # dock light behind the camera
    dl.CreateIntensityAttr(60000.0); dl.CreateRadiusAttr(0.15)
    UsdGeom.Xformable(dl).AddTranslateOp().Set(Gf.Vec3d(camera_x - 1.0, 0.6, 2.2))


# ------------------------------------------------------------------------------------------ SimReady cartons
SKIP = ("_open", "_folded", "_flat", "_long", "_glove", "_paper_office", "composition")


def box_assets(folder):
    """One USD per SimReady carton folder (<name>/simready_usd/sm_<name>*.usd); open/flat/long boxes skipped."""
    if not folder:
        return []
    f = []
    for d in sorted(glob.glob(os.path.join(folder, "box_*", "simready_usd"))):
        name = os.path.basename(os.path.dirname(d))
        if any(k in name for k in SKIP):
            continue
        c = sorted(glob.glob(os.path.join(d, f"sm_{name}*.usd*"))) or sorted(glob.glob(os.path.join(d, "*.usd*")))
        if c:
            f.append(c[0])
    return f


def measure_assets(stage, urls):
    """Native bounding-box size of each model (referenced once under a scratch prim, then removed)."""
    out = []
    for k, url in enumerate(urls):
        path = f"/World/_probe{k}"
        try:
            UsdGeom.Xform.Define(stage, path).GetPrim().GetReferences().AddReference(url)
            ext = np.array(bounds(stage.GetPrimAtPath(path)).GetSize(), float)
            if np.all(ext > 1e-6):
                out.append((url, ext))
        except Exception as e:
            print(f"[trailer] cannot load {url}: {e}")
        stage.RemovePrim(path)
    return out


def pick_model(models, size, rng, used=None, k=4):
    """Among the k models whose proportions are closest to `size` (least stretching of the print), prefer one
    that has been used least so far."""
    t = np.log(np.asarray(size, float))
    cost = np.array([np.ptp(t - np.log(ext)) for _, ext in models])
    best = list(np.argsort(cost)[:k])
    if used is not None:
        n = min(used.get(models[i][0], 0) for i in best)
        best = [i for i in best if used.get(models[i][0], 0) == n]
    url = models[int(rng.choice(best))][0]
    if used is not None:
        used[url] = used.get(url, 0) + 1
    return url


def visual_box(stage, path, url, size):
    """Reference a carton model under `path`/fit, scaled so its bounding box is exactly `size` with the
    bottom centre at the origin of `path`. Physics is stripped: the caller's collider does the physics."""
    fit = UsdGeom.Xform.Define(stage, path + "/fit")
    asset = UsdGeom.Xform.Define(stage, path + "/fit/asset").GetPrim()
    asset.GetReferences().AddReference(url)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    r = cache.ComputeRelativeBound(asset, fit.GetPrim()).ComputeAlignedRange()   # in the fit prim's frame
    ext, ctr = np.array(r.GetSize()), np.array(r.GetMidpoint())
    if np.any(ext <= 1e-6):
        raise RuntimeError("empty bounds")
    s = np.asarray(size) / ext
    fx = UsdGeom.Xformable(fit)
    fx.AddTranslateOp().Set(Gf.Vec3d(-s[0] * ctr[0], -s[1] * ctr[1], -s[2] * r.GetMin()[2]))
    fx.AddScaleOp().Set(Gf.Vec3f(*s))
    strip_physics(fit.GetPrim(), keep_root=False)


# ------------------------------------------------------------------------------------------ packed wall layout
def plan_packed_wall(rng, rows=2, size_min=0.30, size_max=0.55, cube_ness=0.15, yaw_max=1.0, push=0.03,
                     gap=0.004, ceiling_clearance=0.015):
    """Columns across the trailer width, each stacked from the floor up to the ceiling (the last box is sized to
    the space left), rows from the back wall towards the door. Front faces are not aligned (each box has its own
    depth and is pulled towards the door by up to `push`, with a small yaw); widths are shrunk for the yaw so no
    two boxes intersect, which matters because the scene starts with physics on."""
    def half_w(d, w, yaw):
        return 0.5 * (w * abs(math.cos(yaw)) + d * abs(math.sin(yaw)))

    boxes, prev = [], None
    usable = W - 0.03
    z_top = H - ceiling_clearance
    for row in range(rows):
        y = -usable / 2
        row_front = []
        while y < usable / 2 - size_min * 0.8:
            col_w = min(rng.uniform(size_min, size_max), usable / 2 - y)
            if prev is None:
                back_x = L - 0.004
            else:
                behind = [f for (y0, y1, f) in prev if y0 < y + col_w and y1 > y]
                back_x = (min(behind) if behind else min(f for _, _, f in prev)) - gap
            z = 0.0
            while True:
                left = z_top - z
                if left < 0.6 * size_min:
                    break
                s = rng.uniform(size_min, size_max)
                h = s * rng.uniform(1 - cube_ness, 1 + cube_ness)
                if h > left - 0.002 or left - h < 0.6 * size_min:      # last box of the column: fill to the top
                    h = left - 0.002
                d = s * rng.uniform(1 - cube_ness, 1 + cube_ness)
                yaw = math.radians(rng.uniform(-yaw_max, yaw_max))
                w_max = (col_w - gap - abs(d * math.sin(yaw))) / max(abs(math.cos(yaw)), 1e-3)
                w = max(0.5 * col_w, min(w_max, col_w * rng.uniform(0.95, 1.0)))
                hy = half_w(d, w, yaw)
                hx = 0.5 * (d * abs(math.cos(yaw)) + w * abs(math.sin(yaw)))
                cx = back_x - hx - rng.uniform(0.0, push)
                cy = y + col_w / 2 + rng.uniform(-0.5, 0.5) * max(0.0, col_w - gap - 2 * hy)
                boxes.append({"row": row, "size_m": [d, w, h], "center_m": [cx, cy, z + h / 2],
                              "yaw_deg": math.degrees(yaw), "front_face_x_m": cx - hx,
                              "half_extent_xy_m": [hx, hy]})
                row_front.append((cy - hy, cy + hy, cx - hx))
                z += h + 0.002
            y += col_w
        prev = row_front or prev
    return boxes
