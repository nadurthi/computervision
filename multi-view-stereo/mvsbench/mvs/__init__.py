"""MVS method factory. Names:

classical : planesweep_mean, planesweep_bestk, planesweep_bestk_sgm, patchmatch, acmmp, colmap
learned   : patchmatchnet, itermvs_dtu, itermvs_blendedmvs, mvsanywhere, vis_mvsnet, robust_mvd
stereo    : stereo_sgbm, stereo_sgbm_fill, stereo_<any name in mvsbench.stereo.deep.ALL>  (two views only)
"""
import os

CLASSICAL = ["planesweep_mean", "planesweep_bestk", "planesweep_bestk_sgm", "patchmatch", "acmmp", "colmap"]
SCENE_LEVEL = ["acmmp", "colmap"]          # run on a whole view set at once via run_scene()
LEARNED = ["patchmatchnet", "itermvs_dtu", "itermvs_blendedmvs", "mvsanywhere", "vis_mvsnet", "robust_mvd"]
STEREO = ["stereo_sgbm", "stereo_sgbm_fill", "stereo_raft_stereo_middlebury", "stereo_foundation_stereo",
          "stereo_igev_stereo_sceneflow", "stereo_monster_mix_all"]


def build(name, models_root, device="cuda", units_to_m=1.0):
    if name.startswith("planesweep_"):
        from .planesweep import PlaneSweep
        agg = "mean" if "mean" in name else "bestk"
        return PlaneSweep(aggregation=agg, use_sgm=name.endswith("_sgm"), device=device)
    if name == "patchmatch":
        from .patchmatch import PatchMatchMVS
        return PatchMatchMVS(device=device)
    if name == "colmap":
        from .colmap import COLMAP
        return COLMAP()
    if name == "acmmp":
        from .acmmp import ACMMP
        return ACMMP(os.path.join(models_root, "ACMMP", "build", "ACMMP"))
    if name.startswith("stereo_"):
        from .stereo_as_mvs import StereoAsMVS
        sname = name[len("stereo_"):]
        if sname.startswith("sgbm"):
            from ..stereo.sgbm import SGBM
            return StereoAsMVS(SGBM(fill=sname.endswith("fill")))
        from ..stereo.deep import build as build_stereo
        return StereoAsMVS(build_stereo(sname, models_root, device, max_disp=384))
    from .learned import build as build_learned
    return build_learned(name, models_root, device, units_to_m)
