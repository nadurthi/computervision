"""Pinhole geometry shared by every method and by the evaluation.

Conventions (same as MVSNet / COLMAP): K is 3x3, E is the 4x4 world->camera extrinsic, X_cam = E @ X_world,
pixel = K @ X_cam / z. Depth is the camera z coordinate (not ray length). 0 = no depth.
"""
import cv2
import numpy as np

VISIBLE, OCCLUDED, OUT_OF_VIEW, UNKNOWN = 0, 1, 2, 3


def pixel_grid(h, w):
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    return u, v


def relative_pose(E_ref, E_src):
    """T such that X_src = T @ X_ref (homogeneous)."""
    return E_src @ np.linalg.inv(E_ref)


def reproject(depth_ref, K_ref, E_ref, K_src, E_src):
    """Map every ref pixel with depth into the source view -> (u_src, v_src, z_src), all (H,W)."""
    h, w = depth_ref.shape
    u, v = pixel_grid(h, w)
    rays = np.linalg.inv(K_ref) @ np.stack([u.ravel(), v.ravel(), np.ones(h * w)])
    X = rays * depth_ref.ravel()[None]
    T = relative_pose(E_ref, E_src)
    Xs = T[:3, :3] @ X + T[:3, 3:4]
    p = K_src @ Xs
    z = p[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        us, vs = p[0] / z, p[1] / z
    return us.reshape(h, w), vs.reshape(h, w), z.reshape(h, w)


def visibility_in_source(depth_ref, K_ref, E_ref, depth_src, K_src, E_src, rel_tol=0.01, abs_tol=0.0):
    """Classify every ref pixel with GT depth by what the source camera sees there.

    VISIBLE      the same surface point is the first surface along the source ray
    OCCLUDED     something closer to the source camera hides it (z_src > D_src*(1+rel_tol) + abs_tol)
    OUT_OF_VIEW  projects outside the source image or behind the camera
    UNKNOWN      no GT depth at the ref pixel or at the projected source pixel
    """
    h, w = depth_ref.shape
    hs, ws = depth_src.shape
    us, vs, zs = reproject(depth_ref, K_ref, E_ref, K_src, E_src)
    out = np.full((h, w), UNKNOWN, dtype=np.uint8)
    valid = depth_ref > 0
    inside = valid & (zs > 0) & (us >= -0.5) & (us < ws - 0.5) & (vs >= -0.5) & (vs < hs - 0.5)
    out[valid & ~inside] = OUT_OF_VIEW
    ui = np.clip(np.round(us), 0, ws - 1).astype(np.int64)
    vi = np.clip(np.round(vs), 0, hs - 1).astype(np.int64)
    d_s = np.where(inside, depth_src[vi, ui], 0)
    known = inside & (d_s > 0)
    occl = known & (zs > d_s * (1 + rel_tol) + abs_tol)
    out[known & ~occl] = VISIBLE
    out[occl] = OCCLUDED
    return out


def visibility_count(depth_ref, K_ref, E_ref, sources, rel_tol=0.01):
    """sources: list of (depth_src, K_src, E_src). Returns (count of sources that see each pixel, per-source labels)."""
    labels = [visibility_in_source(depth_ref, K_ref, E_ref, d, K, E, rel_tol) for d, K, E in sources]
    count = np.zeros(depth_ref.shape, np.int32)
    for lab in labels:
        count += lab == VISIBLE
    return count, labels


# ------------------------------------------------------------------------------------------- rectification
class RectifiedPair:
    """Rectify (ref, src) with the known calibration so that a standard left/right stereo network can run on it,
    then convert the network's disparity back into a depth map on the *unrectified reference* pixel grid.

    The network always sees the reference as the left image: if the source sits to the left of the reference,
    both images are mirrored; if the baseline is mostly vertical, both images are transposed first.
    """

    def __init__(self, K_ref, E_ref, K_src, E_src, size_wh, max_upscale=2.0):
        w, h = size_wh
        T = relative_pose(E_ref, E_src)
        R, t = np.ascontiguousarray(T[:3, :3], np.float64), np.ascontiguousarray(T[:3, 3:4], np.float64)
        new_size = (w, h)
        for _ in range(2):
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                np.asarray(K_ref, np.float64), None, np.asarray(K_src, np.float64), None, (w, h), R, t, flags=cv2.CALIB_ZERO_DISPARITY, alpha=1, newImageSize=new_size)
            ratio = min(K_ref[0, 0] / P1[0, 0], max_upscale)
            if ratio <= 1.05:
                break
            new_size = (int(round(w * ratio)), int(round(h * ratio)))
        self.R1, self.P1, self.P2, self.size = R1, P1, P2, new_size
        self.vertical = abs(P2[1, 3]) > abs(P2[0, 3])
        t_eff = P2[1, 3] if self.vertical else P2[0, 3]
        self.flip = t_eff > 0                       # source on the -x side: mirror so ref becomes "left"
        self.f = P1[0, 0]
        self.baseline = abs(t_eff) / self.f
        self.map_ref = cv2.initUndistortRectifyMap(K_ref, None, R1, P1[:3, :3], new_size, cv2.CV_32FC1)
        self.map_src = cv2.initUndistortRectifyMap(K_src, None, R2, P2[:3, :3], new_size, cv2.CV_32FC1)
        self.K_ref, self.ref_hw = K_ref, (h, w)

    def _to_net(self, img):
        if self.vertical:
            img = np.ascontiguousarray(np.swapaxes(img, 0, 1))
        if self.flip:
            img = np.ascontiguousarray(img[:, ::-1])
        return img

    def _from_net(self, img):
        if self.flip:
            img = img[:, ::-1]
        if self.vertical:
            img = np.swapaxes(img, 0, 1)
        return np.ascontiguousarray(img)

    def rectify(self, img_ref, img_src):
        """-> (left, right) images for the network (reference is always left)."""
        l = cv2.remap(img_ref, *self.map_ref, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        r = cv2.remap(img_src, *self.map_src, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return self._to_net(l), self._to_net(r)

    def disparity_to_ref_depth(self, disp_net):
        """Network disparity (left image grid, >0) -> depth on the original reference image grid (0 = none)."""
        disp = self._from_net(disp_net.astype(np.float32))
        with np.errstate(divide="ignore"):
            z_rect = np.where(disp > 0, self.f * self.baseline / disp, 0).astype(np.float32)
        h, w = self.ref_hw
        u, v = pixel_grid(h, w)
        r = self.R1 @ (np.linalg.inv(self.K_ref) @ np.stack([u.ravel(), v.ravel(), np.ones(h * w)]))
        p = self.P1[:3, :3] @ r
        mx = (p[0] / p[2]).reshape(h, w).astype(np.float32)
        my = (p[1] / p[2]).reshape(h, w).astype(np.float32)
        z = cv2.remap(z_rect, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return np.where(z > 0, z / r[2].reshape(h, w), 0).astype(np.float32)


# ------------------------------------------------------------------------------------------- resizing
def scale_intrinsics(K, sx, sy):
    K = K.copy()
    K[0] *= sx
    K[1] *= sy
    return K


def resize_view(img, K, size_wh):
    h, w = img.shape[:2]
    out = cv2.resize(img, size_wh, interpolation=cv2.INTER_AREA if size_wh[0] < w else cv2.INTER_LINEAR)
    return out, scale_intrinsics(K, size_wh[0] / w, size_wh[1] / h)


def resize_depth(depth, size_wh):
    """Nearest-neighbour resize so invalid (0) pixels never blend with valid ones."""
    return cv2.resize(depth, size_wh, interpolation=cv2.INTER_NEAREST)
