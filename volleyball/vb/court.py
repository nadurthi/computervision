"""Court geometry: a homography between the image and the court plane, recomputed every frame.

Court frame (metres): X in [0, 9] across, Y in [0, 18] along, net at Y = 9, attack lines at
Y = 6 and Y = 12. H maps court -> pixels:  s [u, v, 1]^T = H [X, Y, 1]^T.

The camera pans and zooms, so H changes every frame. This module seeds H from >= 4 hand-picked
correspondences on the first frame and then follows the camera with a frame-to-frame homography
estimated from background features (player boxes masked out):  H_t = G_{t-1 -> t} . H_{t-1}.
A court-keypoint detector can replace the seed later; the interface stays the same.
"""
from __future__ import annotations

import cv2
import numpy as np

COURT_W, COURT_L, NET_Y = 9.0, 18.0, 9.0

# Named court points (metres) that are usually visible from an end-line camera.
COURT_POINTS: dict[str, tuple[float, float]] = {
    "near_left": (0.0, 0.0), "near_right": (9.0, 0.0),
    "far_left": (0.0, 18.0), "far_right": (9.0, 18.0),
    "near_attack_left": (0.0, 6.0), "near_attack_right": (9.0, 6.0),
    "far_attack_left": (0.0, 12.0), "far_attack_right": (9.0, 12.0),
    "centre_left": (0.0, 9.0), "centre_right": (9.0, 9.0),
}

# Court lines as segments, for drawing and for checking a homography by eye.
COURT_LINES = [((0, 0), (9, 0)), ((0, 18), (9, 18)), ((0, 0), (0, 18)), ((9, 0), (9, 18)),
               ((0, 9), (9, 9)), ((0, 6), (9, 6)), ((0, 12), (9, 12))]


def dlt_homography(court_pts: np.ndarray, img_pts: np.ndarray) -> np.ndarray:
    """Plain DLT (the 2n x 9 system on the blog page), normalised for conditioning."""
    court_pts = np.asarray(court_pts, np.float64)
    img_pts = np.asarray(img_pts, np.float64)
    if len(court_pts) < 4:
        raise ValueError("need at least 4 correspondences")

    def norm(p):
        c = p.mean(0)
        s = np.sqrt(2) / np.mean(np.linalg.norm(p - c, axis=1))
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])

    Tc, Ti = norm(court_pts), norm(img_pts)
    pc = (Tc @ np.c_[court_pts, np.ones(len(court_pts))].T).T
    pi = (Ti @ np.c_[img_pts, np.ones(len(img_pts))].T).T
    rows = []
    for (X, Y, _), (u, v, _) in zip(pc, pi):
        rows.append([X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u])
        rows.append([0, 0, 0, X, Y, 1, -v * X, -v * Y, -v])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Ti) @ Hn @ Tc
    return H / H[2, 2]


def court_homography(court_pts: np.ndarray, img_pts: np.ndarray, ransac_px: float = 3.0):
    """Robust H (court metres -> pixels) with RANSAC; returns (H, inlier_mask)."""
    court_pts = np.asarray(court_pts, np.float32)
    img_pts = np.asarray(img_pts, np.float32)
    if len(court_pts) == 4:
        return dlt_homography(court_pts, img_pts), np.ones(4, bool)
    H, mask = cv2.findHomography(court_pts, img_pts, cv2.RANSAC, ransac_px)
    if H is None:
        raise RuntimeError("homography failed; check the seed points")
    return H / H[2, 2], mask.ravel().astype(bool)


def to_court(H: np.ndarray, px: np.ndarray) -> np.ndarray:
    """Pixels (N,2) on the floor -> court metres (N,2). Only valid for points ON the floor."""
    px = np.asarray(px, np.float32).reshape(-1, 1, 2)
    if len(px) == 0:
        return np.zeros((0, 2), np.float32)
    return cv2.perspectiveTransform(px, np.linalg.inv(H)).reshape(-1, 2)


def to_image(H: np.ndarray, court: np.ndarray) -> np.ndarray:
    court = np.asarray(court, np.float32).reshape(-1, 1, 2)
    if len(court) == 0:
        return np.zeros((0, 2), np.float32)
    return cv2.perspectiveTransform(court, H).reshape(-1, 2)


def side_of(court_xy: np.ndarray) -> np.ndarray:
    """0 = near half (Y < 9), 1 = far half."""
    return (np.asarray(court_xy)[..., 1] >= NET_Y).astype(np.int8)


class HomographyTracker:
    """Keeps H_t up to date while the broadcast camera pans and zooms.

    Frame-to-frame motion is a homography because the court is planar and the camera rotates
    and zooms about (almost) a fixed centre. Features on players move on their own, so player
    boxes are masked out before matching.
    """

    def __init__(self, H0: np.ndarray, ransac_px: float = 3.0, n_features: int = 3000):
        self.H = H0.astype(np.float64)
        self.ransac_px = ransac_px
        self.orb = cv2.ORB_create(n_features, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp = None
        self.prev_des = None
        self.last_inliers = 0
        self.failures = 0

    @staticmethod
    def _mask(shape: tuple[int, int], boxes: np.ndarray | None) -> np.ndarray:
        m = np.full(shape, 255, np.uint8)
        if boxes is not None:
            for x1, y1, x2, y2 in np.asarray(boxes, int):
                m[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0
        return m

    def update(self, gray: np.ndarray, boxes: np.ndarray | None = None) -> np.ndarray:
        kp, des = self.orb.detectAndCompute(gray, self._mask(gray.shape, boxes))
        if self.prev_des is not None and des is not None and len(kp) >= 8:
            matches = self.matcher.match(self.prev_des, des)
            if len(matches) >= 8:
                src = np.float32([self.prev_kp[m.queryIdx].pt for m in matches])
                dst = np.float32([kp[m.trainIdx].pt for m in matches])
                G, inl = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_px)
                n_in = int(inl.sum()) if inl is not None else 0
                if G is not None and n_in >= 15 and self._plausible(G, gray.shape):
                    self.H = G @ self.H
                    self.H /= self.H[2, 2]
                    self.last_inliers = n_in
                else:
                    self.failures += 1       # keep previous H (camera assumed static this frame)
        self.prev_kp, self.prev_des = kp, des
        return self.H

    @staticmethod
    def _plausible(G: np.ndarray, shape: tuple[int, int]) -> bool:
        """Reject wild frame-to-frame jumps (a broadcast camera moves smoothly)."""
        h, w = shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        moved = cv2.perspectiveTransform(corners, G).reshape(-1, 2) - corners.reshape(-1, 2)
        return float(np.abs(moved).max()) < 0.25 * max(h, w)


def draw_court(img: np.ndarray, H: np.ndarray, colour=(0, 255, 255), thickness: int = 2) -> np.ndarray:
    for a, b in COURT_LINES:
        pa, pb = to_image(H, np.array([a, b], np.float32))
        cv2.line(img, tuple(int(v) for v in pa), tuple(int(v) for v in pb), colour, thickness, cv2.LINE_AA)
    return img
