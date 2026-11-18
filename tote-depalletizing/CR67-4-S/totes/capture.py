import numpy as np
from nxlib import NxLib, Camera
from nxlib.constants import *

# Replace with your camera's serial number (or find it via NxView)
CAMERA_SERIAL = "269561"

# Initialize NxLib and open the camera using context managers
with NxLib():
    with Camera.from_serial(CAMERA_SERIAL) as camera:
        # 1. Capture raw stereo images
        camera.capture()

        # 2. Compute the disparity map from the stereo pair
        camera.compute_disparity_map()

        # 3. Compute the 3D point map (returns an H x W x 3 NumPy array in millimeters)
        camera.compute_point_map()
        point_map = camera.get_point_map()

        # 4. Reshape from (H, W, 3) to (N, 3) and remove invalid (NaN) points
        points = point_map.reshape(-1, 3)
        valid_points = points[~np.isnan(points).any(axis=1)]

        print(f"Captured {valid_points.shape[0]} valid 3D points.")