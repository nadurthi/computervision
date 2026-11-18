import numpy as np
import open3d as o3d
from nxlib import NxLib, Camera
from nxlib.constants import *

def capture_and_visualize(serial_number: str):
    # Initialize NxLib and open the camera using context managers
    with NxLib():
        with Camera.from_serial(serial_number) as camera:
            # 1. Capture stereo images and compute 3D point map
            camera.capture()
            camera.compute_disparity_map()
            camera.compute_point_map()

            # 2. Retrieve the point map as a NumPy array of shape (height, width, 3)
            point_map = camera.get_point_map()

    # 3. Reshape to (N, 3) and filter out invalid (NaN) points
    points = point_map.reshape((-1, 3))
    print(points.shape)
    valid_points = points[~np.isnan(points).any(axis=1)]

    # 4. Convert to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)

    # Optional: Save to PLY file
    o3d.io.write_point_cloud("ensenso_capture.ply", pcd)

    # 5. Visualize the point cloud
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Ensenso Point Cloud",
        width=1024,
        height=768
    )

if __name__ == "__main__":
    # Replace with your Ensenso camera's serial number (check via NxView or nx_list_cams.py)
    CAMERA_SERIAL = "269561"
    capture_and_visualize(CAMERA_SERIAL)