import numpy as np
import open3d as o3d
import tifffile

# 1. Load the 3-channel 32-bit float TIFF exported from NxView
# Shape is (Height, Width, 3) representing (X, Y, Z) in millimeters
point_map = tifffile.imread("1.tif")

# 2. Reshape from (H, W, 3) to (N, 3)
points = point_map.reshape(-1, 3)

# 3. Filter out invalid/unmatched pixels (stored as NaN or 0,0,0 by Ensenso)
valid_mask = ~np.isnan(points).any(axis=1) & ~np.all(points == 0, axis=1)
valid_points = points[valid_mask]

# Optional: Convert millimeters to meters if preferred
# valid_points = valid_points / 1000.0

# 4. Create the Open3D PointCloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(valid_points)

# 5. Visualize the point cloud
o3d.visualization.draw_geometries([pcd], window_name="Ensenso Point Cloud")