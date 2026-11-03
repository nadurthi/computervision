#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 21:11:13 2020

@author: nagnanamus
"""


# examples/Python/Basic/visualization.py

import numpy as np
import open3d as o3d
import time


class viz_pointcloud:

    def __init__(self, xyz_points):
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        self.pcd = o3d.geometry.PointCloud()
        xyz_points = np.array([[0, 0, 0], [1, 1, 1], [5, 5, 5], [-5, -5, -5]])
        self.pcd.points = o3d.utility.Vector3dVector(xyz_points)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(self.pcd)
        # print('waiting run')
        # self.vis.run()
        # print('waiting run end')
        # self.pcdsub = rospy.Subscriber("/sensor/velodyne_points", PointCloud2, self.vis_callback)
        self.save_image = False
        self.img_cout = 0

    def vis_callback(self, xyz_points):
        print('callback')
        self.img_cout += 1
        # xyz_points = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(cloud_msg)
        self.pcd.points = o3d.utility.Vector3dVector(xyz_points)
        self.vis.update_geometry(self.pcd)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.run()
        if self.save_image:
            self.vis.capture_screen_image("temp_%04d.jpg" % self.img_cout)

    def destroy_vis(self):
        self.vis.destroy_window()


if __name__ == "__main__":

    # pcd = o3d.io.read_point_cloud("/home/nagnanamus/Downloads/Open3D/examples/TestData/fragment.pcd")
    x = np.linspace(-3, 3, 401)
    mesh_x, mesh_y = np.meshgrid(x, x)
    z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    z_norm = (z - z.min()) / (z.max() - z.min())
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_norm, -1)
    print('xyz')
    print(xyz)

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # o3d.visualization.draw_geometries([pcd])

    vpc = viz_pointcloud(xyz)
    vpc.vis_callback(xyz)

    print('ok')

    cnt = 0
    st = time.time()
    time.sleep(0.1)
    while time.time() - st < 10:
        # time.sleep(0.1)
        print('ok2')
    # vpc.destroy_vis()
    vpc.destroy_vis()
    # print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud(
        "/home/nagnanamus/Downloads/Open3D/examples/TestData/fragment.pcd")
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([pcd])

    # print("Let's draw some primitives")
    # mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0,
    #                                                 height=1.0,
    #                                                 depth=1.0)
    # mesh_box.compute_vertex_normals()
    # mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    # mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
    # mesh_sphere.compute_vertex_normals()
    # mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])

    # mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.3,
    #                                                           height=4.0)
    # mesh_cylinder.compute_vertex_normals()
    # mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    #     size=0.6, origin=[-2, -2, -2])

    # print("We draw a few primitives using collection.")
    # o3d.visualization.draw_geometries(
    #     [pcd,mesh_box, mesh_sphere, mesh_cylinder, mesh_frame])

    # print("We draw a few primitives using + operator of mesh.")
    # o3d.visualization.draw_geometries(
    #     [mesh_box + mesh_sphere + mesh_cylinder + mesh_frame])

    # print("Let's draw a cubic using o3d.geometry.LineSet.")
    # points = [
    #     [0, 0, 0],
    #     [1, 1, 1],
    #     [1, -1, 1],
    #     [-1, -1, 1],
    #     [-1, 1, 1],
    # ]
    # lines = [
    #     [0, 1],
    #     [0, 2],
    #     [0, 3],
    #     [0, 4],
    #     [1, 2],
    #     [2, 3],
    #     [3, 4],
    #     [4, 1],
    # ]
    # colors = [[1, 0, 0] for i in range(len(lines))]
    # line_set = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector(points),
    #     lines=o3d.utility.Vector2iVector(lines),
    # )
    # line_set.colors = o3d.utility.Vector3dVector(colors)
    # o3d.visualization.draw_geometries([pcd, line_set])

    # print("Let's draw a textured triangle mesh from obj file.")
    # textured_mesh = o3d.io.read_triangle_mesh("../../TestData/crate/crate.obj")
    # textured_mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([textured_mesh])
