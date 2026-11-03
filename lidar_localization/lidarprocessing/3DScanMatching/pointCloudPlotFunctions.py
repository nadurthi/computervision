from numba.np.ufunc import parallel
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from numba import njit, prange, jit, cuda
from numba.typed import List, Dict
from numba.core import types
from numpy.core.numeric import identity

float_2Darray = types.float64[:,:]
import heapq
import open3d as o3d
import math
import pickle

import threading
from timeit import repeat


def pointCloudPlot(X1,X2):
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X1)
    # pcd.paint_uniform_color([0.9, 0.1, 0.1])
    colors = np.zeros((X1.shape[0],3))
    zMax = np.amax(X1[:,2])
    colors[:,0] =  1 #X1[:,2]/zMax
    colors[:,2] = 0
    colors[:,1] = .0
    pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(X2)
    colors = np.zeros((X2.shape[0],3))
    zMax = np.amax(X2[:,2])
    colors[:,1] = 0#X2[:,2]/zMax
    colors[:,2] = 1
    colors[:,0] = .0
    pcd2.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd, pcd2])


def pointCloudSeriesPlot(X):
    pcdList = []
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(1920/2), height=int(1080/2))
    
    ctr = vis.get_view_control()
    ctr.change_field_of_view(step=2)

    for i in range(len(X)):
        # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(X[i])
        pcdList.append(pcd)
        # ctr.change_field_of_view(step=fov_step)
        if i == len(X)-1:
            colors = np.zeros((X[i].shape[0],3))
            colors[:,0] =  0 #X1[:,2]/zMax
            colors[:,2] = 0
            colors[:,1] = 0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(pcd)

        # print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
        
        # print("Field of view (after changing) %.2f" % ctr.get_field_of_view())

    # ctr = o3d.visualization.Visualizer.get_view_control()
    # ctr.change_field_of_view(step=10)
    vis.run()
    vis.destroy_window()
    # o3d.visualization.draw_geometries(pcdList,width=int(1920/2), height=int(1080/2))

def pointCloudSeriesAnimation(X):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i in range(len(X)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(X[i])
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(.2)
    vis.destroy_window()
    pointCloudSeriesPlot(X)
    

def pointCloudUpdateAnimation(X,resolution,localTransform, vis,scanToGlobal):
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    X = np.vstack((X.T,np.ones(X.shape[0])))
    localTransform[1,3] += -resolution[0] #subtract the resolution to get the correct point location
    localTransform[0,3] += -resolution[1]
    localTransform[2,3] += -resolution[2]


    scanToGlobal = np.matmul(scanToGlobal,localTransform)
    transform = scanToGlobal

    t=transform[0:3,3]
    print("Transformation: X:%s, Y:%s, Z:%s" %(t[0],t[1],t[2]))
    X_Global = np.matmul(transform,X)
    # print(X_Global.shape)
    
    idx = np.where(X_Global[2,:]< -4)
    X_Global = np.delete(X_Global,idx,axis=1)

    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X_Global[0:3,:].T)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
        # time.sleep(.2)
    
    return scanToGlobal


def downSamplePointCloud(X,voxelSize= 0.05,doPlot=False):
    """Downsamples points to voxel size in meters"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
    # print("Downsample the point cloud with a voxel of 0.05")
    downpcd = pcd.voxel_down_sample(voxelSize)
    # print(downpcd)

    if doPlot:
        o3d.visualization.draw_geometries([downpcd],
                                        zoom=0.3412,
                                        front=[0.4257, -0.2125, -0.8795],
                                        lookat=[2.6172, 2.0475, 1.532],
                                        up=[-0.0694, -0.9768, 0.2024])
    return np.asarray(downpcd.points)
    