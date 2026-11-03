# -*- coding: utf-8 -*-

import pickle as pkl
import numpy as np
import numpy.linalg as nplinalg
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from sklearn.neighbors import KDTree
# from uq.gmm import gmmfuncs as uqgmmfnc
# from utils.plotting import geometryshapes as utpltgmshp
# import time
# from scipy.optimize import minimize, rosen, rosen_der,least_squares
# from scipy import interpolate
# import networkx as nx
# import pdb
# import pandas as pd
# from fastdist import fastdist
import copy
# from lidarprocessing import point2Dprocessing as pt2dproc
# from lidarprocessing import point3Dprocessing as pt3dproc
# from lidarprocessing import point2Dplotting as pt2dplot
# import lidarprocessing.numba_codes.point2Dprocessing_numba as nbpt2Dproc
# from sklearn.neighbors import KDTree
# import os
# import pandas as pd
# time_increment = 1.736111516947858e-05
# angle_increment = 0.004363323096185923
# scan_time = 0.02500000037252903
# range_min, range_max = 0.023000000044703484, 60.0
# angle_min,angle_max =  -2.3518311977386475,2.3518311977386475
# from numba import vectorize, float64,guvectorize,int64,double,int32,int64,float32,uintc,boolean
# from numba import njit, prange,jit
# import scipy.linalg as sclalg
# import scipy.optimize as scopt
from pykitticustom import odometry
import pickle
dtype = np.float64
# from lidarprocessing import icp
import open3d as o3d
# import numba
# from numba import vectorize, float64,guvectorize,int64,double,int32,int64,float32,uintc,boolean
# from numba import njit, prange,jit
# import copy

# import importlib
import utils.parconsumer as parcons

# import pyslam.slam as slam
# importlib.reload(slam)
#%%

basedir ='/media/na0043/misc/DATA/KITTI/odometry/dataset'
# Specify the dataset to load
# sequence = '02'
# sequence = '05'
# sequence = '06'
# sequence = '08'
loop_closed_seq = ['02','05','06','08']
sequence = '05'

dataset = odometry.odometry(basedir, sequence, frames=None) # frames=range(0, 20, 5)
Xtpath=np.zeros((len(dataset),4))
# f3 = plt.figure()    
# ax = f3.add_subplot(111)
# for i in range(len(dataset)):
#     Xtpath[i,:] = dataset.poses[i].dot(np.array([0,0,0,1]))
# ax.plot(Xtpath[:,2],-Xtpath[:,0],'k')
# plt.show()

pose = dataset.poses[1]
velo = dataset.get_velo(2)


#%%
xedges=np.arange(-400,400,0.1)
yedges=np.arange(-10,10,0.1)
zedges=np.arange(-200,500,0.1)

Ht=None

def Fconsumer(inputarg):
    i=inputarg
    X1 = dataset.get_velo(i)
    X1=X1[:,:3]
    idx=(np.abs(X1[:,0])<3) & (np.abs(X1[:,1])<2) & (np.abs(X1[:,2])<3)
    X1=X1[~idx,:]
    H=dataset.calib.T_cam0_velo
    X1=np.hstack([X1,np.ones((X1.shape[0],1))])
    # X1=nplinalg.inv(dataset.poses[i]).dot(X1.T).T
    H=np.dot(nplinalg.inv(dataset.calib.T_cam0_velo),dataset.poses[i].dot(H))
    X1=H.dot(X1.T).T
    
    X1=X1[:,:3]
    
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(X1)
    pcd1, ind = pcd1.remove_radius_outlier(nb_points=5, radius=0.5)
    
    return np.asarray(pcd1.points)

# pc=parcons.ParallelConsumer(Fconsumer,Nproc=3,maxInQ=20000)

# for i in range(0,len(dataset)):
#     pc.pushInputArg(i)



# pcd=None
# for i,res in enumerate(pc.iterateOutput()):
#     # print("output res = ",res)
#     X1=res
#     pcd1 = o3d.geometry.PointCloud()
#     pcd1.points = o3d.utility.Vector3dVector(X1)
    
#     if pcd is None:
#         pcd=pcd1
#     else:
#         pcd=pcd+pcd1
#     pcd = pcd.voxel_down_sample(voxel_size=0.05)    
#     print("recieved ",i)
    
pcd=None
PCD=None
j=None
for i in range(0,len(dataset)):
    print(i,len(dataset))
    X1 = dataset.get_velo(i)
    X1=X1[:,:3]
    idx=(np.abs(X1[:,0])<3) & (np.abs(X1[:,1])<2) & (np.abs(X1[:,2])<3)
    X1=X1[~idx,:]
    H=dataset.calib.T_cam0_velo
    X1=np.hstack([X1,np.ones((X1.shape[0],1))])
    # X1=nplinalg.inv(dataset.poses[i]).dot(X1.T).T
    H=np.dot(nplinalg.inv(dataset.calib.T_cam0_velo),dataset.poses[i].dot(H))
    X1=H.dot(X1.T).T
    
    X1=X1[:,:3]
    
    
        

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(X1)

    # PCD.append(pcd1)
    
    if PCD is None:
        PCD=pcd1
    else:
        PCD=PCD+pcd1
            
    
        
    # if i>15:
    #     break
    # Hf=numba_histogram3D(X1, xedges,yedges,zedges)
    # if Ht is None:
    #     Hf=Ht
    # else:
    #     Hf+=Ht
    
    if j is None or i-j>10:
        j=i
        PCD = PCD.voxel_down_sample(voxel_size=0.015)    
        pcd2, ind = PCD.remove_radius_outlier(nb_points=20, radius=0.1)
        if pcd is None:
            pcd=pcd2
        else:
            pcd=pcd+pcd2
        pcd = pcd.voxel_down_sample(voxel_size=0.05)    
        PCD=None
        
    # if i>950:
    #     break

    
o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries(PCD)
o3d.io.write_point_cloud("kitti-pcd-seq-movingObjRem-%s.pcd"%sequence, pcd)
# #%%
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

# o3d.io.write_point_cloud("kitti-pcd-seq-%s.pcd"%sequence, pcd)

# pcd=o3d.io.read_point_cloud("kitti-pcd-seq-%s.pcd"%sequence)

# # pcd_mltree = o3d.ml.tf.models.KDTree(pcd.points,leaf_size=400)


# pcd_tree = o3d.geometry.KDTreeFlann(pcd)

# [k, idx, _] = pcd_tree.search_hybrid_vector_3d([1,1,1], 2, 2)

# # alpha = 1
# # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
# # mesh.compute_vertex_normals()
# # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

# # radii=o3d.utility.DoubleVector([0.2,0.2,0.2])
# # meshb = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)
# # meshb.compute_vertex_normals()
# # o3d.visualization.draw_geometries([meshb], mesh_show_back_face=True)


# # cube = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
# # scene = o3d.t.geometry.RaycastingScene()
# # cube_id = scene.add_triangles(cube)

# # rays = o3d.core.Tensor([[0.5, 0.5, 1, 1, 0, 0], [0.5, 0.5, 1, 0, 0, -1]],
# #                        dtype=o3d.core.Dtype.Float32)

# # ans = scene.cast_rays(rays)


#%%
pcd=o3d.io.read_point_cloud("kitti-pcd-seq-movingObjRem-%s.pcd"%sequence)
o3d.visualization.draw_geometries([pcd])