#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 22:01:28 2021

@author: nagnanmus
"""

import pickle as pkl
import numpy as np
import numpy.linalg as nplinalg
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from sklearn.neighbors import KDTree
from uq.gmm import gmmfuncs as uqgmmfnc
from utils.plotting import geometryshapes as utpltgmshp
import time
from scipy.optimize import minimize, rosen, rosen_der,least_squares
from scipy import interpolate
import networkx as nx
import pdb
import pandas as pd
from fastdist import fastdist
import copy
from lidarprocessing import point2Dprocessing as pt2dproc
from lidarprocessing import point3Dprocessing as pt3dproc
from lidarprocessing import point2Dplotting as pt2dplot
import lidarprocessing.numba_codes.point2Dprocessing_numba as nbpt2Dproc
import lidarprocessing.numba_codes.binmatchers as binmatchers
from lidarprocessing.numba_codes import gicp
import queue
from sklearn.neighbors import KDTree
import os
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
time_increment = 1.736111516947858e-05
angle_increment = 0.004363323096185923
scan_time = 0.02500000037252903
range_min, range_max = 0.023000000044703484, 60.0
angle_min,angle_max =  -2.3518311977386475,2.3518311977386475
from numba import vectorize, float64,guvectorize,int64,double,int32,int64,float32,uintc,boolean
from numba import njit, prange,jit
import scipy.linalg as sclalg
import scipy.optimize as scopt
from scipy.spatial.transform import Rotation as Rsc
from scipy.spatial.transform import RotationSpline as RscSpl 
dtype = np.float64
from lidarprocessing import icp
import open3d as o3d
from pykitticustom import odometry
from  uq.filters import pf 
from scipy.stats import multivariate_normal
from scipy.interpolate import UnivariateSpline
from sklearn.neighbors import NearestNeighbors
import quaternion
import pickle
from joblib import dump, load
from importlib import reload 
from pyslam import  slam
import json
def quat2omega_scipyspline(tvec,qvec,k=2,s=0.001):
    spl0 = UnivariateSpline(tvec, qvec[:,0], k=k,s=s)
    spl1 = UnivariateSpline(tvec, qvec[:,1], k=k,s=s)
    spl2 = UnivariateSpline(tvec, qvec[:,2], k=k,s=s)
    spl3 = UnivariateSpline(tvec, qvec[:,3], k=k,s=s)
    
    spld0=spl0.derivative()
    spld1=spl1.derivative()
    spld2=spl2.derivative()
    spld3=spl3.derivative()
    
    qdot = np.zeros_like(qvec)
    qvec_sp = np.zeros_like(qvec)
    
    qvec_sp[:,0] = spl0(tvec)
    qvec_sp[:,1] = spl1(tvec)
    qvec_sp[:,2] = spl2(tvec)
    qvec_sp[:,3] = spl3(tvec)
    
    qdot[:,0] = spld0(tvec)
    qdot[:,1] = spld1(tvec)
    qdot[:,2] = spld2(tvec)
    qdot[:,3] = spld3(tvec)
    
    qqdot=quaternion.from_float_array(qdot)
    qqvec=quaternion.from_float_array(qvec_sp)
    
    w=np.zeros((len(qvec),4))
    for i in range(len(qvec)):
        pp=quaternion.as_float_array(qqdot[i]*qqvec[i].inverse())
        w[i]= 2*pp
    
    spl0 = UnivariateSpline(tvec, w[:,0], k=k,s=s)
    spl1 = UnivariateSpline(tvec, w[:,1], k=k,s=s)
    spl2 = UnivariateSpline(tvec, w[:,2], k=k,s=s)
    spl3 = UnivariateSpline(tvec, w[:,3], k=k,s=s)
    
    spld0=spl0.derivative()
    spld1=spl1.derivative()
    spld2=spl2.derivative()
    spld3=spl3.derivative()
    
    alpha = np.zeros((len(qvec),4))
    alpha[:,0] = spld0(tvec)
    alpha[:,1] = spld1(tvec)
    alpha[:,2] = spld2(tvec)
    alpha[:,3] = spld3(tvec)
    
    return tvec,qvec_sp,w,qdot,alpha
#%%
plt.close("all")
# basedir =r'P:\SLAMData\Kitti\visualodo\dataset'

basedir ='/media/na0043/misc/DATA/KITTI/odometry/dataset'
# Specify the dataset to load
# sequence = '02'
# sequence = '05'
# sequence = '06'
# sequence = '08'
loop_closed_seq = ['02','05','06','08']
sequence = '05'



dataset = odometry.odometry(basedir, sequence, frames=None) # frames=range(0, 20, 5)

nd=len(dataset.poses)

Xtpath=np.zeros((len(dataset),10))
f3 = plt.figure()    
ax = f3.add_subplot(111,projection='3d')
for i in range(len(dataset)):
    H=dataset.calib.T_cam0_velo
    H=np.dot(nplinalg.inv(H),dataset.poses[i].dot(H))
    
    Xtpath[i,0:3] = H.dot(np.array([0,0,0,1]))[0:3]
    # Xtpath[i,0:3] = dataset.poses[i].dot(np.array([0,0,0,1]))[0:3]
    
    r = Rsc.from_matrix(H[0:3,0:3])
    
    q =r.as_euler('zyx',degrees=False)
    Xtpath[i,3:6] = q
    
    q =r.as_quat()
    Xtpath[i,6:] = q

tvec,qvec_sp,w,qdot,alpha=quat2omega_scipyspline(dataset.times,Xtpath[:,6:],k=3,s=0.0001)

ax.plot(Xtpath[:nd,0],Xtpath[:nd,1],Xtpath[:nd,2],'k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

ff = plt.figure()    
ax = ff.add_subplot(111,projection='3d')
ax.plot(Xtpath[:nd,0],Xtpath[:nd,1],Xtpath[:nd,2],'k')
ax.set_box_aspect((np.ptp(Xtpath[:,0]), np.ptp(Xtpath[:,1]), np.ptp(Xtpath[:,2])))
plt.show()
    
Velocities=np.zeros((len(dataset),3))
Acc=np.zeros((len(dataset),3))

splx=UnivariateSpline(dataset.times,Xtpath[:,0])
splvx=splx.derivative()
splax=splvx.derivative()

sply=UnivariateSpline(dataset.times,Xtpath[:,1])
splvy=sply.derivative()
splay=splvy.derivative()

splz=UnivariateSpline(dataset.times,Xtpath[:,2])
splvz=splz.derivative()
splaz=splvz.derivative()


Velocities[:,0]=splvx(dataset.times)
Velocities[:,1]=splvy(dataset.times)
Velocities[:,2]=splvz(dataset.times)

Acc[:,0]=splax(dataset.times)
Acc[:,1]=splay(dataset.times)
Acc[:,2]=splaz(dataset.times)

rotations = Rsc.from_euler('zyx', Xtpath[:,3:6], degrees=False)
spline =RscSpl(dataset.times, rotations)
AngRates=spline(dataset.times, 1)
AngAcc=spline(dataset.times, 2)
AngRates=AngRates[:,::-1]
AngAcc=AngAcc[:,::-1]

# AngRates=w[:,1:]
# AngAcc=alpha[:,1:]

# yawspl=UnivariateSpline(dataset.times,Xtpath[:,3],k=5)
# pitchspl=UnivariateSpline(dataset.times,Xtpath[:,4],k=5)
# rollspl=UnivariateSpline(dataset.times,Xtpath[:,5],k=5)

# yawrate=yawspl.derivative()(dataset.times)
# pitchrate=pitchspl.derivative()(dataset.times)
# rollrate=rollspl.derivative()(dataset.times)

# AngRates=np.vstack([yawrate,pitchrate,rollrate]).T
# AngAcc=alpha[:,1:]





f4 = plt.figure()    
ax = f4.subplots(3,3)
ax[0,0].plot(dataset.times[:nd],Xtpath[:nd,3],'r',label='yaw')
ax[1,0].plot(dataset.times[:nd],Xtpath[:nd,4],'b',label='pitch')
ax[2,0].plot(dataset.times[:nd],Xtpath[:nd,5],'g',label='roll')

ax[0,1].plot(dataset.times[:nd],AngRates[:,0],'r',label='Vyaw')
ax[1,1].plot(dataset.times[:nd],AngRates[:,1],'b',label='Vpitch')
ax[2,1].plot(dataset.times[:nd],AngRates[:,2],'g',label='Vroll')

ax[0,2].plot(dataset.times[:nd],AngAcc[:nd,0],'r',label='Ayaw')
ax[1,2].plot(dataset.times[:nd],AngAcc[:nd,1],'b',label='Apitch')
ax[2,2].plot(dataset.times[:nd],AngAcc[:nd,2],'g',label='Aroll')

ax[0,0].legend()
ax[1,0].legend()
ax[2,0].legend()

plt.show()

f5 = plt.figure()    
ax = f5.subplots(3,3)
ax[0,0].plot(dataset.times[:nd],Xtpath[:nd,0],'r',label='x')
ax[1,0].plot(dataset.times[:nd],Xtpath[:nd,1],'r',label='y')
ax[2,0].plot(dataset.times[:nd],Xtpath[:nd,2],'r',label='z')

ax[0,1].plot(dataset.times[:nd],Velocities[:nd,0],'r',label='vx')
ax[1,1].plot(dataset.times[:nd],Velocities[:nd,1],'r',label='vy')
ax[2,1].plot(dataset.times[:nd],Velocities[:nd,2],'r',label='vz')

ax[0,2].plot(dataset.times[:nd],Acc[:nd,0],'r',label='ax')
ax[1,2].plot(dataset.times[:nd],Acc[:nd,1],'r',label='ay')
ax[2,2].plot(dataset.times[:nd],Acc[:nd,2],'r',label='az')

ax[0,0].legend()
ax[1,0].legend()
ax[2,0].legend()
plt.show()



# pose = dataset.poses[1]
# velo = dataset.get_velo(2)


#%%

# folder='lidarprocessing/datasets/kitti'

# i=1
# f1="%06d.bin"%i
# X1 = np.fromfile(folder+'/'+f1, dtype=np.float32)
# X1=X1.reshape((-1, 4))
# X1=X1.astype(dtype)

# # pcd = o3d.geometry.PointCloud()
# # pcd.points = o3d.utility.Vector3dVector(X1[:,:3])
# # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.1)
# # X11=np.asarray(voxel_down_pcd.points)
# # X11=np.ascontiguousarray(X11.T,dtype=dtype)

# X11=X1[1:10000,:3]

# X22=X11+[0,0,1]

# dmax=5
# X11,P1,X22,P2=gicp.gicp_init(X11,X22,dmax=dmax)
# tree1 = KDTree(X11, leaf_size=5)

# x=np.array([0,0,0,0,0,0],dtype=dtype)

# st=time.time()
# res1 = minimize(gicp.gicp_cost, x,args=(tree1,X11,X22,P1,P2,dmax),jac=True,method='BFGS',options={'disp':True,'maxiter':150}) # 'Nelder-Mead'
# et=time.time()
# print("time :",et-st)
# print(res1) 

# gicp.gicp_cost(np.zeros(6),X11,X22,P1,P2,dmax)
#%% PF 


pcd3D=o3d.io.read_point_cloud("kitti-pcd-seq-%s.pcd"%sequence)
pcd3DdownSensorCost=pcd3D.voxel_down_sample(voxel_size=0.5)


pcd3Droadremove=o3d.io.read_point_cloud("kitti-pcd-seq-roadremove-%s.pcd"%sequence)
np.asarray(pcd3Droadremove.points)[:,2]=0
X2Dmap_down=np.array(pcd3Droadremove.voxel_down_sample(voxel_size=0.1).points)
X2Dmap_down=X2Dmap_down[:,:2]


lines = [[i,i+1] for i in range(Xtpath.shape[0]-1)]
colors = [[0, 0, 0] for i in range(len(lines))]
line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(Xtpath[:,:3])
line_set.lines = o3d.utility.Vector2iVector(lines)
line_set.colors = o3d.utility.Vector3dVector(colors)

# o3d.visualization.draw_geometries([pcd,line_set])





# sktree,neigh=load( "kitti-pcd-seq-%s-TREES.joblib"%sequence) 

# min_bound=np.min(np.asarray(pcd.points),axis=0)
# max_bound=np.max(np.asarray(pcd.points),axis=0)
# min_bound[2]=-0.5
# max_bound[2]=5

# bbox3d=o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound,max_bound=max_bound)
# pcdbox=pcd.crop(bbox3d)
# o3d.visualization.draw_geometries([pcdbox,line_set])


# pcd_tree = o3d.geometry.KDTreeFlann(pcd.voxel_down_sample(voxel_size=0.5))



# neigh = NearestNeighbors(n_neighbors=3,radius=20,n_jobs=5,algorithm='kd_tree')
# neigh.fit(np.asarray(pcd.points))

# sktree = KDTree(np.asarray(pcd.voxel_down_sample(voxel_size=0.5).points), leaf_size=30)


# qp=[373.0,   1.3,   6.8]
# radius=20
# maxnn=1
# st=time.time()
# [k, idx, dd] = pcd_tree.search_hybrid_vector_3d(qp, radius, maxnn)
# et=time.time()
# print("kd tree search time = ",et-st)




# st=time.time()
# ddsk,neighbors = neigh.radius_neighbors([qp],radius=radius, return_distance = True)
# et=time.time()
# print("nn sklearn search time = ",et-st)



# st=time.time()
# ddsktree,idxs = sktree.query([qp],k=1, dualtree=True,return_distance = True)
# et=time.time()
# print("sklearn tree search time = ",et-st)

# dump([sktree,neigh], "kitti-pcd-seq-%s-TREES.joblib"%sequence) 



D={"icp":{},
   "gicp":{},
   "gicp_cost":{},
   "ndt":{},
   "sig0":0.5,
   "dmax":10,
   "DoIcpCorrection":0,
   "Localize":{}}

D["icp"]["enable"]=1
D["icp"]["setMaximumIterations"]=50
D["icp"]["setMaxCorrespondenceDistance"]=25
D["icp"]["setRANSACIterations"]=0.0
D["icp"]["setRANSACOutlierRejectionThreshold"]=1.5
D["icp"]["setTransformationEpsilon"]=1e-3
D["icp"]["setEuclideanFitnessEpsilon"]=1



D["gicp_cost"]["enable"]=0


D["gicp"]["enable"]=0
D["gicp"]["setMaxCorrespondenceDistance"]=20
D["gicp"]["setMaximumIterations"]=50.0
D["gicp"]["setMaximumOptimizerIterations"]=20.0
D["gicp"]["setRANSACIterations"]=0
D["gicp"]["setRANSACOutlierRejectionThreshold"]=1.5
D["gicp"]["setTransformationEpsilon"]=1e-9
D["gicp"]["setUseReciprocalCorrespondences"]=1

D["ndt"]["enable"]=0
D["ndt"]["setTransformationEpsilon"]=1e-9
D["ndt"]["setStepSize"]=2.0
D["ndt"]["setResolution"]=1.0
D["ndt"]["setMaximumIterations"]=25.0
D["ndt"]["initialguess_axisangleA"]=0.0
D["ndt"]["initialguess_axisangleX"]=0.0
D["ndt"]["initialguess_axisangleY"]=0.0
D["ndt"]["initialguess_axisangleZ"]=1.0
D["ndt"]["initialguess_transX"]=0.5
D["ndt"]["initialguess_transY"]=0.01
D["ndt"]["initialguess_transZ"]=0.01

D["DON"]={}
D["DON"]["scale1"]=1;
D["DON"]["scale2"]=2;
D["DON"]["threshold"]=0.2;
D["DON"]["threshold_small_z"]=0.5;
D["DON"]["threshold_large_z"]=0.5;

D["DON"]["segradius"]=1;

D["DON"]["threshold_curv_lb"]=0.1;
D["DON"]["threshold_curv_ub"]=100000;
D["DON"]["threshold_small_nz_lb"]=-0.5;
D["DON"]["threshold_small_nz_ub"]=0.5;
D["DON"]["threshold_large_nz_lb"]=-5;
D["DON"]["threshold_large_nz_ub"]=5;

D["Localize"]["sig0"]=0.5
D["Localize"]["dmax"]=10
D["Localize"]["octree"]={"enable":True,
                         "resolution":0.1,
                         "UseApprxNN":True,
                         "UseRadiusSearch":False}



pclloc=slam.Localize(json.dumps(D))

pcd3DdownSensorCostDown=pcd3DdownSensorCost.voxel_down_sample(voxel_size=0.25)
X3Dmap_down = np.asarray(pcd3DdownSensorCostDown.points)
pclloc.setMapX(X3Dmap_down)

opts = json.dumps(D)





#%%
plt.close("all")
opts = json.dumps(D)
pclloc.setOptions(opts)

def down_sample(X,voxel_size):
    X=np.asarray(X)
    NN=X.ndim
    if NN==2:
        X=np.hstack([X,np.zeros((X.shape[0],1))])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    XX=np.asarray(voxel_down_pcd.points)
    return XX[:,:NN]

import heapq
numba_cache=True
dtype=np.float64

from numba.core import types
from numba.typed import Dict
float_2Darray = types.float64[:,:]

def binMatcherAdaptive3(X11,X22,H12,Lmax,thmax,thmin,dxMatch):
    #  X11 are global points
    # X22 are points with respect to a local frame (like origin of velodyne)
    # H12 takes points in the velodyne frame to X11 frame (global)
    
    n=histsmudge =2 # how much overlap when computing max over adjacent hist for levels
    H21comp=np.identity(3)
        
    mn=np.zeros(2)
    mx=np.zeros(2)
    mn_orig=np.zeros(2)
    mn_orig[0] = np.min(X11[:,0])
    mn_orig[1] = np.min(X11[:,1])
    
    mn_orig=mn_orig-dxMatch
    
    
    # R=H12[0:2,0:2]
    # t=H12[0:2,2]
    # X222 = R.dot(X22.T).T+t
    
    H12mn = H12.copy()
    H12mn[0:2,2]=H12mn[0:2,2]-mn_orig
    # print("mn_orig = ",mn_orig)
    
    # X2=X222-mn_orig
    X1=X11-mn_orig
    
    
    # t2 = np.mean(X2,axis=0)
    # X20 = X2-t2
    
    mn[0] = np.min(X1[:,0])
    mn[1] = np.min(X1[:,1])
    mx[0] = np.max(X1[:,0])
    mx[1] = np.max(X1[:,1])
    # rmax=np.max(np.sqrt(X2[:,0]**2+X2[:,1]**2))
    
    t0 = H12mn[0:2,2]
    L0=t0-Lmax
    L1=t0+Lmax
    
    
    
    P = mx-mn

    mxlvl=0
    dx0=mx+dxMatch
    dxs = []
    XYedges=[]
    for i in range(0,100):
        f=2**i
        
        xedges=np.linspace(0,mx[0]+1*dxMatch[0],f+1)
        yedges=np.linspace(0,mx[1]+1*dxMatch[0],f+1)
        XYedges.append((xedges,yedges))
        dx=np.array([xedges[1]-xedges[0],yedges[1]-yedges[0]])
    
        dxs.append(dx)
        
        if np.any(dx<=dxMatch):
            break
        
    mxlvl=len(dxs)
    

    dxs=[dx.astype(np.float64) for dx in dxs]

    
    
    H1match=nbpt2Dproc.numba_histogram2D(X1, XYedges[-1][0],XYedges[-1][1])
    H1match = np.sign(H1match)
    
    
    
    

    levels=[]
    HLevels=[H1match]
    
    
    
    
    
    for i in range(1,mxlvl):
        
    
    
        Hup = HLevels[i-1]
        H=nbpt2Dproc.UpsampleMax(Hup,n)

        HLevels.append(H)
          
    
    mxLVL=len(HLevels)-1
    HLevels=HLevels[::-1]
    HLevels=[np.ascontiguousarray(H).astype(np.int32) for H in HLevels]
    
    # P2=0.001*P
    # Lmax=P2*(np.floor(Lmax/P2)+1)
    # Lmax=np.maximum(Lmax,P)
    
    LmaxOrig=np.zeros(2,dtype=np.float64)
    LmaxOrig[0]=Lmax[0]
    LmaxOrig[1]=Lmax[1]

    SolBoxes_init=[]
    # X2[:,0]=X2[:,0]-LmaxOrig[0]
    # X2[:,1]=X2[:,1]-LmaxOrig[1]
    
    # print("Lmax = ",Lmax)
    # print("LmaxSide = ",2*Lmax)
    L0=dxs[-1]*(np.floor(L0/dxs[-1]))
    L1=dxs[-1]*(np.floor(L1/dxs[-1])+1)
    # print("L0=",L0)
    # print("L1=",L1)
    # print("L0+mn=",L0+mn_orig)
    # print("L1+mn=",L1+mn_orig)
    LmaxNside = L1-L0
    n=np.floor(np.log2(LmaxNside/dxs[-1]))+1
    LmaxNside=dxs[-1]*np.power(2, n)
    # print("LmaxNside = ",LmaxNside)
    # Lmax=dxs[-1]*(np.floor(Lmax/dxs[-1])+1)
    # Lmax=dxs[-1]*(np.floor(Lmax/dxs[-1])+1)
    # print("Lmax = ",Lmax)
    # for xs in np.arange(L0[0],L1[0],dxs[0][0]):
    #     for ys in np.arange(-Lmax[1],Lmax[1],dxs[0][1]):
    #         SolBoxes_init.append( (xs,ys,dxs[0][0],dxs[0][1]) )
    # for xs in np.arange(-Lmax[0],Lmax[0],Lmax[0]):
    #     for ys in np.arange(-Lmax[1],Lmax[1],Lmax[1]):
    #         SolBoxes_init.append( (xs,ys,Lmax[0],Lmax[1]) )        
    
    
    # print(SolBoxes_init)
    
    
    h=[(100000.0,0.0,0.0,0.0,0.0,0.0,0.0)]
    lvl=0
    dx=dxs[lvl]
    H=HLevels[lvl]
    
    Xth= Dict.empty(
        key_type=types.float64,
        value_type=float_2Darray,
    )
    ii=0

    thfineRes = thmin
    thL=np.arange(-thmax,thmax+thfineRes,thfineRes,dtype=np.float64)
    
    
    
    # X2=np.ascontiguousarray(X2)
    for j in range(len(thL)):
        th=thL[j]
        R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        XX=np.transpose(R.dot(X22.T))
        XX=np.transpose(H12mn[0:2,0:2].dot(XX.T))+H12mn[0:2,2]
        Xth[th]=XX
        
        
        # for solbox in SolBoxes_init:
        #     xs,ys,d0,d1 = solbox
        #     Tj=np.array((d0,d1))
        #     Oj = np.array((xs,ys)) 
        d0,d1=Tj=dxs[0]
        xs,ys=Oj = np.array((0,0)) 
        
        cost2=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj)
        h.append((-cost2-np.random.rand()/1e10,xs,ys,d0,d1,lvl,th))
    
    heapq.heapify(h)
    
    XX=np.transpose(H12mn[0:2,0:2].dot(X22.T))+H12mn[0:2,2]
    zz=np.zeros(2,dtype=np.float64)
    # print("dxs[-1] = ",dxs[-1])
    cost0=nbpt2Dproc.getPointCost(HLevels[-1],dxs[-1],XX,zz,dxs[-1])
    # print("cost0 = ",cost0)
    # print("len(heap) = ",len(h))

    while(1):
        (cost,xs,ys,d0,d1,lvl,th)=heapq.heappop(h)
        mainSolbox = (cost,xs,ys,d0,d1,lvl,th)
        if lvl==mxLVL:
            break
        
        nlvl = int(lvl)+1
        dx=dxs[nlvl]
        H=HLevels[nlvl]
        Tj=np.array((d0,d1))
        Oj = np.array((xs,ys))

        
        
        Xg=np.arange(Oj[0],Oj[0]+Tj[0],dx[0])
        Yg=np.arange(Oj[1],Oj[1]+Tj[1],dx[1])
        
        d0,d1=dx[0],dx[1]
        Tj=np.array((d0,d1))
        
        for xs in Xg[:2]:
            for ys in Yg[:2]:
                if xs+t0[0]>L1[0] or ys+t0[1]>L1[1]:
                    continue
                if xs+d0+t0[0]<L0[0] or ys+d1+t0[1]<L0[1]:
                    continue
                
                Oj = np.array((xs,ys))
                cost3=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj) 
                heapq.heappush(h,(-cost3-np.random.rand()/1e10,xs,ys,d0,d1,float(nlvl),th))

    # print("lvl,mxLVL,mxlvl",lvl,mxLVL,mxlvl) 
    t=mainSolbox[1:3]
    # print("t=",t)  
    th = mainSolbox[6]
    cost=-mainSolbox[0]
    

    
    HcompR=np.identity(3)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    HcompR[0:2,0:2]=R
    
    Ht=np.identity(3)
    Ht[0:2,2]=t
    
    # first rotation by HcompR or R
    # then H12mn
    # then translate by t
    # Ht*H12mn*HcompR*X22v
    H12comp=np.dot(Ht.dot(H12mn),HcompR)
    H12comp[0:2,2]=H12comp[0:2,2]+mn_orig
    
    

    
    # H1=np.identity(3)
    # H1[0:2,2]=-mn_orig
    
    # Hlmax=np.identity(3)
    # Hlmax[0:2,2]=-LmaxOrig

    
    # H2=nplinalg.inv(H1)
    # H2=H2.dot(Hcomp)
    # H3=H2.dot(Hlmax)
    # H4=H3.dot(H1)
    # H12comp=H4.dot(H12)

    H21comp=nplinalg.inv(H12comp)
    
    return H21comp,(cost0,cost)
    

Npf=2000





Xlimits_scan=[np.min(np.asarray(pcd3DdownSensorCost.points)[:,0]),np.max(np.asarray(pcd3DdownSensorCost.points)[:,0])]
Ylimits_scan=[np.min(np.asarray(pcd3DdownSensorCost.points)[:,1]),np.max(np.asarray(pcd3DdownSensorCost.points)[:,1])]
Zlimits_scan=[np.min(np.asarray(pcd3DdownSensorCost.points)[:,2]),np.max(np.asarray(pcd3DdownSensorCost.points)[:,2])]


Xlimits=[np.min(Xtpath[:,0]),np.max(Xtpath[:,0])]
Ylimits=[np.min(Xtpath[:,1]),np.max(Xtpath[:,1])]
Zlimits=[np.min(Xtpath[:,2]),np.max(Xtpath[:,2])]

yawlimits=[np.min(Xtpath[:,3]),np.max(Xtpath[:,3])]
pitchlimits=[np.min(Xtpath[:,4]),np.max(Xtpath[:,4])]
rolllimits=[np.min(Xtpath[:,5]),np.max(Xtpath[:,5])]

vv=nplinalg.norm(Velocities,axis=1)
vlimits=[0,np.max(vv)]

Vxlimits=[np.min(Velocities[:,0]),np.max(Velocities[:,0])]
Vylimits=[np.min(Velocities[:,1]),np.max(Velocities[:,1])]
Vzlimits=[np.min(Velocities[:,2]),np.max(Velocities[:,2])]

Axlimits=[np.min(Acc[:,0]),np.max(Acc[:,0])]
Aylimits=[np.min(Acc[:,1]),np.max(Acc[:,1])]
Azlimits=[np.min(Acc[:,2]),np.max(Acc[:,2])]

omgyawlimits=[np.min(AngRates[:,0]),np.max(AngRates[:,0])]
omgpitchlimits=[np.min(AngRates[:,1]),np.max(AngRates[:,1])]
omgrolllimits=[np.min(AngRates[:,2]),np.max(AngRates[:,2])]

dx=[0.1,0.1,0.1]



def pose2Rt(x):
    t=x[0:3].copy()
    phi=x[3]
    xi=x[4]
    zi=x[5]
    
    Rzphi,dRzdphi=gicp.Rz(phi)
    Ryxi,dRydxi=gicp.Ry(xi)
    Rxzi,dRxdzi=gicp.Rx(zi)
    
    R = Rzphi.dot(Ryxi)
    R=R.dot(Rxzi)
    
    return R,t

def Rt2pose(R,t):
    x=np.zeros(6)
    x[:3]=t
    r = Rsc.from_matrix(R)
    phidt,xidt,zidt =r.as_euler('zyx', degrees=False)
    x[3]=phidt
    x[4]=xidt
    x[5]=zidt
    return x

# x is forward


def dynmodelCT3D(xstatepf,dt):
    # xstatepf1=np.zeros_like(xstatepf)
    
    for i in range(xstatepf.shape[0]):
        xx=xstatepf[i]
        
        
        t=xx[0:3]
        phi=xx[3]
        xi=xx[4]
        zi=xx[5]
        
        v=xx[6]
        omgyaw = xx[7]
        omgpitch = xx[8]
        omgroll = xx[9]
        
        a=xx[10]
        
        Rzphi,dRzdphi=gicp.Rz(phi)
        Ryxi,dRydxi=gicp.Ry(xi)
        Rxzi,dRxdzi=gicp.Rx(zi)
        
        R = Rzphi.dot(Ryxi)
        R=R.dot(Rxzi)
        # R=nplinalg.inv(R)
        
        drn = R.dot([1,0,0])
        tdt=t+(v*drn)*dt+0.5*(a*drn)*dt**2
        vdt=v+a*dt
        
        phidt=phi+omgyaw*dt
        xidt=xi+omgpitch*dt
        zidt=zi+omgroll*dt
        
        
        # Rzphi,dRzdphi=gicp.Rz(omgyaw*dt)
        # Ryxi,dRydxi=gicp.Ry(omgpitch*dt)
        # Rxzi,dRxdzi=gicp.Rx(omgroll*dt)
        
        # Romg = Rzphi.dot(Ryxi)
        # Romg=Romg.dot(Rxzi)
        
        # Rdt=R.dot(Romg)
        
        # r = Rsc.from_matrix(Rdt)
        # phidt,xidt,zidt =r.as_euler('zyx', degrees=False)
        
        xstatepf[i,3]=phidt
        xstatepf[i,4]=xidt
        xstatepf[i,5]=zidt
        
        xstatepf[i,0:3]=tdt
        xstatepf[i,6]=vdt

    
    return xstatepf


def dynmodelUM3D(xstatepf,dt):
    # xstatepf1=np.zeros_like(xstatepf)
    # xstatepf = [x,y,z,vx,vy,vz,ax,ay,az]
    for i in range(xstatepf.shape[0]):
        xx=xstatepf[i]
        
        
        t=xx[0:3]
        v=xx[3:6]
        a=xx[6:]
        

        tdt=t+v*dt+0.5*a*dt**2
        xstatepf[i,0:3]=tdt
        
        vdt = v+a*dt
        xstatepf[i,3:6]=vdt
        
        
        
        

    
    return xstatepf

          
# car pose is phi,xi,zi,tpos,v,omgyaw,omgpitch,omgroll    where v is velocity of car, omg is angular velocity

# initialize

# CT
def getCTinitialSamples_origin(Npf):
    xstatepf=np.zeros((Npf,11))
    
    
    xstatepf[:,0]=1*np.random.randn(Npf)
    xstatepf[:,1]=1*np.random.randn(Npf)
    xstatepf[:,2]=0.1*np.random.randn(Npf)
    
    #yaw
    xstatepf[:,3]=5*np.random.randn(Npf)
    #pitch
    xstatepf[:,4]=5*np.random.randn(Npf)
    #roll
    xstatepf[:,5]=0.001*np.random.randn(Npf)
    
    
    
    #vel
    xstatepf[:,6]=5*np.abs(np.random.randn(Npf))
    
    #omgyaw
    xstatepf[:,7]=0.001*np.random.randn(Npf)
    #omgpitch
    xstatepf[:,8]=0.001*np.random.randn(Npf)
    #omgroll
    xstatepf[:,9]=0.001*np.random.randn(Npf)
    
    #acc
    a=nplinalg.norm(Acc,axis=1)
    anormlimits=[np.min(a),np.max(a)]
    xstatepf[:,10]=0.01*np.abs(np.random.randn(Npf))
    
    wpf=np.ones(Npf)/Npf
    
    Q=np.diag([(0.1)**2,(0.1)**2,(0.1)**2, # x-y-z
                (2*np.pi/180)**2,(2*np.pi/180)**2,(0.1*np.pi/180)**2, # angles
                (0.2)**2,   # velocity
                (0.05*np.pi/180)**2,(0.05*np.pi/180)**2,(0.02*np.pi/180)**2,# angle rates
                (0.01)**2]) #acc
    
    return xstatepf,wpf,Q

def getCTinitialSamples(Npf):
    xstatepf=np.zeros((Npf,11))
    
    
    xstatepf[:,0]=np.random.rand(Npf)*(Xlimits[1]-Xlimits[0])+Xlimits[0]
    xstatepf[:,1]=np.random.rand(Npf)*(Ylimits[1]-Ylimits[0])+Ylimits[0]
    xstatepf[:,2]=np.random.rand(Npf)*(Zlimits[1]-Zlimits[0])+Zlimits[0]
    
    #yaw
    xstatepf[:,3]=np.random.rand(Npf)*(yawlimits[1]-yawlimits[0])+yawlimits[0]
    #pitch
    xstatepf[:,4]=np.random.rand(Npf)*(pitchlimits[1]-pitchlimits[0])+pitchlimits[0]
    #roll
    xstatepf[:,5]=np.random.rand(Npf)*(rolllimits[1]-rolllimits[0])+rolllimits[0]
    
    
    
    #vel
    xstatepf[:,6]=np.random.rand(Npf)*(vlimits[1]-vlimits[0])+vlimits[0]
    
    #omgyaw
    xstatepf[:,7]=np.random.rand(Npf)*(omgyawlimits[1]-omgyawlimits[0])+omgyawlimits[0]
    #omgpitch
    xstatepf[:,8]=np.random.rand(Npf)*(omgpitchlimits[1]-omgpitchlimits[0])+omgpitchlimits[0]
    #omgroll
    xstatepf[:,9]=np.random.rand(Npf)*(omgrolllimits[1]-omgrolllimits[0])+omgrolllimits[0]
    
    #acc
    a=nplinalg.norm(Acc,axis=1)
    anormlimits=[np.min(a),np.max(a)]
    xstatepf[:,10]=np.random.rand(Npf)*(anormlimits[1]-anormlimits[0])+anormlimits[0]
    
    wpf=np.ones(Npf)/Npf
    
    Q=np.diag([(0.1)**2,(0.1)**2,(0.1)**2, # x-y-z
                (2*np.pi/180)**2,(2*np.pi/180)**2,(0.1*np.pi/180)**2, # angles
                (0.2)**2,   # velocity
                (0.05*np.pi/180)**2,(0.05*np.pi/180)**2,(0.02*np.pi/180)**2,# angle rates
                (0.01)**2]) #acc
    
    
    return xstatepf,wpf,Q

# UM
def getUMinitialSamples(Npf):
    xstatepf=np.zeros((Npf,9))
    
    
    xstatepf[:,0]=np.random.rand(Npf)*(Xlimits[1]-Xlimits[0])+Xlimits[0]
    xstatepf[:,1]=np.random.rand(Npf)*(Ylimits[1]-Ylimits[0])+Ylimits[0]
    xstatepf[:,2]=np.random.rand(Npf)*(Zlimits[1]-Zlimits[0])+Zlimits[0]
    
    #vx
    xstatepf[:,3]=np.random.rand(Npf)*(Vxlimits[1]-Vxlimits[0])+Vxlimits[0]
    #vy
    xstatepf[:,4]=np.random.rand(Npf)*(Vylimits[1]-Vylimits[0])+Vylimits[0]
    #vy
    xstatepf[:,5]=np.random.rand(Npf)*(Vzlimits[1]-Vzlimits[0])+Vzlimits[0]
    
    
    
    #ax
    xstatepf[:,6]=np.random.rand(Npf)*(Axlimits[1]-Axlimits[0])+Axlimits[0]
    #ay
    xstatepf[:,7]=np.random.rand(Npf)*(Aylimits[1]-Aylimits[0])+Aylimits[0]
    #az
    xstatepf[:,8]=np.random.rand(Npf)*(Azlimits[1]-Azlimits[0])+Azlimits[0]
    
    wpf=np.ones(Npf)/Npf
    
    Q=1*np.diag([(0.1)**2,(0.1)**2,(0.1)**2, # x-y-z
               (1)**2,(1)**2,(0.25)**2, # vel
               (0.5)**2,(0.5)**2,(0.025)**2]) # acc
    
    return xstatepf,wpf,Q





def measModel(xx):
    t=xx[0:3]
    r=nplinalg.norm(t)
    th = np.arccos(t[2]/r)
    phi = np.arctan2(t[1],t[0])
    
    return np.array([r,th,phi])



Rposnoise=np.diag([(1)**2,(1)**2,(1)**2])
    
# sampligfunc = getCTinitialSamples
sampligfunc= getCTinitialSamples_origin

xstatepf, wpf, Q =sampligfunc(Npf)
# xstatepf, wpf, Q =getUMinitialSamples(Npf)

dim=xstatepf.shape[1]
robotpf=pf.Particles(X=xstatepf,wts=wpf)




# figpf = plt.figure()    
# axpf = figpf.add_subplot(111)
# axpf.plot(Xtpath[:,0],Xtpath[:,1],'k')
# axpf.plot(robotpf.X[:,0],robotpf.X[:,1],'r.')
# axpf.set_title("initial pf points")
# plt.show() 
 
dmax=D["Localize"]["dmax"]
# sig=1

#
#
#

XpfTraj=np.zeros((len(dataset),xstatepf.shape[1]))
m,P=robotpf.getEst()
XpfTraj[0]=m

# vis = o3d.visualization.Visualizer()
# vis.create_window()
save_image = False


makeMatPlotLibdebugPlot=True
makeOpen3DdebugPlot=False

if makeMatPlotLibdebugPlot:
    figpf = plt.figure()    
    axpf = figpf.add_subplot(111,projection='3d')

# figpftraj = plt.figure()    
# axpftraj = figpftraj.add_subplot(111,projection='3d')


        


# with open("testpcllocalize.pkl","wb") as FF:
#     pickle.dump([robotpf.X,X1v_down],FF)
    
for k in range(1,len(dataset)):
    print(k)
    st=time.time()
    X1v = dataset.get_velo(k)
    et=time.time()
    print("read velo time = ",et-st)
    
    X1v=X1v[:,:3]
    
    idxx=(X1v[:,0]>-200) & (X1v[:,0]<200) & (X1v[:,1]>-200) & (X1v[:,1]<200 )& (X1v[:,2]>-100) & (X1v[:,2]<100)
    X1v=X1v[idxx,:]
    
    
    # X1v=np.hstack([X1v,np.ones((X1v.shape[0],1))])
    
    
    
    
    
    Hgt=dataset.calib.T_cam0_velo
    
    
    Hgt=np.dot(nplinalg.inv(Hgt),dataset.poses[k].dot(Hgt))
    X1gv=Hgt[0:3,0:3].dot(X1v.T).T+Hgt[0:3,3]
    



    
    # pcdXgv = o3d.geometry.PointCloud()
    # pcdXgv.points = o3d.utility.Vector3dVector(X1gv[:,:3])
    # pcdXgv_down_pcd = pcdXgv.voxel_down_sample(voxel_size=1)
    # X1gv_down = np.asarray(pcdXgv_down_pcd.points)
    
    
    
    dt=dataset.times[k]-dataset.times[k-1]
    
    # propagate
    st=time.time()
    # robotpf.X=dynmodelUM3D(robotpf.X,dt)
    robotpf.X=dynmodelCT3D(robotpf.X,dt)
    
    et=time.time()
    print("dyn model time = ",et-st)
    robotpf.X=robotpf.X+2*np.random.multivariate_normal(np.zeros(dim), Q, Npf)
    
    # measurement update
    
    LiK=[]
    
    pcdXv = o3d.geometry.PointCloud()
    pcdXv.points = o3d.utility.Vector3dVector(X1v[:,:3])
    pcdXv_down_pcd = pcdXv.voxel_down_sample(voxel_size=0.5)
    X1v_down = np.asarray(pcdXv_down_pcd.points)
    # st=time.time()
    # for j in range(Npf):
        
    #     # mm=robotpf.X[j][:3] #measModel(robotpf.X[j])
    #     # ypdf = multivariate_normal.pdf(Xtpath[k,:3], mean=mm, cov=Rposnoise)
    #     # robotpf.wts[j]=(1e-6+ypdf)*robotpf.wts[j]
        
    #     # print("j=",j)
    #     Dv=[]
    #     R,t = pose2Rt(robotpf.X[j][:6])
        
    #     X1gv_pose = R.dot(X1v_down.T).T+t
        
        
    #     # bbox3d=o3d.geometry.AxisAlignedBoundingBox(min_bound=np.min(X1gv_pose,axis=0)-5,max_bound=np.max(X1gv_pose,axis=0)+5)
    #     # pcdbox=pcddown.crop(bbox3d)
        
    #     # pcd_tree_local = o3d.geometry.KDTreeFlann(pcdbox)

        
    #     # Xmap=np.asarray(pcdbox.points)
    #     # for i in range(X1gv_pose.shape[0]):
    #     #     [_, idx, d] = pcd_tree.search_hybrid_vector_3d(X1gv_pose[i], dmax, 1)
    #     #     if len(idx)>0:
    #     #         Dv.append(d[0])
    #     #     else:
    #     #         Dv.append(dmax)
    #     # Dv=np.array(Dv)
        
        
    #     ddsktree,idxs = sktree.query(X1gv_pose,k=1, dualtree=False,return_distance = True)
    #     ddsktree=ddsktree.reshape(-1)
    #     idxs=idxs.reshape(-1)
    #     idxs=idxs[ddsktree<=dmax]
    #     Dv=ddsktree.copy()
    #     Dv[Dv>dmax]=dmax
        
    #     sig=D["sig0"]*np.sqrt(X1gv_pose.shape[0])
    #     likelihood= np.exp(-np.sum(Dv**2)/sig**2)
    #     likelihood=np.max([1e-20,likelihood])
    #     print(j,likelihood)
    #     LiK.append(likelihood)
    #     robotpf.wts[j]=likelihood*robotpf.wts[j]
        
    #     # Hpcl=slam.registrations(X1gv_down,X1gv_pose,json.dumps(D))
    #     # Hpcl=dict(Hpcl)
    #     # H=Hpcl["H_gicp"]
    #     # Hj = np.zeros((4,4))
    #     # Hj[0:3,0:3]=R
    #     # Hj[0:3,3]=t
    #     # Hj[3,3]=1
        
    #     # Hj=nplinalg.inv(Hj)
        
    #     # Hj=Hj.dot(H)
    #     # RR=Hj[0:3,0:3]
    #     # tt=Hj[0:3,3]
    #     # xpose=Rt2pose(RR,tt)
    #     # # pose2Rt
    #     # robotpf.X[j][0:6]=xpose

        
    # et=time.time()
    # print("meas model time = ",et-st)
    
    
    
    #
    idxpf=0
    doBinMatch=0
    if doBinMatch:
        st=time.time()
        ret=pclloc.computeLikelihood(robotpf.X,X1v_down)
        ret=dict(ret)
        beforelikelihood=ret['likelihood'].reshape(-1)
        
        # st=time.time()
        # pcldon = slam.Don(opts)
        # pcldon.setMapX(X1v)
        # pcldon.computeNormals(opts)
        # pcldon.computeDon(opts)
        # ret=pcldon.filter(opts)
        # ret=dict(ret)
        # X1v_roadrem=ret["Xout"]
        # et=time.time()
        # print("time to remove road from scan = ",et-st)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(X1v[:,:3])
        downpcd = pcd.voxel_down_sample(voxel_size=0.05)
        downpcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
        X1vnorm = np.asarray(downpcd.normals)
        idd=np.abs(X1vnorm[:,2])<0.5
        
        
        X1v_roadrem=np.asarray(downpcd.points)
        X1v_roadrem=X1v_roadrem[idd]
        
        X1gv_roadrem=Hgt[0:3,0:3].dot(X1v_roadrem.T).T+Hgt[0:3,3]
        
        dxMatch=np.array([1,1])
        dxBase=np.array([30,30])
        Lmax=np.array([200,200])
        thmax=170*np.pi/180
        thmin=2.5*np.pi/180
        
        idxpfmx=np.argmax(robotpf.wts)
        for idxpf in [idxpfmx]:
            
            Ridx,tidx = pose2Rt(robotpf.X[idxpf][:6])
            phiidx = robotpf.X[idxpf][3]
            Rzphiidx,_=gicp.Rz(phiidx)
            
            H12est=np.identity(3)
            H12est[0:2,0:2]=Rzphiidx[0:2,0:2]
            H12est[0:2,2]=tidx[:2]
            
            X1v2D=X1v_roadrem[:,:2].copy()
            X1v2D=down_sample(X1v2D,dxMatch[0])
            #ax.cla()
            #ax.plot(X1v_roadrem[:,0],X1v_roadrem[:,1],'k.')
        
            X1v2Dgpf=H12est[0:2,0:2].dot(X1v2D.T).T+H12est[0:2,2]
        
        
            X1gvdown=down_sample(X1gv[:,:2],dxMatch[0])
        
            
            st=time.time()
            # Hbin21=binMatcherAdaptive3(X11,X2,H12est,Lmax,thmax,thmin,dxMatch)
            # Hbin21,costs=binMatcherAdaptive3(X2Dmap_down,X1v2D,H12est,Lmax,thmax,thmin,dxMatch)
            Hbin21,cost0,cost,hh,hR=binmatchers.binMatcherAdaptive_super(X2Dmap_down,X1v2D,H12est,Lmax,thmax,thmin,dxMatch,dxBase)
            et=time.time()
            pkl
            Hbin12 = nplinalg.inv(Hbin21)
            X1v2Dgc=Hbin12[0:2,0:2].dot(X1v2D.T).T+Hbin12[0:2,2]
            
            # with open("testBinMatch.pkl","wb") as FF:
            #     pkl.dump([X2Dmap_down,X1v2D,H12est,Lmax,thmax,thmin,dxMatch,dxBase,Hbin12],FF)

            
            figbf = plt.figure("bin-fit")
            if len(figbf.axes)>0:
                ax = figbf.axes[0]
            else:
                ax = figbf.add_subplot(111)
            ax.cla()
            ax.plot(X2Dmap_down[:,0],X2Dmap_down[:,1],'k.')
            ax.plot(X1v2Dgpf[:,0],X1v2Dgpf[:,1],'b.',label='pf-pose')
            ax.plot(X1v2Dgc[:,0],X1v2Dgc[:,1],'r.',label='pf-pose-corrected')
            ax.plot(X1gv_roadrem[:,0],X1gv_roadrem[:,1],'g.',label='ground truth')
            ax.legend()
            ax.axis("equal")
            
            
            tc,thc=nbpt2Dproc.extractPosAngle(Hbin12)
            
            
            if cost>cost0:
                # idxpfleast=np.argmin(robotpf.wts)
                # robotpf.X[idxpfleast]=robotpf.X[idxpf]
                # robotpf.wts[idxpfleast]=robotpf.wts[idxpf]
                # robotpf.X[idxpfleas50t][3]=thc
                # robotpf.X[idxpfleast][:2]=tc
                
                robotpf.X[idxpf][3]=thc
                robotpf.X[idxpf][:2]=tc
                # robotpf.wts[idxpf]=0.8
                # print(thc)
                Ridx,tidx = pose2Rt(robotpf.X[idxpf][:6])
                Hidx=np.identity(4)
                Hidx[0:3,0:3]=Ridx
                Hidx[0:3,3]=tidx
                X1v_down_pfpose = Ridx.dot(X1v_down.T).T+tidx
                
                Hpcl={'H_icp':np.identity(4)}
                bbox3d=o3d.geometry.AxisAlignedBoundingBox(min_bound=np.min(X1v_down_pfpose,axis=0)-5,max_bound=np.max(X1v_down_pfpose,axis=0)+5)
                pcdbox=pcd3DdownSensorCostDown.crop(bbox3d)
                Xmappflocal = np.asarray(pcdbox.points)
                Hpcl=slam.registrations(Xmappflocal.copy(),X1v_down_pfpose.copy(),json.dumps(D))
                Hpcl=dict(Hpcl)
                Ha=Hpcl['H_icp'].dot(Hidx)
                xx=Rt2pose(Ha[0:3,0:3],Ha[0:3,3])
                robotpf.X[idxpf][:6]=xx
                pass
            
        robotpf.renormlizeWts()
    
    print("Pf#%d wt before = ",robotpf.wts[idxpf])
    st=time.time()
    ret=pclloc.computeLikelihood(robotpf.X,X1v_down)
    ret=dict(ret)
    likelihood=ret['likelihood'].reshape(-1)
    # Xposes=ret['Xposes_corrected']
    # likelihood=np.array(likelihood)
    likelihood=np.exp(-likelihood)
    likelihood[likelihood<1e-20]=1e-20
    # break
    robotpf.wts=likelihood*robotpf.wts
    et=time.time()
    print("pcl meas model time = ",et-st)
    
    robotpf.renormlizeWts()
    print("Pf#%d wt after = ",robotpf.wts[idxpf])
    
    # now do icp for the highest wt
    # st=time.time()
    # mxwtidx = np.argmax(robotpf.wts)
    # Rj,tj = pose2Rt(robotpf.X[mxwtidx][:6])
    # iddx=np.abs(X1v_down[:,0])<75
    # iddy=np.abs(X1v_down[:,1])<75
    # iddz=X1v_down[:,2]>0
    # idd = iddx & iddy & iddz
    # X1gv_pose = Rj.dot(X1v_down[idd,:].T).T+tj
    
    # bbox3dmap=o3d.geometry.AxisAlignedBoundingBox(min_bound=np.min(X1gv_pose,axis=0)-5,max_bound=np.max(X1gv_pose,axis=0)+5)
    # pcdboxmap=pcddown.crop(bbox3dmap)
    # Xmaplocal=np.asarray(pcdboxmap.points)
    # Hpcl=slam.registrations(Xmaplocal,X1gv_pose,json.dumps(D))
    # Hpcl=dict(Hpcl)
    # H_gicp=Hpcl["H_gicp"]
    # # H_gicp= nplinalg.inv(H_gicp)
    # Hj = np.zeros((4,4))
    
    # Hj[0:3,0:3]=Rj
    # Hj[0:3,3]=tj
    # Hj[3,3]=1
    
    # X1gv_pose_corr = H_gicp[0:3,0:3].dot(X1gv_pose.T).T+H_gicp[0:3,3]
    
    # figcorr = plt.figure()    
    # axcorr= figcorr.add_subplot(111)
    # axcorr.cla()
    # axcorr.plot(Xmaplocal[:,0],Xmaplocal[:,1],'k.',label='map')
    # axcorr.plot(X1gv_pose[:,0],X1gv_pose[:,1],'r.',label='orig-pose')
    # axcorr.plot(X1gv_pose_corr[:,0],X1gv_pose_corr[:,1],'b.',label='corr-pose')
    # axcorr.legend()
    
    # et=time.time()
    # print("icp correction time = ",et-st)
    
    
    
    Neff=robotpf.getNeff()
    if Neff/Npf<0.25:
        robotpf.bootstrapResample()
    
    # robotpf.regularizedresample(kernel='gaussian',scale=1/25)
    
    m,P=robotpf.getEst()
    XpfTraj[k]=m
    u,v=nplinalg.eig(P)
    
    # if min(u)>50:
    #     xstatepf,wpf,Q=sampligfunc(Npf)
    #     dim=xstatepf.shape[1]
    #     robotpf=pf.Particles(X=xstatepf,wts=wpf)
        
    # downpcd = pcd.voxel_down_sample(voxel_size=0.1)
    
    # vis.update_geometry(downpcd)
    # vis.poll_events()
    # vis.update_renderer()
    # if save_image:
    #     vis.capture_screen_image("kitti_seq-%s_%06d.jpg" % (sequence,k))
    
    if makeMatPlotLibdebugPlot:
        axpf.cla()
        axpf.plot(Xtpath[:k,0],Xtpath[:k,1],Xtpath[:k,2],'k')
        axpf.plot(robotpf.X[:,0],robotpf.X[:,1],robotpf.X[:,2],'r.')
        
        # xmap = np.asarray(pcdbox.points)
        # axpf.plot(xmap[:,0],xmap[:,1],xmap[:,2],'k.')
    
    if makeMatPlotLibdebugPlot:
        st=time.time()
        ArrXpf=[]
        for i in range(robotpf.X.shape[0]):
            xx=robotpf.X[i]
            
            
            R,t = pose2Rt(robotpf.X[i][:6])
            
            drn = R.dot([1,0,0])
            t2=t+5*drn
            ArrXpf.append(t2)
            G=np.vstack([t,t2])
            if makeMatPlotLibdebugPlot:
                axpf.plot(G[:,0],G[:,1],G[:,2],'r')
            
            
            if makeOpen3DdebugPlot:
                if i==0:     
                    X1gv_pose = R.dot(X1gv.T).T+t
                    # axpf.plot(G[:,0],G[:,1],G[:,2],'b')
                    # axpf.plot(X1gv_pose[:,0],X1gv_pose[:,1],X1gv_pose[:,2],'r.')
                    
                    pcdX1gv_pose = o3d.geometry.PointCloud()
                    pcdX1gv_pose.points = o3d.utility.Vector3dVector(X1gv_pose)
                    pcdX1gv_pose.paint_uniform_color([0,1,0]) #green
                    
                    bbox3d=o3d.geometry.AxisAlignedBoundingBox(min_bound=np.min(X1gv_pose,axis=0)-5,max_bound=np.max(X1gv_pose,axis=0)+5)
                    pcdbox=pcd3DdownSensorCost.crop(bbox3d)
                    pcdbox.paint_uniform_color([220,220,220])
    
        et=time.time()
        print("arrow 3d plot time of pf partilees = ",et-st)
    if makeOpen3DdebugPlot :
        # plot PF points as black
        pcdPF = o3d.geometry.PointCloud()
        pcdPF.points = o3d.utility.Vector3dVector(robotpf.X[:,:3])
        pcdPF.paint_uniform_color([0,0,0]) #black
        
        # plot PF arrows as red and pose pf point as green
        ArrXpf=np.array(ArrXpf)
        pfline_set = o3d.geometry.LineSet()
        pfline_set.points = o3d.utility.Vector3dVector( np.vstack([robotpf.X[:,:3],ArrXpf]))
        pflines = [[i,i+Npf] for i in range(Npf)]
        pfline_set.lines = o3d.utility.Vector2iVector(pflines)
        colors = [[1, 0, 0] for i in range(len(pflines))]
        pfline_set.colors = o3d.utility.Vector3dVector(colors)
        np.asarray(pfline_set.colors)[0]=[0,1,0] #green
        
        
        # plot path as black
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(Xtpath[:k,:3])
        lines = [[i,i+1] for i in range(k-1)]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        colors = [[0, 0, 0] for i in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        # pcdbox.paint_uniform_color([0,0,0])
        o3d.visualization.draw_geometries([pcdbox,pcdX1gv_pose,pcdPF,pfline_set,line_set])
    
    if makeMatPlotLibdebugPlot:
        # axpf.set_xlim(Xlimits)
        # axpf.set_ylim(Ylimits)
        # axpf.set_zlim(Zlimits)
        axpf.set_title("time step = %d"%k)
        axpf.set_xlabel('x')
        axpf.set_ylabel('y')
        axpf.set_zlabel('z')
        plt.pause(0.1)
    
    
    # axpftraj.cla()
    # axpftraj.plot(Xtpath[:k,0],Xtpath[:k,1],Xtpath[:k,2],'k')
    # axpftraj.plot(XpfTraj[:k,0],XpfTraj[:k,1],XpfTraj[:k,2],'r')
    # axpftraj.set_xlim(Xlimits)
    # axpftraj.set_ylim(Ylimits)
    # axpftraj.set_zlim(Zlimits)
    # axpftraj.set_title("time step = %d"%k)
    
    # plt.show()
    plt.pause(0.1)
    
    if k>=400:
        break
# vis.destroy_window()    