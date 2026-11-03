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
# from pyslam import  slam
import json

#%%
plt.close("all")
# basedir =r'P:\SLAMData\Kitti\visualodo\dataset'
basedir =r'ftp://homenas.local/home/research/SLAMData/Kitti/visualodo/dataset'
# basedir ='/media/na0043/misc/DATA/KITTI/odometry/dataset'
# Specify the dataset to load
# sequence = '02'
# sequence = '05'
# sequence = '06'
# sequence = '08'
loop_closed_seq = ['02','05','06','08']
sequence = '05'




# pcd3D=o3d.io.read_point_cloud("kitti-pcd-seq-%s.pcd"%sequence)
# pcd3DdownSensorCost=pcd3D.voxel_down_sample(voxel_size=0.5)


pcd3Droadremove=o3d.io.read_point_cloud("kitti-pcd-seq-roadremove-%s.pcd"%sequence)
np.asarray(pcd3Droadremove.points)[:,2]=0
X2Dmap_down=np.array(pcd3Droadremove.voxel_down_sample(voxel_size=0.1).points)
X2Dmap_down=X2Dmap_down[:,:2]









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

def binMatcherAdaptive3(X11,X22,H12,Lmax,thmax,thmin,dxMatch,dxBase):
    #  X11 are global points
    # X22 are points with respect to a local frame (like origin of velodyne)
    # H12 takes points in the velodyne frame to X11 frame (global)
    
    # dxBase=dxMatch*(np.floor(dxBase/dxMatch)+1)
    
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

    
    print("dxs[0]=",dxs[0])
    print("dxs[-1]=",dxs[-1])
    
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
    
    # LmaxOrig=np.zeros(2,dtype=np.float64)
    # LmaxOrig[0]=Lmax[0]
    # LmaxOrig[1]=Lmax[1]

    SolBoxes_init=[]
    # X2[:,0]=X2[:,0]-LmaxOrig[0]
    # X2[:,1]=X2[:,1]-LmaxOrig[1]
    
    # print("Lmax = ",Lmax)
    # print("LmaxSide = ",2*Lmax)
    # L0=dxs[-1]*(np.floor(L0/dxs[-1]))
    # L1=dxs[-1]*(np.floor(L1/dxs[-1])+1)
    # print("L0=",L0)
    # print("L1=",L1)
    # print("L0+mn=",L0+mn_orig)
    # print("L1+mn=",L1+mn_orig)
    # LmaxNside = L1-L0
    # n=np.floor(np.log2(LmaxNside/dxs[-1]))+1
    # LmaxNside=dxs[-1]*np.power(2, n)
    # print("LmaxNside = ",LmaxNside)
    # Lmax=dxs[-1]*(np.floor(Lmax/dxs[-1])+1)
    # Lmax1=dxs[-1]*(np.floor(Lmax/dxs[-1])+1)
    # Lmax0=dxs[-1]*(np.floor(Lmax/dxs[-1])-1)
    
    # print("Lmax = ",Lmax)
    # for xs in np.arange(L0[0],L1[0],dxs[0][0]):
    #     for ys in np.arange(-Lmax[1],Lmax[1],dxs[0][1]):
    #         SolBoxes_init.append( (xs,ys,dxs[0][0],dxs[0][1]) )
    for xs in np.arange(0,dxs[0][0],dxs[0][0]):
        for ys in np.arange(0,dxs[0][1],dxs[0][1]):
            SolBoxes_init.append( (xs,ys,dxs[0][0],dxs[0][1]) )        
    
    
    print(SolBoxes_init)
    
    
    h=[(100000.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)]
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
        XX=np.transpose(H12mn[0:2,0:2].dot(XX.T))#+H12mn[0:2,2]
        Xth[th]=XX
        
        
        for solbox in SolBoxes_init:
            xs,ys,d0,d1 = solbox
            Tj=np.array((d0,d1))
            Oj = np.array((xs,ys)) 
        # d0,d1=Tj=dxs[0]
        # xs,ys=Oj = -Lmax 
        
            cost2=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj)
            h.append((-cost2-np.random.rand()/1e10,xs,ys,d0,d1,lvl,th,mxLVL))
        
    heapq.heapify(h)
    
    print("len heap at 0 = ",len(h))
    
    XX0=np.transpose(H12mn[0:2,0:2].dot(X22.T))+H12mn[0:2,2]
    zz=np.zeros(2,dtype=np.float64)
    # print("dxs[-1] = ",dxs[-1])
    cost0=nbpt2Dproc.getPointCost(HLevels[-1],dxs[-1],XX0,zz,dxs[-1])
    # print("cost0 = ",cost0)
    # print("len(heap) = ",len(h))

    bb1={'x1':t0[0]-Lmax[0],'y1':t0[1]-Lmax[1],'x2':t0[0]+Lmax[0],'y2':t0[1]+Lmax[1]}
    while(1):
        HH=[]
        flg=False
        for i in range(len(h)):
            
            if h[i][3]>dxBase[0] or h[i][4]>dxBase[1]:
                flg=True
                (cost,xs,ys,d0,d1,lvl,th,mxLVL)=h[i]
        
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
                        bb2={'x1':xs,'y1':ys,'x2':xs+d0,'y2':ys+d1}
                        x_left = max(bb1['x1'], bb2['x1'])
                        y_top = max(bb1['y1'], bb2['y1'])
                        x_right = min(bb1['x2'], bb2['x2'])
                        y_bottom = min(bb1['y2'], bb2['y2'])
                    
                        if x_right < x_left or y_bottom < y_top:
                            continue
                        
                        # if np.abs(xs-t0[0])>Lmax[0] or np.abs(ys-t0[1])>Lmax[1]:
                        #     continue
                        # if xs+d0<L0[0] or ys+d1<L0[1]:
                        #     continue
                        
                        Oj = np.array((xs,ys))
                        cost3=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj) 
                        HH.append((-cost3-np.random.rand()/1e10,xs,ys,d0,d1,float(nlvl),th,mxLVL))
            else:
                HH.append(h[i])
        h=HH
        if flg==False:
            break
    
    heapq.heapify(h)
    
    print("len heap at 1 = ",len(h))
    
    
    while(1):
        (cost,xs,ys,d0,d1,lvl,th,mxLVL)=heapq.heappop(h)
        mainSolbox = (cost,xs,ys,d0,d1,lvl,th,mxLVL)
        if lvl==mxLVL:
            break
            cnt=0
            for jj in range(len(h)):
                if np.floor(-h[jj][0])==np.floor(-cost) and h[jj][5]<lvl:
                    cnt+=1
            if cnt==0:
                break
            else:
                continue
            
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
                bb2={'x1':xs,'y1':ys,'x2':xs+d0,'y2':ys+d1}
                x_left = max(bb1['x1'], bb2['x1'])
                y_top = max(bb1['y1'], bb2['y1'])
                x_right = min(bb1['x2'], bb2['x2'])
                y_bottom = min(bb1['y2'], bb2['y2'])
            
                if x_right < x_left or y_bottom < y_top:
                    continue
                
                # if np.abs(xs-t0[0])>Lmax[0] or np.abs(ys-t0[1])>Lmax[1]:
                #     continue
                # if xs+d0<L0[0] or ys+d1<L0[1]:
                #     continue
                
                Oj = np.array((xs,ys))
                cost3=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj) 
                heapq.heappush(h,(-cost3-np.random.rand()/1e10,xs,ys,d0,d1,float(nlvl),th,mxLVL))

    
    # print(h)
    
    # print("lvl,mxLVL,mxlvl",lvl,mxLVL,mxlvl) 
    t=mainSolbox[1:3]-t0
    print("t=",t)  
    th = mainSolbox[6]
    cost=np.floor(-mainSolbox[0])
    d0,d1=mainSolbox[3:5]
    print("mainSolbox d0,d1=",d0,d1)
    
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    XXc=np.transpose(R.dot(X22.T))
    XXc=np.transpose(H12mn[0:2,0:2].dot(XXc.T))+H12mn[0:2,2]+t
    figbf = plt.figure("bin-fit-internal")
    ax = figbf.add_subplot(111)
    ax.plot(X1[:,0],X1[:,1],'k.')
    ax.plot(XX0[:,0],XX0[:,1],'b.',label='pf-pose-orig')
    ax.plot(XXc[:,0],XXc[:,1],'r.',label='pf-pose-corr')
    ax.legend()
    ax.axis("equal")
    ax.set_title("cost0,cost=(%d,%d)"%(cost0,cost))

    
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
    
    hh=[list(ss) for ss in h if np.floor(-ss[0])==np.floor(-mainSolbox[0])]
    for i in range(len(hh)):
        hh[i][1]=hh[i][1]-t0[0]
        hh[i][2]=hh[i][2]-t0[1]
    hh=[tuple(ss) for ss in hh]
    
    hR = []
    # for i in range(len(h)):
    #     g= list(h[i])
    #     t=g[1:3]-t0
    #     R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    #     H0=np.identity(3)
    #     H0[0:2,0:2]=R
    #     H1=np.identity(3)
    #     H1[0:2,2]=t
        
    #     H12i=np.dot(H1.dot(H12mn),H0)
    #     H12i[0:2,2]=H12i[0:2,2]+mn_orig
        
        
    #     g.append(H12i)
    #     hR.append(g)
    
    return H21comp,(cost0,cost),hh,hR
    






def pose2Rt(x):
    t=x[0:3]
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

file='lidarprocessing/000000.bin'
X1v = np.fromfile(file, dtype=np.float32)
X1v=X1v.reshape((-1, 4))

X1v=X1v[:,:3]

idxx=(X1v[:,0]>-200) & (X1v[:,0]<200) & (X1v[:,1]>-200) & (X1v[:,1]<200 )& (X1v[:,2]>-100) & (X1v[:,2]<100)
X1v=X1v[idxx,:]

    

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(X1v[:,:3])
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
downpcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=30))
X1vnorm = np.asarray(downpcd.normals)
idd=np.abs(X1vnorm[:,2])<0.5


X1v_roadrem=np.asarray(downpcd.points)
X1v_roadrem=X1v_roadrem[idd]



dxMatch=np.array([3,3])
dxBase=np.array([70,70])
Lmax=np.array([5100,5100])
thmax=170*np.pi/180
thmin=5*np.pi/180


H12est=np.identity(3)
H12est[0:2,2]=[300,300]

X1v2D=X1v_roadrem[:,:2].copy()
X1v2D=down_sample(X1v2D,dxMatch[0])
#ax.cla()
#ax.plot(X1v_roadrem[:,0],X1v_roadrem[:,1],'k.')

X1v2Dgpf=H12est[0:2,0:2].dot(X1v2D.T).T+H12est[0:2,2]




st=time.time()
# Hbin21=binMatcherAdaptive3(X11,X2,H12est,Lmax,thmax,thmin,dxMatch)
# Hbin21,costs,hh,h=binMatcherAdaptive3(X2Dmap_down,X1v2D,H12est,Lmax,thmax,thmin,dxMatch,dxBase)
Hbin21=binmatchers.binMatcherAdaptive_super(X2Dmap_down,X1v2D,H12est,Lmax,thmax,thmin,dxMatch,dxBase)
et=time.time()
print("time taken = ",et-st)
# print("costs=",costs)
Hbin12 = nplinalg.inv(Hbin21)
X1v2Dgc=Hbin12[0:2,0:2].dot(X1v2D.T).T+Hbin12[0:2,2]

figbf = plt.figure("bin-fit")
if len(figbf.axes)>0:
    ax = figbf.axes[0]
else:
    ax = figbf.add_subplot(111)
ax.cla()
ax.plot(X2Dmap_down[:,0],X2Dmap_down[:,1],'k.')
ax.plot(X1v2Dgpf[:,0],X1v2Dgpf[:,1],'b.',label='pf-pose')
ax.plot(X1v2Dgc[:,0],X1v2Dgc[:,1],'r.',label='pf-pose-corrected')
ax.legend()
ax.axis("equal")
# Xp = np.array([g[-1][0:2,2] for g in h if g[5]>(g[7]-5)])
# ax.plot(Xp[:,0],Xp[:,1],'go')
        
tc,thc=nbpt2Dproc.extractPosAngle(Hbin12)
print("tc=",tc)
print("thc=",thc*180/np.pi)
