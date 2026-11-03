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
import sys
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from threading import Thread
import os
import copy
from utils.plotting import geometryshapes as utpltgmshp   
import pandas as pd
from utils import simmanager as uqsimmanager
from lidarprocessing.numba_codes import gicp
from scipy.spatial.transform import Rotation as Rsc
from scipy.spatial.transform import RotationSpline as RscSpl 
dtype = np.float64
import time
import open3d as o3d
from pykitticustom import odometry
from  uq.filters import pf 
import multiprocessing as mp
import queue
import yaml
from scipy.interpolate import UnivariateSpline

import quaternion
import pickle
from joblib import dump, load

from pyslam import  slam
from pyslam import kittilocal
import argparse
from sklearn.cluster import KMeans
import json





with open('kitti_localize_config.yml', 'r') as outfile:
    D=yaml.safe_load( outfile)

#%%


plt.close("all")

dataset = odometry.odometry(D['Data']['folder'], D['Data']['sequence'], frames=None) # frames=range(0, 20, 5)

try:
    runfilename = __file__
except:
    runfilename = "/home/na0043/Insync/n.adurthi@gmail.com/Google Drive/repos/SLAM/main_kitti_filter_binmatch.py"
    
metalog="""
Journal paper KITTI Bin matching
Author: Venkat
Date: Feb 22 2022

"""
dt=dataset.times[1]-dataset.times[0]
simmanger = uqsimmanager.SimManager(t0=dataset.times[0],tf=dataset.times[-1],dt=dt,dtplot=dt/10,
                                  simname="KITTI-localization-%s"%(D['Data']['sequence']),savepath="simulations",
                                  workdir=os.getcwd())

simmanger.initialize()

simmanger.data['sequence']=D['Data']['sequence']
simmanger.data['basedir']=D['Data']['folder']
simmanger.data['D']=copy.deepcopy(D)

nd=len(dataset.poses)

Xtpath=np.zeros((len(dataset),10))

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




def getmeas(k):
    Hgt=dataset.calib.T_cam0_velo
    Hgt=np.dot(nplinalg.inv(Hgt),dataset.poses[k].dot(Hgt))
    X1v = dataset.get_velo(k)
    X1v=X1v[:,:3]
    
    idxx=(X1v[:,0]>-200) & (X1v[:,0]<200) & (X1v[:,1]>-200) & (X1v[:,1]<200 )& (X1v[:,2]>-100) & (X1v[:,2]<100)
    X1v=X1v[idxx,:]
    
    X1gv=Hgt[0:3,0:3].dot(X1v.T).T+Hgt[0:3,3]
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X1v[:,:3])
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=30))
    X1vnorm = np.asarray(downpcd.normals)
    idd=np.abs(X1vnorm[:,2])<0.5
    X1v_roadrem=np.asarray(downpcd.points)
    X1v_roadrem=X1v_roadrem[idd] 
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X1v_roadrem[:,:3])
    X1v_roadrem = np.asarray(pcd.voxel_down_sample(voxel_size=0.5).points)
    
    X1gv_roadrem=Hgt[0:3,0:3].dot(X1v_roadrem.T).T+Hgt[0:3,3]
    if k==0:
        dt=0
    else:
        dt=dataset.times[k]-dataset.times[k-1]
    tk=dataset.times[k]
    
    return dt,tk,X1v,X1gv,X1v_roadrem,X1gv_roadrem

def getmeas_Q(outQ):
    for k in range(len(dataset.times)):
        outQ.put(getmeas(k))
        print("putted k = ",k)
        
# if __name__=="__main__":
#     outQ = mp.Queue(15)
#     p = mp.Process(target=getmeas_Q, args=(outQ,))
#     p.start()


plt.close("all")
#%%

sequence=D['Data']['sequence']   


Npf = simmanger.data['Npf']=D['PF']['Npf']

k0 = simmanger.data['k0'] = D['PF']['k0']

    
# KL=kittilocal.MapLocalizer(json.dumps(D))
#  /media/na0043/misc/DATA/KITTI/odometry/dataset/sequences/00/velodyne
velofolder = os.path.join(D['Data']['folder'],"sequences",D['Data']['sequence'],"velodyne")
# KL.autoReadMeas_async(velofolder,k0)

# X=KL.getMeasQ_eigen(False)
# X1v = dataset.get_velo(0)

# pcdX1v = o3d.geometry.PointCloud()
# pcdX1v.points = o3d.utility.Vector3dVector(X1v[:,:3])
# pcdX1v.paint_uniform_color([1,0,0]) #green

# pcdX0 = o3d.geometry.PointCloud()
# pcdX0.points = o3d.utility.Vector3dVector(X[0])
# pcdX0.paint_uniform_color([0,0,1]) #green

# o3d.visualization.draw_geometries([pcdX1v,pcdX0])



# pcdX1 = o3d.geometry.PointCloud()
# pcdX1.points = o3d.utility.Vector3dVector(X[1])
# pcdX1.paint_uniform_color([0,1,0]) #green

# o3d.visualization.draw_geometries([pcdX0,pcdX1])
#%% PF 


pcd3D=o3d.io.read_point_cloud("kitti-pcd-seq-%s.pcd"%sequence)
pcd3Droadremove=o3d.io.read_point_cloud("kitti-pcd-seq-roadremove-%s.pcd"%sequence)

pcd3DdownSensorCost=pcd3D.voxel_down_sample(voxel_size=0.5)

Xmap =np.asarray(pcd3D.points)
Xmap2D=np.asarray(pcd3Droadremove.points)


pcd2ddown = o3d.geometry.PointCloud()
Xmap2Dflat=Xmap2D.copy()
Xmap2Dflat[:,2]=0
pcd2ddown.points = o3d.utility.Vector3dVector(Xmap2Dflat[:,:3])
pcd2ddown = pcd2ddown.voxel_down_sample(voxel_size=0.5)

Xmap2Dflat=np.asarray(pcd2ddown.points)



# KL.addMap(Xmap)
# KL.addMap2D(Xmap2D[:,:2])
# KL.setLookUpDist("kitti-pcd-lookupdist-seq-%s.bin"%sequence)
#%%

# KL.resetsim()

time.sleep(2)

# minx,miny,minz,maxx,maxy,maxz=KL.MapPcllimits()







plt.close("all")








        
   
        
def plot2D(k,Xmap2Dflat,X1vroadrem,X1gv_roadrem,binmatchsolss,saveplot=False):
    
        
    figpf2D = plt.figure(figsize=(20, 20))    
    axpf2D = figpf2D.add_subplot(111)

    axpf2D.cla()
    axpf2D.plot(Xmap2Dflat[:,0],Xmap2Dflat[:,1],'k.')

    
    HH=[]
        
    for i in range(len(binmatchsolss)):
        binmatchsol = binmatchsolss[i]
        Hcorr = binmatchsol.H
        HH.append(Hcorr)
        X1vroadrem2Dflatcorr=Hcorr[0:2,0:2].dot(X1vroadrem[:,:2].T).T+Hcorr[0:2,2]
        
        axpf2D.plot(X1vroadrem2Dflatcorr[:,0],X1vroadrem2Dflatcorr[:,1],'g.')
    
    axpf2D.plot(X1gv_roadrem[:,0],X1gv_roadrem[:,1],'b.')

    axpf2D.set_xlabel('x')
    axpf2D.set_ylabel('y')
    axpf2D.axis('equal')

    plt.show(block=False)
    plt.pause(0.1)
    
    if saveplot:
        
        figpf2D.show()

        
        xlim =[Xtpath[k,0]+np.min(X1vroadrem[:,0])-25,Xtpath[k,0]+np.max(X1vroadrem[:,0])+25]
        ylim =[Xtpath[k,1]+np.min(X1vroadrem[:,1])-25,Xtpath[k,1]+np.max(X1vroadrem[:,1])+25]

        
        axpf2D.set_xlim(xlim)
        axpf2D.set_ylim(ylim)
        plt.show(block=False)
        plt.pause(0.1)
        figpf2D.show()
        simmanger.savefigure(figpf2D, ['2Dmap','binmatched'], 'binmatch_'+str(int(k)),data=[k,HH])





Hgt=dataset.calib.T_cam0_velo
Hgt=np.dot(nplinalg.inv(Hgt),dataset.poses[k0].dot(Hgt))


#%%
# D['BinMatch']["dxMatch"]=[1.0,1.0]
# D['BinMatch']["dxBase"]=[100.0,100.0]
# D['BinMatch']["Lmax"]=[500.0,500.0]
# D['BinMatch']["thmax"]=170*np.pi/180
# D['BinMatch']["thfineres"]=2.5*np.pi/180

# bm=kittilocal.BinMatch(json.dumps(D))
# bm.computeHlevels(Xmap2Dflat[:,:2])

# cnt=0
# rmse=[]
# while 1:
#     print(cnt)
#     if cnt > 200:
#         break
    
#     k = np.random.randint(len(dataset.times)-1)

    

#     Hgt=dataset.calib.T_cam0_velo
#     Hgt=np.dot(nplinalg.inv(Hgt),dataset.poses[k].dot(Hgt))
#     dt,tk,X1v,X1gv,X1v_roadrem,X1gv_roadrem=getmeas(k)
    
#     X1v_roadrem=X1v_roadrem[:,:2]
    
#     H12=np.identity(3)
#     st=time.time()
#     binmatchsolss=bm.getmatch(X1v_roadrem,H12)
#     et=time.time()
#     dtime = et-st
    
#     for i in range(len(binmatchsolss)):
#         binmatchsol = binmatchsolss[i]
#         H12corr = binmatchsol.H
#         X1v_roadrem_corr = H12corr[0:2,0:2].dot(X1v_roadrem.T).T + H12corr[0:2,2]
#         Xerr = np.power(nplinalg.norm(X1v_roadrem_corr[:,0:2]-X1gv_roadrem[:,0:2],axis=1),2)
#         rmse.append( [np.sqrt(np.mean(Xerr)),dtime,k] )
        
        
#     # plot2D(k,Xmap2Dflat,X1v_roadrem,X1gv_roadrem,binmatchsolss,saveplot=False)
#     cnt+=1

# np.savez('rmse500.npz', rmse=rmse)


# fig44, ax44= plt.subplots(figsize=(20, 9))
# rmse=np.array(rmse)

# # ax44.boxplot(MTS[methodstrings],positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
# ax44.hist(rmse[:,0],bins=50,rwidth=0.9)
# # ax44.set_xticklabels(methodstrings)
# ax44.set_xlabel("RMSE scan points",fontsize=20)
# # ax44.set_ylabel("count",fontsize=20)
# # ax44.set_xlabel('Methods')
# ax44.tick_params(axis='x', labelsize=20 )
# ax44.tick_params(axis='y', labelsize=20 )
# ax44.grid(True,which="both",axis="y")


# fig5, ax5= plt.subplots(figsize=(20, 9))

# RR=rmse[:,0]
# # ax5.boxplot(MTS[methodstrings],positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
# ax5.hist(RR[RR<10],bins=50,rwidth=0.9)
# # ax5.set_xticklabels(methodstrings)
# ax5.set_xlabel("RMSE scan points",fontsize=20)
# # ax5.set_ylabel("count",fontsize=20)
# # ax5.set_xlabel('Methods')
# ax5.tick_params(axis='x', labelsize=20 )
# ax5.tick_params(axis='y', labelsize=20 )
# ax5.grid(True,which="both",axis="y")


# fig6, ax6= plt.subplots(figsize=(20, 9))

# # ax6.boxplot(MTS[methodstrings],positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
# ax6.hist(rmse[:,1],bins=50,rwidth=0.9)
# # ax6.set_xticklabels(methodstrings)
# ax6.set_xlabel("time taken",fontsize=20)
# # ax6.set_ylabel("count",fontsize=20)
# # ax6.set_xlabel('Methods')
# ax6.tick_params(axis='x', labelsize=20 )
# ax6.tick_params(axis='y', labelsize=20 )
# ax6.grid(True,which="both",axis="y")
#%%
D['BinMatch']["dxMatch"]=[2,2]
D['BinMatch']["dxBase"]=[30.0,30.0]
D['BinMatch']["Lmax"]=[200.0,200.0]
D['BinMatch']["thmax"]=170*np.pi/180
D['BinMatch']["thfineres"]=2.5*np.pi/180

bm=kittilocal.BinMatch(json.dumps(D))
bm.computeHlevels(Xmap2Dflat[:,:2])

cnt=0
rmse=[]
while 1:
    print(cnt)
    if cnt > 200:
        break
    
    k = np.random.randint(len(dataset.times)-1)

    

    Hgt=dataset.calib.T_cam0_velo
    Hgt=np.dot(nplinalg.inv(Hgt),dataset.poses[k].dot(Hgt))
    dt,tk,X1v,X1gv,X1v_roadrem,X1gv_roadrem=getmeas(k)
    
    X1v_roadrem=X1v_roadrem[:,:2]
    
    H12=np.identity(3)
    H12[0:2,2]=Hgt[0:2,3]+90*(2*np.random.rand(1,2)-1)
    st=time.time()
    binmatchsolss=bm.getmatch(X1v_roadrem,H12)
    et=time.time()
    dtime = et-st
    
    for i in range(len(binmatchsolss)):
        binmatchsol = binmatchsolss[i]
        H12corr = binmatchsol.H
        X1v_roadrem_corr = H12corr[0:2,0:2].dot(X1v_roadrem.T).T + H12corr[0:2,2]
        Xerr = np.power(nplinalg.norm(X1v_roadrem_corr[:,0:2]-X1gv_roadrem[:,0:2],axis=1),2)
        rmse.append( [np.sqrt(np.mean(Xerr)),dtime,k] )
        
        
    # plot2D(k,Xmap2Dflat,X1v_roadrem,X1gv_roadrem,binmatchsolss,saveplot=False)
    cnt+=1

np.savez('rmse200.npz', rmse=rmse)


#%%
plt.close('all')
data = np.load('rmse200.npz')
rmse=data['rmse']

fig44, ax44= plt.subplots(figsize=(20, 9))
rmse=np.array(rmse)

# ax44.boxplot(MTS[methodstrings],positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
ax44.hist(rmse[:,0],bins=50,rwidth=0.9)
# ax44.set_xticklabels(methodstrings)
ax44.set_xlabel("Scan-matching RMSE",fontsize=30)
# ax44.set_ylabel("count",fontsize=20)
ax44.set_ylabel('# runs',fontsize=30)
ax44.tick_params(axis='x', labelsize=30 )
ax44.tick_params(axis='y', labelsize=30 )
ax44.grid(True,which="both",axis="y")


fig5, ax5= plt.subplots(figsize=(20, 9))
rmse=np.array(rmse)
RR=rmse[:,0]
# ax5.boxplot(MTS[methodstrings],positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
ax5.hist(RR[RR<10],bins=50,rwidth=0.9)
# ax5.set_xticklabels(methodstrings)
ax5.set_xlabel("Scan-matching RMSE",fontsize=30)
# ax5.set_ylabel("count",fontsize=20)
ax5.set_ylabel('# runs',fontsize=30)
ax5.tick_params(axis='x', labelsize=30 )
ax5.tick_params(axis='y', labelsize=30 )
ax5.grid(True,which="both",axis="y")


fig6, ax6= plt.subplots(figsize=(20, 9))

# ax6.boxplot(MTS[methodstrings],positions = xpos1,boxprops={'linewidth':2},whiskerprops={'linewidth':2})
ax6.hist(rmse[:,1],bins=50,rwidth=0.9)
# ax6.set_xticklabels(methodstrings)
ax6.set_xlabel("time taken",fontsize=20)
# ax6.set_ylabel("count",fontsize=20)
# ax6.set_xlabel('Methods')
ax6.tick_params(axis='x', labelsize=20 )
ax6.tick_params(axis='y', labelsize=20 )
ax6.grid(True,which="both",axis="y")