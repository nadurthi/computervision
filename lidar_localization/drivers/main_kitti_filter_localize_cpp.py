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
from threading import Thread
import os
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

from scipy.interpolate import UnivariateSpline

import quaternion
import pickle
from joblib import dump, load

from pyslam import  slam
from pyslam import kittilocal

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

try:
    runfilename = __file__
except:
    runfilename = "/home/na0043/Insync/n.adurthi@gmail.com/Google Drive/repos/SLAM/main_kitti_filter_localize_cpp.py"
    
metalog="""
Journal paper KITTI localization using Particle filter
Author: Venkat
Date: Feb 22 2022

"""
dt=dataset.times[1]-dataset.times[0]
simmanger = uqsimmanager.SimManager(t0=dataset.times[0],tf=dataset.times[-1],dt=dt,dtplot=dt/10,
                                  simname="KITTI-localization-%s"%(sequence),savepath="simulations",
                                  workdir=os.getcwd())

simmanger.initialize()



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

tvec,qvec_sp,w,qdot,alpha=quat2omega_scipyspline(dataset.times,Xtpath[:,6:],k=3,s=0.0001)


    
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

D={}


D["Localize"]={}
D["Localize"]["sig0"]=0.25
D["Localize"]["dmax"]=10
D["Localize"]["likelihoodsearchmethod"]="octree" # lookup
D["Localize"]["octree"]={"resolution":0.1,
                         "searchmethod":"ApprxNN"}
D["Localize"]["lookup"]={"resolution":[0.25,0.25,0.25]}



#
D["MapManager"]={"map":{},"map2D":{}}
D["MapManager"]["map"]["downsample"]={"enable":True,"resolution":[0.2,0.2,0.2]}
D["MapManager"]["map2D"]["downsample"]={"enable":True,"resolution":[0.1,0.1,0.1]}
D["MapManager"]["map2D"]["removeRoad"]=False

#
D["MeasMenaager"]={"meas":{},"meas2D":{},"meas_Likelihood":{}}
D["MeasMenaager"]["meas"]["downsample"]={"enable":True,"resolution":[0.2,0.2,0.2]}
D["MeasMenaager"]["meas_Likelihood"]["downsample"]={"enable":True,"resolution":[0.3,0.3,0.3]}
D["MeasMenaager"]["meas2D"]["removeRoad"]=False
D["MeasMenaager"]["meas2D"]["downsample"]={"enable":True,"resolution":[0.1,0.1,0.1]}


#
D["BinMatch"]={}
D['BinMatch']["dxMatch"]=list(np.array([1,1],dtype=np.float64))
D['BinMatch']["dxBase"]=list(np.array([30,30],dtype=np.float64))
D['BinMatch']["Lmax"]=list(np.array([200,200],dtype=np.float64))
D['BinMatch']["thmax"]=170*np.pi/180
D['BinMatch']["thfineres"]=2.5*np.pi/180

#
D["mapfit"]={"downsample":{},"gicp":{}}
D["mapfit"]["downsample"]={"resolution":[0.5,0.5,0.5]}
D["mapfit"]["gicp"]["setMaxCorrespondenceDistance"]=10
D["mapfit"]["gicp"]["setMaximumIterations"]=5.0
D["mapfit"]["gicp"]["setMaximumOptimizerIterations"]=5.0
D["mapfit"]["gicp"]["setRANSACIterations"]=0
D["mapfit"]["gicp"]["setRANSACOutlierRejectionThreshold"]=1.5
D["mapfit"]["gicp"]["setTransformationEpsilon"]=1e-6
D["mapfit"]["gicp"]["setUseReciprocalCorrespondences"]=1

#
D["seqfit"]={"downsample":{},"gicp":{}}
D["seqfit"]["gicp"]["setMaxCorrespondenceDistance"]=10
D["seqfit"]["gicp"]["setMaximumIterations"]=30.0
D["seqfit"]["gicp"]["setMaximumOptimizerIterations"]=30.0
D["seqfit"]["gicp"]["setRANSACIterations"]=0
D["seqfit"]["gicp"]["setRANSACOutlierRejectionThreshold"]=1.5
D["seqfit"]["gicp"]["setTransformationEpsilon"]=1e-9
D["seqfit"]["gicp"]["setUseReciprocalCorrespondences"]=1

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
pcd2ddown = pcd2ddown.voxel_down_sample(voxel_size=1)

Xmap2Dflat=np.asarray(pcd2ddown.points)

#%%

KL=kittilocal.MapLocalizer(json.dumps(D))
KL.addMap(Xmap)
KL.addMap2D(Xmap2D[:,:2])
KL.setLookUpDist("kitti-pcd-lookupdist-seq-%s.bin"%sequence)

minx,miny,minz,maxx,maxy,maxz=KL.MapPcllimits()







plt.close("all")


Npf=1000


Xlimits_scan=[minx,maxx]
Ylimits_scan=[miny,maxy]
Zlimits_scan=[minz,maxz]


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
    
    Q=np.diag([(0.5)**2,(0.5)**2,(0.1)**2, # x-y-z
                (5*np.pi/180)**2,(2*np.pi/180)**2,(0.1*np.pi/180)**2, # angles
                (0.1)**2,   # velocity
                (0.05*np.pi/180)**2,(0.05*np.pi/180)**2,(0.02*np.pi/180)**2,# angle rates
                (0.001)**2]) #acc
    
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
    
sampligfunc = getCTinitialSamples
# sampligfunc= getCTinitialSamples_origin

xstatepf, wpf, Q =sampligfunc(Npf)
# xstatepf, wpf, Q =getUMinitialSamples(Npf)

dim=xstatepf.shape[1]
robotpf=pf.Particles(X=xstatepf,wts=wpf)







class plot3dPF:
    def __init__(self,Xtpath,Xmap2Dflat,saveplot=True):
        self.figpf = plt.figure(figsize=(20, 20))    
        self.axpf = self.figpf.add_subplot(111,projection='3d')
        self.Xtpath=Xtpath.copy()
        self.time_taken=[]
        
        self.figpf2D = plt.figure(figsize=(20, 20))    
        self.axpf2D = self.figpf2D.add_subplot(111)
        
        # self.figpf2Dzoom = plt.figure(figsize=(20, 20))    
        # self.axpf2Dzoom = self.figpf2Dzoom.add_subplot(111)
        
        self.Xmap2Dflat=Xmap2Dflat
        self.saveplot=saveplot
        
        
        self.xtpath2Dhandle=None
        self.pf2Dhandle=None
        self.ellipse2Dhandle=None
        self.arrow2Dhandle=[]
        self.binmatch2Dhandle=None
        
        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window()

    def plotOpen3D(self,k,X,X1gv):
        pcdPF = o3d.geometry.PointCloud()
        pcdPF.points = o3d.utility.Vector3dVector(X[:,:3])
        pcdPF.paint_uniform_color([1,0,0]) #black
        

        # axpf.plot(G[:,0],G[:,1],G[:,2],'b')
        # axpf.plot(X1gv_pose[:,0],X1gv_pose[:,1],X1gv_pose[:,2],'r.')
        
        # pcdX1gv_pose = o3d.geometry.PointCloud()
        # pcdX1gv_pose.points = o3d.utility.Vector3dVector(X1gv_pose)
        # pcdX1gv_pose.paint_uniform_color([0,1,0]) #green
        
        bbox3d=o3d.geometry.AxisAlignedBoundingBox(min_bound=np.min(X1gv,axis=0)-15,max_bound=np.max(X1gv,axis=0)+15)
        pcdbox=pcd3DdownSensorCost.crop(bbox3d)
        pcdbox.paint_uniform_color([211/255,211/255,211/255])
        
        ArrXpf=[]
        for i in range(X.shape[0]):
            
            
            R,t = pose2Rt(X[i][:6])
            
            drn = R.dot([1,0,0])
            t2=t+5*drn
            ArrXpf.append(t2)
            
            
        # plot PF arrows as red and pose pf point as green
        ArrXpf=np.array(ArrXpf)
        pfline_set = o3d.geometry.LineSet()
        pfline_set.points = o3d.utility.Vector3dVector( np.vstack([X[:,:3],ArrXpf]))
        pflines = [[i,i+Npf] for i in range(Npf)]
        pfline_set.lines = o3d.utility.Vector2iVector(pflines)
        colors = [[1, 0, 0] for i in range(len(pflines))] # red
        pfline_set.colors = o3d.utility.Vector3dVector(colors)
        # np.asarray(pfline_set.colors)[0]=[0,1,0] #green
        
        
        # plot path as black
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(self.Xtpath[:k,:3])
        lines = [[i,i+1] for i in range(k-1)]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        colors = [[0, 0, 0] for i in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        # pcdbox.paint_uniform_color([0,0,0])
        o3d.visualization.draw_geometries([pcdbox,pcdPF,pfline_set]) #line_set
        # self.vis.update_geometry([pcdbox,pcdPF,pfline_set])
        # self.vis.update_renderer()
        
    def plot3D(self,k,X,m,P):
        st=time.time()
        self.axpf.cla()
        self.axpf.plot(self.Xtpath[:k,0],self.Xtpath[:k,1],self.Xtpath[:k,2],'b')
        self.axpf.plot(X[:,0],X[:,1],X[:,2],'r.')

        for i in range(X.shape[0]):
            # R,t = pose2Rt(robotpf.X[i][:6])
            HH=kittilocal.pose2Hmat(X[i][:6])
            R = HH[0:3,0:3]
            t = HH[0:3,3]
            
            drn = R.dot([1,0,0])
            t2=t+5*drn
            G=np.vstack([t,t2])
            self.axpf.plot(G[:,0],G[:,1],G[:,2],'r')
            

            
        self.axpf.set_title("time step = %d"%k)
        self.axpf.set_xlabel('x')
        self.axpf.set_ylabel('y')
        self.axpf.set_zlabel('z')

        
        et=time.time()
        
        self.time_taken.append(et-st)
        
        plt.show(block=False)
        plt.pause(0.1)
        self.figpf.show()
        simmanger.savefigure(self.figpf, ['3Dmap','snapshot'], 'snapshot_'+str(int(k))+'.png',data=[k,X,m,P])

        
    def plot2D(self,k,X,m,P,X1vroadrem,HHs):


        self.axpf2D.cla()
        self.axpf2D.plot(self.Xmap2Dflat[:,0],self.Xmap2Dflat[:,1],'k.')
        # if self.xtpath2Dhandle is not None:
        #     K=self.xtpath2Dhandle.pop(0)
        #     K.remove()
            
        #     K=self.pf2Dhandle.pop(0)
        #     K.remove()
            
        #     K=self.ellipse2Dhandle.pop(0)
        #     K.remove()
            
        #     while(len(self.arrow2Dhandle)>0):
        #         K=self.arrow2Dhandle.pop(0)
        #         for ss in K:
        #             K.remove(ss)
            
        #     K=self.binmatch2Dhandle.pop(0)
        #     K.remove()
        
        
        self.xtpath2Dhandle=self.axpf2D.plot(self.Xtpath[:k,0],self.Xtpath[:k,1],'b')
        self.pf2Dhandle=self.axpf2D.plot(X[:,0],X[:,1],'r.')
        
        XX=utpltgmshp.getCovEllipsePoints2D(m[0:2],P[0:2,0:2],nsig=1,N=100)
        self.ellipse2Dhandle=self.axpf2D.plot(XX[:,0],XX[:,1],'r--')
        
        for i in range(X.shape[0]):
            # R,t = pose2Rt(robotpf.X[i][:6])
            HH=kittilocal.pose2Hmat(X[i][:6])
            R = HH[0:3,0:3]
            t = HH[0:3,3]
            
            drn = R.dot([1,0,0])
            t2=t+5*drn
            G=np.vstack([t,t2])
 
            
            self.arrow2Dhandle.append( self.axpf2D.plot(G[:,0],G[:,1],'r'))
            
        if HHs is not None:
            idxpf,Hpose,gHkcorr=HHs
            
            pcd2ddown = o3d.geometry.PointCloud()
            X1vroadrem2Dflatcorr=gHkcorr[0:3,0:3].dot(X1vroadrem.T).T+gHkcorr[0:3,3]
            X1vroadrem2Dflatcorr[:,2]=0
            pcd2ddown.points = o3d.utility.Vector3dVector(X1vroadrem2Dflatcorr[:,:3])
            pcd2ddown = pcd2ddown.voxel_down_sample(voxel_size=2)
            
            X1vroadrem2Dflatcorr = np.asarray(pcd2ddown.points)
            X1vroadrem2Dflatcorr=X1vroadrem2Dflatcorr[:,:2]
            self.binmatch2Dhandle=self.axpf2D.plot(X1vroadrem2Dflatcorr[:,0],X1vroadrem2Dflatcorr[:,1],'g.')
            
        self.axpf2D.set_title("time step = %d"%k)
        self.axpf2D.set_xlabel('x')
        self.axpf2D.set_ylabel('y')
        self.axpf2D.axis('equal')

        plt.show(block=False)
        plt.pause(0.1)
        if self.saveplot:
            
            self.figpf2D.show()
            simmanger.savefigure(self.figpf2D, ['2Dmap','snapshot'], 'snapshot_'+str(int(k))+'.png',data=[k,X,m,P,HHs])
            
            xlim =[Xtpath[k,0]+np.min(X1vroadrem[:,0])-25,Xtpath[k,0]+np.max(X1vroadrem[:,0])+25]
            ylim =[Xtpath[k,1]+np.min(X1vroadrem[:,1])-25,Xtpath[k,1]+np.max(X1vroadrem[:,1])+25]
            
            xlim =[Xtpath[k,0]-75,Xtpath[k,0]+75]
            ylim =[Xtpath[k,1]-75,Xtpath[k,1]+75]
            
            
            self.axpf2D.set_xlim(xlim)
            self.axpf2D.set_ylim(ylim)
            # self.axpf2D.axis('equal')
            plt.show(block=False)
            plt.pause(0.1)
            self.figpf2D.show()
            simmanger.savefigure(self.figpf2D, ['2Dmap','snapshot_closeup'], 'snapshot_'+str(int(k))+'.png',data=[k,X,m,P,HHs])
    


kittisimplotter = plot3dPF(Xtpath,Xmap2Dflat,saveplot=False)

doneLocalize=0

dt,tk,X1v,X1gv,X1v_roadrem,X1gv_roadrem=getmeas(0)
KL.addMeas(X1v,X1v_roadrem,tk)


m,P=robotpf.getEst()
XPFmP_history=[(m,P)]
XPFhistory=[(robotpf.X.copy(),robotpf.wts.copy())]
simmanger.data['resampled?']=[]
simmanger.data['BinMatch_idxpf']=[]
simmanger.data['BinMatchedH']={}

for k in range(1,len(dataset)):
    print(k)
    st=time.time()
    # if outQ.empty():
    #     break
    # dt,tk,X1v,X1gv,X1v_roadrem,X1gv_roadrem=outQ.get()
    dt,tk,X1v,X1gv,X1v_roadrem,X1gv_roadrem=getmeas(k)
    KL.addMeas(X1v,X1v_roadrem,tk)
    KL.setRegisteredSeqH_async()
    et=time.time()
    print("get data time = ",et-st)
    
    # propagate
    st=time.time()
    # robotpf.X=dynmodelUM3D(robotpf.X,dt)
    robotpf.X=dynmodelCT3D(robotpf.X,dt)
    
    # start computation of relative states
    KL.setRelStates_async()
    
    et=time.time()
    print("dyn model time = ",et-st)
    robotpf.X=robotpf.X+2*np.random.multivariate_normal(np.zeros(dim), Q, Npf)
    
    ### BIN MATCH
    idxpf=np.random.choice(list(range(Npf)),1,p=robotpf.wts)
    idxpf=idxpf[0]
    
    # idxpf=np.argmin(robotpf.wts)
    
    if doneLocalize==0:
        gg=1
    else:
        gg=10
    simmanger.data['doBinMatch']=doBinMatch=1
    if doBinMatch==1 and k%gg==0:
        print("---doneLocalize ------ = ",doneLocalize)
        simmanger.data['BinMatch_idxpf'].append((k,idxpf))
        # Ridx,tidx = pose2Rt(robotpf.X[idxpf][:6])
        Hpose = kittilocal.pose2Hmat(robotpf.X[idxpf][:6])
        
        t0=k  #max([0,k-5])
        tf=k
        tk=k
        
        st=time.time()
        solret=KL.BMatchseq(t0,tf,tk,Hpose,True)
        et=time.time()
        print("Bin match and gicp time = ",et-st)
        
        if (solret.sols[0].cost>solret.sols[0].cost0):
            # idxpfmin = np.argmin(robotpf.wts)
            # robotpf.X[idxpfmin]=robotpf.X[idxpf].copy()
            robotpf.X[idxpf][:6] = kittilocal.Hmat2pose(solret.gHkcorr)
            Velmeas=KL.getvelocities()
            AngVelmeas=KL.getangularvelocities()
            robotpf.X[idxpf][6]=nplinalg.norm(Velmeas[-1])
            robotpf.X[idxpf][7:10]=AngVelmeas[-1]
            # likelihoods=KL.getLikelihoods_lookup(np.array([robotpf.X[idxpf]]),k)
            
            # likelihoods_exp=np.exp(-likelihoods)
            # likelihoods_exp[likelihoods_exp<1e-70]=1e-70
            
            # lostflg = np.all(likelihoods_exp==1e-70)
            
            # robotpf.wts[idxpf]=likelihoods_exp[0]*robotpf.wts[idxpf]
            # robotpf.renormlizeWts()

            simmanger.data['BinMatchedH'][k]=(idxpf,Hpose,solret.gHkcorr)    
            
        # if doneLocalize==0:
            # robotpf.resetWts()
           
            

    # measurement update
    
    st=time.time()
    # likelihoods_octree=KL.getLikelihoods_octree(robotpf.X,k)
    # likelihoods=KL.getLikelihoods_octree(robotpf.X,k)
    likelihoods=KL.getLikelihoods_lookup(robotpf.X,k)
    
    likelihoods_exp=np.exp(-likelihoods)
    likelihoods_exp[likelihoods_exp<1e-70]=1e-70
    
    lostflg = np.all(likelihoods_exp==1e-70)
    
    robotpf.wts=likelihoods_exp*robotpf.wts
    et=time.time()
    print("pcl meas model time = ",et-st)
    
    robotpf.renormlizeWts()
    
    # lost target
    if np.all(np.isnan(robotpf.wts)):
        doneLocalize=0
        xstatepf, wpf, Q =sampligfunc(Npf)
        robotpf.X=xstatepf
        robotpf.wts=wpf


    
    
    # 
            
    ### boostratp  resample 
    simmanger.data['resampled?'].append((k,False))
    Neff=robotpf.getNeff()
    simmanger.data['Neff/Npf']=0.7
    if doneLocalize==0:
        simmanger.data['fracH']=fracH=0.75
    else:
        simmanger.data['fracH']=fracH=1
        
    if Neff/Npf<simmanger.data['Neff/Npf']:
        print("resampled at k = ",k)
        robotpf.bootstrapResample(fracH=1)
        simmanger.data['resampled?'][-1]=(k,True)
    
    m,P=robotpf.getEst()
    u,v=nplinalg.eig(P)

    pos_std=np.sqrt(u[:3])
    if max(pos_std)<10:
        doneLocalize=1
    
        
    # lost target
    if min(pos_std)>200:
        doneLocalize=0
        
        
        
    ### saving and plotting  resample 
    XPFhistory.append((robotpf.X.copy(),robotpf.wts.copy()))
    
    # robotpf.regularizedresample(kernel='gaussian',scale=1/25)
    
    XPFmP_history.append((m,P))
    
    
    if (k%1==0):
        # kittisimplotter.plot3D(k,robotpf.X,m,P)
        
        if k in simmanger.data['BinMatchedH'].keys():
            HHs=simmanger.data['BinMatchedH'][k]
        else:
            HHs=None
        kittisimplotter.plot2D(k,robotpf.X,m,P,X1v_roadrem,HHs)
        
        # kittisimplotter.plotOpen3D(k,robotpf.X,X1gv)
        

    # plt.show()
    # plt.pause(0.1)
    
    # if k>=400:
    #     break


Velmeas=KL.getvelocities()
AngVelmeas=KL.getangularvelocities()
PosRelmeas=KL.getpositions()

Velmeas=np.vstack(Velmeas)
AngVelmeas=np.vstack(AngVelmeas)
PosRelmeas=np.vstack(PosRelmeas)

nt = PosRelmeas.shape[0]
splx=UnivariateSpline(dataset.times[:nt],PosRelmeas[:,0])
splvx=splx.derivative()
splax=splvx.derivative()

sply=UnivariateSpline(dataset.times[:nt],PosRelmeas[:,1])
splvy=sply.derivative()
splay=splvy.derivative()

splz=UnivariateSpline(dataset.times[:nt],PosRelmeas[:,2])
splvz=splz.derivative()
splaz=splvz.derivative()

Velocities_meas=np.zeros((nt,3))
Velocities_meas[:,0]=splvx(dataset.times[:nt])
Velocities_meas[:,1]=splvy(dataset.times[:nt])
Velocities_meas[:,2]=splvz(dataset.times[:nt])

plt.figure()
plt.plot(dataset.times,Velocities[:,0])
plt.plot(dataset.times[:nt],Velocities_meas[:,0],'r')
plt.plot(dataset.times[:nt],Velmeas[:,0],'g')


plt.figure()
plt.plot(dataset.times,AngRates[:,0])
plt.plot(dataset.times[:nt],AngVelmeas[:,0],'r')
# plt.plot(dataset.times[:nt],AngVelmeas[:,0],'g')



vehiclestates=[Velmeas,AngVelmeas,PosRelmeas]

simmanger.finalize()

simmanger.save(metalog, mainfile=runfilename, vehiclestates=vehiclestates,
               options=D, Npf=Npf,XPFhistory=XPFhistory,
               XPFmP_history=XPFmP_history)

#%%

# gHk=KL.getSeq_gHk()
# i1Hi_seq=KL.geti1Hi_seq()

# cgHk=[np.identity(4)]
# tpos=[[0,0,0]]
# for i in range(len(i1Hi_seq)):
#     iHi1=nplinalg.inv(i1Hi_seq[i])
#     cgHk.append(cgHk[i].dot(iHi1))
#     tpos.append(cgHk[-1][0:3,3])
# tpos=np.vstack(tpos)
# #%%
# Velmeas=KL.getvelocities()
# AngVelmeas=KL.getangularvelocities()
# PosRelmeas=KL.getpositions()

# Velmeas=np.vstack(Velmeas)
# AngVelmeas=np.vstack(AngVelmeas)
# PosRelmeas=np.vstack(PosRelmeas)

# figposrel = plt.figure()    
# axposrel = figposrel.add_subplot(111,projection='3d')
# axposrel.plot(PosRelmeas[:,0],PosRelmeas[:,1],PosRelmeas[:,2],'b.')
# axposrel.set_title("Rel Position Measurement")
# axposrel.set_xlabel('x')
# axposrel.set_ylabel('y')
# axposrel.set_zlabel('z')
# plt.pause(0.1)
# # plt.show()

# #%%
# gHkss=KL.getsetSeq_gHk(1, np.identity(4))
# Xmm=KL.getalignSeqMeas_eigen(0,0,0, np.identity(4),[0.2,0.2,0.2],3)
# # pcd = o3d.geometry.PointCloud()
# # pcd.points = o3d.utility.Vector3dVector(Xmm)

# # o3d.visualization.draw_geometries([pcd])

# Ridx,tidx = pose2Rt(robotpf.X[idxpf][:6])
# Hpose = kittilocal.pose2Hmat(robotpf.X[idxpf][:6])
# xpose = kittilocal.Hmat2pose(Hpose)

# # #%% Bin match
# idxpf=1
# xx=robotpf.X[idxpf][:6].copy()
# xx[:2]=xx[:2]+15

# # Hpose=kittilocal.pose2Hmat(xx)
# # Ridx = Hpose[0:3,0:3]
# # tidx = Hpose[0:3,3]

# t0=5
# tf=5
# tk=5

# Hgt=dataset.calib.T_cam0_velo
# Hgt=np.dot(nplinalg.inv(Hgt),dataset.poses[tk].dot(Hgt))

# # DD={}
# # DD["BinMatch"]={}
# # DD['BinMatch']["dxMatch"]=list(np.array([1,1],dtype=np.float64))
# # DD['BinMatch']["dxBase"]=list(np.array([30,30],dtype=np.float64))
# # DD['BinMatch']["Lmax"]=list(np.array([200,200],dtype=np.float64))
# # DD['BinMatch']["thmax"]=170*np.pi/180
# # DD['BinMatch']["thfineres"]=2.5*np.pi/180

# st=time.time()
# solret=KL.BMatchseq(t0,tf,tk,Hpose,True)
# et=time.time()
# print("Bin match and gicp time = ",et-st)

# Xmap2Dcpp = KL.getmap2D_noroad_res_eigen([0.25,0.25,1],2)
# # Xmeasnorad=KL.getalignSeqMeas_noroad_eigen(t0,tf,tk, Hpose,[D["BinMatch"]["dxMatch"][0],D["BinMatch"]["dxMatch"][1],1],2)

# # bm=kittilocal.BinMatch(json.dumps(DD))
# # bm.computeHlevels(Xmap2Dcpp)

# # # Ridx,tidx = KL.pose2Rt(robotpf.X[idxpf][:6])
# # phiidx = xx[3]
# # tidx = xx[0:3]
# # Rzphiidx,_=gicp.Rz(phiidx)

# # H12est=np.identity(3)
# # H12est[0:2,0:2]=Rzphiidx[0:2,0:2]
# # H12est[0:2,2]=tidx[:2]
# # sol=bm.getmatch(Xmeasnorad,H12est)
# # Rt=np.identity(3)
# # tr=np.zeros(3)
# # Rt[0:2,0:2]=sol[0].H[0:2,0:2]
# # tr[0:2]=sol[0].H[0:2,2]
# # xc= Rt2pose(Rt,tr)

# # xc[2]=xx[2]
# # xc[4:]=xx[4:]
# # Rc,tc=pose2Rt(xc)

# Xmeas2Dcpp_pose = KL.getalignSeqMeas_noroad_eigen(t0,tf,tk, Hpose,[0.25,0.25,1],2)
# Xmeas2Dcpp_posecorrect = KL.getalignSeqMeas_noroad_eigen(t0,tf,tk, solret.gHkcorr,[0.25,0.25,1],2)
# Xmeas2Dcpp_posegt = KL.getalignSeqMeas_noroad_eigen(t0,tf,tk, Hgt,[0.25,0.25,1],2)
# # Xmeas2Dcpp_poseBMcorrect = KL.getalignSeqMeas_noroad_eigen(t0,tf,tk, sol[0].H,[0.25,0.25,1],2)

# lb=np.array([-88.1277 ,-54.7676, -15.1528])
# ub=np.array([90.412, 86.3891, 16.6382 ])
# Xlmp=KL.getmaplocal_eigen(lb,ub)
# res=D["mapfit"]["downsample"]["resolution"]
# Xmeastk=KL.getalignSeqMeas_eigen(t0,tf,tk, solret.gHkcorr,res,3)

# figbf = plt.figure("bin-fit")
# ax=figbf.add_subplot(111)
# ax.cla()
# ax.plot(Xmap2Dcpp[:,0],Xmap2Dcpp[:,1],'k.')
# ax.plot(Xmeas2Dcpp_pose[:,0],Xmeas2Dcpp_pose[:,1],'b.',label='pf-pose')
# ax.plot(Xmeas2Dcpp_posecorrect[:,0],Xmeas2Dcpp_posecorrect[:,1],'r.',label='pf-pose-corrected')
# # ax.plot(Xmeas2Dcpp_poseBMcorrect[:,0],Xmeas2Dcpp_poseBMcorrect[:,1],'y.',label='pf-pose-BM-corrected')
# ax.plot(Xmeas2Dcpp_posegt[:,0],Xmeas2Dcpp_posegt[:,1],'g.',label='ground truth')
# ax.legend()
# ax.axis("equal")



# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(Xlmp)
# pcd.paint_uniform_color([1,0,0]) #green

# pcdmeastk = o3d.geometry.PointCloud()
# pcdmeastk.points = o3d.utility.Vector3dVector(Xmeastk)
# pcdmeastk.paint_uniform_color([0,1,0]) #green

# o3d.visualization.draw_geometries([pcd,pcdmeastk])

# #%%

# st=time.time()
# KL.BMatchseq_async(t0,tf,tk,Hpose,True)
# et=time.time()
# print("Bin match asyc set time = ",et-st)

# solQ=KL.getBMatchseq_async()
