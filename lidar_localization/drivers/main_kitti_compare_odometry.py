# -*- coding: utf-8 -*-

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
from sklearn.neighbors import KDTree
import os
import pandas as pd
time_increment = 1.736111516947858e-05
angle_increment = 0.004363323096185923
scan_time = 0.02500000037252903
range_min, range_max = 0.023000000044703484, 60.0
angle_min,angle_max =  -2.3518311977386475,2.3518311977386475
from numba import vectorize, float64,guvectorize,int64,double,int32,int64,float32,uintc,boolean
from numba import njit, prange,jit
import scipy.linalg as sclalg
import scipy.optimize as scopt
from pykitticustom import odometry

# dtype = np.float32
from lidarprocessing import icp
import open3d as o3d

import importlib
import json

from pyslam import  slam
importlib.reload(slam)
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
f3 = plt.figure()    
ax = f3.add_subplot(111)
for i in range(len(dataset)):
    Xtpath[i,:] = dataset.poses[i].dot(np.array([0,0,0,1]))
ax.plot(Xtpath[:,0],Xtpath[:,2],'k')
plt.show()

pose = dataset.poses[1]
velo = dataset.get_velo(2)


#%%

import faiss
import numpy as np


class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None

    def fit(self, X, y):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init)
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]
    
#%%
params={}

params['REL_POS_THRESH']=0.5# meters after which a keyframe is made
params['REL_ANGLE_THRESH']=30*np.pi/180
params['ERR_THRES']=15
params['n_components']=200
params['reg_covar']=0.2

#%%

def WassersteinDist(m1,P1,m2,P2):
    s1 = sclalg.sqrtm(P1)
    g = np.matmul(np.matmul(s1,P2),s1) 
    d = np.sqrt( nplinalg.norm(m1-m2)+np.trace(  P1+P2-2*sclalg.sqrtm( g  ) ) )
    
    return d


@njit(parallel=True)
def distGmm(MU1,P1,W1,MU2,P2,W2):
    n1 = MU1.shape[0]
    n2 = MU2.shape[0]
    invPP1=np.zeros_like(P1)
    for i in range(n1):
        invPP1[i] = nplinalg.inv(P1[i])
        
    invPP2=np.zeros_like(P2)
    for i in range(n2):
        invPP2[i] = nplinalg.inv(P2[i])
        
        
    D=np.zeros((n1,n2))
    for i in prange(n1):
        for j in range(n2):
            D[i,j] = pt3dproc.BCdist_gassian(MU1[i],P1[i],MU2[j],P2[j])
    
    f=0
    for i in prange(n1): 
        d=np.sort(D[i])
        f+=np.mean(d[d<100])   
        # for j in range(n2):
        #     if D[i,j]>d:
        #         D[i,j]=0
        
    # D=0
    # for i in prange(n1):
    #     for j in prange(n2):
    #         D+= W1[i]*W[j]*(1-BCoeff_gassian(MU1[i],invPP1[i],P1[i],MU2[j],invPP2[j],P2[j]))
    #         D+= W1[i]*W[j]*(1-BCoeff_gassian(MU1[i],invPP1[i],P1[i],MU2[j],invPP2[j],P2[j]))
            
    # O=np.ones(n1)
    # I=np.identity(n1)
    # A=np.zeros((n1+n2,n1*n2))
    # for i in range(n2):
    #     A[i,i*n1:(i*n1+n1)]=O
    #     A[n2:,i*n1:(i+1)*n1]=I
    
    # B=np.hstack([W2,W1])
        
    # res = scopt.linprog(D.reshape(-1), A_ub=None, b_ub=None, A_eq=A, b_eq=B,bounds=(0,1))
    
    return f

# @njit([numba.types.Tuple((float32,float32[:]))(float32[:], float32[:,:], float32[:,:], float32[:,:,:],float32[:]),
#       numba.types.Tuple((float32,float32[:]))(float64[:], float32[:,:], float32[:,:], float32[:,:,:],float32[:])],
#       nopython=True, fastmath=True,nogil=True,parallel=True,cache=False) 
def getcostgradient3Dypr_gmms(x,MU1,P1,W1,MU2,P2,W2):
    x=x.astype(dtype)
    ncomp=MU1.shape[0]
    t=np.zeros(3,dtype=dtype)
    t[0]=x[0]
    t[1]=x[1]
    t[2]=x[2]
    phi=x[3]
    xi=x[4]
    zi=x[5]


    
    Rzphi,dRzdphi=pt3dproc.Rz(phi)
    Ryxi,dRydxi=pt3dproc.Ry(xi)
    Rxzi,dRxdzi=pt3dproc.Rx(zi)

    R = Rzphi.dot(Ryxi)
    R=R.dot(Rxzi)
    
   
 

    G=dRzdphi.dot(Ryxi)
    dRdphi=G.dot(Rxzi)
    
    G=Rzphi.dot(dRydxi)
    dRdxi=G.dot(Rxzi)
    
    G=Rzphi.dot(Ryxi)
    dRdzi=G.dot(dRxdzi)
    

    MU2trans=R.dot(MU1.T).T+t

    # invPP1=np.zeros_like(P1)
    # invPP2=np.zeros_like(P2)
    PP2trans=np.zeros_like(P2)
    MU2trans=np.zeros_like(MU2)
    # denom1 = np.zeros(ncomp,dtype=dtype)
    # denom2 = np.zeros(ncomp,dtype=dtype)
    for i in range(ncomp):
        # invPP1[i] = nplinalg.inv(P1[i])
        # denom1[i] = W1[i]*1/np.sqrt(nplinalg.det(2*np.pi*P1[i]))    
        
        # invPP2[i] = nplinalg.inv(P2[i])
        # denom2[i] = W2[i]*1/np.sqrt(nplinalg.det(2*np.pi*P2[i]))    
        PP2trans[i] = np.dot(R.dot(P2[i]),R.T)


    
    return distGmm(MU1,P1,W1,MU2trans,PP2trans,W2)

#%%
from sklearn import mixture
plt.close("all")
folder='lidarprocessing/'

i=1
f1="%06d.bin"%i
X1 = np.fromfile(folder+'/'+f1, dtype=np.float32)
X1=X1.reshape((-1, 4))
X1=X1.astype(dtype)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(X1[:,:3])
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.1)
X11=np.asarray(voxel_down_pcd.points)
X11=np.ascontiguousarray(X11,dtype=dtype)

X22=X11+1


voxel_down_pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
Xn=np.array(voxel_down_pcd.normals)

X11n=np.hstack([X11,np.array(voxel_down_pcd.normals)])

X11=X11[:10000,:]
X11n=X11n[:10000,:]
X22=X22[:10000,:]

Ncmp=100
# from scipy.cluster.vq import kmeans2
# centroid, label = kmeans2(X11.T, Ncmp, minit='points')

fkm=FaissKMeans(n_clusters=Ncmp, n_init=10, max_iter=50)
fkm.fit(np.ascontiguousarray(X11n,dtype=dtype),None)
Xkmidx=fkm.predict(X11n)
Xkmidx=Xkmidx.reshape(-1)

MU=np.zeros((Ncmp,3),dtype=dtype)
invP=np.zeros((Ncmp,3,3),dtype=dtype)
P=np.zeros((Ncmp,3,3),dtype=dtype)
W=np.ones(Ncmp)/Ncmp
for i in range(Ncmp):
    X=X11[Xkmidx==i,:]
    MU[i]=np.mean(X,axis=0)
    P[i]=np.cov(X.T)
    invP[i]=nplinalg.inv(P[i])

W=W/sum(W)

# MU,P,W=pt3dproc.gmmEM(X11,MU.copy(),P.copy(),W.copy(),250,0.1)

fig = plt.figure("x-y kmeans")
ax = fig.add_subplot(111)
ax.plot(X11[:,0],X11[:,1],'b.')
for i in range(len(W)):
    Xgmm=utpltgmshp.getCovEllipsePoints2D(MU[i,0:2],P[i,0:2,0:2],nsig=1,N=100)
    ax.plot(Xgmm[:,0],Xgmm[:,1],'r')

     
clf = mixture.GaussianMixture(n_components=Ncmp,
                                      covariance_type='full',reg_covar=0.1,weights_init=W,means_init=MU,precisions_init=invP,
                                        warm_start=True,max_iter=10)
        
clf.fit(X11)

MU=np.ascontiguousarray(clf.means_,dtype=dtype)
P=np.ascontiguousarray(clf.covariances_,dtype=dtype)
W=np.ascontiguousarray(clf.weights_,dtype=dtype)

fig = plt.figure("x-y after EM")
ax = fig.add_subplot(111)
ax.plot(X11[:,0],X11[:,1],'b.')
for i in range(len(W)):
    Xgmm=utpltgmshp.getCovEllipsePoints2D(MU[i,0:2],P[i,0:2,0:2],nsig=1,N=100)
    ax.plot(Xgmm[:,0],Xgmm[:,1],'r')
    
    

clf = mixture.GaussianMixture(n_components=Ncmp,
                                      covariance_type='full',reg_covar=0.1,
                                        warm_start=False)
        
clf.fit(X22)

MU2=np.ascontiguousarray(clf.means_,dtype=dtype)
P2=np.ascontiguousarray(clf.covariances_,dtype=dtype)
W2=np.ascontiguousarray(clf.weights_,dtype=dtype)




plotit=False
if plotit:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X11[:,0],X11[:,1],X11[:,2],'b.')
    
    fig = plt.figure("x-y")
    ax = fig.add_subplot(111)
    ax.plot(X11[:,0],X11[:,1],'b.')
    for i in range(len(W)):
        Xgmm=utpltgmshp.getCovEllipsePoints2D(MU[i,0:2],P[i,0:2,0:2],nsig=1,N=100)
        ax.plot(Xgmm[:,0],Xgmm[:,1],'r')
    
    fig = plt.figure("y-z")
    ax = fig.add_subplot(111)
    ax.plot(X11[:,1],X11[:,2],'b.')
    for i in range(len(W)):
        Xgmm=utpltgmshp.getCovEllipsePoints2D(MU[i,1:3],P[i,1:3,1:3],nsig=1,N=100)
        ax.plot(Xgmm[:,0],Xgmm[:,1],'r')
    
    fig = plt.figure("x-z")
    ax = fig.add_subplot(111)
    ax.plot(X11[:,0],X11[:,2],'b.')
    for i in range(len(W)):
        Xgmm=utpltgmshp.getCovEllipsePoints2D(MU[i,[0,2]],P[i,0:3:2,0:3:2],nsig=1,N=100)
        ax.plot(Xgmm[:,0],Xgmm[:,1],'r')
    

x=np.array([-2,-2,-2,np.pi/8,0,0],dtype=dtype)
st=time.time()
f,g=pt3dproc.getcostgradient3Dypr(x,X22.T,MU,P,W)
print("f=",f)
et=time.time()
print("grad :",et-st)

# def gg(x,Xt,MU,P,W):
#     x=np.hstack([x,np.zeros(3)])
#     x=x.astype(dtype)
#     f,g=pt3dproc.getcostgradient3Dypr(x,Xt,MU,P,W)
#     return f

# res = minimize(gg, x[0:3],args=(X22,MU,P,W),jac= '3-point',tol=1e-3,method='BFGS',options={'disp':True,'gtol':1e-3}) # 'Nelder-Mead'
# print(res)
# SLSQP
st=time.time()
res1 = minimize(pt3dproc.getcostgradient3Dypr, x,args=(X22.T,MU,P,W),jac= True,tol=1e-3,method='BFGS',options={'disp':True,'maxiter':10}) # 'Nelder-Mead'
et=time.time()
print("time :",et-st)
print(res1.x) 

st=time.time()
res2 = minimize(pt3dproc.getcostgradient3Dypr_v2, x,args=(X22,MU,P,W),jac= True,tol=1e-1,method='BFGS',options={'disp':True,'maxiter':15}) # 'Nelder-Mead'
et=time.time()
print("time :",et-st)    
print(res2.x) 

st=time.time()
res3 = minimize(getcostgradient3Dypr_gmms, x,args=(MU,P,W,MU2,P2,W2),jac= None,tol=1e-3,method='Nelder-Mead',options={'disp':True}) # 'Nelder-Mead'
et=time.time()
print("time :",et-st)    



#%%
plt.close("all")
D={"icp":{},
   "gicp":{},
   "gicp_cost":{},
   "ndt":{}}

D["icp"]["enable"]=1
D["icp"]["setMaximumIterations"]=500
D["icp"]["setMaxCorrespondenceDistance"]=10
D["icp"]["setRANSACIterations"]=0.0
D["icp"]["setRANSACOutlierRejectionThreshold"]=1.5
D["icp"]["setTransformationEpsilon"]=1e-9
D["icp"]["setEuclideanFitnessEpsilon"]=0.01


D["gicp_cost"]["enable"]=0


D["gicp"]["enable"]=1
D["gicp"]["setMaxCorrespondenceDistance"]=10
D["gicp"]["setMaximumIterations"]=30.0
D["gicp"]["setMaximumOptimizerIterations"]=30.0
D["gicp"]["setRANSACIterations"]=0.0
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


# res2 = minimize(pt3dproc.getcostgradient3Dypr_v2, x,args=(X22,MU,P,W),jac= True,tol=1e-1,method='BFGS',options={'disp':True,'maxiter':15}) # 'Nelder-Mead'
# t=res2.x[:3]
# phi=res2.x[3]
# xi=res2.x[4]
# zi=res2.x[5]
# Rzphi,dRzdphi=pt3dproc.Rz(phi)
# Ryxi,dRydxi=pt3dproc.Ry(xi)
# Rxzi,dRxdzi=pt3dproc.Rx(zi)

# R = Rzphi.dot(Ryxi)
# R=R.dot(Rxzi)
# H=np.hstack([R,t.reshape(-1,1)])
# H=np.vstack([H,[0,0,0,1]])
# H=H.astype(np.float32)

Xlims=[-50,50]
Ylims=[-50,50]
Zlims=[-3,3]
def limitpcd(X):
    X=X[(X[:,0]>=Xlims[0]) & (X[:,0]<=Xlims[1])]
    X=X[(X[:,1]>=Ylims[0]) & (X[:,1]<=Ylims[1])]
    X=X[(X[:,2]>=Zlims[0]) & (X[:,2]<=Zlims[1])]
    return X

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(X11[:,0],X11[:,1],X11[:,2],'b.')

# X11=limitpcd(X11)   
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(X11[:,0],X11[:,1],X11[:,2],'b.')


Hicp=[np.identity(4)]
Hgicp=[np.identity(4)]
Hndt=[np.identity(4)]

Xicp=np.zeros((len(dataset),3))
Xgicp=np.zeros((len(dataset),3))
Xndt=np.zeros((len(dataset),3))

Hgmmtrans=[np.identity(4)]

fig = plt.figure("gg-plot")
ax = fig.add_subplot(111,projection='3d')

for pp in range(1,len(dataset)):
    print(pp)
    X1 = dataset.get_velo(pp-1)
    X2 = dataset.get_velo(pp)
    
    X1=X1[:,:3]
    X2=X2[:,:3]
    # phi=np.pi/8*np.random.randn()
    # xi=np.pi/8*np.random.randn()
    # zi=np.pi/8*np.random.randn()
    # Rzphi,dRzdphi=pt3dproc.Rz(phi)
    # Ryxi,dRydxi=pt3dproc.Ry(xi)
    # Rxzi,dRxdzi=pt3dproc.Rx(zi)

    # R = Rzphi.dot(Ryxi)
    # R=R.dot(Rxzi)
    # X2=X2+2*np.random.randn(X2.shape[0],X2.shape[1])

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(X1)
    voxel_down_pcd1 = pcd1.voxel_down_sample(voxel_size=0.1)
    
    X11=np.asarray(voxel_down_pcd1.points)
    # X11=np.ascontiguousarray(X11,dtype=dtype)
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(X2)
    voxel_down_pcd2 = pcd2.voxel_down_sample(voxel_size=0.1)
    
    X22=np.asarray(voxel_down_pcd2.points)
    # X22=np.ascontiguousarray(X22,dtype=dtype)
    
    
    

    
    
    ##
    voxel_down_pcd1.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    Xn=np.array(voxel_down_pcd1.normals)
    Xnormals = np.array(voxel_down_pcd1.normals)
    X11n=np.hstack([X11,Xnormals])

    X11=limitpcd(X11) 
    X22=limitpcd(X22) 
    X11n=limitpcd(X11n) 
    
    Ncmp=100
    # st=time.time()
    # fkm=FaissKMeans(n_clusters=Ncmp, n_init=5, max_iter=50)
    # fkm.fit(np.ascontiguousarray(X11n,dtype=dtype),None)
    # Xkmidx=fkm.predict(X11n)
    # Xkmidx=Xkmidx.reshape(-1)

    # MU=np.zeros((Ncmp,3),dtype=dtype)
    # invP=np.zeros((Ncmp,3,3),dtype=dtype)
    # P=np.zeros((Ncmp,3,3),dtype=dtype)
    # W=np.ones(Ncmp)/Ncmp
    # for i in range(Ncmp):
    #     X=X11[Xkmidx==i,:]
    #     MU[i]=np.mean(X,axis=0)
    #     P[i]=np.cov(X.T)
    #     invP[i]=nplinalg.inv(P[i])

    # W=W/sum(W)
    
    # clf = mixture.GaussianMixture(n_components=Ncmp,
    #                                       covariance_type='full',reg_covar=0.01,weights_init=W,means_init=MU,precisions_init=invP,
    #                                         warm_start=True)
    # clf = mixture.GaussianMixture(n_components=Ncmp,
    #                                       covariance_type='full',reg_covar=0.01, warm_start=True)
            
    # clf.fit(X11)

    # MU=np.ascontiguousarray(clf.means_,dtype=dtype)
    # P=np.ascontiguousarray(clf.covariances_,dtype=dtype)
    # W=np.ascontiguousarray(clf.weights_,dtype=dtype)
    # et=time.time()
    # print("time taken by EM = ", et-st)
    # st=time.time()
    
    # x=np.array([0.6,0.05,0.01,np.pi/16,0,0],dtype=dtype)
    # res2 = minimize(pt3dproc.getcostgradient3Dypr_v2, x,args=(X22,MU,P,W),jac= True,tol=1e-2,method='BFGS',options={'disp':False,'maxiter':100}) # 'Nelder-Mead'
    # t=res2.x[:3]
    # phi=res2.x[3]
    # xi=res2.x[4]
    # zi=res2.x[5]
    # Rzphi,dRzdphi=pt3dproc.Rz(phi)
    # Ryxi,dRydxi=pt3dproc.Ry(xi)
    # Rxzi,dRxdzi=pt3dproc.Rx(zi)

    # R = Rzphi.dot(Ryxi)
    # R=R.dot(Rxzi)
    # H=np.hstack([R,t.reshape(-1,1)])
    # H=np.vstack([H,[0,0,0,1]])
    # H=H.astype(np.float32)
    # Hgmmtrans.append(H)
    # et=time.time()
    # print("time taken by BFGS = ", et-st)
    
    # X11=limitpcd(X11) 
    # X22=limitpcd(X22) 
    
    Hpcl=slam.registrations(X22,X11,json.dumps(D))
    Hpcl=dict(Hpcl)
    # HH=icp.icp(X22, X11, init_pose=None, max_iterations=100, tolerance=0.001)
    # Hpcl[0]=nplinalg.inv(Hpcl[0])
    # HtransPCL.append(Hpcl)
    
    # ICP path


    Xicp=np.zeros((len(dataset),3))
    Xgicp=np.zeros((len(dataset),3))
    Xndt=np.zeros((len(dataset),3))

    
    H=Hpcl["H_icp"]
    Hicp.append(Hicp[-1].dot(H))
    
    H=Hpcl["H_gicp"]
    Hgicp.append(Hgicp[-1].dot(H))
    
    # H=Hpcl["H_ndt"]
    # Hndt.append(Hndt[-1].dot(H))
    
    # H=Hgmmtrans[i]
    # Hgmm=Hgmm.dot(H)
    
    Xicp[pp]=Hicp[-1][0:3,3]
    Xgicp[pp]=Hgicp[-1][0:3,3]
    # Xndt[pp]=Hndt[-1][0:3,3]
    # Xgmm[i]=Hgmm[0:3,3]
        
    n=len(Hicp)    
    
    ax.cla()
    
    
    # ax.plot(Xtpath[:n,2],-Xtpath[:n,0],'k',label='True')
    ax.plot(Xgicp[:n,0],Xgicp[:n,1],Xgicp[:n,2],'r',label='gicp')
    # ax.plot(Xndt[:n,0],Xndt[:n,1],Xndt[:n,2],'b',label='ndt')
    ax.plot(Xicp[:n,0],Xicp[:n,1],Xicp[:n,2],'g',label='icp')
    # ax.plot(Xgmm[:n,0],Xgmm[:n,1],'b',label='gmm')
    ax.legend()
    ax.set_title("main-all")
    plt.pause(0.1)
    plt.show()
    
    if pp>150:
        break
    
    

n=len(Hicp)


plt.close("all")   
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Xtpath[:n,2],-Xtpath[:n,0],'k',label='True')
ax.set_title("true")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Xtpath[:n,2],-Xtpath[:n,0],'k',label='True')
ax.plot(Xicp[:n,0],Xicp[:n,1],'r',label='icp')
ax.legend()
ax.set_title("icp")


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Xtpath[:n,2],-Xtpath[:n,0],'k',label='True')
ax.plot(Xgicp[:n,0],Xgicp[:n,1],'b',label='gicp')
ax.legend()
ax.set_title("gicp")



# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(Xtpath[:n,2],-Xtpath[:n,0],'k',label='True')
# ax.plot(Xndt[:n,0],Xndt[:n,1],'g',label='ndt')
# ax.legend()
# ax.set_title("ndt")

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(Xtpath[:n,2],-Xtpath[:n,0],'k',label='True')
# ax.plot(Xgmm[:n,0],Xgmm[:n,1],'b',label='gmm')
# ax.legend()
# ax.set_title("gmm")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Xtpath[:n,2],-Xtpath[:n,0],'k',label='True')
ax.plot(Xgicp[:n,0],Xgicp[:n,1],'r',label='gicp')
# ax.plot(Xgmm[:n,0],Xgmm[:n,1],'b',label='gmm')
ax.legend()
ax.set_title("main-all")
