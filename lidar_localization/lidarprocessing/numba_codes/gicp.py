# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
import numpy as np
import numpy.linalg as nplinalg
import numba
from numba import vectorize, float64,guvectorize,int64,double,int32,int64,float32,uintc,boolean
from numba import njit, prange,jit
import copy

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from sklearn.neighbors import KDTree
# from utils.plotting import geometryshapes as utpltgmshp
import time
from scipy.optimize import minimize, least_squares
from scipy import interpolate
from fastdist import fastdist
from uq.gmm import gmmfuncs as uqgmmfnc
import pdb
import networkx as nx
from uq.gmm import gmmfuncs as uqgmmfnc
from scipy.optimize import least_squares
from uq.quadratures import cubatures as uqcub
import math
from scipy.sparse import csr_matrix,lil_matrix

numba_cache=True
dtype=np.float64

@njit
def computeMeanCov(X):
    return np.mean(X),np.cov(X.T)
    
# @njit
def compute_cov(X,nearidxs):
    MU=np.zeros((X.shape[0],3),dtype=dtype)
    P=np.zeros((X.shape[0],3,3),dtype=dtype)
    
    for i in range(X.shape[0]):
        a=nearidxs[i]
        MU[i],P[i]=computeMeanCov(X[a,:])
        u,s,vh=nplinalg.svd(P[i])
        s[s<0.1]=0.1
        s[s>1]=1
        P[i]=u @ np.diag(s) @ vh
        
    return MU,P


@njit( fastmath=True)
def Rz(phi):
    R=np.zeros((3,3),dtype=dtype)
    R[0]=[np.cos(phi),-np.sin(phi),0]
    R[1]=[np.sin(phi),np.cos(phi),0]
    R[2]=[0,0,1]
    
    dR =np.zeros((3,3),dtype=dtype)
    dR[0]=[-np.sin(phi),-np.cos(phi),0]
    dR[1]=[np.cos(phi),-np.sin(phi),0]
    dR[2]=[0,0,0]
    
    return R,dR

@njit( fastmath=True)
def Ry(xi):
    R=np.zeros((3,3),dtype=dtype)
    R[0]=[np.cos(xi),0,np.sin(xi)]
    R[1]=[0,1,0]
    R[2]=[-np.sin(xi),0,np.cos(xi)]
    
    dR =np.zeros((3,3),dtype=dtype)
    dR[0]=[-np.sin(xi),0,np.cos(xi)]
    dR[1]=[0,0,0]
    dR[2]=[-np.cos(xi),0,-np.sin(xi)]
    
    
    return R,dR

@njit( fastmath=True)
def Rx(zi):
    R=np.zeros((3,3),dtype=dtype)
    R[0]=[1,0,0]
    R[1]=[0,np.cos(zi),-np.sin(zi)]
    R[2]=[0,np.sin(zi),np.cos(zi)]
    
    dR =np.zeros((3,3),dtype=dtype)
    dR[0]=[0,0,0]
    dR[1]=[0,-np.sin(zi),-np.cos(zi)]
    dR[2]=[0,np.cos(zi),-np.sin(zi)]
    
    
    return R,dR

@njit(parallel=True)
def gicp_internal_cost(X11,X22,P11,P22,t,R,dRdphi,dRdxi,dRdzi):
    
    e=np.zeros(X11.shape[0],dtype=dtype)
    g=np.zeros((X11.shape[0],6),dtype=dtype)
    for i in prange(X11.shape[0]):
        C=P11[i]+np.dot(R.T,P22[i].dot(R))
        invC=nplinalg.inv(C)
        Xd=X11[i]-R.dot(X22[i])-t
        iCXd=invC.dot(Xd)
        e[i]=np.dot(Xd,iCXd)
        
        g[i,0:3]=-2*iCXd
        dR=dRdphi.dot(X22[i])
        g[i,3]=-2*np.dot(dR,iCXd)
        dR=dRdxi.dot(X22[i])
        g[i,4]=-2*np.dot(dR,iCXd)
        dR=dRdzi.dot(X22[i])
        g[i,5]=-2*np.dot(dR,iCXd)
    
    # print(e.shape,g.shape)
    return e,g


def gicp_cost(x,tree1,X1,X2,P1,P2,dmax):
    x=x.astype(dtype)
    
    t=x[0:3]
    phi=x[3]
    xi=x[4]
    zi=x[5]

    Rzphi,dRzdphi=Rz(phi)
    Ryxi,dRydxi=Ry(xi)
    Rxzi,dRxdzi=Rx(zi)

    R = Rzphi.dot(Ryxi)
    R=R.dot(Rxzi)
    
    G=dRzdphi.dot(Ryxi)
    dRdphi=G.dot(Rxzi)
    
    G=Rzphi.dot(dRydxi)
    dRdxi=G.dot(Rxzi)
    
    G=Rzphi.dot(Ryxi)
    dRdzi=G.dot(dRxdzi)
   
    # now associate
    X22=R.dot(X2.T).T-t
    D21,X21idx=tree1.query(X22, k=1, return_distance=True)
    
    D21=D21.reshape(-1)
    X21idx=X21idx.reshape(-1)
    
    idx=np.argwhere(D21<dmax)
    idx=idx.reshape(-1)
    X21idx=X21idx[idx]

    X11=X1[X21idx]
    X22=X2[idx]
    
    # print(X11-X22)
    # MU11=MU1[X21idx]
    P11=P1[X21idx]
    P22=P2[idx]
    # Xd=X1-R.dot(X2.T).T+t
    # print(X11.shape,X22.shape)
    e,g=gicp_internal_cost(X11,X22,P11,P22,t,R,dRdphi,dRdxi,dRdzi)
    
    return np.mean(e),np.mean(g,axis=0)
    
def gicp_init(X1,X2,dmax=1):

    tree1 = KDTree(X1, leaf_size=15)
    Xnear1 = tree1.query_radius(X1, dmax) 
    cnt=np.array([len(x) for x in Xnear1])
    
    X1=X1[cnt>5,:]
    tree1 = KDTree(X1, leaf_size=15)
    Xnear1 = tree1.query_radius(X1, dmax)
    
    MU1,P1=compute_cov(X1,Xnear1)
    
    tree2 = KDTree(X2, leaf_size=15)
    Xnear2 = tree2.query_radius(X2, dmax) 
    cnt=np.array([len(x) for x in Xnear2])
    
    X2=X2[cnt>5,:]
    tree2 = KDTree(X2, leaf_size=15)
    Xnear2 = tree2.query_radius(X2, dmax) 
    
    print(X2.shape,len(Xnear2))
    MU2,P2=compute_cov(X2,Xnear2)
    
    return X1,P1,X2,P2
    

    
def icp_cost(x,X1,X2,dmax):
    
    tree1 = KDTree(X1, leaf_size=15)
    D21,X21idx=tree1.query(X2, k=1, return_distance=True)
    
    D21=D21.reshape(-1)
    X21idx=X21idx.reshape(-1)
    
    idx=np.argwhere(D21<dmax)
    idx=idx.reshape(-1)
    X21idx=X21idx[idx]

    X11=X1[X21idx]

    # Xd=X11-R.dot(X2.T).T+t
    
    