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
import heapq
import time
numba_cache=True
dtype=np.float64


from numba.core import types
from numba.typed import Dict
float_2Darray = types.float64[:,:]
#%% histogram bins

@njit(cache=numba_cache)
def UpsampleMax(Hup,n):
    H=np.zeros((int(np.ceil(Hup.shape[0]/2)),int(np.ceil(Hup.shape[1]/2))),dtype=np.int32)
    for j in range(H.shape[0]):
        for k in range(H.shape[1]):
            lbx=max([2*j,0])
            ubx=min([2*j+n,Hup.shape[0]-1])+1
            lby=max([2*k,0])
            uby=min([2*k+n,Hup.shape[1]-1])+1
            H[j,k] = np.max( Hup[lbx:ubx,lby:uby] )
    return H



# @jit(int32(int32[:,:], float32[:], float32[:,:], float32[:], float32[:]),nopython=True, nogil=True,cache=numba_cache) 
@njit(cache=numba_cache)
def getPointCost(H,dx,X,Oj,Tj):
    # Tj is the 2D index of displacement
    # X are the points
    # dx is 2D
    # H is the probability histogram
    
    Pn=np.floor((X+Oj)/dx)
    # j=np.floor(Oj/dx)
    # Pn=P+j
    # Idx = np.zeros(Pn.shape[0],dtype=np.int32)
    # for i in range(Pn.shape[0]):
    #     if Pn[i,0]<0 or Pn[i,0]>H.shape[0]-1 or Pn[i,1]<0 or Pn[i,1]>H.shape[1]-1 :
    #         Pn[i,0]=H.shape[0]-1
    #         Pn[i,1]=H.shape[1]-1
            
        # elif Pn[i,0]>H.shape[0]-1:
        #     Pn[i,0]=H.shape[0]-1
        # if Pn[i,1]<0:
        #     Pn[i,1]=0
        # elif Pn[i,1]>H.shape[1]-1:
        #     Pn[i,1]=H.shape[1]-1
        
        # Idx[i]= Pn[i,1]*(H.shape[0]-1)+Pn[i,0]
            


    # c=np.sum(np.take(H, Idx))
    # c=np.sum(np.take(H, np.ravel_multi_index(Pn.T, H.shape,mode='clip')))
    # idx1 = np.all(Pn>=np.zeros(2),axis=1)
    # Pn=Pn[idx1]
    # idx2 = np.all(Pn<H.shape,axis=1)
    # Pn=Pn[idx2]
    idx1=np.logical_and(Pn[:,0]>=0,Pn[:,0]<H.shape[0])
    idx2=np.logical_and(Pn[:,1]>=0,Pn[:,1]<H.shape[1])
    idx=np.logical_and(idx1,idx2 )
    c=0
    # idx=np.all(np.logical_and(Pn>=np.zeros(2) , Pn<H.shape),axis=1 )
    Pn=Pn[idx]
    # if Pn.size>0:
    #     values, counts = np.unique(Pn, axis=0,return_counts=True)
    #     c=np.sum(counts*H[values[:,0],values[:,1]])
        # c=np.sum(H[Pn[:,0],Pn[:,1]])
    for k in range(Pn.shape[0]):
        c+=H[int(Pn[k,0]),int(Pn[k,1])]
        
    return c

@njit(cache=numba_cache)
def binMatcherAdaptive2(X11,X22,H12,Lmax,thmax,dxMatch):
    # dxMax is the max resolution allowed
    # Lmax =[xmax,ymax]
    # search window is [-Lmax,Lmax] and [-thmax,thmax]
    n=histsmudge =2 # how much overlap when computing max over adjacent hist for levels
    
    
    mn=np.zeros(2)
    mx=np.zeros(2)
    mn_orig=np.zeros(2)
    mn_orig[0] = np.min(X11[:,0])
    mn_orig[1] = np.min(X11[:,1])
    
    R=H12[0:2,0:2]
    t=H12[0:2,2]
    X222 = R.dot(X22.T).T+t
    
    
    X2=X222-mn_orig
    X1=X11-mn_orig
    
    # print("mn_orig = ",mn_orig)
    
    mn[0] = np.min(X1[:,0])
    mn[1] = np.min(X1[:,1])
    mx[0] = np.max(X1[:,0])
    mx[1] = np.max(X1[:,1])
    rmax=np.max(np.sqrt(X2[:,0]**2+X2[:,1]**2))
    
    
    # print("mn,mx=",mn,mx)
    P = mx-mn
    
    dxMax=P
    # dxMax[0] = np.min([dxMax[0],Lmax[0]/2,P[0]/2])
    # dxMax[1] = np.min([dxMax[1],Lmax[1]/2,P[1]/2])
    
    nnx=np.ceil(np.log2(P[0]))
    nny=np.ceil(np.log2(P[1]))
    
    xedges=np.arange(mn[0]-dxMatch[0],mx[0]+dxMax[0],dxMatch[0])
    yedges=np.arange(mn[1]-dxMatch[0],mx[1]+dxMax[0],dxMatch[1])
    
    if len(xedges)%2==0:
        
        xedges=np.hstack((xedges,np.array([xedges[-1]+1*dxMatch[0]])))
    if len(yedges)%2==0:
        yedges=np.hstack((yedges,np.array([yedges[-1]+1*dxMatch[1]])))
        
    
    H1match=numba_histogram2D(X1, xedges,yedges)
    
    
    # H1match[H1match>0]=1
    H1match = np.sign(H1match)
    
    # first create multilevel histograms
    
    
    HLevels=[H1match]
    dxs = [dxMatch]
    XYedges=[(xedges,yedges)]
    
    flg=0
    # st=time.time()
    for i in range(1,100):
        
        dx=2*dxs[i-1]
        if np.any(dx>dxMax):
            flg=1
        
        Hup = HLevels[i-1]
        # H=pool2d(Hup, kernel_size=3, stride=2, padding=1, pool_mode='max')
        H=UpsampleMax(Hup,n)
        

        HLevels.append(H)
        dxs.append(dx)
          

        if flg==1:
            break
    HLevels=HLevels[::-1]
    dxs=dxs[::-1]
    # dxs=[dx.astype(np.float32) for dx in dxs]
    # HLevels=[np.ascontiguousarray(H).astype(np.int32) for H in HLevels]

     
    
    LmaxOrig=Lmax.copy()
    SolBoxes_init=[]
    X2=X2-Lmax
    Lmax=dxs[0]*(np.floor(Lmax/dxs[0])+1)
    for xs in np.arange(0,2*Lmax[0],dxs[0][0]):
        for ys in np.arange(0,2*Lmax[1],dxs[0][1]):
            SolBoxes_init.append( (xs,ys,dxs[0][0],dxs[0][1]) )
    
    
    
    
    mxLVL=len(HLevels)-1

    
    h=[(100000.0,0.0,0.0,0.0,0.0,0.0,0.0)]
    # #Initialize with all thetas fixed at Max resolution
    lvl=0
    dx=dxs[lvl]
    H=HLevels[lvl]

    Xth= Dict.empty(
        key_type=types.float64,
        value_type=float_2Darray,
    )
    
    thfineRes = 2*np.pi/180
    thL=np.arange(-thmax,thmax+thfineRes,thfineRes)
    # np.random.shuffle(thL)
    for th in thL:
        # th=thL[i]
        R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        XX=np.transpose(R.dot(X2.T))
        Xth[th]=XX
        
        
        for solbox in SolBoxes_init:            
            xs,ys,d0,d1 = solbox
            Tj=np.array((d0,d1))
            Oj = np.array((xs,ys))
            cost2=getPointCost(H,dx,Xth[th],Oj,Tj)
            # heapq.heappush(h,(-cost2-np.random.rand()/1000,xs,ys,d0,d1,lvl,th))
            h.append((-cost2-np.random.rand()/1000,xs,ys,d0,d1,lvl,th))
            
    heapq.heapify(h)

    # (cost,xs,ys,d0,d1,lvl,th)=heapq.heappop(h)
    mainSolbox=()
    while(1):
        
        # if len(h)==0:
        #     break
        
        (cost,xs,ys,d0,d1,lvl,th)=heapq.heappop(h)
        mainSolbox = (cost,xs,ys,d0,d1,lvl,th)
        if lvl==mxLVL:
            break
        
        nlvl = int(lvl)+1
        dx=dxs[nlvl]
        H=HLevels[nlvl]
        Tj=np.array((d0,d1))
        Oj = np.array((xs,ys))
        # # S=[]
        
        
        Xg=np.arange(Oj[0],Oj[0]+Tj[0],dx[0])
        Yg=np.arange(Oj[1],Oj[1]+Tj[1],dx[1])
        
        d0,d1=dx[0],dx[1]
        Tj=np.array((d0,d1))
        
        for xs in Xg[:2]:
            for ys in Yg[:2]:
                # S.append( (xs,ys,dx[0],dx[1]) )

                # xs,ys,d0,d1 = solbox
                
                Oj = np.array((xs,ys))
                cost3=getPointCost(H,dx,Xth[th],Oj,Tj) 
                heapq.heappush(h,(-cost3-np.random.rand()/1000,xs,ys,d0,d1,float(nlvl),th))

    t=mainSolbox[1:3]
    th = mainSolbox[6]
    cost=-mainSolbox[0]
    H=np.identity(3)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    H[0:2,0:2]=R
    H[0,2]=t[0]-0*LmaxOrig[0]
    H[1,2]=t[1]-0*LmaxOrig[1]
    
    Htotal12 = H.dot(H12)
    RT=Htotal12[0:2,0:2]
    tT=Htotal12[0:2,2]
    
    Rs=H[0:2,0:2]
    ts=H[0:2,2]
    
    t = tT-(Rs.dot(mn_orig)+0*ts)+mn_orig
    Htotal12_updt=Htotal12
    Htotal12_updt[0:2,2]=t
    Htotal21_updt = nplinalg.inv(Htotal12_updt)
    return Htotal21_updt,cost,HLevels

@njit(cache=numba_cache)
def binMatcherAdaptive3(X11,X22,H12,Lmax,thmax,thmin,dxMatch):
    n=histsmudge =2 # how much overlap when computing max over adjacent hist for levels
    H21comp=np.identity(3)
        
    mn=np.zeros(2)
    mx=np.zeros(2)
    mn_orig=np.zeros(2)
    mn_orig[0] = np.min(X11[:,0])
    mn_orig[1] = np.min(X11[:,1])
    
    mn_orig=mn_orig-dxMatch
    
    
    R=H12[0:2,0:2]
    t=H12[0:2,2]
    X222 = R.dot(X22.T).T+t
    
    
    X2=X222-mn_orig
    X1=X11-mn_orig

    mn[0] = np.min(X1[:,0])
    mn[1] = np.min(X1[:,1])
    mx[0] = np.max(X1[:,0])
    mx[1] = np.max(X1[:,1])
    # rmax=np.max(np.sqrt(X2[:,0]**2+X2[:,1]**2))

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

    
    
    H1match=numba_histogram2D(X1, XYedges[-1][0],XYedges[-1][1])
    H1match = np.sign(H1match)
    
    
    
    

    levels=[]
    HLevels=[H1match]
    
    
    
    
    
    for i in range(1,mxlvl):
        
    
    
        Hup = HLevels[i-1]
        H=UpsampleMax(Hup,n)

        HLevels.append(H)
          
    
    mxLVL=len(HLevels)-1
    HLevels=HLevels[::-1]
    HLevels=[np.ascontiguousarray(H).astype(np.int32) for H in HLevels]
    
    P2=0.1*P
    Lmax=P2*(np.floor(Lmax/P2)+1)
    # Lmax=np.maximum(Lmax,P)
    
    LmaxOrig=np.zeros(2,dtype=np.float64)
    LmaxOrig[0]=Lmax[0]
    LmaxOrig[1]=Lmax[1]

    SolBoxes_init=[]
    X2[:,0]=X2[:,0]-LmaxOrig[0]
    X2[:,1]=X2[:,1]-LmaxOrig[1]
    
    Lmax=dxs[0]*(np.floor(Lmax/dxs[0])+1)
    
    for xs in np.arange(0,2*Lmax[0],dxs[0][0]):
        for ys in np.arange(0,2*Lmax[1],dxs[0][1]):
            SolBoxes_init.append( (xs,ys,dxs[0][0],dxs[0][1]) )
    
    
    
    
    
    
    h=[(100000.0,0.0,0.0,0.0,0.0,0.0,0.0)]
    lvl=0
    dx=dxs[lvl]
    H=HLevels[lvl]
    
    Xth= Dict.empty(
        key_type=types.float64,
        value_type=float_2Darray,
    )
    ii=0
    R = np.array([[np.cos(thmax), -np.sin(thmax)],[np.sin(thmax), np.cos(thmax)]])
    X222=np.transpose((R.T).dot(X2.T))
    
    thfineRes = thmin
    thL=np.arange(0,2*thmax+thfineRes,thfineRes,dtype=np.float64)

    # X2=np.ascontiguousarray(X2)
    for j in range(len(thL)):
        th=thL[j]
        R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        XX=np.transpose(R.dot(X222.T))
        Xth[th]=XX
        
        
        for solbox in SolBoxes_init:
            xs,ys,d0,d1 = solbox
            Tj=np.array((d0,d1))
            Oj = np.array((xs,ys))
            cost2=getPointCost(H,dx,Xth[th],Oj,Tj)
            h.append((-cost2-np.random.rand()/1e10,xs,ys,d0,d1,lvl,th))
            
    heapq.heapify(h)
    
    print("len(heap) = ",len(h))

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
                Oj = np.array((xs,ys))
                cost3=getPointCost(H,dx,Xth[th],Oj,Tj) 
                heapq.heappush(h,(-cost3-np.random.rand()/1e10,xs,ys,d0,d1,float(nlvl),th))

    t=mainSolbox[1:3]
    th = mainSolbox[6]+np.pi
    cost=-mainSolbox[0]
    

    
    Hcomp=np.identity(3)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    Hcomp[0:2,0:2]=R
    Hcomp[0:2,2]=t
    

    
    H1=np.identity(3)
    H1[0:2,2]=-mn_orig
    
    Hlmax=np.identity(3)
    Hlmax[0:2,2]=-LmaxOrig
    
    # H1=np.array([[1,0,-mn_orig[0]],[0,1,-mn_orig[1]],[0,0,1]])
    # Hlmax=np.array([[1,0,-LmaxOrig[0]],[0,1,-LmaxOrig[1]],[0,0,1]])
    
    H2=nplinalg.inv(H1)
    H2=H2.dot(Hcomp)
    H3=H2.dot(Hlmax)
    H4=H3.dot(H1)
    H12comp=H4.dot(H12)

    H21comp=nplinalg.inv(H12comp)
    
    return H21comp

@njit(cache=numba_cache)
def binMatcherAdaptive4_good(X11,X22,H12,Lmax,thmax,thmin,dxMatch):
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
    
    # X2=X222-mn_orig
    X1=X11-mn_orig
    
    
    # t2 = np.mean(X2,axis=0)
    # X20 = X2-t2
    
    mn[0] = np.min(X1[:,0])
    mn[1] = np.min(X1[:,1])
    mx[0] = np.max(X1[:,0])
    mx[1] = np.max(X1[:,1])
    # rmax=np.max(np.sqrt(X2[:,0]**2+X2[:,1]**2))

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

    
    
    H1match=numba_histogram2D(X1, XYedges[-1][0],XYedges[-1][1])
    H1match = np.sign(H1match)
    
    
    
    

    levels=[]
    HLevels=[H1match]
    
    
    
    
    
    for i in range(1,mxlvl):
        
    
    
        Hup = HLevels[i-1]
        H=UpsampleMax(Hup,n)

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
    
    # print(Lmax)
    # Lmax=dxs[-1]*(np.floor(Lmax/dxs[-1])+1)
    # print(Lmax)
    for xs in np.arange(-Lmax[0],Lmax[0],dxs[0][0]):
        for ys in np.arange(-Lmax[1],Lmax[1],dxs[0][1]):
            SolBoxes_init.append( (xs,ys,dxs[0][0],dxs[0][1]) )
    
    
    
    
    
    
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
        
        
        for solbox in SolBoxes_init:
            xs,ys,d0,d1 = solbox
            Tj=np.array((d0,d1))
            Oj = np.array((xs,ys))
            cost2=getPointCost(H,dx,Xth[th],Oj,Tj)
            h.append((-cost2-np.random.rand()/1e10,xs,ys,d0,d1,lvl,th))
            
    heapq.heapify(h)
    
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
                Oj = np.array((xs,ys))
                cost3=getPointCost(H,dx,Xth[th],Oj,Tj) 
                heapq.heappush(h,(-cost3-np.random.rand()/1e10,xs,ys,d0,d1,float(nlvl),th))

    t=mainSolbox[1:3]
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
    
    return H21comp
    



@jit(int32[:,:](float64[:,:], float64[:], float64[:]),nopython=True, nogil=True,cache=numba_cache) 
def numba_histogram2D(X, xedges,yedges):
    x_min = np.min(xedges)
    x_max = np.max(xedges)
    nx = len(xedges) 
    
    
    y_min = np.min(yedges)
    y_max = np.max(yedges)
    ny = len(yedges)
    
    H = np.zeros((nx-1,ny-1),dtype=np.int32)
    
    dxy=np.array([x_max-x_min,y_max-y_min])
    xymin = np.array([x_min,y_min])
    xymax = np.array([x_max,y_max])
    
    nxy = np.array([nx-1,ny-1])
    
       
    dd = nxy*((X-xymin)/dxy)

    for i in range(X.shape[0]):
        if np.all( (X[i]-xymin)>=0) and np.all( (xymax-X[i])>=0):
            H[int(dd[i][0]),int(dd[i][1])]+=1   

        
    return H


    
@jit(float32(float64[:], int32[:,:], float64[:,:], float64[:], float64[:], int32),nopython=True, nogil=True,cache=numba_cache) 
def binScanCost(x,P,Xt,xedges,yedges,cntThres):
    # x is the pose
    # P is the Probability of bins of the keyframe
    
    th=x[0]
    t=np.array([x[1],x[2]])
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    Xi=np.dot(R,Xt.T).T
    Xn=Xi+t
    
    H=numba_histogram2D(Xn, xedges,yedges)
    
    nx = len(xedges) 
    ny = len(yedges) 
    
    # Pn is the probability of scan points in the keyframe histogram
    # Pn=np.zeros((nx-1,ny-1),dtype=np.int32)
    Pn=np.sign(H)
    
    # for i in range(nx-1):
    #     for j in range(ny-1):
    #         if H[i,j]>=cntThres:
    #             Pn[i,j]=1

    mbin_and = np.sum(np.logical_and(P,Pn))
    # d2 = np.sum(np.logical_or(P,Pn))
    
    # dd = d.astype(np.int32)
    
    # f = np.sum(dd.reshape(-1))
    # mbin_and,mbin_or,Hist2_ovrlp=Commonbins(Hist1_ovrlp,Xn,xedges_ovrlp,yedges_ovrlp,1)
    activebins1_ovrlp = np.sum(P)
    activebins2_ovrlp = np.sum(Pn)
    
    
    # mbinfrac_ovrlp=mbin_and/mbin_or
    # mbinfrac_ActiveOvrlp = max([mbin_and/activebins1_ovrlp,mbin_and/activebins2_ovrlp])
    mbinfrac_ActiveOvrlp = mbin_and/activebins1_ovrlp

    return mbinfrac_ActiveOvrlp

@jit(numba.types.Tuple((int32[:,:],float64[:],float64[:]))(float64[:,:], float64[:,:], float64[:]),nopython=True, nogil=True,cache=numba_cache) 
def binScanEdges(Xb,Xt,dx):
    mn=np.zeros(2)
    mn[0] = np.min(Xb[:,0])
    mn[1] = np.min(Xb[:,1])
    
    mx=np.zeros(2)
    mx[0] = np.max(Xb[:,0])
    mx[1] = np.max(Xb[:,1])
    
    mnt=np.zeros(2)
    mnt[0] = np.min(Xt[:,0])
    mnt[1] = np.min(Xt[:,1])
    
    mxt=np.zeros(2)
    mxt[0] = np.max(Xt[:,0])
    mxt[1] = np.max(Xt[:,1])
    
    dt= np.max(0.5*(mxt-mnt))
    
    mn = mn-dt
    mx = mx+dt
    
    xedges = np.arange(mn[0],mx[0],dx[0])
    yedges = np.arange(mn[1],mx[1],dx[1])
    
    nx = len(xedges) 
    ny = len(yedges)
    H = numba_histogram2D(Xb, xedges,yedges)
    P=np.sign(H)
    
    return P, xedges,yedges

# @jit(numba.types.Tuple((int32[:,:],float64[:],float64[:]))(float64[:,:], float64[:,:], float64[:]),nopython=True, nogil=True,cache=numba_cache) 
# def posematchMetrics(X1,X2,H12,dx):
#     # transform points to X1 space, bin it and then compute the posematch cost
#     R=H12[0:2,0:2]
#     X12=H12.dot(X2.T)+H12[0:2,2]
    
            


@jit(numba.types.Tuple((float64[:],float64))(float64[:,:], int32[:,:], float64[:,:],float64[:],float64[:], int32),nopython=True, nogil=True,cache=numba_cache) 
def binScanMatcher(Posegrid,P,Xt,xedges,yedges,cntThres):
    # Xb are keyframe points
    # Xt are are scan points
    
    
    # P=np.zeros((nx-1,ny-1),dtype=np.int32)
    # for i in range(nx-1):
    #     for j in range(ny-1):
    #         if H[i,j]>=cntThres:
    #             P[i,j]=1
    
    
    
    m = np.zeros(Posegrid.shape[0])
    for i in numba.prange(Posegrid.shape[0]):
        m[i]=binScanCost(Posegrid[i],P,Xt,xedges,yedges,cntThres) 
        
    ind = np.argmax(m)  
        
    res = Posegrid[ind]
    
    return res,m[ind]

#%%

@jit(float64(float64[:], float64[:,::1], float64[:,:], float64[:,:,:],float64[:]),nopython=True, nogil=True,cache=numba_cache) 
def getcostalign(x,Xt,MU,P,W):
    th=x[0]
    t=np.array([x[1],x[2]])
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    Xi=np.dot(R,Xt.T).T
    Xn=Xi+t
    p=uqgmmfnc.gmm_eval_fast(Xn,MU,P,W)+0.01
    logp = -np.log(p)
    f = np.mean(logp)
    # logp = np.diag(f(Xn[:,0],Xn[:,1]))
    return f



#%%

@jit(numba.types.Tuple((float64[:],float64))(float64[:,:]),nopython=True, nogil=True,parallel=False,cache=numba_cache) 
def extractPosAngle(H):
    theta = np.arctan2(-H[0,1],H[0,0])
    txy = H[:2,2]
    return txy,theta

@jit(float64[:,:](float64, float64[:]),nopython=True, nogil=True,parallel=False,cache=numba_cache) 
def getHmat(th,t):
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    H=np.identity(3)
    H[0:2,0:2]=R
    H[0:2,2]=t
    return H

@jit(float64(float64, float64),nopython=True, nogil=True,parallel=False,cache=numba_cache) 
def anglediff(th1,th2):
    a = th1-th2
    if a>np.pi:
        a-=2*np.pi
    elif a<-np.pi:
        a+=2*np.pi
    elif a==-np.pi:
        a=np.pi
    
    return a


#%%
@jit(float64[:](float64[:,:], float64[:,:], float64[:,:,:],float64[:], float64[:,::1]),nopython=True, nogil=True,parallel=False,cache=numba_cache) 
def gridsearch_alignment(Posegrid,MU,P,W,X):
    
    m=np.zeros(Posegrid.shape[0])
    for i in numba.prange(Posegrid.shape[0]):
        m[i]=getcostalign(Posegrid[i],X,MU,P,W)
    ind = np.argmin(m)
    res = Posegrid[ind]
    return res



@jit(numba.types.Tuple((float64,float64[:]))(float64[:], float64[:,::1], float64[:,:], float64[:,:,:],float64[:]),nopython=True, nogil=True,parallel=False,cache=numba_cache) 
def getcostgradient(x,Xt,MU,P,W):
    npt=Xt.shape[0]
    ncomp=MU.shape[0]
    
    th=x[0]
    t=np.array([x[1],x[2]])
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    Xi=np.dot(R,Xt.T).T
    Xn=Xi+t
    pcomp=uqgmmfnc.gmm_evalcomp_fast(Xn,MU,P,W)
    p = np.sum(pcomp,axis=1)+0.01
    logp = -np.log(p)
    # ind = logp<np.mean(logp)
    f = np.mean(logp)    
    
    invp = 1/p  # this is a vec with each element for one point in Xt
    dRdth = np.array([[-np.sin(th), -np.cos(th)],[np.cos(th), -np.sin(th)]])
    invPP = np.zeros_like(P)
    for i in range(ncomp):
        invPP[i] = nplinalg.inv(P[i])
    z1=np.dot(dRdth,Xt.T).T
    
    dpdth=np.zeros((npt,ncomp),dtype=dtype)
    dpdx=np.zeros((npt,ncomp),dtype=dtype)
    dpdy=np.zeros((npt,ncomp),dtype=dtype)
    for i in range(ncomp): 
        z2=Xn-MU[i]
        z3=np.dot(invPP[i],z2.T).T
        y1=np.multiply(z1,z3)
        dpdth[:,i] = -np.sum(y1,axis=1)
        dpdx[:,i] = -z3[:,0]
        dpdy[:,i] = -z3[:,1]
    
    a1 = -invp*np.sum(pcomp*dpdth,axis=1)
    a2 = -invp*np.sum(pcomp*dpdx,axis=1)
    a3 = -invp*np.sum(pcomp*dpdy,axis=1)
    dth=np.mean(a1)
    dx=np.mean(a2)
    dy=np.mean(a3)
    g=np.array([dth,dx,dy])
    return f,g





#%% Loop closure


@jit(float64(float64[:],int32[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=numba_cache) 
def globalPoseCost(x,Hrelsidx,Hrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    yy=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        yy[s+1,:]= x[s*3:s*3+3]
    

    th=yy[:,0] # global
    txy=yy[:,1:] # global
    F=0

    S = np.identity(3)
    S[0,0]=1
    y=np.zeros(3)
    
    for idx in numba.prange(Hrels.shape[0]):
        i=Hrelsidx[idx,0]
        j=Hrelsidx[idx,1]
        
        thji=Hrels[idx,0]
        tji=Hrels[idx,1:]
        
        
        jHi=getHmat(thji,tji)
        
        thi = th[i]
        ti = txy[i]
        thj = th[j]
        tj = txy[j]
        
        Ri = np.array([[np.cos(thi), -np.sin(thi)],[np.sin(thi), np.cos(thi)]])
        Rj = np.array([[np.cos(thj), -np.sin(thj)],[np.sin(thj), np.cos(thj)]])
        
        # fHi=getHmat(thi,ti)
        # fHj=getHmat(thj,tj)
        
        # jHf = nplinalg.inv(fHj)
        # jHi_var=jHf.dot(fHi)
        
        # tji_var,thji_var=extractPosAngle(jHi_var)
        
        tji_var = Rj.T.dot(ti-tj)
        thji_var = anglediff(thi,thj)
        
        y[0]=thji_var-thji
        y[1] = tji_var[0]-tji[0]
        y[2] = tji_var[1]-tji[1]
        
        
        F= F+0.5*y.dot(S.dot(y)) 

    return F

@jit(numba.types.Tuple((float64,float64[:]))(float64[:],int32[:,:],float64[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=numba_cache) 
def globalPoseCost_Fjac(x,Hrelsidx,Hrels,Hessrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    yy=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        yy[s+1,:]= x[s*3:s*3+3]
    

    th=yy[:,0] # global
    txy=yy[:,1:] # global
    
    m=Hrels.shape[0]*3
    n = len(x)
    
    dydxi = np.zeros((3,3))
    dydxj = np.zeros((3,3))
    
    F=0
    
    
    # S = np.identity(3)
    # S[0,0]=1
    
    for idx in numba.prange(Hrels.shape[0]):
        jac = np.zeros(len(x))
        y=np.zeros(3)
        hess = np.zeros((3,3))
                    
        i=Hrelsidx[idx,0]
        j=Hrelsidx[idx,1]
        
        thji=Hrels[idx,0]
        tji=Hrels[idx,1:]
        # ytrue = np.hstack([thji,tji])
        
        jHi=getHmat(thji,tji)
        
        thi = th[i]
        ti = txy[i]
        thj = th[j]
        tj = txy[j]
        
        hess[0] = Hessrels[idx,0:3]
        hess[1] = Hessrels[idx,3:6]
        hess[2] = Hessrels[idx,6:9]
        # S=np.identity(3)
        S = hess
        
        Ri = np.array([[np.cos(thi), -np.sin(thi)],[np.sin(thi), np.cos(thi)]])
        Rj = np.array([[np.cos(thj), -np.sin(thj)],[np.sin(thj), np.cos(thj)]])
        
        # fHi=getHmat(thi,ti)
        # fHj=getHmat(thj,tj)
        
        # jHf = nplinalg.inv(fHj)
        # jHi_var=jHf.dot(fHi)
        
        # tji_var,thji_var=extractPosAngle(jHi_var)
        
        tji_var = Rj.T.dot(ti-tj)
        thji_var = anglediff(thi,thj)
        

        # for i
        y[0]=thji_var-thji
        y[1] = tji_var[0]-tji[0]
        y[2] = tji_var[1]-tji[1]
        
        # y=np.hstack([thji_var-thji,tji_var-tji])
        
        dydxi[0,0]=1
        dydxi[0,1:]=0
        dydxi[1:,0]=0
        dydxi[1:,1:]=Rj.T
        
        # dJjidi = nplinalg.multi_dot([dydxi.T,S,y]) 
        dJjidi = dydxi.T.dot(S.dot(y)) 
        
        
        # for j
        dRjTdth = np.array([[-np.sin(thj),np.cos(thj)],[-np.cos(thj),-np.sin(thj)]])
        
        dydxj[0,0]=-1
        dydxj[0,1:]=0
        dydxj[1:,0]=dRjTdth.dot(ti-tj)
        dydxj[1:,1:]=-Rj.T
        
        dJjidj = dydxj.T.dot(S.dot(y))  
        #np.matmul(dydxj.T,y)
        if i>0:
            pp=i-1
            jac[3*pp:3*pp+3] = jac[3*pp:3*pp+3]+dJjidi
        
        if j>0:
            pp=j-1
            jac[3*pp:3*pp+3] = jac[3*pp:3*pp+3]+dJjidj
            
        # jac[3*j:3*j+3] = jac[3*j:3*j+3]+dJjidj

        

        F= F+0.5*y.dot(S.dot(y)) 
    
    return F,jac
        
    

@jit(float64[:](float64[:],int32[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=numba_cache) 
def globalPoseCost_lsq(x,Hrelsidx,Hrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    yy=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        yy[s+1,:]= x[s*3:s*3+3]
    

    th=yy[:,0] # global
    txy=yy[:,1:] # global
    F=np.zeros(Hrels.shape[0]*3)

        
    for idx in numba.prange(Hrels.shape[0]):
        i=Hrelsidx[idx,0]
        j=Hrelsidx[idx,1]
        
        thji=Hrels[idx,0]
        tji=Hrels[idx,1:]
        
        
        jHi=getHmat(thji,tji)
        
        thi = th[i]
        ti = txy[i]
        thj = th[j]
        tj = txy[j]
        
        Ri = np.array([[np.cos(thi), -np.sin(thi)],[np.sin(thi), np.cos(thi)]])
        Rj = np.array([[np.cos(thj), -np.sin(thj)],[np.sin(thj), np.cos(thj)]])
        
        # fHi=getHmat(thi,ti)
        # fHj=getHmat(thj,tj)
        
        # jHf = nplinalg.inv(fHj)
        # jHi_var=jHf.dot(fHi)
        
        # tji_var,thji_var=extractPosAngle(jHi_var)
        
        tji_var = Rj.T.dot(ti-tj)
        thji_var = anglediff(thi,thj)
        
        if np.abs(i-j)<=2:
            c=10
        else:
            c=1


        
        F[idx*3] = c*anglediff(thji_var,thji)
        F[idx*3+1] = c*(tji_var[0]-tji[0])
        F[idx*3+2] = c*(tji_var[1]-tji[1])
        
        
    # f=np.sum(F)
    # f=1.3
    return F

# @jit(float64[:,:](float64[:],int32[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=numba_cache) 
def globalPoseCost_lsq_jac(x,Hrelsidx,Hrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    yy=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        yy[s+1,:]= x[s*3:s*3+3]
    
    # now yy includes the first frame with 0,0,0 fixed pose
    th=yy[:,0] # global
    txy=yy[:,1:] # global
    
    m=Hrels.shape[0]*3
    n = len(x)
    jac=lil_matrix((m, n), dtype=dtype)  
    # jac=np.zeros((m, n), dtype=dtype)  
    
    dydxi = np.zeros((3,3))
    dydxj = np.zeros((3,3))
    y=np.zeros(3)
    # print("Hrelsidx")
    # print(Hrelsidx)
    for idx in numba.prange(Hrels.shape[0]):
        i=Hrelsidx[idx,0]
        j=Hrelsidx[idx,1]
        
        thji=Hrels[idx,0]
        tji=Hrels[idx,1:]
        
        
        jHi=getHmat(thji,tji)
        
        thi = th[i]
        ti = txy[i]
        thj = th[j]
        tj = txy[j]
        
        Ri = np.array([[np.cos(thi), -np.sin(thi)],[np.sin(thi), np.cos(thi)]])
        Rj = np.array([[np.cos(thj), -np.sin(thj)],[np.sin(thj), np.cos(thj)]])
        
        # fHi=getHmat(thi,ti)
        # fHj=getHmat(thj,tj)
        
        # jHf = nplinalg.inv(fHj)
        # jHi_var=jHf.dot(fHi)
        
        # tji_var,thji_var=extractPosAngle(jHi_var)
        
        tji_var = Rj.T.dot(ti-tj)
        thji_var = anglediff(thi,thj)
        

        # for i
        y[0]=thji_var-thji
        y[1] = tji_var[0]-tji[0]
        y[2] = tji_var[1]-tji[1]
        
        # np.hstack([thji_var,tji_var])
        
        dydxi[0,0]=1
        dydxi[0,1:]=0
        dydxi[1:,0]=0
        dydxi[1:,1:]=Rj.T
        
        # dJjidi = np.matmul(dydxi.T,y)
        
        
        # for j
        dRjTdth = np.array([[-np.sin(thj),np.cos(thj)],[-np.cos(thj),-np.sin(thj)]])
        
        dydxj[0,0]=-1
        dydxj[0,1:]=0
        dydxj[1:,0]=dRjTdth.dot(ti-tj)
        dydxj[1:,1:]=-Rj.T
        
        if np.abs(i-j)<=2:
            c=10
        else:
            c=1
            
        # dJjidj = np.matmul(dydxj.T,y)
        if i>0:
            pp=i-1
            jac[3*idx,3*pp:3*pp+3] = c*dydxi[0] 
            jac[3*idx+1,3*pp:3*pp+3] =  c*dydxi[1]
            jac[3*idx+2,3*pp:3*pp+3] =  c*dydxi[2]
        
        # print("m,n,idx,i,j,Hrels.shape[0]",m,n,idx,i,j,Hrels.shape[0])
        if j>0:
            pp=j-1
            jac[3*idx,3*pp:3*pp+3] =  c*dydxj[0] 
            jac[3*idx+1,3*pp:3*pp+3] =  c*dydxj[1]
            jac[3*idx+2,3*pp:3*pp+3] =  c*dydxj[2]

        
        # F[idx*3] = (thji_var-thji)
        # F[idx*3+1] = (tji_var[0]-tji[0])
        # F[idx*3+2] = (tji_var[1]-tji[1])
        
        
        
    # f=np.sum(F)
    # f=1.3
    return jac

@jit(float64[:](float64[:],int32[:,:],float64[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=numba_cache) 
def globalPoseCostHess_lsq(x,Hrelsidx,Hrels,Hessrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    yy=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        yy[s+1,:]= x[s*3:s*3+3]
    

    th=yy[:,0] # global
    txy=yy[:,1:] # global
    F=np.zeros(Hrels.shape[0])

    hess = np.zeros((3,3))
    y=np.zeros(3)
    for idx in numba.prange(Hrels.shape[0]):
        i=Hrelsidx[idx,0]
        j=Hrelsidx[idx,1]
        
        thji=Hrels[idx,0]
        tji=Hrels[idx,1:]
        
        hess[0] = Hessrels[idx,0:3]
        hess[1] = Hessrels[idx,3:6]
        hess[2] = Hessrels[idx,6:9]
        
        jHi=getHmat(thji,tji)
        
        thi = th[i]
        ti = txy[i]
        thj = th[j]
        tj = txy[j]
        
        Ri = np.array([[np.cos(thi), -np.sin(thi)],[np.sin(thi), np.cos(thi)]])
        Rj = np.array([[np.cos(thj), -np.sin(thj)],[np.sin(thj), np.cos(thj)]])
        
        # fHi=getHmat(thi,ti)
        # fHj=getHmat(thj,tj)
        
        # jHf = nplinalg.inv(fHj)
        # jHi_var=jHf.dot(fHi)
        
        # tji_var,thji_var=extractPosAngle(jHi_var)
        
        tji_var = Rj.T.dot(ti-tj)
        thji_var = anglediff(thi,thj)
        

        

        
        e = np.array([thji_var-thji,tji_var[0]-tji[0],tji_var[1]-tji[1]])
        e=np.sqrt(np.abs(e))
        y=hess.dot(e)
        y=e*y
        # F[idx] = y
        F[idx*3] = y[0]
        F[idx*3+1] = y[1]
        F[idx*3+2] = y[2]

        
        
    # f=np.sum(F)
    # f=1.3
    return F

@jit(float64[:,:](float64[:],int32[:,:],float64[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=numba_cache) 
def globalPoseCostHess_lsq_jac(x,Hrelsidx,Hrels,Hessrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    yy=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        yy[s+1,:]= x[s*3:s*3+3]
    
    # now yy includes the first frame with 0,0,0 fixed pose
    th=yy[:,0] # global
    txy=yy[:,1:] # global
    
    m=Hrels.shape[0]*3
    n = len(x)
    # jac=lil_matrix((m, n), dtype=dtype)  
    jac=np.zeros((m, n), dtype=dtype)  
    
    dydxi = np.zeros((3,3))
    dydxj = np.zeros((3,3))
    y=np.zeros(3)
    hess = np.zeros((3,3))
    
    # print("Hrelsidx")
    # print(Hrelsidx)
    for idx in numba.prange(Hrels.shape[0]):
        i=Hrelsidx[idx,0]
        j=Hrelsidx[idx,1]
        
        thji=Hrels[idx,0]
        tji=Hrels[idx,1:]
        
        hess[0] = Hessrels[idx,0:3]
        hess[1] = Hessrels[idx,3:6]
        hess[2] = Hessrels[idx,6:9]
        
        jHi=getHmat(thji,tji)
        
        thi = th[i]
        ti = txy[i]
        thj = th[j]
        tj = txy[j]
        
        Ri = np.array([[np.cos(thi), -np.sin(thi)],[np.sin(thi), np.cos(thi)]])
        Rj = np.array([[np.cos(thj), -np.sin(thj)],[np.sin(thj), np.cos(thj)]])
        
        # fHi=getHmat(thi,ti)
        # fHj=getHmat(thj,tj)
        
        # jHf = nplinalg.inv(fHj)
        # jHi_var=jHf.dot(fHi)
        
        # tji_var,thji_var=extractPosAngle(jHi_var)
        
        tji_var = Rj.T.dot(ti-tj)
        thji_var = anglediff(thi,thj)
        

        # for i
        y[0]=thji_var-thji
        y[1] = tji_var[0]-tji[0]
        y[2] = tji_var[1]-tji[1]
        
        # np.hstack([thji_var,tji_var])
        
        dydxi[0,0]=1
        dydxi[0,1:]=0
        dydxi[1:,0]=0
        dydxi[1:,1:]=Rj.T
        
        # dJjidi = np.matmul(dydxi.T,y)
        
        
        # for j
        dRjTdth = np.array([[-np.sin(thj),np.cos(thj)],[-np.cos(thj),-np.sin(thj)]])
        
        dydxj[0,0]=-1
        dydxj[0,1:]=0
        dydxj[1:,0]=dRjTdth.dot(ti-tj)
        dydxj[1:,1:]=-Rj.T
        
        # dJjidj = np.matmul(dydxj.T,y)
        if i>0:
            pp=i-1
            jac[3*idx,3*pp:3*pp+3] = dydxi[0] 
            jac[3*idx+1,3*pp:3*pp+3] =  dydxi[1]
            jac[3*idx+2,3*pp:3*pp+3] =  dydxi[2]
        
        # print("m,n,idx,i,j,Hrels.shape[0]",m,n,idx,i,j,Hrels.shape[0])
        if j>0:
            pp=j-1
            jac[3*idx,3*pp:3*pp+3] =  dydxj[0] 
            jac[3*idx+1,3*pp:3*pp+3] =  dydxj[1]
            jac[3*idx+2,3*pp:3*pp+3] =  dydxj[2]

        
        # F[idx*3] = (thji_var-thji)
        # F[idx*3+1] = (tji_var[0]-tji[0])
        # F[idx*3+2] = (tji_var[1]-tji[1])
        
        
        
    # f=np.sum(F)
    # f=1.3
    return jac
