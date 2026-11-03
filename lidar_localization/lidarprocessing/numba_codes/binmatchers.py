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
import lidarprocessing.numba_codes.point2Dprocessing_numba as nbpt2Dproc
numba_cache=True
dtype=np.float64

from numba.core import types
from numba.typed import Dict
float_2Darray = types.float64[:,:]








@njit(cache=True)
def binMatcherAdaptive_super(X11,X22,H12,Lmax,thmax,thmin,dxMatch,dxBase):
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
    
    

    H12mn = H12.copy()
    H12mn[0:2,2]=H12mn[0:2,2]-mn_orig
    X1=X11-mn_orig
    
    
    mn[0] = np.min(X1[:,0])
    mn[1] = np.min(X1[:,1])
    mx[0] = np.max(X1[:,0])
    mx[1] = np.max(X1[:,1])
    
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
    
    
    SolBoxes_init=[]
    for xs in np.arange(0,dxs[0][0],dxs[0][0]):
        for ys in np.arange(0,dxs[0][1],dxs[0][1]):
            SolBoxes_init.append( (xs,ys,dxs[0][0],dxs[0][1]) )        
    
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
        
            cost2=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj)
            h.append((-cost2-np.random.rand()/1e10,xs,ys,d0,d1,lvl,th,mxLVL))
        
    heapq.heapify(h)
    
    XX0=np.transpose(H12mn[0:2,0:2].dot(X22.T))+H12mn[0:2,2]
    zz=np.zeros(2,dtype=np.float64)
    cost0=nbpt2Dproc.getPointCost(HLevels[-1],dxs[-1],XX0,zz,dxs[-1])
    
    bb1={'x1':t0[0]-Lmax[0],'y1':t0[1]-Lmax[1],'x2':t0[0]+Lmax[0],'y2':t0[1]+Lmax[1]}
    if dxBase[0]>=0:
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
                                 
                            Oj = np.array((xs,ys))
                            cost3=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj) 
                            HH.append((-cost3-np.random.rand()/1e10,xs,ys,d0,d1,float(nlvl),th,mxLVL))
                else:
                    HH.append(h[i])
            h=HH
            if flg==False:
                break
        
        heapq.heapify(h)
        
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
                            
                Oj = np.array((xs,ys))
                cost3=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj) 
                heapq.heappush(h,(-cost3-np.random.rand()/1e10,xs,ys,d0,d1,float(nlvl),th,mxLVL))

    
    t=np.array(mainSolbox[1:3])-t0
    th = mainSolbox[6]
    cost=np.floor(-mainSolbox[0])
    d0,d1=mainSolbox[3:5]
    
    HcompR=np.identity(3)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    HcompR[0:2,0:2]=R
    
    Ht=np.identity(3)
    Ht[0:2,2]=t

    H12comp=np.dot(Ht.dot(H12mn),HcompR)
    H12comp[0:2,2]=H12comp[0:2,2]+mn_orig

    H21comp=nplinalg.inv(H12comp)
    hh=0
    hR=0
    return (H21comp,cost0,cost,hh,hR)
    # return H21comp



@njit
def binMatcherAdaptive_Density(X11,X22,H12,Lmax,thmax,thmin,dxMatch,dxBase):
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
    
    

    H12mn = H12.copy()
    H12mn[0:2,2]=H12mn[0:2,2]-mn_orig
    X1=X11-mn_orig
    
    
    mn[0] = np.min(X1[:,0])
    mn[1] = np.min(X1[:,1])
    mx[0] = np.max(X1[:,0])
    mx[1] = np.max(X1[:,1])
    
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
    # H1match = np.sign(H1match)
    
    levels=[]
    HLevels=[H1match]
    
    for i in range(1,mxlvl):
    
        Hup = HLevels[i-1]
        H=nbpt2Dproc.UpsampleMax(Hup,n)
        HLevels.append(H)
          
    mxLVL=len(HLevels)-1
    HLevels=HLevels[::-1]
    HLevels=[np.ascontiguousarray(H).astype(np.int32) for H in HLevels]
    
    
    SolBoxes_init=[]
    for xs in np.arange(0,dxs[0][0],dxs[0][0]):
        for ys in np.arange(0,dxs[0][1],dxs[0][1]):
            SolBoxes_init.append( (xs,ys,dxs[0][0],dxs[0][1]) )        
    
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
        
            cost2=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj)
            h.append((-cost2-np.random.rand()/1e10,xs,ys,d0,d1,lvl,th,mxLVL))
        
    heapq.heapify(h)
    
    XX0=np.transpose(H12mn[0:2,0:2].dot(X22.T))+H12mn[0:2,2]
    zz=np.zeros(2,dtype=np.float64)
    cost0=nbpt2Dproc.getPointCost(HLevels[-1],dxs[-1],XX0,zz,dxs[-1])
    
    bb1={'x1':t0[0]-Lmax[0],'y1':t0[1]-Lmax[1],'x2':t0[0]+Lmax[0],'y2':t0[1]+Lmax[1]}
    if dxBase[0]>=0:
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
                                 
                            Oj = np.array((xs,ys))
                            cost3=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj) 
                            HH.append((-cost3-np.random.rand()/1e10,xs,ys,d0,d1,float(nlvl),th,mxLVL))
                else:
                    HH.append(h[i])
            h=HH
            if flg==False:
                break
        
        heapq.heapify(h)
        
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
                            
                Oj = np.array((xs,ys))
                cost3=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj) 
                heapq.heappush(h,(-cost3-np.random.rand()/1e10,xs,ys,d0,d1,float(nlvl),th,mxLVL))

    
    t=np.array(mainSolbox[1:3])-t0
    th = mainSolbox[6]
    cost=np.floor(-mainSolbox[0])
    d0,d1=mainSolbox[3:5]
    
    HcompR=np.identity(3)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    HcompR[0:2,0:2]=R
    
    Ht=np.identity(3)
    Ht[0:2,2]=t

    H12comp=np.dot(Ht.dot(H12mn),HcompR)
    H12comp[0:2,2]=H12comp[0:2,2]+mn_orig

    H21comp=nplinalg.inv(H12comp)
    hh=0
    hR=0
    return (H21comp,cost0,cost,hh,hR)
    # return H21comp
