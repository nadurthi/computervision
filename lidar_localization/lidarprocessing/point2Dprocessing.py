# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 21:11:58 2020

@author: nadur
"""

# from numba import set_num_threads,config, njit, threading_layer,get_num_threads

# set_num_threads(1)
# set the threading layer before any parallel target compilation
# config.THREADING_LAYER = 'threadsafe'
# config.NUMBA_NUM_THREADS

# print("#Threads: %d" % get_num_threads())
# print("Threading layer chosen: %s" % threading_layer())

import numpy as np
import pickle as pkl
import numpy as np
import numpy.linalg as nplinalg
import numba
from numba import vectorize, float64,guvectorize,int64,double,int32,float32
from numba import njit, prange,jit
import random
# import g2o
import multiprocessing as mp
import threading
import queue
from utils.plotting import geometryshapes as utpltgmshp
import lidarprocessing.numba_codes.point2Dprocessing_numba as nbpt2Dproc
from lidarprocessing import point2Dplotting as pt2dplot
import copy

import heapq
import numpy as np
from numpy.lib.stride_tricks import as_strided


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
numba_cache=False
import asyncio
# import cupy as cp
from lidarprocessing import icp
from numba.core import types
from numba.typed import Dict

dtype=np.float64
float_2Darray = types.float64[:,:]
#%% Submap- grid



    
def binnerDownSampler(X,dx=0.05,cntThres=1):
    mn = np.min(X,axis=0)
    mx = np.max(X,axis=0)
    mn = mn-dx/2
    mx = mx+dx/2
    
    xedges = np.arange(mn[0],mx[0]+dx,dx)
    yedges = np.arange(mn[1],mx[1]+dx,dx)
    
    # H= nbpt2Dproc.numba_histogram2D(X, xedges,yedges)
    
    H, xedges, yedges = np.histogram2d(X[:,0],X[:,1],bins=(xedges, yedges) )
    
    inds = np.argwhere(H>=cntThres)   
    x=xedges[inds[:,0]]+dx/2
    y=yedges[inds[:,1]]+dx/2                                
    Xd = np.vstack([x,y]).T
    
    return Xd



def binnerDownSamplerProbs(Xlist,dx=0.05,prob=0.5):
    N = len(Xlist)
    X = np.vstack(Xlist)
    mn = np.min(X,axis=0)
    mx = np.max(X,axis=0)
    mn = mn-dx/2
    mx = mx+dx/2
    
    xedges = np.arange(mn[0],mx[0]+dx,dx)
    yedges = np.arange(mn[1],mx[1]+dx,dx)
    
    H= nbpt2Dproc.numba_histogram2D(X, xedges,yedges)
    # H, xedges, yedges = np.histogram2d(X[:,0],X[:,1],bins=(xedges, yedges) )
    
    H= H/N
    
    inds = np.argwhere(H>=prob)   
    x=xedges[inds[:,0]]+dx/2
    y=yedges[inds[:,1]]+dx/2                                
    Xd = np.vstack([x,y]).T
    
    return Xd


    
#%% Pose estimation by keyframe
# - manage keyframes
# - estimate pose to closest keyframe

class Clf:
    def __init__(self,MU,P,W):
        self.means_=MU
        self.covariances_=P
        self.weights_=W
        self.n_components = len(W)

def get0meanIcov(X):
    m=np.mean(X,axis=0)
    Xd=X-m
    return Xd,m

def wcost(W,pcomp,randind):
    pp=np.sum(pcomp*W,axis=1)
    c=0
    for i,j in randind:
        c = c + (pp[i]-pp[j])**4
        # c = c + (np.log(pp[i])-np.log(pp[i-1]))**2
    return c

def wcost2(W,pcomp):
    pp=np.sum(pcomp*W,axis=1)
    c=np.sum((pp-1)**2)
    return c

def wcost2lsq(W,pcomp):
    pp=np.sum(pcomp*W,axis=1)
    c=(pp-1)
    return c

def optWts1(X,MU,P,W):
    
    pcomp=uqgmmfnc.gmm_evalcomp_fast(X,MU,P,W)
    bnds=[]
    for i in range(len(W)):
        bnds.append((0,1000))
    randind = []
    for i in range(100):
        a=np.random.choice(X.shape[0], 2)
        randind.append(a)
        
    res = minimize(wcost, W,args=(pcomp,randind),jac=False,bounds=bnds ,method='SLSQP', tol=1e-4) # 'Nelder-Mead'
    # res = least_squares(wcost_lsq, W,args=(pcomp,),bounds=([0]*len(W),[1000]*len(W)) )
    res.x=res.x/np.sum(res.x)
      
    
    return res


    
def optWts2(X,MU,P,W):
    pcomp=uqgmmfnc.gmm_evalcomp_fast(X,MU,P,W)
    bnds=[]
    for i in range(len(W)):
        bnds.append((0,1000))
    
    # res = minimize(wcost2, W,args=(pcomp,),jac=False,bounds=bnds ,method='BFGS', tol=1e-4) # 'Nelder-Mead'
    res = least_squares(wcost2lsq, W,args=(pcomp,),bounds=([0]*len(W),[1000]*len(W)),xtol=1e-4 )
    res.x=res.x/np.sum(res.x)
    
    return res

    
def getclf(X,params,doReWtopt=True,means_init=None,weights_init=None,precisions_init=None,prevclf=None):
    
    doReWtopt=False
    
    # mn = np.min(X,axis=0)
    # mx = np.max(X,axis=0)
    # dx0=np.max(mx-mn)/50
    
    # for dx in np.arange(dx0,dx0/25,-dx0/25):
    #     Xm=binnerDownSampler(X,dx=dx,cntThres=1)
    #     if np.abs(Xm.shape[0]-params['n_components'])<=5 or Xm.shape[0]>=params['n_components']:
    #         break
    #     else:
    #         continue
    
    # if Xm.shape[0]>params['n_components']:   
    #     means_init=Xm[:params['n_components'],:]
    #     n_components=params['n_components']
    # if Xm.shape[0]<=params['n_components']:   
    #     means_init=Xm
    #     n_components=Xm.shape[0]
    
    # try:
    #     print(n_components)
    # except:
    #     print(dx0)
    #     print(Xm.shape)
        
    Xdb=X   
    if prevclf is None:
        clf = mixture.GaussianMixture(n_components=params['n_components'],
                                      means_init=means_init, 
                                      weights_init=weights_init,
                                      precisions_init=precisions_init,
                                      covariance_type='full',reg_covar=params['reg_covar'],
                                       max_iter=5,warm_start=True)
    else:
        clf = prevclf
        
    clf.fit(Xdb)

    
    MU=np.ascontiguousarray(clf.means_,dtype=dtype)
    P=np.ascontiguousarray(clf.covariances_,dtype=dtype)
    W=np.ascontiguousarray(clf.weights_,dtype=dtype)
    
    # ncomp = MU.shape[0]
    # dim = MU.shape[1]
    # npt = X.shape[0]
    # invPP=np.zeros_like(P)
    # for i in range(ncomp):
    #     invPP[i] = nplinalg.inv(P[i])
    
    if doReWtopt:
        # res=optWts1(X,MU,P,W)
        res=optWts1(X,MU,P,W)
        W=res.x
        clf.weights_=res.x
    

    res={'clf':clf,'success':True}
    return res

#%%


def getgridvec(thset,txset,tyset):
    th,x,y=np.meshgrid(thset,txset,tyset)
    Posegrid = np.vstack([th.ravel(), x.ravel(), y.ravel()]).T
    Posegrid=np.ascontiguousarray(Posegrid,dtype=dtype)
    return Posegrid


#%%



def gridsearch_alignment_opt(Posegrid,MU,P,W,X):
    

    fres=None
    M=1e10
    for i in range(Posegrid.shape[0]):
       res = minimize(nbpt2Dproc.getcostgradient, Posegrid[i],args=(X,MU,P,W),jac=True ,method='BFGS', tol=1e-4) # 'Nelder-Mead'            
       if res.fun < M and res.success:
           M=res.fun
           fres= res

    th=fres.x[0]
    t=fres.x[1:]
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]],dtype=dtype,order='C')

    kHs=np.hstack([R,t.reshape(2,1)])
    kHs = np.vstack([kHs,[0,0,1]])
    sHk = nplinalg.inv(kHs)
    err=fres.fun
    
    # hess_inv is like covariance
    return sHk,err, fres.hess_inv

# @jit(numba.types.Tuple((float64[:,:],float64[:],float64[:,:]))(float64[:], float64[:,::1], float64[:,:], float64[:,:,:],float64[:]),nopython=True, nogil=True,parallel=False,cache=True) 
def gridsearch_alignment_brute(Posegrid,MU,P,W,X):
    

    fres=None
    M=1e10

    fres=nbpt2Dproc.gridsearch_alignment(Posegrid,MU,P,W,X)

    fres = minimize(nbpt2Dproc.getcostgradient, fres,args=(X,MU,P,W),jac=True ,method='BFGS', tol=1e-4)
  
    th=fres.x[0]
    t=fres.x[1:]
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]],dtype=dtype,order='C')

    kHs=np.hstack([R,t.reshape(2,1)])
    kHs = np.vstack([kHs,[0,0,1]])
    sHk = nplinalg.inv(kHs)
    err=fres.fun
    
    # hess_inv is like covariance
    return sHk,err, fres.hess_inv


#%%
def alignscan2keyframe(MU,P,W,X):
    best=np.zeros(3)
    # Posegrid = getgridvec(thset,txset,tyset)
    # best=gridsearch_alignment(Posegrid,MU,P,W,X)
    # print("best = ",best) 
    
    res = minimize(nbpt2Dproc.getcostgradient, best,args=(X,MU,P,W),jac=True ,method='BFGS', tol=1e-2) # 'Nelder-Mead'
    
    
    th=res.x[0]
    t=res.x[1:]
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]],dtype=dtype,order='C')

    kHs=np.hstack([R,t.reshape(2,1)])
    kHs = np.vstack([kHs,[0,0,1]])
    sHk = nplinalg.inv(kHs)
    err=res.fun
    hess_inv=res.hess_inv
    
    # J=res.jac
    # print(J)
    # hess_inv = nplinalg.inv(J.T.dot(J))
    # hess_inv is like covariance
    return sHk,err, hess_inv

def scan2keyframe_bin_match(X1,X2,dstop,H21):
    # X1 and X2 have to be in the scan frames not global frames
    # X1 is fixed and X2 is moved into X1
    # dstop=[dx,dy,dth]
    H12 = nplinalg.inv(H21)
    Xk=np.matmul(H12,np.vstack([X2.T,np.ones(X2.shape[0])])).T  
    X22=Xk[:,:2]
    
    
    
    

    
def scan2keyframe_match(KeyFrameClf,Xclf,X,params,sHk=np.identity(3),method='gmm'):
    # sHrelk is the transformation from k to s
    
    if method=='icp':
        XX=binnerDownSampler(X,dx=0.1,cntThres=1)
        sHk_corrected, distances, i=icp.icp(Xclf, XX, init_pose=sHk, max_iterations=500, tolerance=0.01)
        err=0
        hess_inv=np.identity(3)
        
    if method=='binmatch':
        

        Lmax=np.array([0.5,0.5])
        thmax=20*np.pi/180
        dxMatch=np.array([0.05,0.05])
        thmin=1*np.pi/180
        
        XX=binnerDownSampler(X,dx=dxMatch[0],cntThres=1)
        
        kHs = nplinalg.inv(sHk)
        sHk_corrected=nbpt2Dproc.binMatcherAdaptive3(Xclf,XX,kHs,Lmax,thmax,thmin,dxMatch)
        err=0
        hess_inv=np.identity(3)
        
    if method=='gmm':        
        kHs = nplinalg.inv(sHk)
        Xk=np.matmul(kHs,np.vstack([X.T,np.ones(X.shape[0])])).T  
        Xd=Xk[:,:2]
        # Xd=Xk-m_clf
        MU=np.ascontiguousarray(KeyFrameClf.means_,dtype=dtype)
        P=np.ascontiguousarray(KeyFrameClf.covariances_,dtype=dtype)
        W=np.ascontiguousarray(KeyFrameClf.weights_,dtype=dtype)
        Xd = np.ascontiguousarray(Xd,dtype=dtype) 
        
        sHk_rel,err,hess_inv = alignscan2keyframe(MU,P,W,Xd)
        
        xy_hess_inv_eigval,xy_hess_inv_eigvec = nplinalg.eig( hess_inv[1:,1:] )
        th_hess_inv = hess_inv[0,0]
        txy,theta = nbpt2Dproc.extractPosAngle(sHk_rel)
        
        sHk_corrected=np.matmul(sHk,sHk_rel)
        
    return sHk_corrected,err,hess_inv




def getEdges(Xb,dx):
    mn=np.zeros(2)
    mn[0] = np.min(Xb[:,0])
    mn[1] = np.min(Xb[:,1])
    
    mx=np.zeros(2)
    mx[0] = np.max(Xb[:,0])
    mx[1] = np.max(Xb[:,1])
    
    mn = mn-0.5
    mx = mx+0.5

    xedges = np.arange(mn[0],mx[0],dx[0])
    yedges = np.arange(mn[1],mx[1],dx[1])
    
    return xedges,yedges

def Commonbins(P,Xn,xedges,yedges,cntThres):
    H=nbpt2Dproc.numba_histogram2D(Xn, xedges,yedges)
    
    nx = len(xedges) 
    ny = len(yedges) 
    
    
    # Pn is the probability of scan points in the keyframe histogram
    Pn=np.zeros((nx-1,ny-1),dtype=np.int32)
    
    
    Pn[H>=cntThres]=1

    dand = np.logical_and(P,Pn)
    dor = np.logical_or(P,Pn)
    # dd = d.astype(np.int32)
    
    Fand = np.sum(dand.reshape(-1))
    For = np.sum(dor.reshape(-1))
    return Fand,For,Pn

#%%
def eval_posematch(H21,X2,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp):
    H12 = nplinalg.inv(H21)
    Xn=np.matmul(H12,np.vstack([X2.T,np.ones(X2.shape[0])])).T  
    Xn=Xn[:,:2]
    
    mbin_and,mbin_or,Hist2_ovrlp=Commonbins(Hist1_ovrlp,Xn,xedges_ovrlp,yedges_ovrlp,1)
    activebins2_ovrlp = np.sum(Hist2_ovrlp.reshape(-1))
    
    
    mbinfrac_ovrlp=mbin_and/mbin_or
    
    
    # mbinfrac_ActiveOvrlp = np.max([mbin_and/activebins1_ovrlp,mbin_and/max([1,activebins2_ovrlp])])
    mbinfrac_ActiveOvrlp = mbin_and/activebins1_ovrlp
    # print("\n",mbinfrac_ActiveOvrlp,mbin_and,activebins1_ovrlp,activebins2_ovrlp)
        
    posematch={'activebins1_ovrlp':activebins1_ovrlp,'mbin':mbin_and,'mbinfrac':mbinfrac_ovrlp,'mbinfrac_ActiveOvrlp':mbinfrac_ActiveOvrlp}
    
    return posematch





class CostAndNode:
    def __init__(self, cost, node):
        self.cost = cost
        self.node = node

    # do not compare nodes
    def __lt__(self, other):
        return self.cost < other.cost
    


def getPointCost(H,dx,X,Oj,Tj):
    # Tj is the 2D index of displacement
    # X are the points
    # dx is 2D
    # H is the probability histogram
    
    P=np.floor(X/dx)
    j=np.floor(Oj/dx)
    Pn=P+j
    
    idx=np.all(np.logical_and(Pn>=np.zeros(2) , Pn<H.shape),axis=1 )
    Pn=Pn[idx].astype(int)
    c=0
    if Pn.size>0:
        c=np.sum(H[Pn[:,0],Pn[:,1]])
         
    return c



def UpsampleMax(Hup,n):
    H=np.zeros((int(np.ceil(Hup.shape[0]/2)),int(np.ceil(Hup.shape[1]/2))))
    for j in range(H.shape[0]):
        for k in range(H.shape[1]):
            lbx=max([2*j,0])
            ubx=min([2*j+n,Hup.shape[0]-1])+1
            lby=max([2*k,0])
            uby=min([2*k+n,Hup.shape[1]-1])+1
            H[j,k] = np.max( Hup[lbx:ubx,lby:uby] )
    return H


def pool2d(A, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')

    # Window view of A
    output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                    (A.shape[1] - kernel_size)//stride + 1)
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(A, shape = output_shape + kernel_size, 
                        strides = (stride*A.strides[0],
                                   stride*A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=(1,2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1,2)).reshape(output_shape)
# H=pool2d(Hup, kernel_size=3, stride=2, padding=0, pool_mode='max')

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
        
    
    H1match=nbpt2Dproc.numba_histogram2D(X1, xedges,yedges)
    
    
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

     
    
    
    SolBoxes_init=[]
    Lmax=dxs[0]*(np.floor(Lmax/dxs[0])+1)
    for xs in np.arange(-Lmax[0],Lmax[0]+1*dxs[0][0],dxs[0][0]):
        for ys in np.arange(-Lmax[1],Lmax[1]+1*dxs[0][1],dxs[0][1]):
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
    
    thfineRes = 2.5*np.pi/180
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
    H[0:2,2]=t
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

def binMatcherAdaptive3(X11,X22,H12,Lmax,thmax,dxMatch):
    n=histsmudge =2 # how much overlap when computing max over adjacent hist for levels
        
        
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
    
    # print("mn_orig = ",mn_orig)
    
    mn[0] = np.min(X1[:,0])
    mn[1] = np.min(X1[:,1])
    mx[0] = np.max(X1[:,0])
    mx[1] = np.max(X1[:,1])
    rmax=np.max(np.sqrt(X2[:,0]**2+X2[:,1]**2))
    
    
    # print("mn,mx=",mn,mx)
    P = mx-mn
    
    
    
    
    # dxMax[0] = np.min([dxMax[0],Lmax[0]/2,P[0]/2])
    # dxMax[1] = np.min([dxMax[1],Lmax[1]/2,P[1]/2])
    
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
    
    # dxs=dxs[::-1]
    dxs=[dx.astype(np.float32) for dx in dxs]
    # XYedges=XYedges[::-1]
    
    
    H1match=nbpt2Dproc.numba_histogram2D(X1, XYedges[-1][0],XYedges[-1][1])
    H1match = np.sign(H1match)
    
    
    
    
    # first create multilevel histograms
    levels=[]
    HLevels=[H1match]
    
    
    
    
    
    for i in range(1,mxlvl):
        
    
    
        Hup = HLevels[i-1]
        # H=pool2d(Hup, kernel_size=3, stride=2, padding=1, pool_mode='max')
        H=nbpt2Dproc.UpsampleMax(Hup,n)
        
    
    
        # pt2dproc.plotbins2(XYedges[i][0],XYedges[i][1],H,X1,X2)
    
        HLevels.append(H)
          
    
    mxLVL=len(HLevels)-1
    HLevels=HLevels[::-1]
    HLevels=[np.ascontiguousarray(H).astype(np.int32) for H in HLevels]
    
    # for i in range(mxlvl):
    #     H=HLevels[i]
    #     pt2dproc.plotbins2(XYedges[i][0],XYedges[i][1],H,X1,X2)
        
        
    LmaxOrig=Lmax.copy()
    # et=time.time()
    # print("Time pre-init = ",et-st)
    SolBoxes_init=[]
    X2=X2-Lmax
    Lmax=dxs[0]*(np.floor(Lmax/dxs[0])+1)
    for xs in np.arange(0,2*Lmax[0],dxs[0][0]):
        for ys in np.arange(0,2*Lmax[1],dxs[0][1]):
            SolBoxes_init.append( (xs,ys,dxs[0][0],dxs[0][1]) )
    
    
    
    
    
    h=[(100000.0,0.0,0.0,0.0,0.0,0.0,0.0)]
    #Initialize with all thetas fixed at Max resolution
    lvl=0
    dx=dxs[lvl]
    H=HLevels[lvl]
    
    Xth= Dict.empty(
        key_type=types.float64,
        value_type=float_2Darray,
    )
    ii=0
    thfineRes = 2*np.pi/180
    thL=np.arange(-thmax,thmax+thfineRes,thfineRes,dtype=np.float32)
    # thL=[0]
    # np.random.shuffle(thL)
    for th in thL:
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
    

            
            # cost2=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj)

                
            # ii+=1                
            # heapq.heappush(h,(-(cost2+np.random.rand()/1000),[solbox,lvl,th]))

      

    # st=time.time()
    while(1):
        # print(len(h))
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
    

    
    Hcomp=np.identity(3)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    Hcomp[0:2,0:2]=R
    Hcomp[0:2,2]=t
    
    # t=solboxt[0]
    # print(t,th)
    # H=np.identity(3)
    # R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    # H[0:2,0:2]=R
    # H[0:2,2]=t-R.dot(LmaxOrig)
    # Htotal12 = np.matmul(H,H12)
    # RT=Htotal12[0:2,0:2]
    # tT=Htotal12[0:2,2]
    
    
    
    H1=np.array([[1,0,-mn_orig[0]],[0,1,-mn_orig[1]],[0,0,1]])
    Hlmax=np.array([[1,0,-LmaxOrig[0]],[0,1,-LmaxOrig[1]],[0,0,1]])
    
    H2=nplinalg.inv(H1).dot(Hcomp)
    H3=H2.dot(Hlmax)
    H4=H3.dot(H1)
    H12comp=H4.dot(H12)
    # H12comp=nplinalg.multi_dot([nplinalg.inv(H1),Hcomp,Hlmax,H1,H12])
    H21comp=nplinalg.inv(H12comp)
    
    return H21comp
    
    
    
def binMatcherAdaptive(X11,X22,H12,Lmax,thmax,dxMatch):
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
    
    
    # dxMax[0] = np.min([dxMax[0],Lmax[0]/2,P[0]/2])
    # dxMax[1] = np.min([dxMax[1],Lmax[1]/2,P[1]/2])
    dxMax=P
    print("P=",P)
    nnx=np.ceil(np.log2(P[0]))
    nny=np.ceil(np.log2(P[1]))
    
    xedges=np.arange(mn[0]-dxMatch[0],mx[0]+dxMatch[0],dxMatch[0])
    yedges=np.arange(mn[1]-dxMatch[0],mx[1]+dxMatch[0],dxMatch[1])
    
    if len(xedges)%2==0:
        xedges=np.hstack([xedges,xedges[-1]+1*dxMatch[0]])
    if len(yedges)%2==0:
        yedges=np.hstack([yedges,yedges[-1]+1*dxMatch[1]])
        
    
    H1match=nbpt2Dproc.numba_histogram2D(X1, xedges,yedges)
    
    H1match[H1match>0]=1
    # H1match = np.sign(H1match)
    # H1match=100*H1match/(np.sum(H1match.reshape(-1))*np.prod(dxMatch))
    
    
    
    
    # first create multilevel histograms
    levels=[]
    HLevels=[H1match]
    dxs = [dxMatch]
    XYedges=[(xedges,yedges)]


    flg=False
    # st=time.time()
    for i in range(1,100):
        
        dx=2*dxs[i-1]
        if np.any(dx>dxMax):
            flg=True
        
        Hup = HLevels[i-1]
        # H=pool2d(Hup, kernel_size=3, stride=2, padding=1, pool_mode='max')
        H=nbpt2Dproc.UpsampleMax(Hup,n)
        
        # print(xedges[0],xedges[-1],len(xedges),yedges[0],yedges[-1],len(yedges))
        lx =xedges[-1]
        ly =yedges[-1]
        xedges=xedges[::2]
        yedges=yedges[::2]
        
        # print(xedges[0],xedges[-1],len(xedges),yedges[0],yedges[-1],len(yedges))
        # print("-------------")
        plotbins2(xedges,yedges,H,X1,X2)
        
        XYedges.append((xedges,yedges))  
        if len(xedges)%2==0:
            xedges=np.hstack([xedges,lx])
        if len(yedges)%2==0:
            yedges=np.hstack([yedges,ly])
        
        HLevels.append(H)
        dxs.append(dx)
          

        if flg:
            break
    HLevels=HLevels[::-1]
    dxs=dxs[::-1]
    dxs=[dx.astype(np.float32) for dx in dxs]
    HLevels=[np.ascontiguousarray(H).astype(np.int32) for H in HLevels]
    XYedges=XYedges[::-1]
     
    # et=time.time()
    # print("Time pre-init = ",et-st)
    SolBoxes_init=[]
    print(Lmax,dxs[0])
    Lmax=dxs[0]*(np.floor(Lmax/dxs[0])+1)
    for xs in np.arange(-Lmax[0],Lmax[0],dxs[0][0]):
        for ys in np.arange(-Lmax[1],Lmax[1],dxs[0][1]):
            SolBoxes_init.append( (np.array([xs,ys],dtype=np.float32),dxs[0]) )
    
    # print(SolBoxes_init)
    mxLVL=len(HLevels)-1
    
    # st=time.time()
    decimatedict={}
    h=[]
    #Initialize with all thetas fixed at Max resolution
    lvl=0
    dx=dxs[lvl]
    H=HLevels[lvl]

    Xth={}
    ii=0
    thfineRes = 1*np.pi/180
    # thL=np.arange(-thmax,thmax+thfineRes,thfineRes,dtype=np.float32)
    thL=[0]
    # np.random.shuffle(thL)
    for th in thL:
        R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
        XX=np.transpose(R.dot(X2.T))
        Xth[th]=XX
        
        
        for solbox in SolBoxes_init:
            Tj=solbox[1]
            Oj = solbox[0]
            if not np.all(Tj==dx):
                print(Tj,dx)
                raise Exception("Tj and dx are not equal ")
                
            fig,ax=plotbins2(XYedges[lvl][0],XYedges[lvl][1],HLevels[lvl],X1,Xth[th]+solbox[0])
            
            cost2=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj)
            ax.set_title("lvl=%d,cost=%d, th=%1.2f, %s %s"%(0,cost2,th,str(list(solbox[0])),str(list(solbox[1]))))
            # h.append(CostAndNode(-cost2,[solbox,lvl,th]))
            # if ii==57:
                # plotbins2(XYedges[lvl][0],XYedges[lvl][1],HLevels[lvl],X1,Xth[th],title="original")
            # plotbins2(XYedges[lvl][0],XYedges[lvl][1],HLevels[lvl],X1,Xth[th]+Oj,title=str(cost2)+" "+str(Oj))
                # X=Xth[th]
                # Pn=np.floor((X+Oj)/dx)
                # # j=np.floor(Oj/dx)
                # # Pn=P+j
                # print(dx)
                # # print(P)
                # # print(Oj,j)
                # print(Pn)
                # idx1=np.logical_and(Pn[:,0]>=0,Pn[:,0]<H.shape[0])
                # idx2=np.logical_and(Pn[:,1]>=0,Pn[:,1]<H.shape[1])
                # idx=np.logical_and(idx1,idx2 )
                # # idx=np.all(np.logical_and(Pn>=np.zeros(2) , Pn<H.shape),axis=1 )
                # Pn=Pn[idx].astype(int)
                # print("H=",H[Pn[:,0],Pn[:,1]],"cost2 = ",cost2)
                
            ii+=1                
            heapq.heappush(h,(-(cost2+np.random.rand()/1000),[solbox,lvl,th]))
    

    print(len(h))
  
    cnt=0
    # st=time.time()
    while(1):
        # print(len(h))
        (cost,[solboxt,lvl,th])=heapq.heappop(h)
        
        fig,ax=plotbins2(XYedges[lvl][0],XYedges[lvl][1],HLevels[lvl],X1,Xth[th]+solboxt[0])
        ax.set_title("lvl=%d,cost=%d, th=%1.2f, %s %s"%(lvl,cost,th,str(list(solboxt[0])),str(list(solboxt[1]))))
        plt.savefig("debugPlots/%05d"%cnt)
        plt.close(fig)
        cnt+=1
        # (cost,[solboxt,lvl,th])=(CN.cost,CN.node)
        # print(cost,lvl,len(h),np.round(th*180/np.pi,2),solboxt)
        # if lvl>=mxLVL:
        #     continue
        if lvl==mxLVL:
            print("done")
            break
        
        
        dx=dxs[lvl+1]
        H=HLevels[lvl+1]
        Tj=solboxt[1]
        Oj=solboxt[0]
        S=[]

        Xg=np.arange(Oj[0],Oj[0]+Tj[0],dx[0])
        Yg=np.arange(Oj[1],Oj[1]+Tj[1],dx[1])
        for xs in Xg:
            for ys in Yg:
                S.append( (np.array([xs,ys]),dx) )
        

        for solbox in S:
            Tj=solbox[1]
            Oj = solbox[0]
            if not np.all(Tj==dx):
                print(Tj,dx)
                raise Exception("Tj and dx are not equal ")            
            cost3=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj) 

                
            # print(lvl,cost,lvl+1,-cost3)
            heapq.heappush(h,(-(cost3+np.random.rand()/1000),[solbox,lvl+1,th]))

          
    # et=time.time()
    # print("time for final run heap push pop = ", et-st)

    
    t=solboxt[0]
    print(t,th)
    H=np.identity(3)
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    H[0:2,0:2]=R
    H[0:2,2]=t
    Htotal12 = np.matmul(H,H12)
    RT=Htotal12[0:2,0:2]
    tT=Htotal12[0:2,2]
    
    Rs=H[0:2,0:2]
    ts=H[0:2,2]
    
    t = tT-(Rs.dot(mn_orig)+0*ts)+mn_orig
    Htotal12_updt=Htotal12.copy()
    Htotal12_updt[0:2,2]=t
    Htotal21_updt = nplinalg.inv(Htotal12_updt)
    return Htotal21_updt,cost,HLevels,dxs



def getCombinedNode(poseGraph,idx,nn,params,Doclf=True):
    Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    G=poseGraph.subgraph(Lkey[Lkey.index(idx)-nn:Lkey.index(idx)+nn])
    # Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    L=list(filter(lambda x: idx in x,list(nx.simple_cycles(G))))
    L=list(filter(lambda x: idx in x[int(len(x)/3):int(2*len(x)/3)],L))
    L=sorted(L,key=lambda x:len(x))
    if len(L)==0:
        print("bruteforce side combine")
        pii=idx
        lp=[idx]

        for ii in Lkey[Lkey.index(idx)+1:Lkey.index(idx)+nn]:
            if (pii,ii) in poseGraph.edges:
                if poseGraph.edges[pii,ii]['posematch']['mbinfrac_ActiveOvrlp']>=params["Side_Combine_Overlap"]:
                    lp.append(ii)
            pii=ii
        pii=idx
        for ii in Lkey[Lkey.index(idx)-nn:Lkey.index(idx)][::-1]:
            if (ii,pii) in poseGraph.edges:
                if poseGraph.edges[ii,pii]['posematch']['mbinfrac_ActiveOvrlp']>=params["Side_Combine_Overlap"]:
                    lp.append(ii)
            pii=ii
        lp=list(set(lp))
            
        
    else:
        lp=L[-1]
    # for lp in L:
    X=[poseGraph.nodes[idx]['X']]
    iHg=poseGraph.nodes[idx]['sHg']
    for jj in lp:
        if jj!=idx:
            jHg = poseGraph.nodes[jj]['sHg']
            iHj=np.matmul(iHg,nplinalg.inv(jHg))
          
            Xj=poseGraph.nodes[jj]['X']
            
            XX=np.matmul(iHj,np.vstack([Xj.T,np.ones(Xj.shape[0])])).T 
            X.append(XX[:,:2])
            
    
    
    X=binnerDownSamplerProbs(X,dx=params['BinDownSampleKeyFrame_dx'],prob=params['BinDownSampleKeyFrame_probs'])
    X=np.ascontiguousarray(X,dtype=dtype)

    if Doclf:
        res = getclf(X,params,doReWtopt=True)
        clf = res['clf']
    else:
        clf = None
        
    return X,clf

def poseGraph_keyFrame_matcher_long(poseGraph,idx1,idx2,params,PoseGrid,isPoseGridOffset,isBruteForce,histplot=False):
    # fromidx is idx1, toidx is idx2 
    # X1,clf1=getCombinedNode(poseGraph,idx1,5,params,Doclf=True)
    # X2,_=getCombinedNode(poseGraph,idx2,5,params,Doclf=False)
    
    if params["USE_Side_Combine"]:
        clf1=poseGraph.nodes[idx1]['clflc']
        X1 = poseGraph.nodes[idx1]['Xlc']
        X2 = poseGraph.nodes[idx2]['Xlc']
    else:
        clf1=poseGraph.nodes[idx1]['clf']
        X1 = poseGraph.nodes[idx1]['X']
        X2 = poseGraph.nodes[idx2]['X']
    
    sHg_1 = poseGraph.nodes[idx1]['sHg']
    sHg_2 = poseGraph.nodes[idx2]['sHg']
    
    H21_est = np.matmul(sHg_2,nplinalg.inv(sHg_1))

    
    
    
    
    dxcomp = params['LOOPCLOSE_BIN_MIN_FRAC_dx']
    Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc.binScanEdges(X1,X2,dxcomp)
    activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))
    

    if PoseGrid is None:
        H21,err,hess_inv=scan2keyframe_match(clf1,X1,X2,params,sHk=H21_est)
        posematch=eval_posematch(H21,X2,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
        posematch['H']=H21
        posematch['err']=err
        posematch['hess_inv']=hess_inv
    else:
        # print("-----------------Inside-----------------")
        
        if isPoseGridOffset:
            t,th=nbpt2Dproc.extractPosAngle(H21_est) 
            PoseGrid[:,0]+= th
            PoseGrid[:,1]+= t[0]
            PoseGrid[:,2]+= t[1]
        # elif isBruteForce:
        #     m1 = np.mean(X1,axis=0)
        #     m2 = np.mean(X2,axis=0)
        #     v = m1-m2
        #     thset = np.linspace(0,2*np.pi,PoseGrid[0])
        #     txset = np.linspace(xedges_ovrlp[0],xedges_ovrlp[-1],PoseGrid[1])+v[0]
        #     tyset = np.linspace(yedges_ovrlp[0],yedges_ovrlp[-1],PoseGrid[2])+v[1]
        #     PoseGrid=getgridvec(thset,txset,tyset)
        

        M=-1
        posematch=None
        for i in range(PoseGrid.shape[0]):
            th=PoseGrid[i,0]
            t=np.array([PoseGrid[i,1],PoseGrid[i,2]])
            HH = nbpt2Dproc.getHmat(th,t) 
            H21,err,hess_inv=scan2keyframe_match(clf1,X1,X2,params,sHk=HH)
            
            posematch2=eval_posematch(H21,X2,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
            if posematch2['mbinfrac_ActiveOvrlp']>params['LOOPCLOSE_BIN_MAXOVRL_FRAC_COMPLETE']:
                posematch=copy.deepcopy(posematch2)
                posematch['H']=H21
                posematch['err']=err
                posematch['hess_inv']=hess_inv
                break
            
            if posematch2['mbinfrac_ActiveOvrlp']>M:
                posematch=copy.deepcopy(posematch2)
                posematch['H']=H21
                posematch['err']=err
                posematch['hess_inv']=hess_inv
                M = posematch2['mbinfrac_ActiveOvrlp']
  
    if histplot:
        plotposematch(xedges_ovrlp,yedges_ovrlp,Hist1_ovrlp,X1,X2,H21,params)
    return posematch

def poseGraph_keyFrame_matcher_binmatch(poseGraph,idx1,idx2,params,dx0=1.5,L0=10,th0=np.pi/3,DoCLFmatch=False,PoseGrid=None,isPoseGridOffset=True,isBruteForce=False,H21_est=None):
    if H21_est is None:
        sHg_1 = poseGraph.nodes[idx1]['sHg']
        sHg_2 = poseGraph.nodes[idx2]['sHg']
        
        H21_est = np.matmul(sHg_2,nplinalg.inv(sHg_1))
        
    # H21_est = np.identity(3)
    # H21_est[0,2]=2
    X1 = poseGraph.nodes[idx1]['X']
    X2 = poseGraph.nodes[idx2]['X']
    
    H12_est = nplinalg.inv(H21_est)
    X=np.matmul(H12_est,np.vstack([X2.T,np.ones(X2.shape[0])])).T  
    X22=X[:,:2]
    
    dxcomp = params['LOOPCLOSE_BIN_MIN_FRAC_dx']
    

    Lmax=np.array([10,10])
    thmax=45*np.pi/180
    dxMatch=np.array([0.1,0.1])
    thmin=1*np.pi/180

    H21_corrected=nbpt2Dproc.binMatcherAdaptive3(X1,X2,H12_est,Lmax,thmax,thmin,dxMatch)
    
    
    # print("H21_est=",H21_est)
    
    # flg=False
    # H21_corrected = H21_est
    # # print(r,dth,dxx)
    # for i in range(9):
    #     # print(i)
    #     if i==0:
    #         dx=np.array([dx0,dx0],dtype=np.float64)
    #         Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc.binScanEdges(X1,X2,dx)
    #         activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))    
        
    #         dxx = 0.9*dx
    #         r = 0.5*nplinalg.norm([np.max(xedges_ovrlp)-np.min(xedges_ovrlp),np.max(yedges_ovrlp)-np.min(yedges_ovrlp)])
    #         L=L0
            
    #         th0=th0
    #         dth = 1*np.max(dx)/r
            
            
        
    #     else:
    #         dx = dx*(0.5**i)
    #         if np.all(dx<=dxcomp):
    #             dx=0.9*dxcomp
    #             flg=True
    #         Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc.binScanEdges(X1,X2,dx)
    #         activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))    
        
    #         L=3*np.max(dx)
    #         dxx = 0.9*dx
    #         r = 0.5*nplinalg.norm([np.max(xedges_ovrlp)-np.min(xedges_ovrlp),np.max(yedges_ovrlp)-np.min(yedges_ovrlp)])
            
    #         dth = 1*np.max(dx)/r
    #         th0=5*dth
            
        
    #     # print("activebins1_ovrlp=",activebins1_ovrlp)
                
    #     thset = np.arange(-th0,th0+dth,dth)
    #     txset = np.arange(-L,L+dxx[0],dxx[0]) # 
    #     tyset = np.arange(-L,L+dxx[1],dxx[1]) #
    #     PoseGrid=getgridvec(thset,txset,tyset)
        
    #     # print("PoseGrid.shape = ",PoseGrid.shape)
    #     pose,mbinfrac_ActiveOvrlp = nbpt2Dproc.binScanMatcher(PoseGrid,Hist1_ovrlp,X22,xedges_ovrlp,yedges_ovrlp,1)
    #     # print("mbinfrac_ActiveOvrlp=",mbinfrac_ActiveOvrlp)
        
    #     H12 = nbpt2Dproc.getHmat(pose[0],pose[1:]) 
    #     H21=nplinalg.inv(H12)

    #     H21_corrected=np.matmul(H21_corrected,H21)
    #     H12_corrected=nplinalg.inv(H21_corrected)
    #     # print("H21_corrected=",H21_corrected)
        
    #     if flg:
    #         break
    #     X=np.matmul(H12_corrected,np.vstack([X2.T,np.ones(X2.shape[0])])).T  
    #     X22=X[:,:2]
        
        
        
    Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc.binScanEdges(X1,X2,dxcomp)
    activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))    
            
    if DoCLFmatch:
        clf1=poseGraph.nodes[idx1]['clf']
        H21_corrected,err,hess_inv=scan2keyframe_match(clf1,X1,X2,params,sHk=H21_corrected)
        
        posematch=eval_posematch(H21_corrected,X2,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
        posematch['H']=H21_corrected
        posematch['err']=err
        posematch['hess_inv']=hess_inv
    else:
        posematch=eval_posematch(H21_corrected,X2,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
        posematch['H']=H21_corrected
        posematch['err']=1
        posematch['hess_inv']=np.identity(3)
        
    # H21_corrected=np.matmul(H21_est,H21)

    # print('H21_corrected=',H21_corrected)
    # plotposematch(xedges_ovrlp,yedges_ovrlp,Hist1_ovrlp,X1,X2,H21_est,params)
    # plotposematch(xedges_ovrlp,yedges_ovrlp,Hist1_ovrlp,X1,X2,H21_corrected,params)
    # pt2dplot.plotcomparisons_points(X1,X2,idx1,idx2,clf1,UseLC=False,H21_est=H21_est,H12=nplinalg.inv( posematch['H']) ,err=posematch['mbinfrac_ActiveOvrlp'])
    
    # print('posematch[H]=',posematch['H'])
    return posematch

def plotposematch(xedges,yedges,Hist1,X1,X2,H21,params):
    H12 = nplinalg.inv(H21)
    Xn=np.matmul(H12,np.vstack([X2.T,np.ones(X2.shape[0])])).T  
    Xn=Xn[:,:2]
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.pcolormesh(xedges,yedges,Hist1.T,shading='flat',alpha=0.4 )
    ax.plot(X1[:,0],X1[:,1],'r.')
    ax.plot(Xn[:,0],Xn[:,1],'bo')
    ax.axis('equal')
    
    plt.show()    
    
    
    
def plotbins(xedges,yedges,Hist1,idx1,idx2,poseGraph,params,posematch=None):
    if posematch is None:
        posematch = poseGraph.nodes[idx1]['posematch']
        
    X1 = poseGraph.nodes[idx1]['X']
    X2 = poseGraph.nodes[idx2]['X']
    print(posematch)
    H21=posematch['H']
    H12 = nplinalg.inv(H21)
    Xn=np.matmul(H12,np.vstack([X2.T,np.ones(X2.shape[0])])).T  
    Xn=Xn[:,:2]
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.pcolormesh(xedges,yedges,Hist1.T,shading='flat',alpha=0.4 )
    ax.plot(X1[:,0],X1[:,1],'r.')
    ax.plot(Xn[:,0],Xn[:,1],'bo')
    ax.axis('equal')
    
    plt.show()    
    

def plotbins2(xedges,yedges,Hist1,X1,X2):

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.pcolormesh(xedges,yedges,Hist1.T,shading='flat',alpha=0.4 )
    ax.plot(X1[:,0],X1[:,1],'r.',label="hist points")
    ax.plot(X2[:,0],X2[:,1],'b.',label="matching points")
    ax.axis('equal')
    ax.legend()
    plt.show()    
    return fig,ax






#%% poseGraph manipulation

def addNewKeyFrame(poseGraph,KeyFrameClf,X_previdx,XXidx,idx,KeyFrame_prevIdx,sHk_idx,sHg_idx,sHk_prevframe,params,keepOtherScans=False):
    """
    idx : current keyframe id
    KeyFrame_prevIdx: previous keyframe id
    
    Find the mid frame (which should be a scan frame)
    
    Make combined X for KeyFrame_prevIdx, delete the scans of the previous to KeyFrame_prevIdx
    """
    KeyPrevIdxs_succ = list(poseGraph.successors(KeyFrame_prevIdx))
    KeyPrevIdxs_succ = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="scan",KeyPrevIdxs_succ))
    1
    KeyPrevPrev = list(poseGraph.predecessors(KeyFrame_prevIdx))
    KeyPrevPrev = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",KeyPrevPrev))
    # KeyPrevIdxs_pred = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="scan",KeyPrevIdxs_pred))
    
    if len(KeyPrevPrev)==0: # no prev prev keyframe
        KeyPrevPrev_idx=[]
        KeyPrevPrevIdxs_succ=[]
    else:
        KeyPrevPrev_idx = max(KeyPrevPrev)        
        KeyPrevPrevIdxs_succ = list(poseGraph.successors(KeyPrevPrev_idx))
        KeyPrevPrevIdxs_succ = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="scan",KeyPrevPrevIdxs_succ))
    
    # pdb.set_trace()
    sHg_prevIdx=poseGraph.nodes[KeyFrame_prevIdx]['sHg']
    
    scansCombinedprevprev=[]
    scansCombined=[]
    
    X= [poseGraph.nodes[KeyFrame_prevIdx]['X']]
    for ix in KeyPrevPrevIdxs_succ:
        posematch=poseGraph.edges[KeyPrevPrev_idx,ix]['posematch']
        if poseGraph.nodes[ix]['frametype']=='scan' and posematch['mbinfrac_ActiveOvrlp']>params["Scan2Key_Overlap"]:
            sHg=poseGraph.nodes[ix]['sHg']
            gHs = nplinalg.inv(sHg)
            H = np.matmul(sHg_prevIdx,gHs)
            XX=np.matmul(H,np.vstack([poseGraph.nodes[ix]['X'].T,np.ones(poseGraph.nodes[ix]['X'].shape[0])])).T  
            X.append(XX[:,:2])
            scansCombinedprevprev.append(ix)
            
    Xidx = [XXidx]
    for ix in KeyPrevIdxs_succ:
        posematch=poseGraph.edges[KeyFrame_prevIdx,ix]['posematch']
        if poseGraph.nodes[ix]['frametype']=='scan' and posematch['mbinfrac_ActiveOvrlp']>params["Scan2Key_Overlap"]:
            sHg=poseGraph.nodes[ix]['sHg']
            gHs = nplinalg.inv(sHg)
            H = np.matmul(sHg_prevIdx,gHs)
            XX=np.matmul(H,np.vstack([poseGraph.nodes[ix]['X'].T,np.ones(poseGraph.nodes[ix]['X'].shape[0])])).T  
            X.append(XX[:,:2])
            
            H = np.matmul(sHg_idx,gHs)
            XX=np.matmul(H,np.vstack([poseGraph.nodes[ix]['X'].T,np.ones(poseGraph.nodes[ix]['X'].shape[0])])).T  
            Xidx.append(XX[:,:2])
            
            scansCombined.append(ix)
    
    
    # Xprevidx=binnerDownSampler(np.vstack(X),dx=params['BinDownSampleKeyFrame_dx'],cntThres=1)
    Xprevidx=binnerDownSamplerProbs(X,dx=params['BinDownSampleKeyFrame_dx'],prob=params['BinDownSampleKeyFrame_probs'])
    poseGraph.nodes[KeyFrame_prevIdx]['X']=Xprevidx
    if 'SideScansCombine' in poseGraph.nodes[KeyFrame_prevIdx]:
        poseGraph.nodes[KeyFrame_prevIdx]['SideScansCombine']=poseGraph.nodes[KeyFrame_prevIdx]['SideScansCombine']+scansCombined
    else:
        poseGraph.nodes[KeyFrame_prevIdx]['SideScansCombine']=scansCombined+scansCombinedprevprev
        
        
        
    res = getclf(Xprevidx,params,doReWtopt=True,means_init=None)
    clf=res['clf']
    poseGraph.nodes[KeyFrame_prevIdx]['clf']=clf
    
    idbmx = params['INTER_DISTANCE_BINS_max']
    idbdx=params['INTER_DISTANCE_BINS_dx']
    # h=get2DptFeat(X,bins=np.arange(0,idbmx,idbdx))
    h=np.array([0,0])
    poseGraph.nodes[KeyFrame_prevIdx]['h']=h
    
    ## now add the idx node as a keyframe
    # Xidx=binnerDownSampler(np.vstack(Xidx),dx=params['BinDownSampleKeyFrame_dx'],cntThres=1)
    Xidx=binnerDownSamplerProbs(Xidx,dx=params['BinDownSampleKeyFrame_dx'],prob=params['BinDownSampleKeyFrame_probs'])
    res = getclf(Xidx,params,doReWtopt=True,means_init=None)
    clf=res['clf']    
    
    sHk,serrk,shessk_inv = scan2keyframe_match(KeyFrameClf,X_previdx,Xidx,params,sHk=sHk_prevframe)
    
    kHg = poseGraph.nodes[KeyFrame_prevIdx]['sHg'] #global pose to the prev keyframe
    sHg = np.matmul(sHk,kHg) # global pose to the current frame: global to current frame
    gHs=nplinalg.inv(sHg) 
       
    tpos=np.matmul(gHs,np.array([0,0,1])) 

    idbmx = params['INTER_DISTANCE_BINS_max']
    idbdx=params['INTER_DISTANCE_BINS_dx']
    # h=get2DptFeat(X,bins=np.arange(0,idbmx,idbdx))
    h=np.array([0,0])
    poseGraph.add_node(idx,frametype="keyframe",X=Xidx,clf=clf,time=idx,sHg=sHg_idx,pos=(tpos[0],tpos[1]),h=h,color='g',LoopDetectDone=False,SideScansCombine=scansCombined)
    
    poseGraph.add_edge(KeyFrame_prevIdx,idx,H=sHk,H_prevframe=sHk_idx,err=serrk,hess_inv=shessk_inv,edgetype="Key2Key",color='k')
            
    

    # bin match key frame to keyframe
    dxcomp = params['LOOPCLOSE_BIN_MIN_FRAC_dx']
    Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc.binScanEdges(Xprevidx,Xidx,dxcomp)
    activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))
    H12=poseGraph.edges[KeyFrame_prevIdx,idx]['H']
    posematch=eval_posematch(H12,Xidx,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
    posematch['method']='GMMmatch'
    posematch["H"]=H12
    poseGraph.edges[KeyFrame_prevIdx,idx]['posematch']=posematch
    print("%d-%d-posematch['mbinfrac_ActiveOvrlp']="%(KeyFrame_prevIdx,idx),posematch['mbinfrac_ActiveOvrlp'])
    # if posematch['mbinfrac_ActiveOvrlp']<params["Key2Key_Overlap"]:
    #     # res = getclf(Xprevidx,params,doReWtopt=True,means_init=None)
    #     # clf=res['clf']
    #     # poseGraph.nodes[KeyFrame_prevIdx]['clf']=clf
        
    #     posematch = poseGraph_keyFrame_matcher_binmatch(poseGraph,KeyFrame_prevIdx,idx,params,DoCLFmatch=True,dx0=0.8,L0=5,th0=np.pi/4,PoseGrid=None,isPoseGridOffset=True,isBruteForce=False,H21_est=sHk_prevframe)
    #     posematch['method']='binmatch'
    #     print("%d-%d-posematch['mbinfrac_ActiveOvrlp']="%(KeyFrame_prevIdx,idx),posematch['mbinfrac_ActiveOvrlp'])
        
    #     poseGraph.edges[KeyFrame_prevIdx,idx]['H']=posematch['H']
    #     poseGraph.edges[KeyFrame_prevIdx,idx]['posematchGMM']=poseGraph.edges[KeyFrame_prevIdx,idx]['posematch']
    #     poseGraph.edges[KeyFrame_prevIdx,idx]['posematch']=posematch
        
    #     sHg = np.matmul(posematch['H'],kHg)
    #     gHs=nplinalg.inv(sHg)    
    #     tpos=np.matmul(gHs,np.array([0,0,1])) 
    #     poseGraph.nodes[idx]['pos']=(tpos[0],tpos[1])
    #     poseGraph.nodes[idx]['sHg']=sHg
        
    # delete rest of the scan-ids as they are useless
    if keepOtherScans is False:
        for ix in KeyPrevPrevIdxs_succ:
            if poseGraph.nodes[ix]['frametype']=='scan':
                poseGraph.remove_node(ix)



def addNewKeyFrameAndScan(poseGraph,KeyFrame_prevIdx,KeyFrame_newIdx,ScanFrame_idx,Xscan,Tscan,
                          params,timeMetrics,keepOtherScans=False,debugMode=False):
    """
    idx : current keyframe id
    KeyFrame_prevIdx: previous keyframe id
    
    Find the mid frame (which should be a scan frame)
    
    Make combined X for KeyFrame_prevIdx, delete the scans of the previous to KeyFrame_prevIdx
    """
    # args = copy.deepcopy([poseGraph,KeyFrame_prevIdx,KeyFrame_newIdx,ScanFrame_idx,Xscan,Tscan,
    #                       params])
    
    if params['DoSideCombineNewKeyframe']:
        KeyPrevIdxs_succ = list(poseGraph.successors(KeyFrame_prevIdx))
        KeyPrevIdxs_succ = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="scan",KeyPrevIdxs_succ))
        
        KeyPrevPrev = list(poseGraph.predecessors(KeyFrame_prevIdx))
        KeyPrevPrev = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",KeyPrevPrev))
        # KeyPrevIdxs_pred = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="scan",KeyPrevIdxs_pred))
        
        if len(KeyPrevPrev)==0: # no prev prev keyframe
            KeyPrevPrev_idx=[]
            KeyPrevPrevIdxs_succ=[]
        else:
            KeyPrevPrev_idx = max(KeyPrevPrev)        
            KeyPrevPrevIdxs_succ = list(poseGraph.successors(KeyPrevPrev_idx))
            KeyPrevPrevIdxs_succ = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="scan",KeyPrevPrevIdxs_succ))
        
        # pdb.set_trace()
        sHg_prevIdx=poseGraph.nodes[KeyFrame_prevIdx]['sHg']
        
        scansCombinedprevprev=[]
        scansCombined=[]
        
        X= [poseGraph.nodes[KeyFrame_prevIdx]['X']]
        for ix in KeyPrevPrevIdxs_succ:
            posematch=poseGraph.edges[KeyPrevPrev_idx,ix]['posematch']
            if poseGraph.nodes[ix]['frametype']=='scan' and posematch['mbinfrac_ActiveOvrlp']>params["Scan2Key_Overlap"]:
                sHg=poseGraph.nodes[ix]['sHg']
                gHs = nplinalg.inv(sHg)
                H = np.matmul(sHg_prevIdx,gHs)
                XX=np.matmul(H,np.vstack([poseGraph.nodes[ix]['X'].T,np.ones(poseGraph.nodes[ix]['X'].shape[0])])).T  
                X.append(XX[:,:2])
                scansCombinedprevprev.append(ix)
                
        X_newKeyidx = [poseGraph.nodes[KeyFrame_newIdx]['X']]
        sHg_newKeyidx=poseGraph.nodes[KeyFrame_newIdx]['sHg']
        for ix in KeyPrevIdxs_succ:
            posematch=poseGraph.edges[KeyFrame_prevIdx,ix]['posematch']
            if ix!=KeyFrame_newIdx and poseGraph.nodes[ix]['frametype']=='scan' and posematch['mbinfrac_ActiveOvrlp']>params["Scan2Key_Overlap"]:
                sHg=poseGraph.nodes[ix]['sHg']
                gHs = nplinalg.inv(sHg)
                H = np.matmul(sHg_prevIdx,gHs)
                XX=np.matmul(H,np.vstack([poseGraph.nodes[ix]['X'].T,np.ones(poseGraph.nodes[ix]['X'].shape[0])])).T  
                X.append(XX[:,:2])
                
                H = np.matmul(sHg_newKeyidx,gHs)
                XX=np.matmul(H,np.vstack([poseGraph.nodes[ix]['X'].T,np.ones(poseGraph.nodes[ix]['X'].shape[0])])).T  
                X_newKeyidx.append(XX[:,:2])
                
                scansCombined.append(ix)
        
        
        # Xprevidx=binnerDownSampler(np.vstack(X),dx=params['BinDownSampleKeyFrame_dx'],cntThres=1)
        Xprevidx=binnerDownSamplerProbs(X,dx=params['BinDownSampleKeyFrame_dx'],prob=params['BinDownSampleKeyFrame_probs'])    
        poseGraph.nodes[KeyFrame_prevIdx]['X']=Xprevidx
        poseGraph.nodes[KeyFrame_prevIdx]['clf']=None
        # res = getclf(Xprevidx,params,doReWtopt=True,means_init=None,prevclf=poseGraph.nodes[KeyFrame_prevIdx]['clf'])
        # clf=res['clf']
        # poseGraph.nodes[KeyFrame_prevIdx]['clf']=clf
        
        
        idbmx = params['INTER_DISTANCE_BINS_max']
        idbdx=params['INTER_DISTANCE_BINS_dx']
        # h=get2DptFeat(X,bins=np.arange(0,idbmx,idbdx))
        h=np.array([0,0])
        poseGraph.nodes[KeyFrame_prevIdx]['h']=h
        
        if 'SideScansCombine' in poseGraph.nodes[KeyFrame_prevIdx]:
            poseGraph.nodes[KeyFrame_prevIdx]['SideScansCombine']=poseGraph.nodes[KeyFrame_prevIdx]['SideScansCombine']+scansCombined
        else:
            poseGraph.nodes[KeyFrame_prevIdx]['SideScansCombine']=scansCombined+scansCombinedprevprev
            
        # new keyframe
        X_newKeyidx=binnerDownSamplerProbs(X_newKeyidx,dx=params['BinDownSampleKeyFrame_dx'],prob=params['BinDownSampleKeyFrame_probs'])
        poseGraph.nodes[KeyFrame_newIdx]['X']=X_newKeyidx
        poseGraph.nodes[KeyFrame_newIdx]['SideScansCombine']=scansCombined

        
        
        
        
    else:
        Xprevidx=poseGraph.nodes[KeyFrame_prevIdx]['X']    
        X_newKeyidx=poseGraph.nodes[KeyFrame_newIdx]['X']   
        
    res = getclf(X_newKeyidx,params,doReWtopt=True,means_init=None)
    clf=res['clf']        
    poseGraph.nodes[KeyFrame_newIdx]['clf']=clf
    
    poseGraph.nodes[KeyFrame_newIdx]['frametype']="keyframe"
    poseGraph.nodes[KeyFrame_newIdx]['color']="g"
    poseGraph.nodes[KeyFrame_newIdx]['LoopDetectDone']=False
    
    
    if debugMode:
        if 'Xorig' not in poseGraph.nodes[KeyFrame_prevIdx]:
            poseGraph.nodes[KeyFrame_prevIdx]['Xorig']=[]
        poseGraph.nodes[KeyFrame_prevIdx]['Xorig'].append(poseGraph.nodes[KeyFrame_prevIdx]['X'])

        if 'Xorig' not in poseGraph.nodes[KeyFrame_newIdx]:
            poseGraph.nodes[KeyFrame_newIdx]['Xorig']=[]
        poseGraph.nodes[KeyFrame_newIdx]['Xorig'].append(poseGraph.nodes[KeyFrame_newIdx]['X'])

    if 'debugMsg' not in poseGraph.nodes[KeyFrame_newIdx]:
        poseGraph.nodes[KeyFrame_newIdx]['debugMsg']=["node updated in addNewKeyFrameAndScan from scan to key"]
    
    poseGraph.edges[KeyFrame_prevIdx,KeyFrame_newIdx]['edgetype']="Key2Key"
    poseGraph.edges[KeyFrame_prevIdx,KeyFrame_newIdx]['color']="k"
    
    kHg = poseGraph.nodes[KeyFrame_prevIdx]['sHg'] #global pose to the prev keyframe
    sHk = poseGraph.edges[KeyFrame_prevIdx,KeyFrame_newIdx]['H']
    sHg = np.matmul(sHk,kHg) # global pose to the current frame: global to current frame
    gHs=nplinalg.inv(sHg) # current frame to global
    tpos=np.matmul(gHs,np.array([0,0,1]))
    
    poseGraph.nodes[KeyFrame_newIdx]['sHg']=sHg
    poseGraph.nodes[KeyFrame_newIdx]['pos']=(tpos[0],tpos[1])

    # now add the scan frame
    KeyFrameClf = poseGraph.nodes[KeyFrame_newIdx]['clf']
    Xclf = poseGraph.nodes[KeyFrame_newIdx]['X']
    sHk_prevframe = np.identity(3)
    
    sHk,serrk,shessk_inv = scan2keyframe_match(KeyFrameClf,Xclf,Xscan,params,sHk=sHk_prevframe,method=params['ScanMatchMethod'])
    
   
    
    dxcomp = params['LOOPCLOSE_BIN_MIN_FRAC_dx']
    Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc.binScanEdges(Xclf,Xscan,dxcomp)
    activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))
    posematch=eval_posematch(sHk,Xscan,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
    posematch['method']='GMMmatch'
    posematch["H"]=sHk
    posematch['when']="Scan to key in add2key"
    
    kHg = poseGraph.nodes[KeyFrame_newIdx]['sHg'] #global pose to the prev keyframe
    sHg = np.matmul(sHk,kHg) # global pose to the current frame: global to current frame
    gHs=nplinalg.inv(sHg) # current frame to global
    tpos=np.matmul(gHs,np.array([0,0,1]))
    
    poseGraph.add_node(ScanFrame_idx,frametype="scan",time=Tscan,X=Xscan,sHg=sHg,pos=(tpos[0],tpos[1]),color='r',LoopDetectDone=False)
    poseGraph.add_edge(KeyFrame_newIdx,ScanFrame_idx,H=sHk,H_prevframe=sHk_prevframe,err=serrk,hess_inv=shessk_inv,edgetype="Key2Scan",color='r')
    poseGraph.edges[KeyFrame_newIdx,ScanFrame_idx]['posematch']=posematch
    
    if debugMode:
        if 'ScanMatch' in poseGraph.edges[KeyFrame_newIdx,ScanFrame_idx]:
            poseGraph.edges[KeyFrame_newIdx,ScanFrame_idx]['ScanMatch2']={'input':[KeyFrameClf,Xclf,Xscan,sHk_prevframe],
                                                                        'output':[sHk,serrk,shessk_inv],
                                                                        'clf_X_sidecombine':poseGraph.nodes[KeyFrame_newIdx]['SideScansCombine'],
                                                                        'X_sidecombine':None}
        else:
            poseGraph.edges[KeyFrame_newIdx,ScanFrame_idx]['ScanMatch']={'input':[KeyFrameClf,Xclf,Xscan,sHk_prevframe],
                                                                        'output':[sHk,serrk,shessk_inv],
                                                                        'clf_X_sidecombine':poseGraph.nodes[KeyFrame_newIdx]['SideScansCombine'],
                                                                        'X_sidecombine':None}
        
    print("key %d- scan %d-posematch['mbinfrac_ActiveOvrlp']="%(KeyFrame_newIdx,ScanFrame_idx),posematch['mbinfrac_ActiveOvrlp'])
    
    if posematch['mbinfrac_ActiveOvrlp']<0.1:
        # with open("debugPlots/FailedNewScanFit_%d_%d_%d.pkl"%(KeyFrame_prevIdx,KeyFrame_newIdx,ScanFrame_idx),'wb') as F:
        #     pkl.dump(args,F)
            
        # res = getclf(Xprevidx,params,doReWtopt=True,means_init=None)
        # clf=res['clf']
        # poseGraph.nodes[KeyFrame_prevIdx]['clf']=clf
        
        posematch2 = poseGraph_keyFrame_matcher_binmatch(poseGraph,KeyFrame_newIdx,ScanFrame_idx,params,DoCLFmatch=True,dx0=0.8,L0=2,th0=np.pi/6,PoseGrid=None,isPoseGridOffset=True,isBruteForce=False,H21_est=None)
        posematch2['method']='binmatch'
        print("%d-%d-posematch['mbinfrac_ActiveOvrlp'] scan to key in addkeyandscan="%(KeyFrame_newIdx,ScanFrame_idx),posematch['mbinfrac_ActiveOvrlp'])
        if posematch2['mbinfrac_ActiveOvrlp']>posematch['mbinfrac_ActiveOvrlp']:
            poseGraph.edges[KeyFrame_newIdx,ScanFrame_idx]['H']=posematch2['H']
            poseGraph.edges[KeyFrame_newIdx,ScanFrame_idx]['posematchGMM']=poseGraph.edges[KeyFrame_newIdx,ScanFrame_idx]['posematch']
            poseGraph.edges[KeyFrame_newIdx,ScanFrame_idx]['posematch']=posematch2
            
            kHg = poseGraph.nodes[KeyFrame_newIdx]['sHg']
            sHg = np.matmul(posematch['H'],kHg)
            gHs=nplinalg.inv(sHg)    
            tpos=np.matmul(gHs,np.array([0,0,1])) 
            poseGraph.nodes[ScanFrame_idx]['pos']=(tpos[0],tpos[1])
            poseGraph.nodes[ScanFrame_idx]['sHg']=sHg
        
    # delete rest of the scan-ids as they are useless
    if keepOtherScans is False:
        for ix in KeyPrevPrevIdxs_succ:
            if poseGraph.nodes[ix]['frametype']=='scan':
                poseGraph.remove_node(ix)
                
#%% features


def get2DptFeat(X,bins=np.arange(0,11,0.25)):
    db=bins[1]-bins[0]
    D=fastdist.matrix_pairwise_distance(X, fastdist.euclidean, "euclidean", return_matrix=False)
    D = D.reshape(1,-1)
    h,b=np.histogram(D, bins=bins)
    h=h/np.sum(db*h)
    return h
    


#%% bundle adjustment
def loopExists(poseGraph,idx1,idx2):
    Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    Lkey = [x for x in Lkey if x>=idx1 and x<=idx2]
    Lkey.sort()
    
    for idx in Lkey:
        if poseGraph.in_degree(idx)>1:
            return True

def ProgressiveLoopClosures(idx,poseGraph,params,returnCopy=False):
    """
    Do loop closure to all the previous nodes to idx
    Do a subgraph combine  for idx and combine it to some 

    """
    params['SubGraphCombine_mbin']
    params['SubGraphCombine_err']
    params['SubGraphCombine_UseOdom']
    
    


def matchdetect(qin,qout,ExitFlag,Lkey,poseGraph,params) :
    while(True):
        idx = None
        previdx = None
        try:
            idx,previdx = qin.get(True,0.2)
        except queue.Empty:
            idx = None
            previdx = None
        
        if idx is not None and previdx is not None:
            h1=poseGraph.nodes[idx]['h']
            p1=poseGraph.nodes[idx]['pos']
            i1=Lkey.index(idx)
            i2=Lkey.index(previdx)
            if poseGraph.has_edge(idx,previdx) is False and poseGraph.has_edge(previdx,idx) is False:
                
                h2=poseGraph.nodes[previdx]['h']
                
                p2=poseGraph.nodes[previdx]['pos']
                d=nplinalg.norm(h1-h2,ord=1)
                
                c1 = nplinalg.norm(np.array(p1)-np.array(p2),ord=2)<=params['LOOP_CLOSURE_POS_THES']
                c2 = nplinalg.norm(np.array(p1)-np.array(p2),ord=2)>=params['LOOP_CLOSURE_POS_MIN_THES']
                if d<=params['LOOP_CLOSURE_D_THES'] and c1 and c2:
                    # add loop closure edge
                    
                    posematch=poseGraph_keyFrame_matcher_long(poseGraph,idx,previdx,params,params['NearLoopClose']['PoseGrid'],
                                                          params['NearLoopClose']['isPoseGridOffset'],
                                                          params['NearLoopClose']['isBruteForce'])
                    
                    # a1=posematch['err'] < params['LOOP_CLOSURE_ERR_THES']
                    # a2=posematch['mbinfrac']>=params['LOOPCLOSE_BIN_MIN_FRAC']
                    a3=posematch['mbinfrac_ActiveOvrlp']>=params['LOOPCLOSE_BIN_MAXOVRL_FRAC_LOCAL']
                    # print("Potential Loop closure ",a1,a2,a3)
                    if a3:
                        qout.put(['edge',idx,previdx,posematch['H'],posematch['err'],posematch['hess_inv'],d,posematch])
            
            
            # qout.put(['status',idx,'LoopDetectDone',True])
                
            
        if (ExitFlag.is_set() and qin.empty()):
            break
    
    print("thread done")              

def matchdetectLong(qin,qout,ExitFlag,Lkey,poseGraph,params) :
    while(True):
        idx = None
        previdx = None
        try:
            idx,previdx = qin.get(True,0.2)
        except queue.Empty:
            idx = None
            previdx = None
            
        if idx is not None and previdx is not None:
            h1=poseGraph.nodes[idx]['h']
            p1=poseGraph.nodes[idx]['pos']
            i1=Lkey.index(idx)
            i2=Lkey.index(previdx)
            if poseGraph.has_edge(idx,previdx) is False and poseGraph.has_edge(previdx,idx) is False:
                
                h2=poseGraph.nodes[previdx]['h']
                
                p2=poseGraph.nodes[previdx]['pos']
                d=nplinalg.norm(h1-h2,ord=1)
                
                c1 = nplinalg.norm(np.array(p1)-np.array(p2),ord=2)<=params['LOOP_CLOSURE_POS_THES']
                c2 = nplinalg.norm(np.array(p1)-np.array(p2),ord=2)>=params['LOOP_CLOSURE_POS_MIN_THES']
                if d<=params['LOOP_CLOSURE_D_THES'] and c1 and c2:
                    # add loop closure edge
                    
                    posematch=poseGraph_keyFrame_matcher_long(poseGraph,idx,previdx,params,params['LongLoopClose']['PoseGrid'],
                                                            params['LongLoopClose']['isPoseGridOffset'],
                                                            params['LongLoopClose']['isBruteForce'])
                    # # posematch2=poseGraph_keyFrame_matcher_long(poseGraph,previdx,idx,params,params['LongLoopClose']['PoseGrid'],
                    #                                      params['LongLoopClose']['isPoseGridOffset'],
                    #                                      params['LongLoopClose']['isBruteForce'])
                    if posematch['mbinfrac_ActiveOvrlp']<params['LOOPCLOSE_BIN_MAXOVRL_FRAC_COMPLETE']:
                        posematch = poseGraph_keyFrame_matcher_binmatch(poseGraph,idx,previdx,params,dx0=params['LongLoopClose']['Bin_Match_dx0'],L0=params['LongLoopClose']['Bin_Match_L0'],th0=params['LongLoopClose']['Bin_Match_th0'],DoCLFmatch=params['LongLoopClose']['DoCLFmatch'],PoseGrid=None,isPoseGridOffset=True,isBruteForce=False)
                    
                    
                    # a1=posematch['err'] < params['LOOP_CLOSURE_ERR_THES']
                    # a2=posematch['mbinfrac']>=params['LOOPCLOSE_BIN_MIN_FRAC']
                    a3=posematch['mbinfrac_ActiveOvrlp']>=params['LOOPCLOSE_BIN_MAXOVRL_FRAC_COMPLETE']
                    if a3:
                        qout.put(['edge',idx,previdx,posematch['H'],posematch['err'],posematch['hess_inv'],d,posematch])
                        # print(idx,previdx)
                        
                    # print("Potential Loop closure ",a1,a2,a3)
                    
                    
                    # if posematch1['mbinfrac_ActiveOvrlp']>=params['LOOPCLOSE_BIN_MAXOVRL_FRAC_current'] and posematch1['mbinfrac_ActiveOvrlp']>posematch2['mbinfrac_ActiveOvrlp']:
                    #     # qout.put(['edge',idx,previdx,posematch['H'],posematch['err'],posematch['hess_inv'],d,posematch])
                    #     qout.put(['edge',idx,previdx,posematch1['H'],posematch1['err'],posematch1['hess_inv'],d,posematch1])
                    # elif posematch2['mbinfrac_ActiveOvrlp']>=params['LOOPCLOSE_BIN_MAXOVRL_FRAC_current'] and posematch2['mbinfrac_ActiveOvrlp']>posematch1['mbinfrac_ActiveOvrlp']:
                    #     posematch2['H']=nplinalg.inv(posematch2['H'])
                    #     qout.put(['edge',idx,previdx,posematch2['H'],posematch2['err'],posematch2['hess_inv'],d,posematch2])
                    # elif posematch1['mbinfrac_ActiveOvrlp']>=params['LOOPCLOSE_BIN_MAXOVRL_FRAC_current']:
                    #     qout.put(['edge',idx,previdx,posematch1['H'],posematch1['err'],posematch1['hess_inv'],d,posematch1])
                    # elif posematch2['mbinfrac_ActiveOvrlp']>=params['LOOPCLOSE_BIN_MAXOVRL_FRAC_current']:
                    #     posematch2['H']=nplinalg.inv(posematch2['H'])
                    #     qout.put(['edge',idx,previdx,posematch2['H'],posematch2['err'],posematch2['hess_inv'],d,posematch2])    
            # qout.put(['status',idx,'LoopDetectDone',True])
                
            
        if (ExitFlag.is_set() and qin.empty()):
            break
    
    print("thread done")  
       
def evalposeMatch2(poseGraph,idx1,idx2,params,H12=None,Hdefault='global'):
    
    dxcomp = params['LOOPCLOSE_BIN_MIN_FRAC_dx']
    X1=poseGraph.nodes[idx1]['X']
    X2=poseGraph.nodes[idx2]['X']
    Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc.binScanEdges(X1,X2,dxcomp)
    activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))
    if H12 is None:
        if Hdefault=='global':
            s1Hg=poseGraph.nodes[idx1]['sHg']
            s2Hg=poseGraph.nodes[idx2]['sHg']
            H12=s1Hg.dot(nplinalg.inv(s2Hg))
        else:
            H12=nplinalg.inv( poseGraph.edges[idx1,idx2]['H'] )
    H21 = nplinalg.inv(H12) 
    posematch=eval_posematch(H21,X2,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
    return posematch
    
    
def detectAllLoopClosures(poseGraph,params,idxlb=None,idxub=None,returnCopy=False,parallel=True):
    """
    idx is index of current "keyframe"
    idx is the current pose. detect loop closures to all previous key frames

    """
    ctx = mp

    qin = ctx.Queue()
    qout = ctx.Queue()
    ExitFlag = ctx.Event()
    ExitFlag.clear()

    Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    if idxlb is None:
        idxlb=Lkeys[0]    
    if idxub is None:
        idxub=Lkeys[-1]
    LkeysPrevIdx = copy.deepcopy(Lkeys)
    
    Lkeys=[ll for ll in Lkeys if ll>=idxlb and ll<=idxub]
    Lkeyloop_edges = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key",poseGraph.edges))       
    Lkeyloop_edges=[ss for ss in Lkeyloop_edges if min(ss)>=idxlb and max(ss)<=idxub]
    
    for nn in Lkeys:
        if 'modified' not in poseGraph.nodes[nn]:
            poseGraph.nodes[nn]['modified']=[]
            
        if poseGraph.nodes[nn]['clf'] is None or 'clf' not in poseGraph.nodes[nn]:
            X = poseGraph.nodes[nn]['X']
            res = getclf(X,params,doReWtopt=True,means_init=None)
            clf=res['clf']
            poseGraph.nodes[nn]['clf']=clf
            poseGraph.nodes[nn]['modified'].append('clf')
            
    
    for previdx,idx  in Lkeyloop_edges:
        if 'modified' not in poseGraph.edges[previdx,idx]:
            poseGraph.edges[previdx,idx]['modified']=[]
            
            
        posematch=poseGraph.edges[previdx,idx ]['posematch']
        
        if poseGraph.edges[previdx,idx ].get('DoneBinMatch',False) is False:
            print("%d-%d-posematch['mbinfrac_ActiveOvrlp']="%(previdx,idx),posematch['mbinfrac_ActiveOvrlp'])
            if posematch['mbinfrac_ActiveOvrlp']<params["Key2Key_Overlap"]:
                
                posematch2 = poseGraph_keyFrame_matcher_binmatch(poseGraph,previdx,idx,params,DoCLFmatch=True,dx0=params['Key2KeyBinMatch_dx0'],L0=params['Key2KeyBinMatch_L0'],th0=params['Key2KeyBinMatch_th0'],PoseGrid=None,isPoseGridOffset=True,isBruteForce=False,H21_est=None)
                posematch2['method']='binmatch'
                print("%d-%d-posematch['mbinfrac_ActiveOvrlp']="%(previdx,idx),posematch['mbinfrac_ActiveOvrlp'])
                if posematch2['mbinfrac_ActiveOvrlp']>posematch['mbinfrac_ActiveOvrlp']:
                    poseGraph.edges[previdx,idx]['H']=posematch2['H']
                    poseGraph.edges[previdx,idx]['posematchGMM']=poseGraph.edges[previdx,idx]['posematch']
                    poseGraph.edges[previdx,idx]['posematch']=posematch2
                    
                    poseGraph.edges[previdx,idx]['modified'].append('H')
                    poseGraph.edges[previdx,idx]['modified'].append('posematchGMM')
                    poseGraph.edges[previdx,idx]['modified'].append('posematch')
                    # sHg = np.matmul(posematch['H'],kHg)
                    # gHs=nplinalg.inv(sHg)    
                    # tpos=np.matmul(gHs,np.array([0,0,1])) 
                    # poseGraph.nodes[idx]['pos']=(tpos[0],tpos[1])
                    # poseGraph.nodes[idx]['sHg']=sHg
                
        poseGraph.edges[previdx,idx ]['DoneBinMatch']=True
        poseGraph.edges[previdx,idx]['modified'].append('DoneBinMatch')
    
    # poseGraph=updated_sHg(poseGraph)
    
    # Lkeyloop_edges = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key",poseGraph.edges))
    # for previdx,idx  in Lkeyloop_edges[-50:]:
    #     dxcomp = params['LOOPCLOSE_BIN_MIN_FRAC_dx']
    #     Xp=poseGraph.nodes[previdx]['X']
    #     Xi=poseGraph.nodes[idx]['X']
    #     Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc.binScanEdges(Xp,Xi,dxcomp)
    #     activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))
    #     sHk=poseGraph.edges[previdx,idx]['H']
    #     posematch=eval_posematch(sHk,Xi,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
    #     if posematch['mbinfrac_ActiveOvrlp']<=0.1:
    #         posematch2= poseGraph_keyFrame_matcher_binmatch(poseGraph,previdx,idx,params,DoCLFmatch=True,dx0=0.9,L0=5,th0=np.pi/4,PoseGrid=None,isPoseGridOffset=True,isBruteForce=False)
    #         posematch2['when']="DoAllLoopClosures-Redo"
    #         posematch2['method']='binmatch'
    #         print(previdx,idx,posematch['mbinfrac_ActiveOvrlp'],posematch2['mbinfrac_ActiveOvrlp'])
    #         if posematch2['mbinfrac_ActiveOvrlp']>posematch['mbinfrac_ActiveOvrlp']:        
    #             poseGraph.edges[previdx,idx]['H']=posematch2['H']
    #             poseGraph.edges[previdx,idx]['posematchBinMatchRedo']=poseGraph.edges[previdx,idx]['posematch']
    #             poseGraph.edges[previdx,idx]['posematch']=posematch2
                
    # poseGraph=updated_sHg(poseGraph)
    
    
    # nn2=params['LOOP_CLOSURE_COMBINE_MAX_NODES']    
    # if params["USE_Side_Combine"]:
    #     mxndcmb=params['MAX_NODES_ADJ_COMBINE']
    #     for nn in Lkeys:
    #         if poseGraph.nodes[nn].get('DoneAdjCombine',False) is False:
    #             X1,clf1=getCombinedNode(poseGraph,nn,mxndcmb,params,Doclf=True)
    #             poseGraph.nodes[nn]['clflc']=clf1
    #             poseGraph.nodes[nn]['Xlc']=X1
    #             poseGraph.nodes[nn]['DoneAdjCombine']=True
                
    #             poseGraph.nodes[nn]['modified']=poseGraph.nodes[nn]['modified']+['clflc','Xlc','DoneAdjCombine']
    
    

            
    
    lenlist= params['LongLoopClose']['AlongPathNearFracLength']
    Npickstotal = params['LongLoopClose']['#TotalRandomPicks']
    Npickfrac= int(params['LongLoopClose']['AlongPathNearFracCountNodes']*Npickstotal)
    an=params['LongLoopClose']['AdjSkipList']
            
    cnt=0
    offsetNodesBy=params['offsetNodesBy']    
    if offsetNodesBy is None or offsetNodesBy==0:
        LkeysIdxs = Lkeys[::-1]
    else:
        LkeysIdxs = Lkeys[:-offsetNodesBy][::-1]
    for idx in LkeysIdxs :
        # if poseGraph.nodes[idx]['LoopDetectDone'] is True:
        #     continue
        


            
        if 'LocalLoopClosed' in poseGraph.nodes[idx]:
            if poseGraph.nodes[idx]['LocalLoopClosed']==False:
                continue
        
        # poseGraph.nodes[idx]['LoopDetectDone'] = True     
        

        if 'LongLoopDonePrevIdxs' not in poseGraph.nodes[idx]:
            poseGraph.nodes[idx]['LongLoopDonePrevIdxs']=[]
        
        
        
        LL=[]
        SkipIt=[]
        for previdx in LkeysPrevIdx[:max([LkeysPrevIdx.index(idx)-1,0])]:
            if previdx >=idx:
                continue
            if previdx in SkipIt:
                continue
            if previdx in poseGraph.nodes[idx]['LongLoopDonePrevIdxs']:
                continue
        
            if poseGraph.has_edge(idx,previdx) is True or poseGraph.has_edge(previdx,idx) is True:
                mn1 = max([0,LkeysPrevIdx.index(previdx)-an])
                mx1 = min([len(LkeysPrevIdx)-1,LkeysPrevIdx.index(previdx)+an])
                
                SkipIt=SkipIt+Lkeys[mn1:mx1]
                continue
            
            p1=poseGraph.nodes[idx]['pos']
            p2=poseGraph.nodes[previdx]['pos']
            
            
            c1 = nplinalg.norm(np.array(p1)-np.array(p2),ord=2)<=params['LOOP_CLOSURE_POS_THES']
            c2 = nplinalg.norm(np.array(p1)-np.array(p2),ord=2)>=params['LOOP_CLOSURE_POS_MIN_THES']
            if c1 and c2:
                LL.append(previdx)
                
        LL=list(set(LL))
        LL.sort(reverse=True)
        LL1=copy.deepcopy(LL[:int(lenlist*len(LL))])
        LL2=copy.deepcopy(LL[int(lenlist*len(LL)):])
        PrevIdxList=[]
        
        for i in range(Npickfrac):
            if len(LL1)==0:
                break
            
            a=random.choice(LL1)
            if a not in PrevIdxList:
                PrevIdxList.append(a)
                LL1.remove(a)
                    
        for i in range(Npickstotal-Npickfrac):
            if len(LL2)==0:
                break
            a=random.choice(LL2)
            if a not in PrevIdxList:
                PrevIdxList.append(a)
                LL2.remove(a)
        # print(idx,PrevIdxList)        
        for previdx in PrevIdxList:
            cnt+=1
            
            nn2=params['LOOP_CLOSURE_COMBINE_MAX_NODES']    
            if params["USE_Side_Combine"]:
                mxndcmb=params['MAX_NODES_ADJ_COMBINE']
                for ij in [idx,previdx]:
                    if poseGraph.nodes[ij].get('DoneAdjCombine',False) is False:
                        X1,clf1=getCombinedNode(poseGraph,ij,mxndcmb,params,Doclf=True)
                        poseGraph.nodes[ij]['clflc']=clf1
                        poseGraph.nodes[ij]['Xlc']=X1
                        poseGraph.nodes[ij]['DoneAdjCombine']=True
                    
                    # poseGraph.nodes[idx]['modified']=poseGraph.nodes[idx]['modified']+['clflc','Xlc','DoneAdjCombine']
                    
            qin.put([idx,previdx])

            poseGraph.nodes[idx]['LongLoopDonePrevIdxs'].append(previdx)
        
        if cnt >=params['LongLoopClose']['TotalCntComp'] :
            break
        
        
        poseGraph.nodes[idx]['modified'].append('LongLoopDonePrevIdxs')
    print("cnt = ",cnt)   
    
    Ncore = params['#ThreadsLoopClose']
    processes = []
    if parallel:
        for i in range(Ncore):
            p = ctx.Process(target=matchdetectLong, args=(qin,qout,ExitFlag,LkeysPrevIdx,poseGraph,params))
            processes.append( p )
            p.start()
            print("created thread")    
            time.sleep(0.1)  
            
    # params['LOOPCLOSE_BIN_MAXOVRL_FRAC_current'] = params['LOOPCLOSE_BIN_MAXOVRL_FRAC_COMPLETE']
    
            
    if not parallel:
        ExitFlag.set()
        matchdetectLong(qin,qout,ExitFlag,Lkeys,poseGraph,params)
        
    flg=0
    while True:
        res=None
        try:
            res=qout.get(block=True, timeout=0.1)
        except queue.Empty:
            time.sleep(0.1)
            
            
        if res is not None:
            if 'status' == res[0]:
                poseGraph.nodes[res[1]][res[2]] = res[3]
            elif 'edge' == res[0]:
                poseGraph.add_edge(res[1],res[2],H=res[3],err = res[4],hess_inv = res[5],edgetype="Key2Key-LoopClosure",d=res[6],color='b',posematch=res[7],modified=[])
                
            # print("res=",res[:3])
                
            flg=0
            
        
        
        if qin.empty() and qout.empty():
            ExitFlag.set()
            Palive=0
            for i in range(len(processes)):
                if parallel:
                    if processes[i].is_alive():
                        Palive+=1
            
            if Palive==0:
                break
    
    if parallel:
        for i in range(len(processes)):
            print("Joining %d"%i)
            processes[i].join()

    # params.pop('LOOPCLOSE_BIN_MAXOVRL_FRAC_current')
    print("detect loop closes done")
    
    if returnCopy:
        return copy.deepcopy(poseGraph)
    
    
    
    
    return poseGraph

# class PoseGraphOptimization(g2o.SparseOptimizer):
#     def __init__(self):
#         super().__init__()
#         solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
#         solver = g2o.OptimizationAlgorithmLevenberg(solver)
#         super().set_algorithm(solver)

#     def optimize(self, max_iterations=50):
#         super().initialize_optimization()
#         super().optimize(max_iterations)

#     def add_vertex(self, id, pose, fixed=False):
#         v_se3 = g2o.VertexSE3()
#         v_se3.set_id(id)
#         v_se3.set_estimate(pose)
#         v_se3.set_fixed(fixed)
#         super().add_vertex(v_se3)

#     def add_edge(self, vertices, measurement, 
#             information=np.identity(6),
#             robust_kernel=g2o.RobustKernelHuber()):

#         edge = g2o.EdgeSE3()
#         for i, v in enumerate(vertices):
#             if isinstance(v, int):
#                 v = self.vertex(v)
#             edge.set_vertex(i, v)

#         edge.set_measurement(measurement)  # relative pose
#         edge.set_information(information)
#         if robust_kernel is not None:
#             edge.set_robust_kernel(robust_kernel)
#         super().add_edge(edge)

#     def get_pose(self, id):
#         return self.vertex(id).estimate()



def adjustPosesG2o(poseGraph,pgopt=None):
    if pgopt is None:
        pgopt=PoseGraphOptimization()
    # nodeList = [nn for nn in poseGraph.nodes if nn>=idx0 and nn<=idx1]
    # nodeList.sort()
    
    for nn in poseGraph.nodes:
        if poseGraph.nodes[nn]['frametype']!="keyframe":
            continue
        
        if pgopt.vertex(nn) is not None:
            continue
        
        gHs=nplinalg.inv(poseGraph.nodes[nn]['sHg'])
        R=np.identity(3)
        R[0:2,0:2]=gHs[0:2,0:2]
        tpos=np.zeros(3)
        tpos[0:2]=gHs[0:2,2]
        
        if nn==0:
            pgopt.add_vertex(nn,g2o.Isometry3d(R, tpos),fixed=True)
        else:
            pgopt.add_vertex(nn,g2o.Isometry3d(R, tpos),fixed=False)
    
    L=list(pgopt.edges())
    L=map(lambda x: x.vertices(),L)
    edgeList=list(map(lambda x: (x[0].id(),x[1].id()),L))
    
    for (e1,e2) in poseGraph.edges:
        
        
        # if max([e1,e2])<idx0 or min([e1,e2])>idx1:
        #     continue
        if (e1,e2) in edgeList:
            continue
        if poseGraph.edges[e1,e2]['edgetype']=="Key2Key" or poseGraph.edges[e1,e2]['edgetype']=="Key2Key-LoopClosure":
            pass
        else:
            continue
        
        H=nplinalg.inv(poseGraph.edges[e1,e2]['H'])
        R=np.identity(3)
        R[0:2,0:2]=H[0:2,0:2]
        tpos=np.zeros(3)
        tpos[0:2]=H[0:2,2]
        
        err = poseGraph.edges[e1,e2]['posematch']['mbinfrac_ActiveOvrlp'] 
        Hess = np.identity(6)
        Hess[0:3,0:3]=1*poseGraph.edges[e1,e2]['hess_inv']
        pgopt.add_edge([e1,e2],g2o.Isometry3d(R, tpos),information=Hess)
    
    
    pgopt.optimize()
    
    for nn in poseGraph.nodes:   
        if poseGraph.nodes[nn]['frametype']!="keyframe":
            continue
        
        v1=pgopt.get_pose(nn)
        tpos=v1.position()
        R = v1.rotation_matrix()
        H=np.identity(3)
        H[0:2,0:2]=R[0:2,0:2]
        H[0:2,2]=tpos[0:2]
        
        poseGraph.nodes[nn]['sHg']=nplinalg.inv(H)
        poseGraph.nodes[nn]['pos']=(tpos[0],tpos[1])
    

    for ns in list(poseGraph.nodes):
        # if ns<=nodeList[-1] and ns>=nodeList[0]:            
        for pidx in poseGraph.predecessors(ns):
            if poseGraph.nodes[pidx]['frametype']=="keyframe": # and pidx in sHg_updated
                if poseGraph.edges[pidx,ns]['edgetype']=="Key2Scan":
                    psHg=poseGraph.nodes[pidx]['sHg']
                    nsHps=poseGraph.edges[pidx,ns]['H']
                    nsHg = nsHps.dot(psHg)
                    poseGraph.nodes[ns]['sHg']=nsHg
                    gHns=nplinalg.inv(nsHg)
                    tpos=np.matmul(gHns,np.array([0,0,1]))
                    poseGraph.nodes[ns]['pos'] = (tpos[0],tpos[1])
                    break
                    
        # if ns>nodeList[-1]:
        # for pidx in poseGraph.predecessors(ns):
        #     if poseGraph.nodes[pidx]['frametype']=="keyframe": # and pidx in sHg_updated
        #         if poseGraph.edges[pidx,ns]['edgetype']=="Key2Key" or poseGraph.edges[pidx,ns]['edgetype']=="Key2Scan":
        #             psHg=poseGraph.nodes[pidx]['sHg']
        #             nsHps=poseGraph.edges[pidx,ns]['H']
        #             nsHg = nsHps.dot(psHg)
        #             poseGraph.nodes[ns]['sHg']=nsHg
        #             gHns=nplinalg.inv(nsHg)
        #             tpos=np.matmul(gHns,np.array([0,0,1]))
        #             poseGraph.nodes[ns]['pos'] = (tpos[0],tpos[1])
        #             break
    return poseGraph,pgopt


def detectAllLoopClosures_closebyNodes(poseGraph,params,returnCopy=False,parallel=True):
    """
    idx is index of current "keyframe"
    idx is the current pose. detect loop closures to all previous key frames

    """
    
    ctx = mp

    qin = ctx.Queue()
    qout = ctx.Queue()
    ExitFlag = ctx.Event()
    ExitFlag.clear()

    Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    
    for nn in Lkeys:
        if poseGraph.nodes[nn]['clf'] is None:
            X = poseGraph.nodes[nn]['X']
            res = getclf(X,params,doReWtopt=True,means_init=None)
            clf=res['clf']
            poseGraph.nodes[nn]['clf']=clf
    
    if params["USE_Side_Combine"]:
        mxndcmb=params['MAX_NODES_ADJ_COMBINE']
        for nn in Lkeys:
            if poseGraph.nodes[nn].get('DoneAdjCombine',False) is False:
                X1,clf1=getCombinedNode(poseGraph,nn,mxndcmb,params,Doclf=True)
                poseGraph.nodes[nn]['clflc']=clf1
                poseGraph.nodes[nn]['Xlc']=X1
                poseGraph.nodes[nn]['DoneAdjCombine']=True
            
    
    params['LOOPCLOSE_BIN_MAXOVRL_FRAC_current'] = params['LOOPCLOSE_BIN_MAXOVRL_FRAC_LOCAL']
    
    Ncore = params['#ThreadsLoopClose']
    processes = []
    if parallel:
        for i in range(Ncore):
            p = ctx.Process(target=matchdetect, args=(qin,qout,ExitFlag,Lkeys,poseGraph,params))
            processes.append( p )
            p.start()
            print("created thread")    
            time.sleep(0.01)  
    

    nn2=params['LOOP_CLOSURE_COMBINE_MAX_NODES']    
    offsetNodesBy=params['offsetNodesBy']    
    for idx in Lkeys[:-offsetNodesBy]:
        if 'LocalLoopClosed' in poseGraph.nodes[idx]:
            if poseGraph.nodes[idx]['LocalLoopClosed']==True:
                continue
            
        PPidx = Lkeys[max([Lkeys.index(idx)-nn2,0]):max([Lkeys.index(idx)-1,0])]
        for previdx in PPidx:
            if 'LocalLoopClosed' in poseGraph.nodes[previdx]:
                if poseGraph.nodes[previdx]['LocalLoopClosed']==True:
                    continue
                
            qin.put([idx,previdx])
        
    for idx in Lkeys[:-offsetNodesBy]:
        poseGraph.nodes[idx]['LocalLoopClosed']=True
    
    if not parallel:
        ExitFlag.set()
        matchdetect(qin,qout,ExitFlag,Lkeys,poseGraph,params)
        
    flg=0
    while True:
        res=None
        try:
            res=qout.get(block=True, timeout=0.1)
        except queue.Empty:
            time.sleep(0.01)
            
            
        if res is not None:
            if 'status' == res[0]:
                poseGraph.nodes[res[1]][res[2]] = res[3]
            elif 'edge' == res[0]:
                poseGraph.add_edge(res[1],res[2],H=res[3],err = res[4],hess_inv = res[5],edgetype="Key2Key-LoopClosure",d=res[6],color='m',posematch=res[7],isCloseByNodes=True)
            flg=0
            
        
        
        if qin.empty() and qout.empty():
            ExitFlag.set()
            Palive=0
            for i in range(len(processes)):
                if parallel:
                    if processes[i].is_alive():
                        Palive+=1
            
            if Palive==0:
                break
    
    if parallel:
        for i in range(len(processes)):
            print("Joining %d"%i)
            processes[i].join()
                

        
    if returnCopy:
        return copy.deepcopy(poseGraph)
    
    params.pop('LOOPCLOSE_BIN_MAXOVRL_FRAC_current')
    print("detect loop closes done")
    return poseGraph

def LoopCLose_CloseByNodes(poseGraph,params):
    Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
     
    LkeyCloseByloop_edges = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key-LoopClosure" and poseGraph.edges[x].get('isCloseByNodes',False)==True ,poseGraph.edges))
    if len(LkeyCloseByloop_edges)<=2:
        G=poseGraph
    else:
        LkeyCloseByloop_nodes = np.array(LkeyCloseByloop_edges).reshape(-1)
        mnnode = min(LkeyCloseByloop_nodes)
        mxnode = max(LkeyCloseByloop_nodes)
        G=poseGraph.subgraph([ss for ss in Lkeys if ss>=mnnode and ss<=mxnode])
        
    Seqs=[]      
    for seq in nx.simple_cycles(G):
        seq = sorted(list(seq))
        Seqs.append(seq)
        
    # Seqs=sorted(Seqs,key=lambda x: len(x),reverse=True)
    Seqs=sorted(Seqs,key=lambda x: len(x))
    Seqs_todo = []
    for seq in Seqs:
        S=[Lkeys.index(s) for s in seq]
        LloopCls = []
        flg=False
        for s1 in seq:
            for s2 in poseGraph.successors(s1):
                if poseGraph.edges[s1,s2]['edgetype']=='Key2Key-LoopClosure':
                    if poseGraph.edges[s1,s2].get('isCloseByNodes',False)==True:
                        flg=True
                        break
                
        if flg and np.all(np.abs(np.diff(S))==1) and len(S)<=params['LOOP_CLOSURE_COMBINE_MAX_NODES']: 
           Seqs_todo.append(seq)
    
    # pdb.set_trace()
    
    print("Seqs_todo = ",Seqs_todo)  
    if len(Seqs_todo)==0:
        Lkeyloop = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key-LoopClosure",poseGraph.edges))
    
        for ee in Lkeyloop:
            if 'isCloseByNodes' in poseGraph.edges[ee]:
                if poseGraph.edges[ee]['isCloseByNodes']:
                    poseGraph.remove_edge(ee[0],ee[1])
                    
        return poseGraph
    
    removedNodes=[]
    # Seqs_todo is the list of sorted decreasing sequences
    # do pose optimization and update all the global poses

    Seqs_todo=sorted(Seqs_todo,key=lambda x: len(x),reverse=True)
    for i,seq in enumerate(Seqs_todo):

        if set(seq) & set(poseGraph.nodes)!=set(seq):
            continue
        
        # if max(seq)<=3502 or min(seq)>=3952:# def fails upto 8070 ; 3898 is bad     ; 3111 is okay  ;3952 and idx>=3502
        s1 = seq[0]
        s2 = seq[-1]
        st=time.time()
        res,sHg_updated,sHg_previous=adjustPoses(poseGraph,s1,s2,maxiter=None,algo='trf',xtol=1e-4)
        if res.success:
            poseGraph=updateGlobalPoses(copy.deepcopy(poseGraph),sHg_updated,updateRelPoses=True)
        else:
            print(seq)
            print("opt is failure")
            print(res)
        
        mn = seq[int(len(seq)/2)]
        pii=mn
        lp=[mn]
        for ii in seq[seq.index(mn)+1:]:
            if poseGraph.edges[pii,ii]['posematch']['mbinfrac_ActiveOvrlp']>=params["Side_Combine_Overlap"]:
                lp.append(ii)
            pii=ii
        lp2=[]
        pii=mn
        for ii in seq[:seq.index(mn)][::-1]:
            if poseGraph.edges[ii,pii]['posematch']['mbinfrac_ActiveOvrlp']>=params["Side_Combine_Overlap"]:
                lp2.append(ii)
            pii=ii
        lp=lp2[::-1]+lp
        if set(lp) == set(seq):
            seq=lp
        else:
            continue
           
        mn = seq[int(len(seq)/2)]

        X=[poseGraph.nodes[mn]['X']]
        mnHg=poseGraph.nodes[mn]['sHg']
        gHmn=nplinalg.inv(mnHg)
        for nn in seq:
            if nn==mn:
                continue
            nnHg=poseGraph.nodes[nn]['sHg']
            gHnn = nplinalg.inv(nnHg)
            mnHnn = np.matmul(mnHg,gHnn)
            XX=np.matmul(mnHnn,np.vstack([poseGraph.nodes[nn]['X'].T,np.ones(poseGraph.nodes[nn]['X'].shape[0])])).T  
            X.append(XX[:,:2])
            
            
        X=binnerDownSamplerProbs(X,dx=params['BinDownSampleKeyFrame_dx'],prob=params['BinDownSampleKeyFrame_probs'])
        poseGraph.nodes[mn]['X']=X
        res = getclf(X,params,doReWtopt=True,means_init=None)
        clf=res['clf']
        poseGraph.nodes[mn]['clf']=clf
        idbmx = params['INTER_DISTANCE_BINS_max']
        idbdx=params['INTER_DISTANCE_BINS_dx']
        # h=get2DptFeat(X,bins=np.arange(0,idbmx,idbdx))
        h=np.array([0,0])
        poseGraph.nodes[mn]['h']=h
        
        poseGraph.nodes[mn]['LocalLoopClosed']=True
        
        ln = seq[-1]
        for sidx in poseGraph.successors(ln):
            if poseGraph.nodes[sidx]['frametype']=="keyframe" and poseGraph.edges[ln,sidx]['edgetype']=="Key2Key":
                sHln = poseGraph.edges[ln,sidx]['H']
                
                lnHg=poseGraph.nodes[ln]['sHg']
                lnHmn = np.matmul(lnHg,gHmn)
                
                sHmn = np.matmul(sHln,lnHmn)
                
                H_prevframe=poseGraph.edges[ln,sidx]['H_prevframe']
                
                serrk=poseGraph.edges[ln,sidx]['err']
                shessk_inv=poseGraph.edges[ln,sidx]['hess_inv']
                poseGraph.add_edge(mn,sidx,H=sHmn,H_prevframe=H_prevframe,err=serrk,hess_inv=shessk_inv,edgetype="Key2Key",color='k')
    
                break
        
        fn = seq[0]
        for pidx in poseGraph.predecessors(fn):
            if poseGraph.nodes[pidx]['frametype']=="keyframe" and poseGraph.edges[pidx,fn]['edgetype']=="Key2Key":
                fnHp = poseGraph.edges[pidx,fn]['H']
                
                fnHg=poseGraph.nodes[fn]['sHg']
                gHfn=nplinalg.inv(fnHg)
                mnHfn = np.matmul(mnHg,gHfn)
                
                mnHp = np.matmul(mnHfn,fnHp)
                
                H_prevframe=poseGraph.edges[pidx,fn]['H_prevframe']
                
                serrk=poseGraph.edges[pidx,fn]['err']
                shessk_inv=poseGraph.edges[pidx,fn]['hess_inv']
                poseGraph.add_edge(pidx,mn,H=mnHp,H_prevframe=H_prevframe,err=serrk,hess_inv=shessk_inv,edgetype="Key2Key",color='k')
    
                break
        
        for nn in seq:
            if nn!=mn:
                removedNodes.append(nn)
                poseGraph.remove_node(nn)

                    
    Lkeyloop = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key-LoopClosure",poseGraph.edges))
    
    for ee in Lkeyloop:
        if 'isCloseByNodes' in poseGraph.edges[ee]:
            if poseGraph.edges[ee]['isCloseByNodes']:
                poseGraph.remove_edge(ee[0],ee[1])
    

    dxcomp = params['LOOPCLOSE_BIN_MIN_FRAC_dx']
    for ee in poseGraph.edges:
        if poseGraph.edges[ee[0],ee[1]].get('posematch',None) is None:
            Xclf = poseGraph.nodes[ee[0]]['X']
            X = poseGraph.nodes[ee[1]]['X']
            Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc.binScanEdges(Xclf,X,dxcomp)
            activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))
            H12=poseGraph.edges[ee[0],ee[1]]['H']
            posematch=eval_posematch(H12,X,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
            poseGraph.edges[ee[0],ee[1]]['posematch']=posematch
            # print("posematch['mbinfrac_ActiveOvrlp']=",posematch['mbinfrac_ActiveOvrlp'])
            # if posematch['mbinfrac_ActiveOvrlp']<0.5:
            #     posematch = pt2dproc.poseGraph_keyFrame_matcher_binmatch(poseGraph,KeyFrame_prevIdx,idx,params,DoCLFmatch=True,dx0=0.8,L0=1,th0=np.pi/12,PoseGrid=None,isPoseGridOffset=True,isBruteForce=False)
            #     poseGraph.edges[KeyFrame_prevIdx,idx]['H']=posematch['H']
            #     poseGraph.edges[KeyFrame_prevIdx,idx]['posematch']=posematch
                
            #     sHg = np.matmul(posematch['H'],kHg)
            #     gHs=nplinalg.inv(sHg)    
            #     tpos=np.matmul(gHs,np.array([0,0,1])) 
            #     poseGraph.nodes[idx]['pos']=(tpos[0],tpos[1])
            #     poseGraph.nodes[idx]['sHg']=sHg
        
    return poseGraph


def LoopCLose_CloseByNodes_orig(poseGraph,params):
    Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    
    Seqs=[]
    
    for seq in nx.simple_cycles(poseGraph):
        seq = sorted(list(seq))
        Seqs.append(seq)
        
    # Seqs=sorted(Seqs,key=lambda x: len(x),reverse=True)
    Seqs=sorted(Seqs,key=lambda x: x[0])
    Seqs_todo = []
    for seq in Seqs:
        S=[Lkeys.index(s) for s in seq]
        LloopCls = []
        flg=False
        for s1 in seq:
            for s2 in poseGraph.successors(s1):
                if poseGraph.edges[s1,s2]['edgetype']=='Key2Key-LoopClosure':
                    if poseGraph.edges[s1,s2].get('isCloseByNodes',False)==True:
                        flg=True
                        break
                
        if flg and np.all(np.abs(np.diff(S))==1) and len(S)<=params['LOOP_CLOSURE_COMBINE_MAX_NODES']: 
           Seqs_todo.append(seq)
    
    # pdb.set_trace()
    
    print("Seqs_todo = ",Seqs_todo)  
    if len(Seqs_todo)==0:
        Lkeyloop = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key-LoopClosure",poseGraph.edges))
    
        for ee in Lkeyloop:
            if 'isCloseByNodes' in poseGraph.edges[ee]:
                if poseGraph.edges[ee]['isCloseByNodes']:
                    poseGraph.remove_edge(ee[0],ee[1])
                    
        return poseGraph
    
    removedNodes=[]
    # Seqs_todo is the list of sorted decreasing sequences
    # do pose optimization and update all the global poses
    SS=[]
    for i,seq in enumerate(Seqs_todo):
        # if max(seq)<=3502 or min(seq)>=3952:# def fails upto 8070 ; 3898 is bad     ; 3111 is okay  ;3952 and idx>=3502
        s1 = seq[0]
        s2 = seq[-1]
        st=time.time()
        res,sHg_updated,sHg_previous=adjustPoses(poseGraph,s1,s2,maxiter=None,algo='trf',xtol=1e-4)
        if res.success:
            poseGraph=updateGlobalPoses(copy.deepcopy(poseGraph),sHg_updated,updateRelPoses=True)
        else:
            print("opt is failure")
            print(res)
        
        mn = seq[int(len(seq)/2)]
        pii=mn
        lp=[mn]
        for ii in seq[seq.index(mn)+1:]:
            if poseGraph.edges[pii,ii]['posematch']['mbinfrac_ActiveOvrlp']>=0.5:
                lp.append(ii)
            pii=ii
        lp2=[]
        pii=mn
        for ii in seq[:seq.index(mn)][::-1]:
            if poseGraph.edges[ii,pii]['posematch']['mbinfrac_ActiveOvrlp']>=0.5:
                lp2.append(ii)
            pii=ii
        lp=lp2[::-1]+lp
        if set(lp) == set(seq):
            SS.append(lp)
        
        # print(mn,"----",seq,lp)
        
    Seqs_todo = SS
    
    Seqs_todo=sorted(Seqs_todo,key=lambda x: len(x),reverse=True)
    for i,seq in enumerate(Seqs_todo):
        if set(seq) & set(poseGraph.nodes)!=set(seq):
            continue
            
        mn = seq[int(len(seq)/2)]
        
        
        
        X=[poseGraph.nodes[mn]['X']]
        mnHg=poseGraph.nodes[mn]['sHg']
        gHmn=nplinalg.inv(mnHg)
        for nn in seq:
            if nn==mn:
                continue
            nnHg=poseGraph.nodes[nn]['sHg']
            gHnn = nplinalg.inv(nnHg)
            mnHnn = np.matmul(mnHg,gHnn)
            XX=np.matmul(mnHnn,np.vstack([poseGraph.nodes[nn]['X'].T,np.ones(poseGraph.nodes[nn]['X'].shape[0])])).T  
            X.append(XX[:,:2])
            
            
        X=binnerDownSamplerProbs(X,dx=params['BinDownSampleKeyFrame_dx'],prob=params['BinDownSampleKeyFrame_probs'])
        poseGraph.nodes[mn]['X']=X
        res = getclf(X,params,doReWtopt=True,means_init=None)
        clf=res['clf']
        poseGraph.nodes[mn]['clf']=clf
        idbmx = params['INTER_DISTANCE_BINS_max']
        idbdx=params['INTER_DISTANCE_BINS_dx']
        # h=get2DptFeat(X,bins=np.arange(0,idbmx,idbdx))
        h=np.array([0,0])
        poseGraph.nodes[mn]['h']=h
        
        poseGraph.nodes[mn]['LocalLoopClosed']=True
        
        ln = seq[-1]
        for sidx in poseGraph.successors(ln):
            if poseGraph.nodes[sidx]['frametype']=="keyframe" and poseGraph.edges[ln,sidx]['edgetype']=="Key2Key":
                sHln = poseGraph.edges[ln,sidx]['H']
                
                lnHg=poseGraph.nodes[ln]['sHg']
                lnHmn = np.matmul(lnHg,gHmn)
                
                sHmn = np.matmul(sHln,lnHmn)
                
                H_prevframe=poseGraph.edges[ln,sidx]['H_prevframe']
                
                serrk=poseGraph.edges[ln,sidx]['err']
                shessk_inv=poseGraph.edges[ln,sidx]['hess_inv']
                poseGraph.add_edge(mn,sidx,H=sHmn,H_prevframe=H_prevframe,err=serrk,hess_inv=shessk_inv,edgetype="Key2Key",color='k')
    
                break
        
        fn = seq[0]
        for pidx in poseGraph.predecessors(fn):
            if poseGraph.nodes[pidx]['frametype']=="keyframe" and poseGraph.edges[pidx,fn]['edgetype']=="Key2Key":
                fnHp = poseGraph.edges[pidx,fn]['H']
                
                fnHg=poseGraph.nodes[fn]['sHg']
                gHfn=nplinalg.inv(fnHg)
                mnHfn = np.matmul(mnHg,gHfn)
                
                mnHp = np.matmul(mnHfn,fnHp)
                
                H_prevframe=poseGraph.edges[pidx,fn]['H_prevframe']
                
                serrk=poseGraph.edges[pidx,fn]['err']
                shessk_inv=poseGraph.edges[pidx,fn]['hess_inv']
                poseGraph.add_edge(pidx,mn,H=mnHp,H_prevframe=H_prevframe,err=serrk,hess_inv=shessk_inv,edgetype="Key2Key",color='k')
    
                break
        
        for nn in seq:
            if nn!=mn:
                removedNodes.append(nn)
                poseGraph.remove_node(nn)

                    
    Lkeyloop = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key-LoopClosure",poseGraph.edges))
    
    for ee in Lkeyloop:
        if 'isCloseByNodes' in poseGraph.edges[ee]:
            if poseGraph.edges[ee]['isCloseByNodes']:
                poseGraph.remove_edge(ee[0],ee[1])
    

    dxcomp = params['LOOPCLOSE_BIN_MIN_FRAC_dx']
    for ee in poseGraph.edges:
        if poseGraph.edges[ee[0],ee[1]].get('posematch',None) is None:
            Xclf = poseGraph.nodes[ee[0]]['X']
            X = poseGraph.nodes[ee[1]]['X']
            Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc.binScanEdges(Xclf,X,dxcomp)
            activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))
            H12=poseGraph.edges[ee[0],ee[1]]['H']
            posematch=eval_posematch(H12,X,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
            poseGraph.edges[ee[0],ee[1]]['posematch']=posematch
            # print("posematch['mbinfrac_ActiveOvrlp']=",posematch['mbinfrac_ActiveOvrlp'])
            # if posematch['mbinfrac_ActiveOvrlp']<0.5:
            #     posematch = pt2dproc.poseGraph_keyFrame_matcher_binmatch(poseGraph,KeyFrame_prevIdx,idx,params,DoCLFmatch=True,dx0=0.8,L0=1,th0=np.pi/12,PoseGrid=None,isPoseGridOffset=True,isBruteForce=False)
            #     poseGraph.edges[KeyFrame_prevIdx,idx]['H']=posematch['H']
            #     poseGraph.edges[KeyFrame_prevIdx,idx]['posematch']=posematch
                
            #     sHg = np.matmul(posematch['H'],kHg)
            #     gHs=nplinalg.inv(sHg)    
            #     tpos=np.matmul(gHs,np.array([0,0,1])) 
            #     poseGraph.nodes[idx]['pos']=(tpos[0],tpos[1])
            #     poseGraph.nodes[idx]['sHg']=sHg
        
    return poseGraph


#%%

def adjustPoses(poseGraph,idx1,idx2,maxiter=None,algo='BFGS',tol=1e-3,xtol=1e-4):
    # adjust the bundle from idx1 to idx2
    # i.e. optimize the global pose matrices, with constraint local poses 
    # idx1 is assumed to be fixed (like origin)
    Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    Lkey = [x for x in Lkey if x>=idx1 and x<=idx2]
    Lkey.sort()
    
    Lkey_dict={x:i for i,x in enumerate(Lkey)}
    
    Hrels=[] # Hrels=[[i,j,thji,txji,tyji],...]
    Hrelsidx=[]
    Hessrels=[] #Hessrels=[[hess_flattened],...]

    for j in range(1,len(Lkey)):
        idx = Lkey[j]
        for idx_prev in poseGraph.predecessors(idx):
            if idx_prev not in Lkey_dict:
                continue
            i = Lkey_dict[idx_prev]
            # if 'updated' in poseGraph.edges[idx_prev,idx]:
            #     jHi=poseGraph.edges[idx_prev,idx]['updated'].get('H',poseGraph.edges[idx_prev,idx]['H'])
            # else:
            jHi=poseGraph.edges[idx_prev,idx]['H']
            j_hess_inv_i = poseGraph.edges[idx_prev,idx]['hess_inv']
            if poseGraph.edges[idx_prev,idx]['edgetype']=="Key2Key-LoopClosure":
                j_hess_i = 1*nplinalg.inv(j_hess_inv_i)
            else:
                j_hess_i = 1*nplinalg.inv(j_hess_inv_i)
            
            # if np.abs(i-j)>=3 and np.abs(i-j)<=30:
            #     continue
                
            j_hess_i = 1*j_hess_i
            tji,thji = nbpt2Dproc.extractPosAngle(jHi)
            
            idxpos=np.array(poseGraph.nodes[idx]['pos'])
            idxprevpos=np.array(poseGraph.nodes[idx_prev]['pos'])
            # if nplinalg.norm(idxpos-idxprevpos)>15:
            #     print(idxpos,idxprevpos)
            #     continue
            
            Hrels.append([thji,tji[0],tji[1] ])        
            Hessrels.append(j_hess_i.reshape(-1))
            Hrelsidx.append([i,j])
            
    Hrels = np.array(Hrels,dtype=dtype)
    Hessrels = np.array(Hessrels,dtype=dtype)

    Hrelsidx = np.array(Hrelsidx,dtype=np.int32)

    # pdb.set_trace()
    
    # x0=x0.reshape(-1)
    x0 = np.zeros((len(Lkey)-1,3))   #as first frame is 0,0,0          
    firstHg=poseGraph.nodes[Lkey[0]]['sHg']
    gHfirst = nplinalg.inv(firstHg)
    sHg_original={}
    bounds=[[-np.pi/4,np.pi/4],[-2,2],[-2,2]]*(len(Lkey)-1)
    for j in range(1,len(Lkey)):
        idx = Lkey[j]
        # if 'updated' in poseGraph.nodes[idx]:
        #     sHg=poseGraph.nodes[idx]['updated'].get('sHg',poseGraph.nodes[idx]['sHg'])
        # else:
        sHg=poseGraph.nodes[idx]['sHg']
        sHg_original[idx] = sHg
        
        jHf=np.matmul(sHg,gHfirst) # first to j (local global frame)
        fHj=nplinalg.inv(jHf)
        tj,thj = nbpt2Dproc.extractPosAngle(fHj)
        # tj = Hjf[:2,2]  
        x0[j-1,0]=thj
        x0[j-1,1:]=tj
        
        bounds[(j-1)*3][0]=thj+bounds[(j-1)*3][0]
        bounds[(j-1)*3][1]=thj+bounds[(j-1)*3][1]
        
        bounds[(j-1)*3+1][0]=tj[0]+bounds[(j-1)*3+1][0]
        bounds[(j-1)*3+1][1]=tj[0]+bounds[(j-1)*3+1][1]
        
        bounds[(j-1)*3+2][0]=tj[1]+bounds[(j-1)*3+2][0]
        bounds[(j-1)*3+2][1]=tj[1]+bounds[(j-1)*3+2][1]
        
    x0=x0.reshape(-1)
    # x0 = x0+1*np.random.randn(len(x0))
    
    lb = np.array([bnd[0] for bnd in bounds])
    ub = np.array([bnd[1] for bnd in bounds])
    
    # print("len(x) = ",len(x0))
    # print("Hrelsidx = ",Hrelsidx.shape)
    # print("Hrels = ",Hrels.shape)
    
    # pdb.set_trace()
    # print("starting cost globalPoseCost= ",nbpt2Dproc.globalPoseCost(x0,Hrelsidx,Hrels))
    # gg = nbpt2Dproc.globalPoseCost_lsq(x0,Hrelsidx,Hrels)
    # print("starting cost globalPoseCost_lsq= ",np.sum(gg**2))
    # st=time.time()
    # res = minimize(globalPoseCost, x0,args=(Hrelsidx,Hrels) ,method='BFGS', tol=1e-5)
    # success=res.success
    # et=time.time()
    # print(res.fun," time: ",et-st)
    # print(idx1,idx2,Hrelsidx,Hrels)
    # print(Hrelsidx.shape)
    st=time.time()
    if algo in ['lm','trf']:
        # res = least_squares(nbpt2Dproc.globalPoseCostHess_lsq, x0,args=(Hrelsidx,Hrels,Hessrels),method=algo)
        # res = least_squares(nbpt2Dproc.globalPoseCost_lsq, x0,args=(Hrelsidx,Hrels),jac=nbpt2Dproc.globalPoseCost_lsq_jac,method=algo,xtol=1e-3) #
        # res = least_squares(nbpt2Dproc.globalPoseCostHess_lsq, x0,args=(Hrelsidx,Hrels,Hessrels) ,method='trf',xtol=1e-3)
        
        res = least_squares(nbpt2Dproc.globalPoseCost_lsq, x0,args=(Hrelsidx,Hrels),jac=nbpt2Dproc.globalPoseCost_lsq_jac ,method='trf',xtol=xtol)
    # res = least_squares(nbpt2Dproc.globalPoseCost_lsq, x0,args=(Hrelsidx,Hrels),jac=nbpt2Dproc.globalPoseCost_lsq_jac ,method='lm',xtol=1e-4,ftol=1e-4) #max_nfev
    # res = least_squares(nbpt2Dproc.globalPoseCost_lsq, x0,args=(Hrelsidx,Hrels),method='lm')
    
    # res = least_squares(nbpt2Dproc.globalPoseCostHess_lsq, x0,args=(Hrelsidx,Hrels,Hessrels) ,method='lm')
    # res = least_squares(nbpt2Dproc.globalPoseCost_lsq, x0,args=(Hrelsidx,Hrels) ,method='trf',loss='huber')
    # res = least_squares(nbpt2Dproc.globalPoseCostHess_lsq, x0,args=(Hrelsidx,Hrels,Hessrels) ,method='trf',loss='huber')
    if algo in ['BFGS','CG','Newton-CG']:
        res = minimize(nbpt2Dproc.globalPoseCost_Fjac, x0,args=(Hrelsidx,Hrels,Hessrels) ,jac=True,method=algo,options={'maxiter':maxiter}) #
    
    # res = minimize(nbpt2Dproc.globalPoseCost, x0,args=(Hrelsidx,Hrels) ,method='BFGS', tol=1e-4,options={'maxiter':3})
    
    
    
    success=res.success
    et=time.time()
    print(" time loop-closure-opt: ",et-st)


    
    x = res.x
    x=x.reshape(-1,3)   
    sHg_updated={}
    for j in range(1,len(Lkey)):     
        thj=x[j-1,0] # global
        tj=x[j-1,1:] # global
    
        fHj=nbpt2Dproc.getHmat(thj,tj)
        jHf=nplinalg.inv(fHj)
        sHg_updated[Lkey[j]] = np.matmul(jHf,firstHg)
        
    return res,sHg_updated,sHg_original

def updateGlobalPoses(poseGraph,sHg_updated,updateRelPoses=True):
    # loop closure optimization
    # Lkeycycle = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))    
    # Lkeycycle=list(nx.simple_cycles(poseGraph.subgraph(Lkeycycle)))
    # Lkeycycle=[sorted(m) for m in Lkeycycle]
    # Lkeycycle.sort(key=lambda x: len(x))
    

    # first update the global poses of the updated ones
    for ns in sHg_updated.keys():
        sHg = sHg_updated[ns]
        gHs=nplinalg.inv(sHg)
        tpos=np.matmul(gHs,np.array([0,0,1]))
        poseGraph.nodes[ns]['pos'] = (tpos[0],tpos[1])
        poseGraph.nodes[ns]['sHg'] = sHg

        
    # # now update the relative poses of the ones between the updated poses
    
    # for n1 in sHg_updated.keys():
    #     for n2 in poseGraph.successors(n1):
    #         if n2 in sHg_updated.keys():
    #             # if poseGraph.edges[n1,n2]['edgetype']=="Key2Key-LoopClosure":
    #             #     continue
    #             n1Hg = poseGraph.nodes[n1]['sHg']
    #             n2Hg = poseGraph.nodes[n2]['sHg']
    #             Hnew = np.matmul(n2Hg,nplinalg.inv(n1Hg))
                
    #             if (n1,n2) in poseGraph.edges:
    #                 poseGraph.edges[n1,n2]['H'] = Hnew
                    
    
    # lastupdatedNode = max(sHg_updated.keys())
    
    # # now update other frames key/scan frames that were not part of the optimization
    # # Lkeys = list(filter(lambda x: x>lastupdatedNode,Lkeys))
    # # Lkeys.sort()
    
    # for ns in list(poseGraph.nodes):
    #     for pidx in poseGraph.predecessors(ns):
    #         if poseGraph.nodes[pidx]['frametype']=="keyframe": # and pidx in sHg_updated
    #             if poseGraph.edges[pidx,ns]['edgetype']=="Key2Scan": #poseGraph.edges[pidx,ns]['edgetype']=="Key2Key" or 
    #                 psHg=poseGraph.nodes[pidx]['sHg']
    #                 nsHps=poseGraph.edges[pidx,ns]['H']
    #                 nsHg = nsHps.dot(psHg)
    #                 poseGraph.nodes[ns]['sHg']=nsHg
    #                 gHns=nplinalg.inv(nsHg)
    #                 tpos=np.matmul(gHns,np.array([0,0,1]))
    #                 poseGraph.nodes[ns]['pos'] = (tpos[0],tpos[1])

                    
                 
                    
    #                 break
    
    return poseGraph



def updated_sHg(poseGraph):
    for ns in list(poseGraph.nodes):
        for pidx in poseGraph.predecessors(ns):
            if poseGraph.nodes[pidx]['frametype']=="keyframe": # and pidx in sHg_updated
                if poseGraph.edges[pidx,ns]['edgetype']=="Key2Key" or poseGraph.edges[pidx,ns]['edgetype']=="Key2Scan":
                    psHg=poseGraph.nodes[pidx]['sHg']
                    nsHps=poseGraph.edges[pidx,ns]['H']
                    nsHg = nsHps.dot(psHg)
                    poseGraph.nodes[ns]['sHg']=nsHg
                    gHns=nplinalg.inv(nsHg)
                    tpos=np.matmul(gHns,np.array([0,0,1]))
                    poseGraph.nodes[ns]['pos'] = (tpos[0],tpos[1])
                    
                    if 'modified' not in poseGraph.nodes[ns]:
                        poseGraph.nodes[ns]['modified']=[]
                        
                    poseGraph.nodes[ns]['modified'].append('pos')
                    poseGraph.nodes[ns]['modified'].append('sHg')
                    
                    break
    
    return poseGraph

#%%        


if __name__=="__main__":
    # scanfilepath = 'C:/Users/Nagnanamus/Google Drive/repos/SLAM/lidarprocessing/houseScan_std.pkl'
    scanfilepath = 'lidarprocessing/houseScan_std.pkl'
    with open(scanfilepath,'rb') as fh:
        dataset=pkl.load(fh)
    
    def getscanpts(dataset,idx):
        # ranges = dataset[i]['ranges']
        rngs = np.array(dataset[idx]['ranges'])
        
        angle_min=dataset[idx]['angle_min']
        angle_max=dataset[idx]['angle_max']
        angle_increment=dataset[idx]['angle_increment']
        ths = np.arange(angle_min,angle_max+angle_increment,angle_increment)
        p=np.vstack([np.cos(ths),np.sin(ths)])
        
        rngidx = (rngs>= dataset[idx]['range_min']) & (rngs<= dataset[idx]['range_max'])
        ptset = rngs.reshape(-1,1)*p.T
        # ptset=np.ascontiguousarray(ptset,dtype=dtype)
        return ptset[rngidx,:]
    X1=getscanpts(dataset,0)
    X2=getscanpts(dataset,50)
    
    Xd,m = get0meanIcov(X1)
    clf,MU,P,W = getclf(Xd)
    sHk_prevframe = np.identity(3)
    
    sHk,err,hess_inv = scan2keyframe_match(clf,m,X2,sHk=sHk_prevframe)
    
    st=time.time()
    sHk,err,hess_inv = scan2keyframe_match(clf,m,X2,sHk=sHk_prevframe)
    et = time.time()
    print("time taken: ",et-st)
    
    x=np.array([np.pi/4,0.1,0.1])
    
    
    