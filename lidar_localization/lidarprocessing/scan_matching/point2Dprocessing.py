# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 21:11:58 2020

@author: nadur
"""

# from numba import set_num_threads,config, njit, threading_layer,get_num_threads
import numpy as np
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
import gmmfuncs as uqgmmfnc
import pdb
import networkx as nx

numba_cache=False


dtype=np.float64
#%% Pose estimation by keyframe
# - manage keyframes
# - estimate pose to closest keyframe

def get0meanIcov(X):
    m=np.mean(X,axis=0)
    Xd=X-m
    return Xd,m

def getclf(X,reg_covar=0.001):
    clf = mixture.GaussianMixture(n_components=25, covariance_type='full',reg_covar=reg_covar)
    clf.fit(X)
    MU=np.ascontiguousarray(clf.means_,dtype=dtype)
    P=np.ascontiguousarray(clf.covariances_,dtype=dtype)
    W=np.ascontiguousarray(clf.weights_,dtype=dtype)
    
    ncomp = MU.shape[0]
    dim = MU.shape[1]
    npt = X.shape[0]
    invPP=np.zeros_like(P)
    for i in range(ncomp):
        invPP[i] = nplinalg.inv(P[i])

    # remcomp=[]
    # for i in range(len(W)):
    #     z=X-MU[i]
    #     x=np.dot(z,invPP[i])
    #     y=x*z
    #     g=np.sum(y,axis=1)
    #     if np.sum(g<2)<3:
    #         remcomp.append(i)
    # remcomp = np.array(remcomp,dtype=np.int)
    
    # MU = np.delete(MU,remcomp,axis=0)
    # P = np.delete(P,remcomp,axis=0)
    # W = np.delete(W,remcomp,axis=0)
    # W=W/np.sum(W)
               
    return clf,MU,P,W

#%%
@jit(float64(float64[:], float64[:,::1], float64[:,:], float64[:,:,:],float64[:]),nopython=True, nogil=True,cache=True) 
def getcostalign(x,Xt,MU,P,W):
    # Xt are the red points
    th=x[0]
    t=np.array([x[1],x[2]])
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    Xi=np.dot(R,Xt.T).T
    Xn=Xi+t
    p=uqgmmfnc.gmm_eval_fast(Xn,MU,P,W)+1
    logp = -np.log(p)
    m = np.mean(logp)
    # logp = np.diag(f(Xn[:,0],Xn[:,1]))
    return m


def getgridvec(thset,txset,tyset):
    # convert all possible combinations into a column vector with 3 columns ([th, tx,ty])
    th,x,y=np.meshgrid(thset,txset,tyset)
    Posegrid = np.vstack([th.ravel(), x.ravel(), y.ravel()]).T
    Posegrid=np.ascontiguousarray(Posegrid,dtype=dtype)
    return Posegrid
#%%
@jit(float64[:](float64[:,:], float64[:,:], float64[:,:,:],float64[:], float64[:,::1]),nopython=True, nogil=True,parallel=False,cache=True) 
def gridsearch_alignment(Posegrid,MU,P,W,X):
    # brute force way to find best pose.
    # evaluate getcostalign for each pose and pick the minimum error combinations
    m=np.zeros(Posegrid.shape[0])
    for i in numba.prange(Posegrid.shape[0]):
        m[i]=getcostalign(Posegrid[i],X,MU,P,W)
    ind = np.argmin(m)
    res = Posegrid[ind]
    return res

#%%
@jit(numba.types.Tuple((float64,float64[:]))(float64[:], float64[:,::1], float64[:,:], float64[:,:,:],float64[:]),nopython=True, nogil=True,parallel=False,cache=True) 
def getcostgradient(x,Xt,MU,P,W):
    # Xt the red points
    # this is function is used with BFGS optimizer with gradient
    npt=Xt.shape[0]
    ncomp=MU.shape[0]
    
    th=x[0]
    t=np.array([x[1],x[2]])
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    Xi=np.dot(R,Xt.T).T
    Xn=Xi+t
    pcomp=uqgmmfnc.gmm_evalcomp_fast(Xn,MU,P,W)
    p = np.sum(pcomp,axis=1)+1
    logp = -np.log(p)
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


@jit(numba.types.Tuple((float64,float64[:]))(float64[:], float64[:,::1], float64[:,:], float64[:,:,:],float64[:]),nopython=True, nogil=True,parallel=False,cache=True) 
def getcostgradient_prop(x,Xt,MU,P,W):
    npt=Xt.shape[0]
    ncomp=MU.shape[0]
    
    th=x[0]
    t=np.array([x[1],x[2]])
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    Xi=np.dot(R,Xt.T).T
    Xn=Xi+t
    pcomp=uqgmmfnc.gmm_evalcomp_fast(Xn,MU,P,W)
    p = np.sum(pcomp,axis=1)+1
    # logp = -np.log(p)
    f = np.mean(p)    
    
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
    
    a1 = np.sum(pcomp*dpdth,axis=1)
    a2 = np.sum(pcomp*dpdx,axis=1)
    a3 = np.sum(pcomp*dpdy,axis=1)
    dth=np.mean(a1)
    dx=np.mean(a2)
    dy=np.mean(a3)
    g=np.array([dth,dx,dy])
    return -f,-g

# def getcostgradient_wrapper(x,Xt,MU,P,W):
#     g=getcostgradient(x,Xt,MU,P,W)
#     return g[0],g[1:]

#%%

@jit(numba.types.Tuple((float64[:],float64))(float64[:,:]),nopython=True, nogil=True,parallel=False,cache=True) 
def extractPosAngle(H):
    theta = np.arctan2(-H[0,1],H[0,0])
    txy = H[:2,2]
    return txy,theta

@jit(float64[:,:](float64, float64[:]),nopython=True, nogil=True,parallel=False,cache=True) 
def getHmat(th,t):
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    H=np.identity(3)
    H[0:2,0:2]=R
    H[0:2,2]=t
    return H

#%%
def alignscan2keyframe(thset,txset,tyset,MU,P,W,X):
    # get sHk: takes a point from keyframe k to scan frame s
    
    # pdb.set_trace()
    # st=time.time()

    # et=time.time()
    # print("time taken by gridsearch : ",et-st)
    
    # st=time.time()
    # res = minimize(getcostalign, best,args=(X,MU,P,W) ,method='BFGS', tol=1e-5) # 'Nelder-Mead'
    # et=time.time()
    # print("time taken by minimize : ",et-st)
    
    # st=time.time()
    best=np.zeros(3)
    res = minimize(getcostgradient, best,args=(X,MU,P,W),jac=True ,method='BFGS', tol=1e-5) # 'Nelder-Mead'
    if np.isfinite(res.fun)==0 or np.all(np.isfinite(res.x))==0 or res.success is False:
        print("BFGS scan to scan failed")
        Posegrid = getgridvec(thset,txset,tyset)
        best=gridsearch_alignment(Posegrid,MU,P,W,X)

        res = minimize(getcostgradient, best,args=(X,MU,P,W),jac=True ,method='SLSQP', tol=1e-5) # 'Nelder-Mead'
    
    # et=time.time()
    # print("time taken by gradient minimize : ",et-st)
    
    # print(res2.fun,res.fun)
    # print(res2.x,res.x)
    
    th=res.x[0]
    t=res.x[1:]
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]],dtype=dtype,order='C')

    kHs=np.hstack([R,t.reshape(2,1)])
    kHs = np.vstack([kHs,[0,0,1]])
    sHk = nplinalg.inv(kHs)
    err=res.fun
    
    return sHk,err

def scan2keyframe_match(KeyFrameClf,m_clf,X,sHk=np.identity(3),doGridInit=False):
    # KeyFrameClf is the just the Gaussian mixture computed from a set of black points
    # the black points have mean m_clf
    # m_clf is the shift in the centroid
    # for the red points X, just do X-m_clf
    # 
    # sHrelk is the transformation from k to s
    if doGridInit is True:
        thset = np.linspace(-np.pi/3,np.pi/3,9)
        txset = np.linspace(-0.4,0.4,9) # 
        tyset = np.linspace(-0.4,0.4,9) #
    else:
        thset = np.linspace(-np.pi/3,np.pi/3,9)
        txset = np.linspace(-0.2,0.2,9) # 
        tyset = np.linspace(-0.2,0.2,9) #
        
    kHs = nplinalg.inv(sHk)
    Xk=np.matmul(kHs,np.vstack([X.T,np.ones(X.shape[0])])).T  
    Xk=Xk[:,:2]
    Xd=Xk-m_clf
    MU=np.ascontiguousarray(KeyFrameClf.means_,dtype=dtype)
    P=np.ascontiguousarray(KeyFrameClf.covariances_,dtype=dtype)
    W=np.ascontiguousarray(KeyFrameClf.weights_,dtype=dtype)
    Xd = np.ascontiguousarray(Xd,dtype=dtype) 
    sHk_rel,err = alignscan2keyframe(thset,txset,tyset,MU,P,W,Xd)
    if np.all(np.isfinite(sHk_rel))==0 or np.isfinite(err)==0:
        print("scan failed")
        # pdb.set_trace()
        
    sHk_corrected=np.matmul(sHk,sHk_rel)
    return sHk_corrected,err

def poseGraph_keyFrame_matcher(poseGraph,idx1,idx2):
    # fromidx is idx1, toidx is idx2 
    clf1=poseGraph.nodes[idx1]['clf']
    m_clf1=poseGraph.nodes[idx1]['m_clf']
    sHg_1 = poseGraph.nodes[idx1]['sHg']
    sHg_2 = poseGraph.nodes[idx2]['sHg']
    H21_est = np.matmul(sHg_2,nplinalg.inv(sHg_1))
    X2 = poseGraph.nodes[idx2]['X']
    H21,err21 = scan2keyframe_match(clf1,m_clf1,X2,sHk=H21_est,doGridInit=True)
    
    return H21,err21

#%% Pose estimation by tree
    
def getcostalign_tree(x,Xt,tree):
    th=x[0]
    t=x[1:]
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    Xn=np.matmul(R,Xt.T).T+t
    dist, ind = tree.query(Xn, k=1)  
    d = np.mean(np.mean(dist,axis=1))
    return d

def alignscan2tree(thset,txset,tyset,base_tree,X):
    # get sHk: takes a point from keyframe k to scan frame s
    m=1e10
    best=[thset[0],txset[0],tyset[0]]
    for th in thset:
        for tx in txset:
            for ty in tyset:
                c=getcostalign_tree([th,tx,ty],X,base_tree)
                if c < m:
                    m=c
                    best=[th,tx,ty] 
    
 
    res = minimize(getcostalign_tree, best,args=(X,base_tree) ,method='Nelder-Mead', tol=1e-5)
    th=res.x[0]
    t=res.x[1:]

    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])

    kHs=np.hstack([R,t.reshape(2,1)])
    kHs = np.vstack([kHs,[0,0,1]])
    sHk = nplinalg.inv(kHs)
    err=res.fun
    
    return sHk,err

#%% generic method entry functions
def scan2scan_match(X1,X2,method="GMM",H21=None,skipgrid=False):
    # estimate H21 from 1 to 2
    # method="GMM"
    # method="tree"
    if H21 is None:
        H21 = np.array([[1,0,0],[0,1,0],[0,0,1]])
        thset = np.linspace(-2*np.pi/3,2*np.pi/3,13)
        txset = np.linspace(-0.7,0.7,5) # 
        tyset = np.linspace(-0.7,0.7,5) #
    else:
        thset = np.linspace(-np.pi/4,np.pi/4,7)
        txset = np.linspace(-0.3,0.3,5) # 
        tyset = np.linspace(-0.3,0.3,5) #
    
    if skipgrid:
        thset = [0]
        txset = [0] # 
        tyset = [0] #
        
    H12 = nplinalg.inv(H21)
    X12=np.matmul(H12,np.vstack([X2.T,np.ones(X2.shape[0])])).T   
    X12=X12[:,:2]
    
    
    
    if method=='tree':
        tree = KDTree(X1, leaf_size=3)  
        H21_rel,err = alignscan2tree(thset,txset,tyset,tree,X12)
        H21_corrected=np.matmul(H21,H21_rel)
    if method=='GMM':
        clf,MU,P,W=getclf(X1)
        H21_rel,err = alignscan2keyframe(thset,txset,tyset,MU,P,W,X12)
        H21_corrected=np.matmul(H21,H21_rel)
    
    return H21_corrected,err


#%% features
bins=np.arange(0,11,0.25)
db=bins[1]-bins[0]
def get2DptFeat(X):
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
        
@jit(float64(float64[:],int32[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=True) 
def globalPoseCost(x,Hrelsidx,Hrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    y=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        y[s+1,:]= x[s*3:s*3+3]
    

    th=y[:,0] # global
    txy=y[:,1:] # global
    F=np.zeros(Hrels.shape[0])

        
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
        
        fHi=getHmat(thi,ti)
        fHj=getHmat(thj,tj)
        
        jHf = nplinalg.inv(fHj)
        
        e=(jHi-jHf.dot(fHi))
        eR=e[0:2,0:2]
        et=e[0:2,2]
        F[idx] = 1*(eR[0,0]**2+eR[1,1]**2) + 1*(et[0]**2 + et[1]**2)
    
    f=np.sum(F)
    # f=1.3
    return f

@jit(float64[:](float64[:],int32[:,:],float64[:,:]),nopython=True, nogil=True,parallel=False,cache=True) 
def globalPoseCost_lsq(x,Hrelsidx,Hrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    # x=x.reshape(-1,3)
    
    # z=x.reshape(-1,3)
    # y=np.vstack([np.zeros((1,3)),x])     
    
    y=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        y[s+1,:]= x[s*3:s*3+3]
    

    th=y[:,0] # global
    txy=y[:,1:] # global
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
        
        fHi=getHmat(thi,ti)
        fHj=getHmat(thj,tj)
        
        jHf = nplinalg.inv(fHj)
        jHi_var=jHf.dot(fHi)
        
        tji_var,thji_var=extractPosAngle(jHi_var)
        

        F[idx*3] = thji_var-thji
        F[idx*3+1] = tji_var[0]-tji[0]
        F[idx*3+2] = tji_var[1]-tji[1]
        
    # f=np.sum(F)
    # f=1.3
    return F

def adjustPoses(poseGraph,idx1,idx2):
    # adjust the bundle from idx1 to idx2
    # i.e. optimize the global pose matrices, with constraint local poses 
    # idx1 is assumed to be fixed (like origin)
    Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    Lkey = [x for x in Lkey if x>=idx1 and x<=idx2]
    Lkey.sort()
    
    Lkey_dict={x:i for i,x in enumerate(Lkey)}
    
    Hrels=[] # Hrels=[[i,j,thji,txji,tyji],...]
    Hrelsidx=[]
    for j in range(1,len(Lkey)):
        idx = Lkey[j]
        for idx_prev in poseGraph.predecessors(idx):
            if idx_prev not in Lkey_dict:
                continue
            i = Lkey_dict[idx_prev]
            jHi=poseGraph.edges[idx_prev,idx]['H']
            tji,thji = extractPosAngle(jHi)
            Hrels.append([thji,tji[0],tji[1] ])        
            Hrelsidx.append([i,j])
    Hrels = np.array(Hrels,dtype=dtype)
    Hrelsidx = np.array(Hrelsidx,dtype=np.int32)
    
    # x0=x0.reshape(-1)
    x0 = np.zeros((len(Lkey)-1,3))            
    firstHg=poseGraph.nodes[Lkey[0]]['sHg']
    gHfirst = nplinalg.inv(firstHg)
    sHg_original={}
    bounds=[[-np.pi/4,np.pi/4],[-2,2],[-2,2]]*(len(Lkey)-1)
    for j in range(1,len(Lkey)):
        idx = Lkey[j]
        sHg=poseGraph.nodes[idx]['sHg']
        sHg_original[idx] = sHg
        
        jHf=np.matmul(sHg,gHfirst) # first to j (local global frame)
        fHj=nplinalg.inv(jHf)
        tj,thj = extractPosAngle(fHj)
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
    x0 = x0+0.001*np.random.randn(len(x0))
    
    lb = np.array([bnd[0] for bnd in bounds])
    ub = np.array([bnd[1] for bnd in bounds])
    

    # st=time.time()
    # res = minimize(globalPoseCost, x0,args=(Hrelsidx,Hrels) ,method='BFGS', tol=1e-5)
    # success=res.success
    # et=time.time()
    # print(res.fun," time: ",et-st)

    st=time.time()
    res = least_squares(globalPoseCost_lsq, x0,args=(Hrelsidx,Hrels) ,method='lm')
    success=res.success
    et=time.time()
    print(" time loop-closure-opt: ",et-st)


    
    x = res.x
    x=x.reshape(-1,3)   
    sHg_updated={}
    for j in range(1,len(Lkey)):     
        thj=x[j-1,0] # global
        tj=x[j-1,1:] # global
    
        fHj=getHmat(thj,tj)
        jHf=nplinalg.inv(fHj)
        sHg_updated[Lkey[j]] = np.matmul(jHf,firstHg)
        
    return res,sHg_updated,sHg_original

def updateGlobalPoses(poseGraph,sHg_updated):
    # loop closure optimization
    # Lkeycycle = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))    
    # Lkeycycle=list(nx.simple_cycles(poseGraph.subgraph(Lkeycycle)))
    # Lkeycycle=[sorted(m) for m in Lkeycycle]
    # Lkeycycle.sort(key=lambda x: len(x))
    


    for ns in sHg_updated.keys():
        sHg = sHg_updated[ns]
        gHs=nplinalg.inv(sHg)
        tpos=np.matmul(gHs,np.array([0,0,1]))
        poseGraph.nodes[ns]['pos'] = (tpos[0],tpos[1])
        poseGraph.nodes[ns]['sHg'] = sHg
        
    # now update other non-keyframes
    for ns in poseGraph.nodes:
        if poseGraph.nodes[ns]['frametype']!="keyframe":
            for pidx in poseGraph.predecessors(ns):
                if poseGraph.nodes[pidx]['frametype']=="keyframe" and pidx in sHg_updated:
                    psHg=poseGraph.nodes[pidx]['sHg']
                    nsHps=poseGraph.edges[pidx,ns]['H']
                    nsHg = nsHps.dot(psHg)
                    poseGraph.nodes[ns]['sHg']=nsHg
                    gHns=nplinalg.inv(nsHg)
                    tpos=np.matmul(gHns,np.array([0,0,1]))
                    poseGraph.nodes[ns]['pos'] = (tpos[0],tpos[1])
        
                    break
    
    return poseGraph


def adjustPoses_byPtAlign(posegraph,idx1,idx2):
    # adjust the bundle from idx1 to idx2
    # i.e. optimize the global pose matrices, with Gaussian overlap with scan data
    # idx1 is assumed to be fixed (like origin)
    pass


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
    
    sHk,err = scan2keyframe_match(clf,m,X2,sHk=sHk_prevframe)
    
    st=time.time()
    sHk,err = scan2keyframe_match(clf,m,X2,sHk=sHk_prevframe)
    et = time.time()
    print("time taken: ",et-st)
    
    x=np.array([np.pi/4,0.1,0.1])
    
    
    