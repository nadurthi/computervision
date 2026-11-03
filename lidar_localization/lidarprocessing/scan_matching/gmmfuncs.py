# -*- coding: utf-8 -*-

# from numba import set_num_threads,config, njit, threading_layer,get_num_threads
# import numpy as np
# set_num_threads(1)

from numba import cuda, float32

import numpy as np
import math
import numba
from numba import vectorize, float64,guvectorize,int64,double,int32
import numpy.linalg as nplg
from numpy.linalg import multi_dot
from numba import njit, prange,jit
import time 

numba_cache = False

dtype=np.float64


# @vectorize([float64(float64, float64)],nopython=True)
# def f(x, y):
#     return x + y

# @guvectorize([(int64[:], int64, int64[:])], '(n),()->(n)')
# def g(x, y, res):
#     for i in range(x.shape[0]):
#         res[i] = x[i] + y

# @guvectorize([(float64[:,:], float64[:], float64[:,:], float64[:])], '(n,m),(m),(m,m)->(n)', nopython=True, target="cpu")
# def gaussianpdf_guvec(X, mu,P, res):
#     invP = nplg.inv(P)
#     denom = 1/np.sqrt(nplg.det(2*np.pi*P))  
#     for i in range(X.shape[0]):
#         z=X[i,:]-mu
#         x=np.dot(z,invP)
#         y=np.dot(x,z)
#         res[i] = denom*np.exp(-0.5*y)



# def gaussian_pdf_eval(X,mu,P):
# , nopython=True, nogil=True, parallel=True    


@jit(float64[:](float64[:,:], float64[:,:], float64[:,:,:],float64[:]),nopython=True, nogil=True,parallel=False,cache=True )
def gmm_eval(X,MU,PP,W):
    # numba eval
    # X vector
    # MU vector of means
    # PP is concatenated covariances
    ncomp = MU.shape[0]
    dim = MU.shape[1]
    npt = X.shape[0]
    invPP=np.zeros_like(PP)
    denom = np.zeros(ncomp)
    for i in range(ncomp):
        invPP[i] = nplg.inv(PP[i])
        denom[i] = W[i]*1/np.sqrt(nplg.det(2*np.pi*PP[i]))    
    
    s=np.zeros(npt,dtype=dtype)
    
    for i in numba.prange(ncomp):   
        for j in numba.prange(npt):
            z=X[j,:]-MU[i]
            x=np.dot(z,invPP[i])
            y=np.dot(x,z)
            s[j]=s[j]+denom[i]*np.exp(-0.5*y)
    
    # ss = np.sum(s,axis=1)
    return s

@jit(float64[:](float64[:,:], float64[:,:], float64[:,:,:],float64[:]),nopython=True, nogil=True,parallel=False,cache=True )
def gmm_eval_fast(X,MU,PP,W):
    # numba eval
    # X vector
    # MU vector of means
    # PP is concatenated covariances
    ncomp = MU.shape[0]
    dim = MU.shape[1]
    npt = X.shape[0]
    invPP=np.zeros_like(PP)
    denom = np.zeros(ncomp)
    for i in range(ncomp):
        invPP[i] = nplg.inv(PP[i])
        denom[i] = W[i]*1/np.sqrt(nplg.det(2*np.pi*PP[i]))    
    
    s=np.zeros((npt,ncomp),dtype=dtype)
    
    for i in numba.prange(ncomp):   
        z=X-MU[i]
        x=np.dot(z,invPP[i])
        y=x*z
        g=np.sum(y,axis=1)
        s[:,i]=s[:,i]+denom[i]*np.exp(-0.5*g)
    
    ss = np.sum(s,axis=1)
    return ss

@jit(float64[:,:](float64[:,:], float64[:,:], float64[:,:,:],float64[:]),nopython=True, nogil=True,parallel=False,cache=True )
def gmm_evalcomp_fast(X,MU,PP,W):
    # numba eval
    # X vector
    # MU vector of means
    # PP is concatenated covariances
    ncomp = MU.shape[0]
    dim = MU.shape[1]
    npt = X.shape[0]
    invPP=np.zeros_like(PP)
    denom = np.zeros(ncomp)
    for i in range(ncomp):
        invPP[i] = nplg.inv(PP[i])
        denom[i] = W[i]*1/np.sqrt(nplg.det(2*np.pi*PP[i]))    
    
    s=np.zeros((npt,ncomp),dtype=dtype)
    
    for i in numba.prange(ncomp):   
        z=X-MU[i]
        x=np.dot(z,invPP[i])
        y=x*z
        g=np.sum(y,axis=1)
        s[:,i]=s[:,i]+denom[i]*np.exp(-0.5*g)
    
    
    return s



if __name__ == '__main__': 
    import uq.uqutils.random as uqrnd
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from uq.gmm import gmmbase as uqgmm
    
    plt.close('all')
    
    ncomp=100
    npt=10000
    dim=5
    X=np.random.randn(npt,dim)
    X=np.ascontiguousarray(X,dtype=dtype)
    
    MU=np.random.randn(ncomp,dim)       
    MU=np.ascontiguousarray(MU,dtype=dtype)
    
    P =np.zeros((ncomp,dim,dim))
    P=np.ascontiguousarray(P,dtype=dtype)
    
    W = np.ones(ncomp)/ncomp
    W=np.ascontiguousarray(W,dtype=dtype)
    
    for i in range(ncomp):
        P[i]=uqrnd.genRandomCov(dim,meanP=None,sigP=None)
        
    
    s1=gmm_eval(X,MU,P,W)
    st=time.time()
    s2=gmm_eval(X,MU,P,W)
    print("time taken : ",time.time()-st)
    
    s3=gmm_eval_fast(X,MU,P,W)
    st=time.time()
    s4=gmm_eval_fast(X,MU,P,W)
    print("time taken : ",time.time()-st)
    
    st=time.time()
    gmb = uqgmm.GMM(MU,P,W,0)
    sb=gmb.pdf(X)
    print("time taken : ",time.time()-st)
    
    s4-sb
    
    fig=plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(X[:,0],X[:,1],s4,'bo')
    
    
    
