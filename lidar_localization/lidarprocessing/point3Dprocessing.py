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
numba_cache=True
dtype=np.float32

from numba.core import types
from numba.typed import Dict
float_2Darray = types.float32[:,:]



#%%

# @guvectorize([(int64[:], int64, int64[:])], '(n),()->(n)')
# def get(X, mu, P):
#     for i in range(X.shape[0]):
#         pass
        
# @jit([numba.types.Tuple((float32,float32[:]))(float32[:], float32[:,:], float32[:,:], float32[:,:,:],float32[:]),
#       numba.types.Tuple((float32,float32[:]))(float64[:], float32[:,:], float32[:,:], float32[:,:,:],float32[:])],
#       nopython=True, fastmath=True,nogil=True,parallel=True,cache=False) 
def getcostgradient3D(x,Xt,MU,P,W):
    x=x.astype(dtype)
    npt=Xt.shape[0]
    ncomp=MU.shape[0]
    
    t=x[0:3]
    w=x[3:]
    th=np.sqrt(w[0]**2+w[1]**2+w[2]**2)
    w=w/th
    K=np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]],dtype=dtype)
    K2=K.dot(K)
    
    
    R = np.identity(3)+np.sin(th)*K+(1-np.cos(th))*K2
    R=R.astype(dtype)
    
    Xn=R.dot(Xt.T).T+t

    invPP=np.zeros_like(P)
    denom = np.zeros(ncomp,dtype=dtype)
    for i in range(ncomp):
        invPP[i] = nplinalg.inv(P[i])
        denom[i] = W[i]*1/np.sqrt(nplinalg.det(2*np.pi*P[i]))    
    
    # pcomp=np.zeros((npt,ncomp),dtype=dtype)
    

        

    
    invPP = np.zeros_like(P)
    for i in range(ncomp):
        invPP[i] = nplinalg.inv(P[i])
        
    dKdw1=np.array([[0,0,0],[0,0,-1],[0,1,0]],dtype=dtype)
    dKdw2=np.array([[0,0,1],[0,0,0],[-1,0,0]],dtype=dtype)
    dKdw3=np.array([[0,-1,0],[1,0,0],[0,0,0]],dtype=dtype)
    
    w1=w[0]
    w2=w[1]
    w3=w[2]
    
    D=np.array([[-w3**2-w2**2,w1*w2,w1*w3],[w1*w2,-w3**2-w1**2,w3*w2],[w1*w3,w2*w3,-w2**2-w1**2]],dtype=dtype)
    wpow4=th**4
    wpow2=th**2
    
    dK2dw1=-2*(w1/wpow4)*D+(1/wpow2)*np.array([[0,w2,w3],[w2,-2*w1,0],[w3,0,-2*w1]],dtype=dtype)
    dK2dw2=-2*(w2/wpow4)*D+(1/wpow2)*np.array([[-2*w2,w1,0],[w1,0,w3],[0,w3,-2*w2]],dtype=dtype)
    dK2dw3=-2*(w3/wpow4)*D+(1/wpow2)*np.array([[-2*w3,0,w1],[0,-2*w3,w2],[w1,w2,0]],dtype=dtype)
    
    dRdw1=np.cos(th)*w1/th*K+np.sin(th)*dKdw1+np.sin(th)*w1/th*K2+(1-np.cos(th))*dK2dw1
    dRdw2=np.cos(th)*w2/th*K+np.sin(th)*dKdw2+np.sin(th)*w2/th*K2+(1-np.cos(th))*dK2dw2
    dRdw3=np.cos(th)*w3/th*K+np.sin(th)*dKdw3+np.sin(th)*w3/th*K2+(1-np.cos(th))*dK2dw3
    
    zz1=np.dot(dRdw1.astype(dtype),Xt.T).T
    zz2=np.dot(dRdw2.astype(dtype),Xt.T).T
    zz3=np.dot(dRdw3.astype(dtype),Xt.T).T
    
    # dpdw1=np.zeros((npt,ncomp),dtype=dtype)
    # dpdw2=np.zeros((npt,ncomp),dtype=dtype)
    # dpdw3=np.zeros((npt,ncomp),dtype=dtype)
    # dpdx=np.zeros((npt,ncomp),dtype=dtype)
    # dpdy=np.zeros((npt,ncomp),dtype=dtype)
    # dpdz=np.zeros((npt,ncomp),dtype=dtype)
    
    a1=np.zeros(npt,dtype=dtype)
    a2=np.zeros(npt,dtype=dtype)
    a3=np.zeros(npt,dtype=dtype)
    a4=np.zeros(npt,dtype=dtype)
    a5=np.zeros(npt,dtype=dtype)
    a6=np.zeros(npt,dtype=dtype)
    
    p=np.zeros(npt,dtype=dtype)
    
    for i in range(ncomp): 

        
        z2=Xn-MU[i]
        z3=np.dot(invPP[i],z2.T).T
        
        # g = np.sum(z2*z3,axis=1)
        gg=np.exp(-0.5*np.sum(z2*z3,axis=1)).astype(dtype)
        ## pp is[npts,1]
        
        p = p +denom[i]*gg
        
        y1=np.multiply(zz1,z3)
        y2=np.multiply(zz2,z3)
        y3=np.multiply(zz3,z3)
        
        # dpdw1[:,i] = -np.sum(y1,axis=1)
        # dpdw2[:,i] = -np.sum(y2,axis=1)
        # dpdw3[:,i] = -np.sum(y3,axis=1)
        
        # dpdx[:,i] = -z3[:,0]
        # dpdy[:,i] = -z3[:,1]
        # dpdz[:,i] = -z3[:,2]
        
        a1+=gg*(-z3[:,0])
        a2+=gg*(-z3[:,1])
        a3+=gg*(-z3[:,2])
        
        a4+=gg*(-np.sum(y1,axis=1))
        a5+=gg*(-np.sum(y2,axis=1))
        a6+=gg*(-np.sum(y3,axis=1))
        
    p +=0.0000000000000001
    logp = -np.log(p)
    # ind = logp<np.mean(logp)
    f = np.mean(logp)    
    
    invp = 1/p  # this is a vec with each element for one point in Xt
    
    
    # a1 = -invp*np.sum(pcomp*dpdx,axis=1)
    # a2 = -invp*np.sum(pcomp*dpdy,axis=1)
    # a3 = -invp*np.sum(pcomp*dpdz,axis=1)
    
    # a4 = -invp*np.sum(pcomp*dpdw1,axis=1)
    # a5 = -invp*np.sum(pcomp*dpdw2,axis=1)
    # a6 = -invp*np.sum(pcomp*dpdw3,axis=1)
    
    
    
    dx=np.mean(-invp*a1)
    dy=np.mean(-invp*a2)
    dz=np.mean(-invp*a3)
    
    dw1=np.mean(-invp*a4)
    dw2=np.mean(-invp*a5)
    dw3=np.mean(-invp*a6)
    
    g=np.array([dx,dy,dz,dw1,dw2,dw3],dtype=dtype)
    return f,g

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

@njit([numba.types.Tuple((float32,float32[:]))(float32[:], float32[:,:], float32[:,:], float32[:,:,:],float32[:]),
      numba.types.Tuple((float32,float32[:]))(float64[:], float32[:,:], float32[:,:], float32[:,:,:],float32[:])],
      nopython=True, fastmath=True,nogil=True,parallel=True,cache=False) 
def getcostgradient3Dypr(x,Xt,MU,P,W):
    x=x.astype(dtype)
    npt=Xt.shape[1]
    ncomp=MU.shape[0]
    t=np.zeros((3,1),dtype=dtype)
    t[0]=x[0]
    t[1]=x[1]
    t[2]=x[2]
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
    

    Xn=R.dot(Xt)+t

    invPP=np.zeros_like(P)
    denom = np.zeros(ncomp,dtype=dtype)
    for i in range(ncomp):
        invPP[i] = nplinalg.inv(P[i])
        denom[i] = W[i]*1/np.sqrt(nplinalg.det(2*np.pi*P[i]))    
    

      
    
    zz1=dRdphi.dot(Xt)
    zz2=dRdxi.dot(Xt)
    zz3=dRdzi.dot(Xt)

    
    a1=np.zeros(npt,dtype=dtype)
    a2=np.zeros(npt,dtype=dtype)
    a3=np.zeros(npt,dtype=dtype)
    a4=np.zeros(npt,dtype=dtype)
    a5=np.zeros(npt,dtype=dtype)
    a6=np.zeros(npt,dtype=dtype)
    
    p=np.zeros(npt,dtype=dtype)
     
    
    M=np.zeros((3,1),dtype=dtype)


    for i in range(ncomp): 

        M[0]=MU[i,0]
        M[1]=MU[i,1]
        M[2]=MU[i,2]
        
        z2=Xn-M
        z3=invPP[i].dot(z2)
        
        # g = np.sum(z2*z3,axis=1)
        gg=np.exp(-0.5*np.sum(z2*z3,axis=0)).astype(dtype)
        ## pp is[npts,1]
        
        p+=denom[i]*gg
        
        y1=zz1*z3
        y2=zz2*z3
        y3=zz3*z3
        
        
        a1+=denom[i]*gg*(-z3[0,:])
        a2+=denom[i]*gg*(-z3[1,:])
        a3+=denom[i]*gg*(-z3[2,:])
        
        a4+=denom[i]*gg*(-np.sum(y1,axis=0))
        a5+=denom[i]*gg*(-np.sum(y2,axis=0))
        a6+=denom[i]*gg*(-np.sum(y3,axis=0))
        
    p =p+1e-20
    logp = -np.log(p)
    f = np.mean(logp)    
    
    invp = 1/p  # this is a vec with each element for one point in Xt
    

    
    g=np.zeros(6,dtype=dtype)
    
    g[0]=dx=np.mean(-invp*a1)
    g[1]=dy=np.mean(-invp*a2)
    g[2]=dz=np.mean(-invp*a3)
    
    g[3]=dw1=np.mean(-invp*a4)
    g[4]=dw2=np.mean(-invp*a5)
    g[5]=dw3=np.mean(-invp*a6)
    
    # g=np.array([dx,dy,dz,dw1,dw2,dw3],dtype=dtype)
    # g=np.array([dx,dy,dz],dtype=dtype)
    
    # return 0,np.zeros(6,dtype=dtype)
    return f,g




@njit([numba.types.Tuple((float32,float32[:]))(float32[:], float32[:,:], float32[:,:], float32[:,:,:],float32[:]),
      numba.types.Tuple((float32,float32[:]))(float64[:], float32[:,:], float32[:,:], float32[:,:,:],float32[:])],
      nopython=True, fastmath=True,nogil=True,parallel=True,cache=False) 
def getcostgradient3Dypr_v2(x,Xt,MU,P,W):
    x=x.astype(dtype)
    npt=Xt.shape[0]
    ncomp=MU.shape[0]

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
    

    Xn=R.dot(Xt.T).T+t

    invPP=np.zeros_like(P)
    denom = np.zeros(ncomp,dtype=dtype)
    for i in range(ncomp):
        invPP[i] = nplinalg.inv(P[i])
        denom[i] = W[i]*1/np.sqrt(nplinalg.det(2*np.pi*P[i]))    
    

      
    
    zz1=dRdphi.dot(Xt.T).T
    zz2=dRdxi.dot(Xt.T).T
    zz3=dRdzi.dot(Xt.T).T

    
    a1=np.zeros(npt,dtype=dtype)
    a2=np.zeros(npt,dtype=dtype)
    a3=np.zeros(npt,dtype=dtype)
    a4=np.zeros(npt,dtype=dtype)
    a5=np.zeros(npt,dtype=dtype)
    a6=np.zeros(npt,dtype=dtype)
    
    p=np.zeros(npt,dtype=dtype)
     
    



    for i in range(ncomp): 


        
        z2=Xn-MU[i]
        z3=invPP[i].dot(z2.T).T
        
        # g = np.sum(z2*z3,axis=1)
        gg=np.exp(-0.5*np.sum(z2*z3,axis=1)).astype(dtype)
        ## pp is[npts,1]
        
        p+=denom[i]*gg
        
        
        
        a1+=denom[i]*gg*(-z3[:,0])
        a2+=denom[i]*gg*(-z3[:,1])
        a3+=denom[i]*gg*(-z3[:,2])
        
        a4+=denom[i]*gg*(-np.sum(zz1*z3,axis=1))
        a5+=denom[i]*gg*(-np.sum(zz2*z3,axis=1))
        a6+=denom[i]*gg*(-np.sum(zz3*z3,axis=1))
        
    p =p+1e-20
    logp = -np.log(p)
    f = np.mean(logp)    
    
    invp = 1/p  # this is a vec with each element for one point in Xt
    

    
    g=np.zeros(6,dtype=dtype)
    
    g[0]=dx=np.mean(-invp*a1)
    g[1]=dy=np.mean(-invp*a2)
    g[2]=dz=np.mean(-invp*a3)
    
    g[3]=dw1=np.mean(-invp*a4)
    g[4]=dw2=np.mean(-invp*a5)
    g[5]=dw3=np.mean(-invp*a6)
    
    # g=np.array([dx,dy,dz,dw1,dw2,dw3],dtype=dtype)
    # g=np.array([dx,dy,dz],dtype=dtype)
    
    # return 0,np.zeros(6,dtype=dtype)
    return f,g


@njit([numba.types.Tuple((float32[:,:],float32[:,:,:],float32[:]))(float32[:,:], float32[:,:], float32[:,:,:], float32[:],int32,float32),
       numba.types.Tuple((float32[:,:],float32[:,:,:],float32[:]))(float32[:,:], float32[:,:], float32[:,:,:], float32[:],int32,float32)],
      nopython=True, fastmath=True,nogil=True,parallel=True,cache=True) 
def gmmEM(X,MU,P,W,nitr,reg):
    npt=X.shape[1]
    ndim=X.shape[0]
    ncomp = MU.shape[0]
    wt=np.zeros((npt,ncomp),dtype=dtype)
    

    M=np.zeros((3,1),dtype=dtype)
    invPP=np.zeros_like(P)
    denom = np.zeros(ncomp,dtype=dtype)
    for itr in range(nitr):
        
        for i in range(ncomp):
            invPP[i] = nplinalg.inv(P[i])
            denom[i] = W[i]*1/np.sqrt(nplinalg.det(2*np.pi*P[i]))  
            
        # E step
        for i in prange(ncomp):
            M[0]=MU[i,0]
            M[1]=MU[i,1]
            M[2]=MU[i,2]
            
            z2=X-M
            z3=invPP[i].dot(z2)
            
            # g = np.sum(z2*z3,axis=1)
            gg=np.exp(-0.5*np.sum(z2*z3,axis=0)).astype(dtype)
            ## pp is[npts,1]
            
            wt[:,i]=denom[i]*gg
            
        # npt,ncomp
        p=np.sum(wt,axis=1)
        wt=(wt.T/p).T
        
        # Q step
        Nk=np.sum(wt,axis=0)
        W=Nk/npt
        
        for i in range(ncomp):
            M[0]=MU[i,0]
            M[1]=MU[i,1]
            M[2]=MU[i,2]
            x=wt[:,i]*(X-M)
            P[i]=(X-M).dot(x.T)/Nk[i]+reg*np.identity(ndim)
            MU[i]= np.sum(wt[:,i]*X,axis=1)/Nk[i]
    
    return MU,P,W 


#%% GMM

@njit
def BCoeff_gassian(m1,P1inv,P1,m2,P2inv,P2):
    a1=m1.dot(P1inv)
    a2=m2.dot(P2inv)
    a = a1+ a2
    b = nplinalg.inv(P1inv+P2inv)
    
    d = np.dot(a1,m1) + np.dot(a2,m2)
    C = 0.5*np.dot(a.dot(b),a)-0.5*d
    bc = np.sqrt( nplinalg.det(2*nplinalg.inv(P1inv+P2inv)) /np.sqrt(nplinalg.det(np.dot(P1,P2))) ) * np.exp(C/2)
    if bc>1:
        bc=1
    return bc

@njit
def BCdist_gassian(m1,P1,m2,P2):
    S=0.5*(P1+P2)
    Sinv=nplinalg.inv(S)
    Sinv=Sinv.astype(dtype)
    Db=1/8*np.dot((m1-m2).dot(Sinv),m1-m2)+0.5*np.log(nplinalg.det(S)/np.sqrt(nplinalg.det(P1)*nplinalg.det(P2)))
    return Db

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
            D[i,j] = BCdist_gassian(MU1[i],P1[i],MU2[j],P2[j])
    
    f=0
    for i in prange(n1): 
        d=np.sort(D[i])
        f+=np.mean(d[d<10])   
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
    
    