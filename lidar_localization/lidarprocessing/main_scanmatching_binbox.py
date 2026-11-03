# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:46:31 2020

@author: nadur
"""

#%%


#%%


dtype = np.float64

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
import copy
from lidarprocessing import point2Dprocessing as pt2dproc
from lidarprocessing import point2Dplotting as pt2dplot
import lidarprocessing.numba_codes.point2Dprocessing_numba as nbpt2Dproc
from sklearn.neighbors import KDTree
import os
import pandas as pd
import heapq
import numpy.linalg as nplinalg 
from lidarprocessing import icp
#%%
plt.close("all")
with open("DeutchesMeuseum_g2oTest_good2.pkl",'rb') as fh:
    poseGraph,params,_=pkl.load(fh)




Lkeyloop_edges = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key-LoopClosure",poseGraph.edges))
e1=Lkeyloop_edges[1][0]
e2=Lkeyloop_edges[1][1]
X1=poseGraph.nodes[e1]['X']
X2=poseGraph.nodes[e2]['X']
H21=np.identity(3) 
th=0*np.pi/180
H21[0:2,2]=[1,1]
R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
H21[0:2,0:2]=R
# H21=poseGraph.edges[e1,e2]['H']
# H21[0:2,2]=H21[0:2,2]+5
# H12 = nplinalg.inv(H21)
# X2=R.dot(X1.T).T+H21[0:2,2]

plt.figure()
plt.plot(X1[:,0],X1[:,1],'b.')
plt.plot(X2[:,0],X2[:,1],'r.')


Lmax=np.array([5,5])
thmax=15*np.pi/180
dxMatch=np.array([0.5,0.5])
# dxMax=np.array([5,5])
st=time.time()
# X2small = pt2dproc.binnerDownSampler(X2,dx=0.2,cntThres=1)
# Hbin21,cost,HLevels=pt2dproc.binMatcherAdaptive(X1,X2,H12,Lmax,thmax,dxMatch)
print("----------------")
H12est = np.identity(3) 
Hbin21,cost,HLevels2=pt2dproc.binMatcherAdaptive2(X1,X2,H12est,Lmax,thmax,dxMatch)
# Hbin21,cost,HLevels,dxs=pt2dproc.binMatcherAdaptive(X1,X2,H12est,Lmax,thmax,dxMatch)
et=time.time()
# X1=poseGraph.nodes[e1]['X']
# X2=poseGraph.nodes[e2]['X']
print("Complete in time = ",et-st)
Hbin12 = nplinalg.inv(Hbin21)
R=Hbin12[0:2,0:2]
t=Hbin12[0:2,2]
X22 = R.dot(X2.T).T+t

plt.figure()
plt.plot(X1[:,0],X1[:,1],'b.')
plt.plot(X22[:,0],X22[:,1],'r.')

plt.show()

#%%
plt.close("all")
with open("DeutchesMeuseum_g2oTest_good2.pkl",'rb') as fh:
    poseGraph,params,_=pkl.load(fh)

with open("DeutchesMuseum-NoLoopICP.pkl",'rb') as fh:
    poseGraph,params=pkl.load(fh)

Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
Lkeyloop_edges = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key",poseGraph.edges))
pt2dplot.plot_keyscan_path(poseGraph,Lkey[0],Lkey[-1],params,makeNew=True,skipScanFrame=True,plotGraphbool=False,
                                    forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True,plotKeyFrameNodesTraj=True)


Lmax=np.array([3,3])
thmax=25*np.pi/180
dxMatch=np.array([0.3,0.3])
H12est = np.identity(3) 
thmin=1*np.pi/180

for ee in Lkeyloop_edges:
    X1=poseGraph.nodes[ee[0]]['X']
    X2=poseGraph.nodes[ee[1]]['X']
    
    X1=pt2dproc.binnerDownSampler(X1,dx=dxMatch[0],cntThres=1)
    X2=pt2dproc.binnerDownSampler(X2,dx=dxMatch[0],cntThres=1)
    
    H21est=poseGraph.edges[ee[0],ee[1]]['H']
    H12est=nplinalg.inv(H12est)
    H12est=np.identity(3)
    Hbin21=nbpt2Dproc.binMatcherAdaptive3(X1,X2,H12est,Lmax,thmax,thmin,dxMatch)
    poseGraph.edges[ee[0],ee[1]]['H']=Hbin21
    
    Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc.binScanEdges(X1,X2,dxMatch)
    activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))
    posematch=pt2dproc.eval_posematch(Hbin21,X2,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
    print(ee,posematch['mbinfrac_ActiveOvrlp'])
    
    Hbin12=nplinalg.inv(Hbin21)
    X2to1 = Hbin12[0:2,0:2].dot(X2.T).T+Hbin12[0:2,2]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(X1[:,0],X1[:,1],'r.')
    ax.plot(X2to1[:,0],X2to1[:,1],'b.')
    
    ax.set_title(str(ee)+str(posematch['mbinfrac_ActiveOvrlp']))
    plt.show()
    
    

poseGraph=pt2dproc.updated_sHg(poseGraph)
pt2dplot.plot_keyscan_path(poseGraph,Lkey[0],Lkey[-1],params,makeNew=True,skipScanFrame=True,plotGraphbool=False,
                                    forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True,plotKeyFrameNodesTraj=True)

   
    
e1=Lkeyloop_edges[1][0]
e2=Lkeyloop_edges[1][1]
H21true=poseGraph.edges[e1,e2]['H']

# H21=np.identity(3) 
# th=0*np.pi/180
# H21[0:2,2]=[-2,2]
# R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
# H21[0:2,0:2]=R
#%%
plt.close("all")
Lmax=np.array([2,2])
thmax=15*np.pi/180
dxMatch=np.array([0.05,0.05])
H12est = np.identity(3) 
thmin=1*np.pi/180

X1=pt2dproc.binnerDownSampler(X1,dx=dxMatch[0],cntThres=1)
X2=pt2dproc.binnerDownSampler(X2,dx=dxMatch[0],cntThres=1)
Hbin21=nbpt2Dproc.binMatcherAdaptive3(X1,X2,H12est,Lmax,thmax,dxMatch)


X1main=poseGraph.nodes[e1]['X'].copy()
X1main=pt2dproc.binnerDownSampler(X1main,dx=0.125,cntThres=1)

X2main=poseGraph.nodes[e2]['X'].copy()
X2main=pt2dproc.binnerDownSampler(X2main,dx=0.125,cntThres=1)

X1main=np.ascontiguousarray(X1main,dtype=np.float64)
X2main=np.ascontiguousarray(X2main,dtype=np.float64)

X1=pt2dproc.binnerDownSampler(X1,dx=dxMatch[0],cntThres=1)
X2=pt2dproc.binnerDownSampler(X2,dx=dxMatch[0],cntThres=1)
Hbin21=nbpt2Dproc.binMatcherAdaptive3(X1,X2,H12est,Lmax,thmax,dxMatch)
# et=time.time()
# print(et-st)

# st=time.time()
# Hbin21=pt2dproc.binMatcherAdaptive3(X1main,X2main,H12est,Lmax,thmax,dxMatch)
# et=time.time()
# print(et-st)


# with open("DeutchesMuseum-NoLoopICP.pkl",'rb') as fh:
#     poseGraph,params=pkl.load(fh)

# Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
# Lkeyloop_edges = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key",poseGraph.edges))

# ee = Lkeyloop_edges[20]
# X11=poseGraph.nodes[ee[0]]['X']
# X22=poseGraph.nodes[ee[1]]['X']

X11=getscanpts(2571)
X22=getscanpts(2572)
X11=pt2dproc.binnerDownSampler(X11,dx=dxMatch[0],cntThres=1)
X22=pt2dproc.binnerDownSampler(X22,dx=dxMatch[0],cntThres=1)
# X11=pt2dproc.binnerDownSampler(X11,dx=0.125,cntThres=1)
# X22=pt2dproc.binnerDownSampler(X22,dx=0.125,cntThres=1)

H12=H12est.copy()





n=histsmudge =1 # how much overlap when computing max over adjacent hist for levels
    
    
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
H1match[H1match>0]=1




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

for i in range(mxlvl):
    H=HLevels[i]
    pt2dproc.plotbins2(XYedges[i][0],XYedges[i][1],H,X1,X2)
    
    
LmaxOrig=Lmax.copy()
# et=time.time()
# print("Time pre-init = ",et-st)
SolBoxes_init=[]
X2L=X2-Lmax
Lmax=dxs[0]*(np.floor(Lmax/dxs[0])+1)
for xs in np.arange(0,2*Lmax[0],dxs[0][0]):
    for ys in np.arange(0,2*Lmax[1],dxs[0][1]):
        SolBoxes_init.append( (np.array([xs,ys],dtype=np.float32),dxs[0]) )





h=[]
#Initialize with all thetas fixed at Max resolution
lvl=0
dx=dxs[lvl]
H=HLevels[lvl]

Xth={}
ii=0
thfineRes = thmin
thL=np.arange(-thmax,thmax+thfineRes,thfineRes,dtype=np.float32)
# thL=[0]

# np.random.shuffle(thL)
for th in thL:
    R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    XX=np.transpose(R.dot(X2L.T))
    Xth[th]=XX
    
    
    for solbox in SolBoxes_init:
        Tj=solbox[1]
        Oj = solbox[0]
        if not np.all(Tj==dx):
            print(Tj,dx)
            raise Exception("Tj and dx are not equal ")
            
        # fig,ax=pt2dproc.plotbins2(XYedges[lvl][0],XYedges[lvl][1],HLevels[lvl],X1,Xth[th]+solbox[0])
        
        cost2=nbpt2Dproc.getPointCost(H,dx,Xth[th],Oj,Tj)
        # ax.set_title("lvl=%d,cost=%d, th=%1.2f, %s %s"%(0,cost2,th,str(list(solbox[0])),str(list(solbox[1]))))
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
    
    # fig,ax=pt2dproc.plotbins2(XYedges[lvl][0],XYedges[lvl][1],HLevels[lvl],X1,Xth[th]+solboxt[0])
    # ax.set_title("lvl=%d,cost=%d, th=%1.2f, %s %s"%(lvl,cost,th,str(list(solboxt[0])),str(list(solboxt[1]))))
    # plt.savefig("debugPlots/%05d"%cnt)
    # plt.close(fig)
    # cnt+=1
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
    # print(len(S))

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

fig,ax=pt2dproc.plotbins2(XYedges[lvl][0],XYedges[lvl][1],HLevels[lvl],X1,Xth[th]+solboxt[0])
ax.set_title("final")

Hcomp=np.identity(3)
t=solboxt[0]
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

H12comp=nplinalg.multi_dot([nplinalg.inv(H1),Hcomp,Hlmax,H1,H12])
H21comp=nplinalg.inv(H12comp)

RT=H12comp[0:2,0:2]
tT=H12comp[0:2,2]
X2to1=RT.dot(X22.T).T+tT
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(X11[:,0],X11[:,1],'r.')
ax.plot(X2to1[:,0],X2to1[:,1],'b.')

ax.set_title("final in main points")
plt.show()




#%%
import progressbar

# for i in progressbar.progressbar(range(100)):
    
Lkeyloop_edges = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key",poseGraph.edges))
for i in progressbar.progressbar(range(len(Lkeyloop_edges))):
    e1=Lkeyloop_edges[1][0]
    e2=Lkeyloop_edges[1][1]
    H21true=poseGraph.edges[e1,e2]['H']
    
    # H21=np.identity(3) 
    # th=0*np.pi/180
    # H21[0:2,2]=[-2,2]
    # R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    # H21[0:2,0:2]=R
    
    Lmax=np.array([7,7])
    thmax=25*np.pi/180
    dxMatch=np.array([0.25,0.25])
    H12est = np.identity(3) 
    
    X1main=poseGraph.nodes[e1]['X'].copy()
    X1main=pt2dproc.binnerDownSampler(X1main,dx=0.125,cntThres=1)
    
    X2main=poseGraph.nodes[e2]['X'].copy()
    X2main=pt2dproc.binnerDownSampler(X2main,dx=0.125,cntThres=1)
    
    # X2main=R.dot(X1main.T).T+H21[0:2,2]
    # Hbin21=pt2dproc.binMatcherAdaptive3(X1main,X2main,H12est,Lmax,thmax,dxMatch)
    st=time.time()
    Hbin21=nbpt2Dproc.binMatcherAdaptive3(X1main,X2main,H12est,Lmax,thmax,dxMatch)
    et=time.time()
    print(et-st)
    ttr,thtr=nbpt2Dproc.extractPosAngle(H21true)
    tes,thes=nbpt2Dproc.extractPosAngle(Hbin21)
    tdiff=nplinalg.norm(ttr-tes)
    thdiff=np.abs(thtr-thes)
    if thdiff>3*pi/180 or tdiff>1:
        print(i,e1,e2)
    
    
#%%
# scanfilepath = 'C:/Users/nadur/Google Drive/repos/SLAM/lidarprocessing/houseScan_std.pkl'
# scanfilepath = 'C:/Users/Nagnanamus/Google Drive/repos/SLAM/lidarprocessing/houseScan_complete.pkl'
# scanfilepath = 'C:/Users/Nagnanamus/Google Drive/repos/SLAM/lidarprocessing/houseScan_std.pkl'
# scanfilepath = 'lidarprocessing/datasets/DeutchMeuseum/b2-2015-07-07-11-27-05.pkl'
# scanfilefolder = 'lidarprocessing/datasets/DeutchMeuseum/b2-2015-07-07-11-27-05/'
# scanfilepath = 'lidarprocessing/datasets/DeutchMeuseum/b2-2014-11-24-14-33-46.pkl'
scanfilefolder = 'lidarprocessing/datasets/DeutchMeuseum/b2-2014-11-24-14-33-46/'

# scanfilepath = 'lidarprocessing/datasets/DeutchMeuseum/b2-2016-04-27-12-31-41.pkl'
# scanfilefolder = 'lidarprocessing/datasets/DeutchMeuseum/b2-2016-04-27-12-31-41/'

# os.makedirs(scanfilefolder)

# with open(scanfilepath,'rb') as fh:
#     dataset=pkl.load(fh)


# for i in range(len(dataset['scan'])):
#     with open(os.path.join(scanfilefolder,'scan_%d.pkl'%i),'wb') as F:
#         pkl.dump(dataset['scan'][i],F)

def getscanpts_deutches(idx):
    # if idx >= 20079 and idx<=20089:
    #     return None
    try:
        with open(os.path.join(scanfilefolder,'scan_%d.pkl'%idx),'rb') as F:
            scan=pkl.load(F)
    except:
        return None
    rngs = list(map(lambda x: np.max(x) if len(x)>0 else 120,scan['scan']))
    rngs = np.array(rngs)
    
    # ranges = dataset[i]['ranges']
    # rngs = np.array(dataset[idx]['ranges'])
    
    ths = np.arange(angle_min,angle_max,angle_increment)
    p=np.vstack([np.cos(ths),np.sin(ths)])
    
    rngidx = (rngs> (range_min+0.1) ) & (rngs< (range_max-5))
    ptset = rngs.reshape(-1,1)*p.T
    
    X=ptset[rngidx,:]
    
    Xd=pt2dproc.binnerDownSampler(X,dx=0.025,cntThres=1)
                
    # now filter silly points
    tree = KDTree(Xd, leaf_size=5)
    cnt = tree.query_radius(Xd, 0.25,count_only=True) 
    Xd= Xd[cnt>=2,:]
    
    cnt = tree.query_radius(Xd, 0.5,count_only=True) 
    Xd = Xd[cnt>=5,:]
    
    return Xd


class IntelData:
    def __init__(self):
        intelfile = "lidarprocessing/datasets/Freiburg/Intel Research Lab.clf"
        ff=open(intelfile)
        self.inteldata=ff.readlines()
        self.inteldata = list(filter(lambda x: '#' not in x,self.inteldata))
        self.flaserdata = list(filter(lambda x: 'FLASER' in x[:8],self.inteldata))
        self.odomdata = list(filter(lambda x: 'ODOM' in x[:8],self.inteldata))

        
    def __len__(self):
        return len(self.flaserdata)
    
    def getflaser(self,idx):
        g = self.flaserdata[idx]
        glist = g.strip().split(' ')
        # print(glist)
        cnt = int(glist[1])
        rngs =[]
        ths=np.arange(0,1*np.pi,1*np.pi/180)
        for i in range(cnt):
            rngs.append(float(glist[2+i]))
        odom=np.array([float(ss) for ss in glist[182:188]])
        rngs = np.array(rngs)
        p=np.vstack([np.cos(ths),np.sin(ths)])
        ptset = rngs.reshape(-1,1)*p.T
        rngidx = (rngs> (0+0.1) ) & (rngs< (25))
        return ptset[rngidx,:],odom
    
dataset = IntelData()       
def getscanpts_intel(idx):
    # ranges = dataset[i]['ranges']
    
    X,odom=dataset.getflaser(idx)
    
    # now filter silly points
    # tree = KDTree(X, leaf_size=5)
    # cnt = tree.query_radius(X, 0.0125,count_only=True) 
    # X = X[cnt>=2,:]
    
    # cnt = tree.query_radius(X, 0.015,count_only=True) 
    # X = X[cnt>=5,:]
    
    return X,odom

getscanpts = getscanpts_deutches

# plt.figure()
# Xpos=[]
#%%

plt.close("all")
poses=[]
poseGraph = nx.DiGraph()


# Xr=np.zeros((len(dataset),3))
ri=0
KeyFrames=[]

params={}

params['REL_POS_THRESH']=0.05# meters after which a keyframe is made
params['REL_ANGLE_THRESH']=10*np.pi/180
params['ERR_THRES']=15
params['n_components']=35
params['reg_covar']=0.002

params["Key2Key_Overlap"]=0.3
params["Scan2Key_Overlap"]=0.3

params['Key2KeyBinMatch_dx0']=2
params['Key2KeyBinMatch_L0']=7
params['Key2KeyBinMatch_th0']=np.pi/4

params['BinDownSampleKeyFrame_dx']=0.15
params['BinDownSampleKeyFrame_probs']=0.05

params['Plot_BinDownSampleKeyFrame_dx']=0.15
params['Plot_BinDownSampleKeyFrame_probs']=0.0001

params['doLoopClosure'] = False
params['doLoopClosureLong'] = False

params['Loop_CLOSURE_PARALLEL'] = True
params['LOOP_CLOSURE_D_THES']=31.4
params['LOOP_CLOSURE_POS_THES']=30
params['LOOP_CLOSURE_POS_MIN_THES']=0.1
params['LOOP_CLOSURE_ERR_THES']= 3
# params['LOOPCLOSE_BIN_MATCHER_dx'] = 4
# params['LOOPCLOSE_BIN_MATCHER_L'] = 13
params['LOOPCLOSE_BIN_MIN_FRAC_dx'] = np.array([0.15,0.15],dtype=np.float64)

params['LOOPCLOSE_BIN_MIN_FRAC'] = 0.2
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_LOCAL']=0.6
params['LOOPCLOSE_BIN_MAXOVRL_FRAC_COMPLETE']=0.4
params['LOOP_CLOSURE_COMBINE_MAX_NODES']= 8

params['offsetNodesBy'] = 0


params['MAX_NODES_ADJ_COMBINE']=5
params["USE_Side_Combine"]=True
params["Side_Combine_Overlap"]=0.3


params['NearLoopClose'] = {}
params['NearLoopClose']['Method']='GMM'
params['NearLoopClose']['PoseGrid']=None #pt2dproc.getgridvec(np.linspace(-np.pi/12,np.pi/12,3),np.linspace(-1,1,3),np.linspace(-1,1,3))
params['NearLoopClose']['isPoseGridOffset']=True
params['NearLoopClose']['isBruteForce']=False


# meters. skip loop closure of current node if there is a loop closed node within radius along the path
params['LongLoopClose'] = {}
params['LongLoopClose']['Method'] = 'GMM'
params['LongLoopClose']['SkipLoopCloseIfNearCLosedNodeWithin'] = 5 
A=pt2dproc.getgridvec([0],np.linspace(-5,5,5),np.linspace(-5,5,5))
ind = np.lexsort((np.abs(A[:,0]),np.abs(A[:,1]),np.abs(A[:,2])))
params['LongLoopClose']['PoseGrid']= None #A[ind]
params['LongLoopClose']['isPoseGridOffset']=True
params['LongLoopClose']['isBruteForce']=False
params['LongLoopClose']['Bin_Match_dx0'] = 2
params['LongLoopClose']['Bin_Match_L0'] = 7
params['LongLoopClose']['Bin_Match_th0'] = np.pi/4
params['LongLoopClose']['DoCLFmatch'] = True

params['LongLoopClose']['AlongPathNearFracCountNodes'] = 0.3
params['LongLoopClose']['AlongPathNearFracLength'] = 0.3
params['LongLoopClose']['#TotalRandomPicks'] = 10
params['LongLoopClose']['AdjSkipList'] = 3
params['LongLoopClose']['TotalCntComp'] = 100

# params['Do_GMM_FINE_FIT']=False

# params['Do_BIN_FINE_FIT'] = False

params['Do_BIN_DEBUG_PLOT-dx']=False
params['Do_BIN_DEBUG_PLOT']= False

params['xy_hess_inv_thres']=100000000*0.4
params['th_hess_inv_thres']=100000000*0.4


params['#ThreadsLoopClose']=8

params['INTER_DISTANCE_BINS_max']=120
params['INTER_DISTANCE_BINS_dx']=1


params['LOOPCLOSE_AFTER_#KEYFRAMES']=2



timeMetrics={'scan2keyMain':[],'addNewKeyScan':[],'SendPlot':[],'SendPoseGraphLoop':[],'RecievePoseGraphLoop':[],
                          'UpdatePoseGraphLoop':[],'PrevPrevScanPtsStack':[],'PrevScanPtsStack':[],'PrevPrevScanPtsCombine':[],'PrevScanPtsCombine':[],'NewKeyFrameClf':[],'scan2keyNew':[],'OverlapScan2keyNew':[]}
DoneLoops=[]
# fig = plt.figure("Full Plot")
# ax = fig.add_subplot(111)

# figg = plt.figure("Graph Plot")
# axgraph = figg.add_subplot(111)

Nframes = len(os.listdir(scanfilefolder))
# Nframes = len(dataset)

idx1=0 #19970 #16000 #27103 #14340
idxLast = Nframes
previdx_loopclosure = idx1
previdx_loopdetect = idx1
previdx_loopdetect_long=idx1

Lmax=np.array([2,2])
thmax=15*np.pi/180
dxMatch=np.array([0.1,0.1])
thmin=1*np.pi/180

for idx in range(idx1,idxLast): 
    # ax.cla()
    # if idx>=20083 and idx <=20085:
    #     continue
    X=getscanpts(idx)
    X=pt2dproc.binnerDownSampler(X,dx=dxMatch[0],cntThres=1)

    if X is None:
        continue
    if len(poseGraph.nodes)==0:
        # Xd,m = pt2dproc.get0meanIcov(X)
        clf=None
        H=np.hstack([np.identity(2),np.zeros((2,1))])
        H=np.vstack([H,[0,0,1]])
        idbmx = params['INTER_DISTANCE_BINS_max']
        idbdx=params['INTER_DISTANCE_BINS_dx']
        # h=pt2dproc.get2DptFeat(X,bins=np.arange(0,idbmx,idbdx))
        h = np.array([0,0])
        
        poseGraph.add_node(idx,frametype="keyframe",X=X,clf=clf,time=idx,sHg=H,pos=(0,0),h=h,color='g',LoopDetectDone=False)
        # poseData[idx]={'X':X}
        

        continue
    
    
    Xprev = poseGraph.nodes[idx-1]['X']
    
    
    # if (idx-KeyFrame_prevIdx)<=1:
    #     sHk_prevframe = np.identity(3)
    # elif KeyFrame_prevIdx==previdx:
    #     sHk_prevframe = np.identity(3)
    # else:
    #     sHk_prevframe = poseGraph.edges[KeyFrame_prevIdx,previdx]['H']
    
    sHk_prevframe = np.identity(3)
    kHs_prevframe=nplinalg.inv(sHk_prevframe)
    
    
    
    
    
    
    # assuming sHk_prevframe is very close to sHk
    st=time.time()
    # sHk,serrk,shessk_inv = pt2dproc.scan2keyframe_match(KeyFrameClf,Xclf,X,params,sHk=sHk_prevframe)
    sHk=nbpt2Dproc.binMatcherAdaptive3(Xprev,X,kHs_prevframe,Lmax,thmax,thmin,dxMatch)
    # sHk, distances, i=icp.icp(Xprev, X, init_pose=None, max_iterations=100, tolerance=0.01)
    serrk=None
    shessk_inv=None
    et = time.time()
    
    dxcomp = params['LOOPCLOSE_BIN_MIN_FRAC_dx']
    Hist1_ovrlp, xedges_ovrlp,yedges_ovrlp=nbpt2Dproc.binScanEdges(Xprev,X,dxcomp)
    activebins1_ovrlp = np.sum(Hist1_ovrlp.reshape(-1))
    posematch=pt2dproc.eval_posematch(sHk,X,Hist1_ovrlp,activebins1_ovrlp,xedges_ovrlp,yedges_ovrlp)
    posematch['method']='GMMmatch'
    posematch['when']="Scan to key in main"
    
    print("idx = ",idx," Error = ",serrk," , and time taken = ",et-st," posematch=",posematch['mbinfrac_ActiveOvrlp'])
    
    
    # publish pose
    kHg = poseGraph.nodes[idx-1]['sHg']
    sHg = np.matmul(sHk,kHg)
    poses.append(sHg)
    gHs=nplinalg.inv(sHg)
    
    
    # get relative frame from idx-1 to idx
    # iHim1 = np.matmul(sHk,nplinalg.inv(sHk_prevframe))
    tprevK,thprevK = nbpt2Dproc.extractPosAngle(kHg)
    tcurr,thcurr = nbpt2Dproc.extractPosAngle(sHg)
    thdiff = np.abs(nbpt2Dproc.anglediff(thprevK,thcurr))
    # print("thdiff = ",thdiff)
    # check if to make this the keyframe
    
    tpos=np.matmul(gHs,np.array([0,0,1]))
    poseGraph.add_node(idx,frametype="scan",time=idx,X=X,sHg=sHg,pos=(tpos[0],tpos[1]),color='r',LoopDetectDone=False) 
    poseGraph.add_edge(idx-1,idx,H=sHk,H_prevframe=sHk_prevframe,err=serrk,hess_inv=shessk_inv,edgetype="Key2Scan",color='r')
    poseGraph.edges[idx-1,idx]['posematch']=posematch
        
        
    
    
    
    # plotting
    if idx%5==0 or idx==idxLast-1:
        st = time.time()
        pt2dplot.plot_keyscan_path(poseGraph,idx1,idx,params,makeNew=False,skipScanFrame=True,plotGraphbool=False,
                                    forcePlotLastidx=True,plotLastkeyClf=False,plotLoopCloseOnScanPlot=False,plotKeyFrameNodesTraj=True)
        et = time.time()
        print("plotting time : ",et-st)
        plt.show()
        plt.pause(0.01)
    
    previdx = idx
    
    
N=len(poseGraph)
Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
Lscan = list(filter(lambda x: poseGraph.nodes[x]['frametype']!="keyframe",poseGraph.nodes))
print(N,len(Lkey),len(Lscan))
df=pd.DataFrame({'type':['keyframe']*len(Lkey)+['scan']*len(Lscan),'idx':Lkey+Lscan})
df.sort_values(by=['idx'],inplace=True)
df

with open("PoseGraph-deutchesMesuemDebug-binmatcher.pkl",'wb') as fh:
    pkl.dump([poseGraph,params],fh)


    