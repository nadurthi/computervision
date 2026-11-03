# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:46:31 2020

@author: nadur
"""

import pickle as pkl
import numpy as np
import numpy.linalg as nplinalg
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture
from sklearn.neighbors import KDTree
import gmmfuncs as uqgmmfnc
import geometryshapes as utpltgmshp
import time
from scipy.optimize import minimize, rosen, rosen_der,least_squares
from scipy import interpolate
import networkx as nx
import pdb
import pandas as pd
from fastdist import fastdist
import copy
import point2Dprocessing as pt2dproc
import point2Dplotting as pt2dplot

dtype = np.float64
 #%%
# scanfilepath = 'C:/Users/nadur/Google Drive/repos/SLAM/lidarprocessing/houseScan_std.pkl'
# scanfilepath = 'C:/Users/Nagnanamus/Google Drive/repos/SLAM/lidarprocessing/houseScan_complete.pkl'
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
    
    X=ptset[rngidx,:]
    
    # now filter silly points
    # tree = KDTree(X, leaf_size=5)
    # dist, ind = tree.query(X, k=10) 
    # dist=np.mean(dist[:,1:],axis=1)
    # dm=np.percentile(dist,95)
        
    # return X[dist<dm,:]
    return X


#%% TEST::::::: Pose estimation by keyframe
plt.close("all")
poses=[]
poseGraph = nx.DiGraph()
Xr=np.zeros((len(dataset),3))
ri=0
KeyFrames=[]

REL_POS_THRESH=0.3 # meters after which a keyframe is made
ERR_THRES=1.2

LOOP_CLOSURE_D_THES=1.5
LOOP_CLOSURE_POS_THES=4
LOOP_CLOSURE_ERR_THES=2

DoneLoops=[]
# fig = plt.figure("Full Plot")
# ax = fig.add_subplot(111)

# figg = plt.figure("Graph Plot")
# axgraph = figg.add_subplot(111)

idx1=1000
previdx_loopclosure = idx1

for idx in range(idx1,len(dataset)):
    # ax.cla()
    X=getscanpts(dataset,idx)
    if len(poseGraph.nodes)==0:
        Xd,m = pt2dproc.get0meanIcov(X)
        clf,MU,P,W = pt2dproc.getclf(Xd)
        H=np.hstack([np.identity(2),np.zeros((2,1))])
        H=np.vstack([H,[0,0,1]])
        h=pt2dproc.get2DptFeat(X)
        poseGraph.add_node(idx,frametype="keyframe",clf=clf,X=X,m_clf=m,time=0,sHg=H,pos=(0,0),h=h,color='g')
        KeyFrame_prevIdx=idx
        KeyFrames.append(np.array([0,0,0]))
        continue
    
    # estimate pose to last keyframe
    KeyFrameClf = poseGraph.nodes[KeyFrame_prevIdx]['clf']
    m_clf = poseGraph.nodes[KeyFrame_prevIdx]['m_clf']
    if (idx-KeyFrame_prevIdx)<=1:
        sHk_prevframe = np.identity(3)
    else:
        sHk_prevframe = poseGraph.edges[KeyFrame_prevIdx,idx-1]['H']
    # assuming sHk_prevframe is very close to sHk
    st=time.time()
    sHk,err = pt2dproc.scan2keyframe_match(KeyFrameClf,m_clf,X,sHk=sHk_prevframe)
    et = time.time()
    
    if np.isfinite(err)==0:
        pdb.set_trace()
        pt2dplot.plotcomparisons_posegraph(poseGraph,KeyFrame_prevIdx,idx,H12=nplinalg.inv( sHk_prevframe) )
        
        
    print("idx = ",idx," Error = ",err," , and time taken = ",et-st)
    # publish pose
    kHg = poseGraph.nodes[KeyFrame_prevIdx]['sHg']
    sHg = np.matmul(sHk,kHg)
    poses.append(sHg)
    gHs=nplinalg.inv(sHg)
    
    # get relative frame from idx-1 to idx
    # iHim1 = np.matmul(sHk,nplinalg.inv(sHk_prevframe))
    
    # check if to make this the keyframe
    if err>ERR_THRES or nplinalg.norm(sHk[:2,2])>REL_POS_THRESH or (idx-KeyFrame_prevIdx)>100:
        print("New Keyframe")
        Xd,m = pt2dproc.get0meanIcov(X)
        clf,MU,P,W = pt2dproc.getclf(Xd)
        tpos=np.matmul(gHs,np.array([0,0,1])) 

        h=pt2dproc.get2DptFeat(X)
        poseGraph.add_node(idx,frametype="keyframe",clf=clf,X=X,m_clf=m,time=idx,sHg=sHg,pos=(tpos[0],tpos[1]),h=h,color='g')
        poseGraph.add_edge(KeyFrame_prevIdx,idx,H=sHk,edgetype="Key2Key",color='k')
        
        
        # now delete previous scan data up-until the previous keyframe
        # this is to save space. but keep 1. Also complete pose estimation to this scan
        Lidxs = list(poseGraph.successors(KeyFrame_prevIdx))
        Lidxs = list(filter(lambda x: poseGraph.nodes[x]['frametype']!="keyframe",Lidxs))
        if len(Lidxs)>0:
            scanid = Lidxs[int(np.floor(len(Lidxs)/2))]
            # estimate pose to new keyframe
            Xs = poseGraph.nodes[scanid]['X']

            KeyFrameClf = poseGraph.nodes[idx]['clf']
            m_clf = poseGraph.nodes[idx]['m_clf']
            msHk = poseGraph.edges[KeyFrame_prevIdx,scanid]['H']
            sHk_newkeyframe =  np.matmul(msHk,nplinalg.inv(sHk))
            sHk_corrected,err = pt2dproc.scan2keyframe_match(KeyFrameClf,m_clf,Xs,sHk=sHk_newkeyframe)
            if np.isfinite(err)==0:
                pdb.set_trace()
                pt2dplot.plotcomparisons_posegraph(poseGraph,idx,scanid,H12=nplinalg.inv( sHk_newkeyframe) )
                pt2dplot.plotcomparisons(scanid,idx,H12=sHk_newkeyframe)
                
            poseGraph.add_edge(idx,scanid,H=sHk_corrected,edgetype="Key2Scan",color='r')
            
            # delete rest of the scan-ids as they are useless
            for i in Lidxs:
                if i!=scanid and poseGraph.in_degree(i)==1:
                    poseGraph.remove_node(i)
            

            
        KeyFrame_prevIdx = idx
        KeyFrames.append(np.matmul(gHs,np.array([0,0,1]))  )
        
        
        # detect loop closure and add the edge
        
        # Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
        # Lkey.sort()
        # idx_n = Lkey.index(idx)
        # for previdx in Lkey:
        #     if previdx < Lkey[idx_n-3] and poseGraph.has_edge(idx,previdx) is False and poseGraph.has_edge(previdx,idx) is False:
        #         h1=poseGraph.nodes[idx]['h']
        #         h2=poseGraph.nodes[previdx]['h']
        #         p1=poseGraph.nodes[idx]['pos']
        #         p2=poseGraph.nodes[previdx]['pos']
        #         d=nplinalg.norm(h1-h2,ord=1)
        #         if d<=LOOP_CLOSURE_D_THES and nplinalg.norm(np.array(p1)-np.array(p2),ord=2)<=LOOP_CLOSURE_POS_THES:
        #             # add loop closure edge
        #             print("Potential Loop closure")
        #             piHi,pi_err_i=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,idx,previdx)
        #             iHpi,i_err_pi=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,previdx,idx)
        #             if pi_err_i < LOOP_CLOSURE_ERR_THES and pi_err_i<i_err_pi:
        #                 poseGraph.add_edge(idx,previdx,H=piHi,edgetype="Key2Key-LoopClosure",d=d,color='b')
        #                 print("Adding Loop closure")
        #             elif  i_err_pi < LOOP_CLOSURE_ERR_THES and i_err_pi<i_err_pi:
        #                 poseGraph.add_edge(idx,previdx,H=nplinalg.inv(iHpi),edgetype="Key2Key-LoopClosure",d=d,color='b')
        #                 print("Adding Loop closure")
        #             elif pi_err_i < LOOP_CLOSURE_ERR_THES:
        #                 poseGraph.add_edge(idx,previdx,H=piHi,edgetype="Key2Key-LoopClosure",d=d,color='b')
        #                 print("Adding Loop closure")
        #             elif i_err_pi < LOOP_CLOSURE_ERR_THES:
        #                 poseGraph.add_edge(idx,previdx,H=nplinalg.inv(iHpi),edgetype="Key2Key-LoopClosure",d=d,color='b')
        #                 print("Adding Loop closure")
        #             else:
        #                 print("No loop closure this time")
                    
                    
    else: #not a keyframe
        tpos=np.matmul(gHs,np.array([0,0,1]))

        poseGraph.add_node(idx,frametype="scan",time=idx,X=X,sHg=sHg,pos=(tpos[0],tpos[1]),color='r')
        poseGraph.add_edge(KeyFrame_prevIdx,idx,H=sHk,edgetype="Key2Scan",color='r')
    
    
    
    
    


    
    
    
    # plotting
    if idx%25==0 or idx==len(dataset)-1:
        st = time.time()
        pt2dplot.plot_keyscan_path(poseGraph,idx1,idx,makeNew=False)
        et = time.time()
        print("plotting time : ",et-st)
        plt.show()
        plt.pause(0.01)
    # Lidxs = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    # plt.figure("Graph Plot")
    # axgraph.cla()
    # plotGraph(poseGraph,Lidxs,ax=axgraph)
    # plt.show()
    # plt.pause(0.1)
    
    # # plotting
    # # if idx % 1==0 or idx==len(dataset)-1:
    # plt.figure("Full Plot")
    # Lidxs.sort()
    # for i in Lidxs:
    #     gHs = nplinalg.inv(poseGraph.nodes[i]['sHg'])
    #     XX = poseGraph.nodes[i]['X']
    #     XX=np.matmul(gHs,np.vstack([XX.T,np.ones(XX.shape[0])])).T   
    #     ax.plot(XX[:,0],XX[:,1],'b.')
        
    # gHs=nplinalg.inv(poseGraph.nodes[idx]['sHg'])
    # # Xg=np.matmul(gHs,np.vstack([X.T,np.ones(X.shape[0])])).T   

    # # Xg=Xg[:,:2]
    # Xr[ri,:] =np.matmul(gHs,np.array([0,0,1]))   
    # ri=ri+1
    # # ax.plot(Xg[:,0],Xg[:,1],'b.')

    # ax.plot(Xr[:ri,0],Xr[:ri,1],'r')
    # ax.plot(Xr[ri-1,0],Xr[ri-1,1],'ro')
    
    # XX=np.vstack(KeyFrames)
    # ax.plot(XX[:,0],XX[:,1],'gs')
    
    # ax.set_title(str(idx))
    
    # plt.show()
    # plt.pause(0.1)




N=len(poseGraph)
Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
Lscan = list(filter(lambda x: poseGraph.nodes[x]['frametype']!="keyframe",poseGraph.nodes))
print(N,len(Lkey),len(Lscan))
df=pd.DataFrame({'type':['keyframe']*len(Lkey)+['scan']*len(Lscan),'idx':Lkey+Lscan})
df.sort_values(by=['idx'],inplace=True)
df

with open("PoseGraph-Sample",'wb') as fh:
    pkl.dump([poseGraph],fh)


#%% detect loop clusure, local bundle adjustment and global bundle adjustment
with open("PoseGraph-Sample",'rb') as fh:
    poseGraph,=pkl.load(fh)
    
#%% posegraph with loop-closure detections
plt.close("all")
ee=[]
E = list(poseGraph.edges)
for e in E:
    if 'edgetype' in poseGraph.edges[e[0],e[1]]:
        if poseGraph.edges[e[0],e[1]]["edgetype"]=="Key2Key-LoopClosure":
            ee.append(e)
for e in ee:
    poseGraph.remove_edge(*e)


# for idx in poseGraph.nodes:
#     gHs = nplinalg.inv(poseGraph.nodes[idx]['sHg'])
#     t=np.matmul(gHs,np.array([0,0,1]))  
#     poseGraph.nodes[idx]['pos']=(t[0],t[1])
         
LOOP_CLOSURE_D_THES=1.5
LOOP_CLOSURE_POS_THES=4
LOOP_CLOSURE_ERR_THES=-0.7
Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
for idx in Lkey:
    X=poseGraph.nodes[idx]['X']
    # h=pt2dproc.get2DptFeat(X)
    h=poseGraph.nodes[idx]['h']
    gHs = nplinalg.inv(poseGraph.nodes[idx]['sHg'])
    t=np.matmul(gHs,np.array([0,0,1]))  
    # poseGraph.nodes[idx]['pos']=(t[0],t[1])
    # poseGraph.nodes[idx]['h']=h
    flag=False
    for previdx in Lkey:
        if previdx < idx-2 and poseGraph.has_edge(idx,previdx) is False and poseGraph.has_edge(previdx,idx) is False:
            h1=poseGraph.nodes[idx]['h']
            h2=poseGraph.nodes[previdx]['h']
            p1=poseGraph.nodes[idx]['pos']
            p2=poseGraph.nodes[previdx]['pos']
            d=nplinalg.norm(h1-h2,ord=1)
            if d<=LOOP_CLOSURE_D_THES and nplinalg.norm(np.array(p1)-np.array(p2),ord=2)<=LOOP_CLOSURE_POS_THES:
                # add loop closure edge
                print("Potential Loop closure")
                piHi,pi_err_i=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,idx,previdx)
                iHpi,i_err_pi=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,previdx,idx)
                if pi_err_i < LOOP_CLOSURE_ERR_THES and pi_err_i<i_err_pi:
                    poseGraph.add_edge(idx,previdx,H=piHi,edgetype="Key2Key-LoopClosure",d=d,color='b')
                    print("Adding Loop closure")
                elif  i_err_pi < LOOP_CLOSURE_ERR_THES and i_err_pi<i_err_pi:
                    poseGraph.add_edge(idx,previdx,H=nplinalg.inv(iHpi),edgetype="Key2Key-LoopClosure",d=d,color='b')
                    print("Adding Loop closure")
                elif pi_err_i < LOOP_CLOSURE_ERR_THES:
                    poseGraph.add_edge(idx,previdx,H=piHi,edgetype="Key2Key-LoopClosure",d=d,color='b')
                    print("Adding Loop closure")
                elif i_err_pi < LOOP_CLOSURE_ERR_THES:
                    poseGraph.add_edge(idx,previdx,H=nplinalg.inv(iHpi),edgetype="Key2Key-LoopClosure",d=d,color='b')
                    print("Adding Loop closure")
                else:
                    print("No loop closure this time")
    
    # subgraph = poseGraph.subgraph([n for n in poseGraph.nodes if n<=idx])
    # keyframe_nodes = list(filter(lambda x: subgraph.nodes[x]['frametype']=="keyframe",subgraph.nodes))    
    # L=list(nx.simple_cycles(subgraph))
    # L=[sorted(m) for m in L]
    # L.sort(key=lambda x: len(x))
    
Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
pt2dplot.plotGraph(poseGraph,Lkey)
plt.show()
#%%
idx=2576
previdx= 2563
piHi,pi_err_i=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,idx,previdx)
iHpi,i_err_pi=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,previdx,idx)
h1=poseGraph.nodes[idx]['h']
h2=poseGraph.nodes[previdx]['h']
p1=poseGraph.nodes[idx]['pos']
p2=poseGraph.nodes[previdx]['pos']
d=nplinalg.norm(h1-h2,ord=1)

pt2dplot.plotcomparisons(previdx,idx,H12=piHi)

#%%
idx1=1000
idx2=9898
pt2dplot.plotKeyGmm_ScansPts(poseGraph,idx1,idx2)

#%% loop closure optimization

    
def globalPoses_byPtAlign_cost(x,scanidxs,keyidxs,Xs0,MU0,P0,W0):
    # x are the global poses 
    # are the MU0,P0,W0 are the means and covs for local gmm frame
    # MU0 have ones
    # MU0=[mu11,mu12,mu13...,mu21,m22,...]
    # keyidxs=[1,1,1,...2,2,...]
    
    # Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    # Lscan = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="scan",poseGraph.nodes))
    
    y=np.zeros((int(len(x)/3)+1,3),dtype=dtype)
    for s in range(int(len(x)/3)):
        y[s+1,:]= x[s*3:s*3+3]
    

    th=y[:,0] # global
    txy=y[:,1:] # global
    MU=np.zeros_like(MU0)
    P=np.zeros_like(P0)
    for i in  range(keyidxs.shape[0]):
        j=keyidxs[i,0]
        i1= keyidxs[i,1]
        i2= keyidxs[i,2]
        thj = th[j]
        tj = txy[j]
        gHj=pt2dproc.getHmat(thj,tj)
        gRj = gHj[0:2,0:2]
        
        MU[i1:i2]=gHj.dot(MU0[i1:i2].T).T
        
        P[i1:i2]=(gRj.T.dot(P0[i1:i2])).dot(gRj)
    
    Xs=np.zeros_like(Xs0)
    for i in  range(scanidxs.shape[0]):
        j=scanidxs[i,0]
        i1= scanidxs[i,1]
        i2= scanidxs[i,2]
        thj = th[j]
        tj = txy[j]
        gHj=pt2dproc.getHmat(thj,tj)

        Xs[i1:i2]=gHj.dot(Xs0[i1:i2].T).T
        
    p=uqgmmfnc.gmm_eval_fast(Xs,MU,P,W0)+1
    logp = -np.log(p)
    m = np.mean(logp)
    
    return m   


    

# def adjustPoses(poseGraph,idx1,idx2):
# adjust the bundle from idx1 to idx2
# i.e. optimize the global pose matrices, with constraint local poses 
# idx1 is assumed to be fixed (like origin)
idx1=1000
idx2=9898
# idx2=8660
pt2dproc.loopExists(poseGraph,idx1,idx2)


Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))

pt2dplot.plot_keyscan_path(poseGraph,idx1,idx2,makeNew=True)
plt.show()


res,sHg_updated,sHg_previous=pt2dproc.adjustPoses(poseGraph,idx1,idx2)

poseGraph2=pt2dproc.updateGlobalPoses(copy.deepcopy(poseGraph),sHg_updated)

pt2dplot.plot_keyscan_path(poseGraph2,idx1,idx2,makeNew=True)
plt.show()


#%%
poseGraph2 = copy.deepcopy(poseGraph)

plot_keyscan_path(poseGraph2,idx1,idx2,makeNew=True)
if res.success:
    for ns in sHg_updated.keys():
        sHg = sHg_updated[ns]
        gHs=nplinalg.inv(sHg)
        tpos=np.matmul(gHs,np.array([0,0,1]))
        poseGraph2.nodes[ns]['pos'] = (tpos[0],tpos[1])
        poseGraph2.nodes[ns]['sHg'] = sHg
        
    # now update other non-keyframes
    for ns in poseGraph2.nodes:
        if poseGraph2.nodes[ns]['frametype']!="keyframe":
            for pidx in poseGraph2.predecessors(ns):
                if poseGraph2.nodes[pidx]['frametype']=="keyframe":
                    psHg=poseGraph2.nodes[pidx]['sHg']
                    nsHps=poseGraph2.edges[pidx,ns]['H']
                    nsHg = nsHps.dot(psHg)
                    poseGraph2.nodes[ns]['sHg']=nsHg
                    gHns=nplinalg.inv(nsHg)
                    tpos=np.matmul(gHns,np.array([0,0,1]))
                    poseGraph2.nodes[ns]['pos'] = (tpos[0],tpos[1])
        
                    break
plot_keyscan_path(poseGraph2,idx1,idx2,makeNew=True)
#%% cycles
Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))    
L=list(nx.simple_cycles(poseGraph.subgraph(Lkey)))
L=[sorted(m) for m in L]
L.sort(key=lambda x: len(x))
# L=[set(m) for m in L]
print(L)

nx.find_cycle(poseGraph,source=5000, orientation="original")
for i in range(0,len(L),5):        
    plotGraph(poseGraph,L[i])
plt.show()
#%%

#%% plotting functions
plt.close("all")
xg = np.linspace(np.min(X[:,0])-10., np.min(X[:,0])+10,100)
yg = np.linspace(np.min(X[:,1])-10., np.min(X[:,1])+10,100)

clf,MU,P,W = pt2dproc.getclf(X,reg_covar=0.02)

Xg, Yg = np.meshgrid(xg, yg)
XXg = np.array([Xg.ravel(), Yg.ravel()]).T
Zg = -clf.score_samples(XXg)
Zg = Zg.reshape(Xg.shape)

# f = interpolate.interp2d(Xg, Yg, Zg,kind='cubic')

plt.figure()
CS = plt.contour(Xg, Yg, Zg, norm=LogNorm(vmin=1.0, vmax=1000.0),
                  levels=np.logspace(0, 3, 10))
plt.plot(X[:,0],X[:,1],'bo')

fig=plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(Xg,Yg,Zg)
plt.plot(X[:,0],X[:,1],'bo')


plt.figure()
plt.plot(X[:,0],X[:,1],'bo')
for i in range(clf.n_components):
    m = clf.means_[i]
    P = clf.covariances_[i]
    Xe= utpltgmshp.getCovEllipsePoints2D(m,P,nsig=1,N=100)
    plt.plot(Xe[:,0],Xe[:,1],'g')

theta = np.random.uniform(-np.pi/2,np.pi/2)
rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
Xtest = np.matmul(rotation_matrix,X.T).T + 5*np.random.rand(1,X.shape[1])

plt.figure()
plt.plot(X[:,0],X[:,1],'bo')
plt.plot(Xtest[:,0],Xtest[:,1],'ro')


# m=np.mean(Xtest,axis=0)
# Xtest=Xtest-m

# np.mean(Xtest,axis=0)

# # brute-force grid search
# sttime = time.time()
# thset=np.linspace(-np.pi/2,np.pi/2,15)
# txset=np.linspace(-2,2,5) # 
# tyset=np.linspace(-2,2,5) # 
# def getcostalign(x,Xt,clf):
#     th=x[0]
#     t=x[1:]
#     R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
#     Xn=np.matmul(R,Xt.T).T+t
#     logp = -clf.score_samples(Xn)
#     # logp = np.diag(f(Xn[:,0],Xn[:,1]))
#     return sum(logp)
# m=1e10
# best=[]
# for th in thset:
#     for tx in txset:
#         for ty in tyset:
#             t=[tx,ty]
#             c=getcostalign([th,tx,ty],Xtest,clf)
#             # x0=[th,tx,ty]
#             # res = minimize(getcostalign, x0,args=(Xtest,clf) ,method='BFGS', tol=1e-1,options={'maxiter': 1})
            
#             # th=res.x[0]            
#             # t=res.x[1:]
#             # c=res.fun 
#             if c < m:
#                 m=c
#                 best=[th,t]                

# endtime = time.time()-sttime
# print("time taken: ",endtime )
# th=best[0]
# t=best[1]
# # th=theta
# # t=[0,0]
# R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
# Xn=np.matmul(R,Xtest.T).T+t
    
# plt.figure()
# plt.plot(X[:,0],X[:,1],'bo')
# plt.plot(Xn[:,0],Xn[:,1],'ro')
# plt.title("grid based optimization")

# sttime = time.time()
# x0=[best[0],best[1][0],best[1][1]]
# # x0=[0,1,1]
# res = minimize(getcostalign, x0,args=(Xtest,clf) ,method='Nelder-Mead', tol=1e-6)
# endtime = time.time()-sttime                                 #Nelder-Mead
# print("opt-clf-time taken: ",endtime )

# th=res.x[0]
# t=res.x[1:]
# R = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
# Xn=np.matmul(R,Xtest.T).T+t


# plt.figure()
# plt.plot(X[:,0],X[:,1],'bo')
# plt.plot(Xn[:,0],Xn[:,1],'ro')
# plt.title("Direct optimization")
# for i in range(clf.n_components):
#     m = clf.means_[i]
#     P = clf.covariances_[i]
#     Xe= utpltgmshp.getCovEllipsePoints2D(m,P,nsig=1,N=100)
#     plt.plot(Xe[:,0],Xe[:,1],'g')
# # interpolate cost function and derivatives of keyframe

# #%%
