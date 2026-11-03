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
from utils.plotting import geometryshapes as utpltgmshp
import time
from scipy.optimize import minimize, rosen, rosen_der
from scipy import interpolate
import networkx as nx
import pdb
import pandas as pd
from fastdist import fastdist

from lidarprocessing import point2Dprocessing as pt2dproc


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
    # ptset=np.ascontiguousarray(ptset,dtype=np.float64)
    return ptset[rngidx,:]

def plotGraph(poseGraph,Lkey,ax=None):
    pos = nx.get_node_attributes(poseGraph, "pos")
    # poskey = {k:v for k,v in pos.items() if k in Lkey}
    
    # node_color  = []
    # for n in poseGraph.nodes:
    #     if n in Lkey:
    #         if poseGraph.nodes[n]["frametype"]=='keyframe':
    #             node_color.append('g')
    #         elif poseGraph.nodes[n]["frametype"]=='scan':
    #             node_color.append('r')
    node_color_dict = nx.get_node_attributes(poseGraph, "color")
    node_color = [node_color_dict[n] for n in Lkey]
    
    edge_color_dict=nx.get_edge_attributes(poseGraph, "color")
    edge_type_dict=nx.get_edge_attributes(poseGraph, "edgetype")
    
    edge_color = [edge_color_dict[e] for e in poseGraph.edges if e[0] in Lkey and e[1] in Lkey]
    edgelist = [e for e in poseGraph.edges if e[0] in Lkey and e[1] in Lkey]
    
    # edgelist =[]
    # for e in poseGraph.edges:
    #     if e[0] in Lkey and e[1] in Lkey:
    #         edgelist.append(e)
    #         if 'edgetype' in poseGraph.edges[e[0],e[1]]:
    #             if poseGraph.edges[e[0],e[1]]['edgetype']=='Key2Key-LoopClosure':
    #                 edge_color.append('b')
    #             else:
    #                 edge_color.append('r')
    #         else:
    #             edge_color.append('k')
    
    if ax is None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    
    nx.draw_networkx(poseGraph,pos=pos,nodelist =Lkey,edgelist=edgelist,edge_color=edge_color,with_labels=True,font_size=6,node_size=200,ax=ax)


def plot_keyscan_path(poseGraph,idx1,idx2):
    Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    Lkey = [x for x in Lkey if x>=idx1 and x<=idx2]
    Lkey.sort()

    
    fig = plt.figure("Full Plot")
    if len(fig.axes)==0:
        ax = fig.add_subplot(111)
    else:
        ax = fig.axes[0]
        
    ax.cla()
    
    figg = plt.figure("Graph Plot")
    if len(figg.axes)==0:
        axgraph = figg.add_subplot(111)
    else:
        axgraph = figg.axes[0]
        
    axgraph.cla()
    
    plotGraph(poseGraph,Lkey,ax=axgraph)
    plt.show()

    
    # plotting
    # if idx % 1==0 or idx==len(dataset)-1:
    plt.figure("Full Plot")
    # plot scans in global frame
    for i in Lkey:
        gHs = nplinalg.inv(poseGraph.nodes[i]['sHg'])
        XX = poseGraph.nodes[i]['X']
        XX=np.matmul(gHs,np.vstack([XX.T,np.ones(XX.shape[0])])).T   
        ax.plot(XX[:,0],XX[:,1],'b.')
        
    # gHs=nplinalg.inv(poseGraph.nodes[idx]['sHg'])
    # Xg=np.matmul(gHs,np.vstack([X.T,np.ones(X.shape[0])])).T   

    
    posdict = nx.get_node_attributes(poseGraph, "pos")
    
    # plot robot path
    Xr=[]
    KeyFrames=[]
    for idx in range(idx1,idx2+1):
        if idx in posdict:
            Xr.append(posdict[idx])
        if idx in Lkey:
            KeyFrames.append(posdict[idx])
            
    Xr=np.array(Xr)
    KeyFrames=np.array(KeyFrames)
    
    ax.plot(Xr[:,0],Xr[:,1],'r')
    ax.plot(Xr[-1,0],Xr[-1,1],'ro')
    

    ax.plot(KeyFrames[:,0],KeyFrames[:,1],'gs')
    
    ax.set_title(str(idx1)+" to "+str(idx2))
    
    plt.show()
    plt.pause(0.05)
    

def plotcomparisons(idx1,idx2,H12=None):
    # H12: from 2 to 1
    
    fig = plt.figure(figsize=(20,10))
    if H12 is None:
        ax = fig.subplots(nrows=1, ncols=2)
    else:
        ax = fig.subplots(nrows=1, ncols=3)
    # idx=6309
    # idx2=8761
    X1=getscanpts(dataset,idx1)
    X2=getscanpts(dataset,idx2)
    # X12: points in 2 , transformed to 1
    X12 = np.dot(H12,np.hstack([X2,np.ones((X2.shape[0],1))]).T).T
    X12=X12[:,0:2]
    h1=poseGraph.nodes[idx1]['h']
    h2=poseGraph.nodes[idx2]['h']
    
    X12=X12-np.mean(X1,axis=0)            
    X1=X1-np.mean(X1,axis=0)
    X2=X2-np.mean(X2,axis=0)
    
    ax[0].cla()
    ax[1].cla()
    ax[0].plot(X1[:,0],X1[:,1],'b.')
    ax[1].plot(X2[:,0],X2[:,1],'r.')
    ax[0].set_xlim(-4,4)
    ax[1].set_xlim(-4,4)
    ax[0].set_ylim(-4,4)
    ax[1].set_ylim(-4,4)
    ax[0].set_title(str(idx1))
    ax[1].set_title(str(idx2))
    ax[0].axis('equal')
    ax[1].axis('equal')
    if H12 is not None:
        ax[2].cla()
        ax[2].plot(X1[:,0],X1[:,1],'b.')
        ax[2].plot(X12[:,0],X12[:,1],'k.')
        ax[2].set_xlim(-4,4)
        ax[2].set_ylim(-4,4)
        ax[2].axis('equal')
    plt.show()
    plt.pause(1)
#%% TEST::::::: Pose estimation by keyframe
plt.close("all")
poses=[]
poseGraph = nx.DiGraph()
Xr=np.zeros((len(dataset),3))
ri=0
KeyFrames=[]

REL_POS_THRESH=0.5 # meters after which a keyframe is made
ERR_THRES=1.6

LOOP_CLOSURE_D_THES=0.8
LOOP_CLOSURE_POS_THES=2
LOOP_CLOSURE_ERR_THES=1.0

# fig = plt.figure("Full Plot")
# ax = fig.add_subplot(111)

# figg = plt.figure("Graph Plot")
# axgraph = figg.add_subplot(111)

idx1=5000
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
        if np.all(np.isfinite(tpos)) is False:
            pdb.set_trace()
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
            m_clf = poseGraph.nodes[KeyFrame_prevIdx]['m_clf']
            msHk = poseGraph.edges[KeyFrame_prevIdx,scanid]['H']
            sHk_newkeyframe =  np.matmul(msHk,nplinalg.inv(sHk))
            sHk,err = pt2dproc.scan2keyframe_match(KeyFrameClf,m_clf,Xs,sHk=sHk_newkeyframe)

            poseGraph.add_edge(idx,scanid,H=sHk,edgetype="Key2Scan",color='r')
            
            # delete rest of the scan-ids as they are useless
            for i in Lidxs:
                if i!=scanid and poseGraph.in_degree(i)==1:
                    poseGraph.remove_node(i)
            
            # pdb.set_trace()
            
        KeyFrame_prevIdx = idx
        KeyFrames.append(np.matmul(gHs,np.array([0,0,1]))  )
        
        
        # detect loop closure and add the edge
        Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
        Lkey.sort()
        for previdx in Lkey:
            if previdx < idx and poseGraph.has_edge(idx,previdx) is False and poseGraph.has_edge(previdx,idx) is False:
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
                    if min([pi_err_i,i_err_pi]) <LOOP_CLOSURE_ERR_THES:
                        print("Adding Loop closure")
                        poseGraph.add_edge(idx,previdx,H=piHi,edgetype="Key2Key-LoopClosure",d=d,color='b')
                    
    else: #not a keyframe
        tpos=np.matmul(gHs,np.array([0,0,1]))
        if np.all(np.isfinite(tpos)) is False:
            pdb.set_trace()
        poseGraph.add_node(idx,frametype="scan",time=idx,X=X,sHg=sHg,pos=(tpos[0],tpos[1]),color='r')
        poseGraph.add_edge(KeyFrame_prevIdx,idx,H=sHk,edgetype="Key2Scan",color='r')
    
    
    
    
    
    # loop closure optimization
    if pt2dproc.loopExists(poseGraph,idx1,idx) and poseGraph.nodes[idx]['frametype']=="keyframe":
        if idx-previdx_loopclosure>=2:
            print("performing loop closure optimization")
            successflag,sHg_updated,sHg_previous=pt2dproc.adjustPoses(poseGraph,idx1,idx)
            if successflag:
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
                            if poseGraph.nodes[pidx]['frametype']=="keyframe":
                                psHg=poseGraph.nodes[pidx]['sHg']
                                nsHps=poseGraph.edges[pidx,ns]['H']
                                nsHg = nsHps.dot(psHg)
                                poseGraph.nodes[ns]['sHg']=nsHg
                                gHns=nplinalg.inv(nsHg)
                                tpos=np.matmul(gHns,np.array([0,0,1]))
                                poseGraph.nodes[ns]['pos'] = (tpos[0],tpos[1])
                    
                                break
                    
                previdx_loopclosure = idx
    
    
    
    
    # plotting
    plot_keyscan_path(poseGraph,idx1,idx)
    
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
    poseGraph=pkl.load(fh)
    
H=[]
for idx in range(5000,len(dataset)):
    print(idx)
    X=getscanpts(dataset,idx)
    h=pt2dproc.get2DptFeat(X)
    H.append(h)

thes=0.5
fig = plt.figure(figsize=(20,10))
ax = fig.subplots(nrows=1, ncols=2)
for idx in range(6309,len(dataset)):
    for idx2 in range(idx+50,len(dataset)):
        d=nplinalg.norm(H[idx2-5000]-H[idx-5000],ord=1)
        if d<=thes:
            X1=getscanpts(dataset,idx)
            X2=getscanpts(dataset,idx2)
            X1=X1-np.mean(X1,axis=0)
            X2=X2-np.mean(X2,axis=0)
            ax[0].cla()
            ax[1].cla()
            ax[0].plot(X1[:,0],X1[:,1],'b.')
            ax[1].plot(X2[:,0],X2[:,1],'r.')
            ax[0].set_xlim(-4,4)
            ax[1].set_xlim(-4,4)
            ax[0].set_ylim(-4,4)
            ax[1].set_ylim(-4,4)
            ax[0].set_title(str(idx))
            ax[1].set_title(str(idx2))
            ax[0].axis('equal')
            ax[1].axis('equal')
            plt.show()
            plt.pause(1)

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
         
LOOP_CLOSURE_THES=0.8
LOOP_CLOSURE_POS_THES=2
LOOP_CLOSURE_ERR_THES=1
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
        if previdx < idx and poseGraph.has_edge(idx,previdx) is False and poseGraph.has_edge(previdx,idx) is False:
            h1=poseGraph.nodes[idx]['h']
            h2=poseGraph.nodes[previdx]['h']
            p1=poseGraph.nodes[idx]['pos']
            p2=poseGraph.nodes[previdx]['pos']
            d=nplinalg.norm(h1-h2,ord=1)
            if d<=LOOP_CLOSURE_THES and nplinalg.norm(np.array(p1)-np.array(p2),ord=2)<=LOOP_CLOSURE_POS_THES:
                print("Candidate Loop closure")
                piHi,pi_err_i=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,idx,previdx)
                iHpi,i_err_pi=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,previdx,idx)
                # flag=True
                # plotcomparisons(previdx,idx,H12=piHi)
                # plotcomparisons(idx,previdx,H12=iHpi)
                if min([pi_err_i,i_err_pi]) <LOOP_CLOSURE_ERR_THES:
                    print("Adding Loop closure")
                    poseGraph.add_edge(idx,previdx,H=piHi,edgetype="Key2Key-LoopClosure",d=d,color='b')
    
    subgraph = poseGraph.subgraph([n for n in poseGraph.nodes if n<=idx])
    keyframe_nodes = list(filter(lambda x: subgraph.nodes[x]['frametype']=="keyframe",subgraph.nodes))    
    L=list(nx.simple_cycles(subgraph))
    L=[sorted(m) for m in L]
    L.sort(key=lambda x: len(x))
    
plotGraph(poseGraph,Lkey)
plt.show()
#%%
idx=8508
previdx= 8407
piHi,pi_err_i=pt2dproc.poseGraph_keyFrame_matcher(poseGraph,idx,previdx)
h1=poseGraph.nodes[idx]['h']
h2=poseGraph.nodes[previdx]['h']
p1=poseGraph.nodes[idx]['pos']
p2=poseGraph.nodes[previdx]['pos']
d=nplinalg.norm(h1-h2,ord=1)

plotcomparisons(previdx,idx,H12=piHi)

#%% loop closure optimization
with open("PoseGraph-Sample",'rb') as fh:
    poseGraph,Xr=pkl.load(fh)
    

def globalPoseCost(x,Hrelsidx,Hrels):
    # x is global poses
    # Hrels=[[i,j,thji,txji,tyji],...]
    x=x.reshape(-1,3)
    x=np.vstack([np.zeros((1,3)),x])        
    th=x[:,0] # global
    txy=x[:,1:] # global
    F=np.zeros(Hrels.shape[0])
    for idx in range(Hrels.shape[0]):
        i=Hrelsidx[idx,0]
        j=Hrelsidx[idx,1]
        
        thji=Hrels[idx,0]
        tji=Hrels[idx,1:]
        
        jHi=pt2dproc.getHmat(thji,tji)
        
        thi = th[i]
        ti = txy[i]
        thj = th[j]
        tj = txy[j]
        
        fHi=pt2dproc.getHmat(thi,ti)
        fHj=pt2dproc.getHmat(thj,tj)
        
        jHf = nplinalg.inv(fHj)
        
        e=(jHi-jHf.dot(fHi))
        eR=e[0:2,0:2]
        et=e[0:2,2]
        F[idx] = eR[0,0]**2+eR[1,1]**2 + 5*(et[0]**2 + et[1]**2)
    
    f=np.sum(F)
    return f
    
        
# def adjustPoses(poseGraph,idx1,idx2):
# adjust the bundle from idx1 to idx2
# i.e. optimize the global pose matrices, with constraint local poses 
# idx1 is assumed to be fixed (like origin)
idx1=5782
# idx2=6368
idx2=6368
pt2dproc.loopExists(poseGraph,idx1,idx2)

Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
Lkey = [x for x in Lkey if x>=idx1 and x<=idx2]
Lkey.sort()

Lkey_dict={x:i for i,x in enumerate(Lkey)}

Hrels=[] # Hrels=[[i,j,thji,txji,tyji],...]
Hrelsidx=[]
for j in range(1,len(Lkey)):
    idx = Lkey[j]
    for idx_prev in poseGraph.predecessors(idx):
        i = Lkey_dict[idx_prev]
        jHi=poseGraph.edges[idx_prev,idx]['H']
        tji,thji = pt2dproc.extractPosAngle(jHi)
        Hrels.append([thji,tji[0],tji[1] ])        
        Hrelsidx.append([i,j])
Hrels = np.array(Hrels,dtype=np.float64)
Hrelsidx = np.array(Hrelsidx,dtype=np.int32)

# x0=x0.reshape(-1)
x0 = np.zeros((len(Lkey)-1,3))            
firstHg=poseGraph.nodes[Lkey[0]]['sHg']
gHfirst = nplinalg.inv(firstHg)
sHg_original={}
for j in range(1,len(Lkey)):
    idx = Lkey[j]
    sHg=poseGraph.nodes[idx]['sHg']
    sHg_original[idx] = sHg
    
    jHf=np.matmul(sHg,gHfirst) # first to j (local global frame)
    fHj=nplinalg.inv(jHf)
    tj,thj = pt2dproc.extractPosAngle(fHj)
    # tj = Hjf[:2,2]  
    x0[j-1,0]=thj
    x0[j-1,1:]=tj
    
x0=x0.reshape(-1)

st=time.time()
f=globalPoseCost(x0,Hrelsidx,Hrels)
et=time.time()
print(f," time: ",et-st)

st=time.time()
f=pt2dproc.globalPoseCost(x0,Hrelsidx,Hrels)
et=time.time()
print(f," time: ",et-st)

st=time.time()
res = minimize(globalPoseCost, x0+0.1*np.random.randn(len(x0)),args=(Hrelsidx,Hrels) ,method='BFGS', tol=1e-5)
et=time.time()
print(res.fun," time: ",et-st)

st=time.time()
res = minimize(pt2dproc.globalPoseCost, x0+0.1*np.random.randn(len(x0)),args=(Hrelsidx,Hrels) ,method='BFGS', tol=1e-5)
et=time.time()
print(res.fun," time: ",et-st)



x = res.x
x=x.reshape(-1,3)   
sHg_updated={}
for j in range(1,len(Lkey)):     
    thj=x[j-1,0] # global
    tj=x[j-1,1:] # global

    fHj=pt2dproc.getHmat(thj,tj)
    jHf=nplinalg.inv(fHj)
    sHg_updated[Lkey[j]] = np.matmul(jHf,firstHg)
     

idx1=5000
idx2=9884   

sHg_updated,sHg_previous=pt2dproc.adjustPoses(poseGraph,idx1,idx2)

for idx in sHg_updated.keys():
    sHg = sHg_updated[idx]
    gHs=nplinalg.inv(sHg)
    tpos=np.matmul(gHs,np.array([0,0,1]))
    poseGraph.nodes[idx]['pos'] = (tpos[0],tpos[1])
    poseGraph.nodes[idx]['sHg'] = sHg
    
# plotGraph(poseGraph,Lkey)
# plt.show()

plot_keyscan_path(poseGraph,idx1,idx2)

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

# #%% plotting functions
# plt.close("all")
# xg = np.linspace(np.min(X[:,0])-10., np.min(X[:,0])+10,100)
# yg = np.linspace(np.min(X[:,1])-10., np.min(X[:,1])+10,100)

# clf,MU,P,W = pt2dproc.getclf(X,reg_covar=0.02)

# Xg, Yg = np.meshgrid(xg, yg)
# XXg = np.array([Xg.ravel(), Yg.ravel()]).T
# Zg = -clf.score_samples(XXg)
# Zg = Zg.reshape(Xg.shape)

# # f = interpolate.interp2d(Xg, Yg, Zg,kind='cubic')

# plt.figure()
# CS = plt.contour(Xg, Yg, Zg, norm=LogNorm(vmin=1.0, vmax=1000.0),
#                  levels=np.logspace(0, 3, 10))
# plt.plot(X[:,0],X[:,1],'bo')

# fig=plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(Xg,Yg,Zg)
# plt.plot(X[:,0],X[:,1],'bo')


# plt.figure()
# plt.plot(X[:,0],X[:,1],'bo')
# for i in range(clf.n_components):
#     m = clf.means_[i]
#     P = clf.covariances_[i]
#     Xe= utpltgmshp.getCovEllipsePoints2D(m,P,nsig=1,N=100)
#     plt.plot(Xe[:,0],Xe[:,1],'g')

# theta = np.random.uniform(-np.pi/2,np.pi/2)
# rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
# Xtest = np.matmul(rotation_matrix,X.T).T + 5*np.random.rand(1,X.shape[1])

# plt.figure()
# plt.plot(X[:,0],X[:,1],'bo')
# plt.plot(Xtest[:,0],Xtest[:,1],'ro')


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
