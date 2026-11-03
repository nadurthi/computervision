import pickle as pkl
import numpy as np
import numpy.linalg as nplinalg
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import geometryshapes as utpltgmshp
import networkx as nx

# from lidarprocessing import point2Dprocessing as pt2dproc


 #%%
# scanfilepath = 'C:/Users/nadur/Google Drive/repos/SLAM/lidarprocessing/houseScan_std.pkl'
# scanfilepath = 'C:/Users/Nagnanamus/Google Drive/repos/SLAM/lidarprocessing/houseScan_complete.pkl'
# scanfilepath = 'C:/Users/Nagnanamus/Google Drive/repos/SLAM/lidarprocessing/houseScan_std.pkl'
# scanfilepath = 'lidarprocessing/houseScan_std.pkl'



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


def plot_keyscan_path(poseGraph,idx1,idx2,makeNew=False):
    Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    Lkey = [x for x in Lkey if x>=idx1 and x<=idx2]
    Lkey.sort()

    if makeNew:
        fig = plt.figure()
    else:
        fig = plt.figure("Full Plot")
        
    if len(fig.axes)==0:
        ax = fig.add_subplot(111)
    else:
        ax = fig.axes[0]
        
    ax.cla()
    
    if makeNew:
        figg=plt.figure()
    else:
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
    # plt.figure("Full Plot")
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
    

    ax.plot(KeyFrames[:,0],KeyFrames[:,1],'gs', markersize=3)
    
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
    
    
def plotcomparisons_posegraph(poseGraph,idx1,idx2,H12=None):
    # H12: from 2 to 1
    
    fig = plt.figure(figsize=(20,10))
    if H12 is None:
        ax = fig.subplots(nrows=1, ncols=2)
        ax[0].cla()
        ax[1].cla()
    else:
        ax = fig.subplots(nrows=1, ncols=3)
        ax[2].cla()
        ax[0].cla()
        ax[1].cla()
    # idx=6309
    # idx2=8761
    m_clf1=poseGraph.nodes[idx1]['m_clf']
    clf1=poseGraph.nodes[idx1]['clf']
    
    X1=poseGraph.nodes[idx1]['X']
    if idx2 in poseGraph.nodes:
        X2=poseGraph.nodes[idx2]['X']
    else:
        X2 = getscanpts(dataset,idx2)
    # X12: points in 2 , transformed to 1
    X12 = np.dot(H12,np.hstack([X2,np.ones((X2.shape[0],1))]).T).T
    X12=X12[:,0:2]

    
    X12=X12-m_clf1    
    X1=X1-m_clf1
    X2=X2-np.mean(X2,axis=0)
    
    for i in range(clf1.n_components):
        # print("ok")
        m = clf1.means_[i]
        P = clf1.covariances_[i]
        Xe= utpltgmshp.getCovEllipsePoints2D(m,P,nsig=1,N=100)
        ax[0].plot(Xe[:,0],Xe[:,1],'g')
        if H12 is not None:
            ax[2].plot(Xe[:,0],Xe[:,1],'g')
            
   
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
        ax[2].plot(X1[:,0],X1[:,1],'b.')
        ax[2].plot(X12[:,0],X12[:,1],'k.')
        ax[2].set_xlim(-4,4)
        ax[2].set_ylim(-4,4)
        ax[2].axis('equal')
    plt.show()
    plt.pause(1)
    
    
    
def plotKeyGmm_ScansPts(poseGraph,idx1,idx2):
    Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    Lkey=[k for k in Lkey if k>=idx1 and k<=idx2]
    
    Lscan_base = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="scan",poseGraph.nodes))
    Lscan=[]
    for idx in Lscan_base:
        if len(set(Lkey)&set(poseGraph.predecessors(idx)))>=2:
            Lscan.append(idx)
    
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    # plot scan points
    for idx in Lscan:
        sHg = poseGraph.nodes[idx]['sHg']
        gHs = nplinalg.inv(sHg)
        Xs=poseGraph.nodes[idx]['X']
        Xg=np.matmul(gHs,np.vstack([Xs.T,np.ones(Xs.shape[0])])).T
        ax.plot(Xg[:,0],Xg[:,1],'b.')
    
    # now plot gmms
    for idx in Lkey:
        sHg = poseGraph.nodes[idx]['sHg']
        gHs = nplinalg.inv(sHg)
        
        m_clf = poseGraph.nodes[idx]['m_clf']
        clf = poseGraph.nodes[idx]['clf']
        
        MU0=clf.means_ + m_clf
        P0=clf.covariances_
        W0=clf.weights_
        
        
        MU=gHs.dot(np.vstack([MU0.T,np.ones(MU0.shape[0])])).T
        MU=MU[:,0:2]
        gRs = gHs[0:2,0:2]
        P=np.zeros_like(P0)
        for i in range(len(W0)): 
            P[i]=(gRs.dot(P0[i])).dot(gRs.T)
        
        print(MU.shape,P.shape,W0.shape)
        
        for i in range(len(W0)):
            Xe = utpltgmshp.getCovEllipsePoints2D(MU[i],P[i],nsig=1,N=100)
            ax.plot(Xe[:,0],Xe[:,1],'g')
        
        
    
    