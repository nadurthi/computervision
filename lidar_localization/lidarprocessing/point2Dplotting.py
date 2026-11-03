import pickle as pkl
import numpy as np
import numpy.linalg as nplinalg
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from utils.plotting import geometryshapes as utpltgmshp
import networkx as nx
from uq.gmm import gmmfuncs as uqgmmfnc
# from lidarprocessing import point2Dprocessing as pt2dproc
from lidarprocessing import point2Dprocessing as pt2dproc
import lidarprocessing.numba_codes.point2Dprocessing_numba as nbpt2Dproc
dtype=np.float64


import matplotlib.cm as cm
 #%%
# scanfilepath = 'C:/Users/nadur/Google Drive/repos/SLAM/lidarprocessing/houseScan_std.pkl'
# scanfilepath = 'C:/Users/Nagnanamus/Google Drive/repos/SLAM/lidarprocessing/houseScan_complete.pkl'
# scanfilepath = 'C:/Users/Nagnanamus/Google Drive/repos/SLAM/lidarprocessing/houseScan_std.pkl'
# scanfilepath = 'lidarprocessing/houseScan_std.pkl'

def debugplotgmm(clf1,X1,X2,a=1):
    
    MU=np.ascontiguousarray(clf1.means_,dtype=dtype)
    P=np.ascontiguousarray(clf1.covariances_,dtype=dtype)
    W=np.ascontiguousarray(clf1.weights_,dtype=dtype)
    p1=uqgmmfnc.gmm_eval_fast(X1,MU,P,W)+a
    p2=uqgmmfnc.gmm_eval_fast(X2,MU,P,W)+a
    
    X = np.vstack([X1,X2])
    mn = np.min(X,axis=0)
    mx = np.max(X,axis=0)
    mn = mn-5
    mx = mx+5
    
    xedges = np.linspace(mn[0],mx[0],150)
    yedges = np.linspace(mn[1],mx[1],150)
    xgrid,ygrid=np.meshgrid(xedges,yedges)
    
    XX = np.hstack([xgrid.reshape(-1,1),ygrid.reshape(-1,1)])
    pgrid = uqgmmfnc.gmm_eval_fast(XX,MU,P,W)+a
    pgrid=pgrid.reshape(xgrid.shape[0],xgrid.shape[1])
    
    # pgrid = np.zeros_like(xgrid)
    # for i in range(xgrid.shape[0]):
    #     for j in range(xgrid.shape[1]):
            
    #         pgrid[i,j] = uqgmmfnc.gmm_eval_fast(X1,MU,P,W)
            
    H1, xedges, yedges = np.histogram2d(X1[:,0], X1[:,1], bins=(xedges, yedges))
    H2, xedges, yedges = np.histogram2d(X2[:,0], X2[:,1], bins=(xedges, yedges))
    
    
    fig = plt.figure('debugplotgmm-X1')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xgrid,ygrid,pgrid,alpha=0.4)
    ax.plot(X1[:,0],X1[:,1],p1,'ro',label='X1')
    ax.legend()
    
    fig = plt.figure('debugplotgmm-both')
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xgrid,ygrid,pgrid,alpha=0.4)
    ax.plot(X1[:,0],X1[:,1],p1,'ro',label='X1')
    ax.plot(X2[:,0],X2[:,1],p2,'b.',label='X2')
    ax.legend()
    
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
    ax.axis('equal')

def plot_keyscan_path(poseGraphMain,idx1,idx2,params,makeNew=False,skipScanFrame=True,plotGraphbool=True,
                      forcePlotLastidx=False,plotLastkeyClf=False,plotLoopCloseOnScanPlot=False,plotKeyFrameNodesTraj=False,CloseUpRadiusPlot=30,GlobalBinMap=None):
    Lkey = list(filter(lambda x: poseGraphMain.nodes[x]['frametype']=="keyframe",poseGraphMain.nodes))
    Lkey = [x for x in Lkey if x>=idx1 and x<=idx2]
    
    poseGraph = poseGraphMain.subgraph(Lkey)
    Lkey = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
    Lkey = [x for x in Lkey if x>=idx1 and x<=idx2]
    Lkey.sort()

    if makeNew:
        fig = plt.figure()
        if CloseUpRadiusPlot is not None:
            closeupfig = plt.figure()
    else:
        fig = plt.figure("Full Plot")
        if CloseUpRadiusPlot is not None:
            closeupfig = plt.figure("Close-Up Plot")

    if CloseUpRadiusPlot is not None:
        if len(closeupfig.axes)==0:
            closeupax = closeupfig.add_subplot(111)
        else:
            closeupax = closeupfig.axes[0]
        
        closeupax.cla()
        
    if len(fig.axes)==0:
        ax = fig.add_subplot(111)
    else:
        ax = fig.axes[0]
        
    ax.cla()
    
    if plotGraphbool:
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
        
        plt.draw()
        # plt.show()

    
    Xcomb=[]
    XcombClose=[]
    ClosebyIdxs=[]
    gHslast = nplinalg.inv(poseGraph.nodes[Lkey[-1]]['sHg'])
    tposlast=np.matmul(gHslast,np.array([0,0,1]))
    for i in Lkey:
        gHs = nplinalg.inv(poseGraph.nodes[i]['sHg'])
        tpos=np.matmul(gHs,np.array([0,0,1]))
        
                
        # Ti = poseGraph.nodes[i]['time']
        XX = poseGraph.nodes[i]['X']
        XX=np.matmul(gHs,np.vstack([XX.T,np.ones(XX.shape[0])])).T   
        Xcomb.append(XX[:,:2])
        
        if CloseUpRadiusPlot is not None:
            if nplinalg.norm(tpos-tposlast)<=CloseUpRadiusPlot:
                XcombClose.append(XX[:,:2])
                ClosebyIdxs.append(i)                    
        if i==Lkey[-1]:
            Xlast = XX[:,:2]
            
            
    Xcomb=np.vstack(Xcomb)
    Xcombfil=pt2dproc.binnerDownSampler(Xcomb,dx=0.1,cntThres=2)
    # Xcomb=pt2dproc.binnerDownSamplerProbs(Xcomb,dx=params['Plot_BinDownSampleKeyFrame_dx'],prob=params['Plot_BinDownSampleKeyFrame_probs'])    
    if CloseUpRadiusPlot is not None:
        XcombClose=np.vstack(XcombClose)
        XcombClosefil=pt2dproc.binnerDownSampler(XcombClose,dx=0.1,cntThres=2)
    # XcombClose=pt2dproc.binnerDownSamplerProbs(XcombClose,dx=params['Plot_BinDownSampleKeyFrame_dx'],prob=params['Plot_BinDownSampleKeyFrame_probs'])    
    
    if CloseUpRadiusPlot is not None:
        closeupax.plot(XcombClosefil[:,0],XcombClosefil[:,1],'k.',linewidth=0.2, markersize=2)
        closeupax.plot(Xlast[:,0],Xlast[:,1],'b.',linewidth=0.2, markersize=2)
    
    ax.plot(Xcombfil[:,0],Xcombfil[:,1],'k.',linewidth=0.2, markersize=2)
    ax.plot(Xlast[:,0],Xlast[:,1],'b.',linewidth=0.2, markersize=2)
    
    
    if GlobalBinMap is not None:
        xedges=GlobalBinMap['xedges']
        yedges=GlobalBinMap['yedges']
        globBinfig = plt.figure("Full Plot-Bin Plot")
        if len(globBinfig.axes)==0:
            globBinax = globBinfig.add_subplot(111)
        else:
            globBinax = globBinfig.axes[0]
        for i in poseGraph.nodes:
            gHs = nplinalg.inv(poseGraph.nodes[i]['sHg'])
            tpos=np.matmul(gHs,np.array([0,0,1]))
            XX = poseGraph.nodes[i]['X']
            XX=np.matmul(gHs,np.vstack([XX.T,np.ones(XX.shape[0])])).T   
            if GlobalBinMap['H'] is None:
                GlobalBinMap['H']=nbpt2Dproc.numba_histogram2D(XX[:,:2], xedges,yedges)
            else:
                GlobalBinMap['H']+=nbpt2Dproc.numba_histogram2D(XX[:,:2], xedges,yedges)
        
        dd=(xedges[1]-xedges[0])*(yedges[1]-yedges[0])
        Hprob=GlobalBinMap['H']/np.sum(GlobalBinMap['H']*dd)
        q=np.quantile(Hprob.reshape(-1,1),0.9)
        print(q)
        Hprob[Hprob>q]=1
        Hprob[Hprob<=q]=0
        # prob = numpy.histogram2d(Xcomb[:,0],Xcomb[:,1],bins=[xedges,yedges],density=True)
        # globBinax.hist2d(Xcomb[:,0],Xcomb[:,1],bins=[xedges,yedges],density=True)
        globBinax.pcolormesh(xedges,yedges, (1-Hprob).T,shading='flat',cmap=cm.gray)
        globBinax.axis('equal')
        # plt.show()
        globBinfig.canvas.draw()
    
    
    if plotLastkeyClf:
        clf1=poseGraph.nodes[Lkey[-1]]['clf']
        gHs = nplinalg.inv(poseGraph.nodes[Lkey[-1]]['sHg'])
        for i in range(clf1.n_components):
            m = clf1.means_[i]
            P = clf1.covariances_[i]
            Xe= utpltgmshp.getCovEllipsePoints2D(m,P,nsig=2,N=100)
            XX=np.matmul(gHs,np.vstack([Xe.T,np.ones(Xe.shape[0])])).T   
            ax.plot(XX[:,0],XX[:,1],'g')
            
            if CloseUpRadiusPlot is not None:
                closeupax.plot(XX[:,0],XX[:,1],'g')
    
    if forcePlotLastidx:
        gHs = nplinalg.inv(poseGraphMain.nodes[idx2]['sHg'])
        # Tidx2 = poseGraph.nodes[idx2]['time']
        XX = poseGraphMain.nodes[idx2]['X']
        XX=np.matmul(gHs,np.vstack([XX.T,np.ones(XX.shape[0])])).T   
        ax.plot(XX[:,0],XX[:,1],'r.',linewidth=0.2, markersize=2)
        if CloseUpRadiusPlot is not None:
            closeupax.plot(XX[:,0],XX[:,1],'r.',linewidth=0.2, markersize=2)
                
    # gHs=nplinalg.inv(poseGraph.nodes[idx]['sHg'])
    # Xg=np.matmul(gHs,np.vstack([X.T,np.ones(X.shape[0])])).T   

    if plotLoopCloseOnScanPlot:
        LedgesLoop = list(filter(lambda x: poseGraph.edges[x]['edgetype']=="Key2Key-LoopClosure",poseGraph.edges))
        
        for x in LedgesLoop:
            idx_pos=poseGraph.nodes[x[0]]['pos']
            previdx_pos=poseGraph.nodes[x[1]]['pos']
            ax.arrow(idx_pos[0],idx_pos[1],previdx_pos[0]-idx_pos[0],previdx_pos[1]-idx_pos[1],color='b',linestyle='--',length_includes_head=True)
            
            if CloseUpRadiusPlot is not None and x[0] in ClosebyIdxs and x[1] in ClosebyIdxs :
                closeupax.arrow(idx_pos[0],idx_pos[1],previdx_pos[0]-idx_pos[0],previdx_pos[1]-idx_pos[1],color='b',linestyle='--',length_includes_head=True)
            
    posdict = nx.get_node_attributes(poseGraphMain, "pos")
    
    # plot robot path
    Xr=[]
    KeyFrames=[]
    Xr_closeby=[]
    KeyFrames_closeb=[]
    for idx in range(idx1,idx2+1):
        if idx in posdict:
            Xr.append(posdict[idx])
        if idx in Lkey:
            KeyFrames.append(posdict[idx])
        if idx in ClosebyIdxs:
            Xr_closeby.append(posdict[idx])
            KeyFrames_closeb.append(posdict[idx])
            
    Xr=np.array(Xr)
    Xr_closeby=np.array(Xr_closeby)
    KeyFrames=np.array(KeyFrames)
    KeyFrames_closeb=np.array(KeyFrames_closeb)
    
    ax.plot(Xr[:,0],Xr[:,1],'r')
    ax.plot(Xr[-1,0],Xr[-1,1],'ro')
    
    if CloseUpRadiusPlot is not None:
        closeupax.plot(Xr[:,0],Xr[:,1],'r')
        closeupax.plot(Xr[-1,0],Xr[-1,1],'ro')
        
    if plotKeyFrameNodesTraj:
        ax.plot(KeyFrames[:,0],KeyFrames[:,1],'gs', markersize=3)
        if CloseUpRadiusPlot is not None:
            closeupax.plot(KeyFrames_closeb[:,0],KeyFrames_closeb[:,1],'gs', markersize=3)
    
    ax.set_title(str(idx1)+" to "+str(idx2))
    ax.axis('equal')
    plt.draw()
    # plt.show()
    fig.canvas.draw()
    if plotGraphbool:
        fig.canvas.draw()
    
    if CloseUpRadiusPlot is not None:
        closeupax.set_title(str(idx1)+" to "+str(idx2))
        closeupax.axis('equal')
        gHs = nplinalg.inv(poseGraphMain.nodes[idx2]['sHg'])
        tpos=gHs[0:2,2]
        closeupax.set_xlim([tpos[0]-CloseUpRadiusPlot,tpos[0]+CloseUpRadiusPlot])
        closeupax.set_ylim([tpos[1]-CloseUpRadiusPlot,tpos[1]+CloseUpRadiusPlot])
        plt.draw()
        # plt.show()
        closeupfig.canvas.draw()

            
    plt.pause(0.05)
    
    
    if plotGraphbool:      
        return fig,ax,figg,axgraph
    else:
        return fig,ax,None,None
    
    
def plotcomparisons_points(X1,X2,idx1,idx2,clf1,UseLC=False,H21_est=None,H12=None,err=None):
    # X12: points in 2 , transformed to 1
    
    fig = plt.figure("ComparisonPlot",figsize=(20,10))
    ax = fig.subplots(nrows=1, ncols=4)
    
    # H21_est = nplinalg.inv(H12)
    
    X12 = np.dot(H12,np.hstack([X2,np.ones((X2.shape[0],1))]).T).T
    X12=X12[:,0:2]

    if H21_est is not None:
        H12_est = nplinalg.inv(H21_est)
        X12est = np.dot(H12_est,np.hstack([X2,np.ones((X2.shape[0],1))]).T).T
        X12est=X12est[:,0:2]
    
    # X12=X12-np.mean(X1,axis=0)            
    # X1=X1-np.mean(X1,axis=0)
    # X2=X2-np.mean(X2,axis=0)
    
    ax[0].cla()
    ax[1].cla()
    ax[2].cla()
    ax[3].cla()
    
    ax[0].plot(X1[:,0],X1[:,1],'b.')
    ax[1].plot(X2[:,0],X2[:,1],'r.')
    ax[2].plot(X1[:,0],X1[:,1],'b.')
    if H21_est is not None:
        ax[2].plot(X12est[:,0],X12est[:,1],'r.')
    ax[3].plot(X1[:,0],X1[:,1],'b.')
    ax[3].plot(X12[:,0],X12[:,1],'r.')
    
    # ax[0].set_xlim(-4,4)
    # ax[1].set_xlim(-4,4)
    # ax[2].set_xlim(-4,4)
    
    # ax[0].set_ylim(-4,4)
    # ax[1].set_ylim(-4,4)
    # ax[2].set_ylim(-4,4)
    
    
    ax[0].set_title(str(idx1))
    ax[1].set_title(str(idx2))
    
    ax[0].axis('equal')
    ax[1].axis('equal')
    ax[2].axis('equal')
    ax[3].axis('equal')

        
        
        
        
        
    
        
    for i in range(clf1.n_components):
        # print("ok")
        m = clf1.means_[i]
        P = clf1.covariances_[i]
        Xe= utpltgmshp.getCovEllipsePoints2D(m,P,nsig=2,N=100)
        ax[0].plot(Xe[:,0],Xe[:,1],'g')
        ax[3].plot(Xe[:,0],Xe[:,1],'g')
    
    ax[3].set_title("err = %f"%(err,))
    
    plt.draw()
    # plt.show()
    fig.canvas.draw()
    
    plt.pause(1)
    return fig,ax
def plotcomparisons(poseGraph,idx1,idx2,UseLC=False,H12=None,err=None):
    # H12: from 2 to 1
    sHg_1 = poseGraph.nodes[idx1]['sHg']
    sHg_2 = poseGraph.nodes[idx2]['sHg']
    
    H21_est = np.matmul(sHg_2,nplinalg.inv(sHg_1))
    
    if UseLC:
        clf1=poseGraph.nodes[idx1]['clflc']
    else:
        clf1=poseGraph.nodes[idx1]['clf']
    
    # idx=6309
    # idx2=8761
    if UseLC:
        X1 = poseGraph.nodes[idx1]['Xlc']
        X2 = poseGraph.nodes[idx2]['Xlc']
    else:
        X1 = poseGraph.nodes[idx1]['X']
        X2 = poseGraph.nodes[idx2]['X']
        
    plotcomparisons_points(X1,X2,idx1,idx2,clf1,UseLC=UseLC,H21_est=H21_est,H12=H12,err=err)
    
    
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
    # m_clf1=poseGraph.nodes[idx1]['m_clf']
    clf1=poseGraph.nodes[idx1]['clf']
    
    X1=poseGraph.nodes[idx1]['X']
    X2=poseGraph.nodes[idx2]['X']
    
    X12 = np.dot(H12,np.hstack([X2,np.ones((X2.shape[0],1))]).T).T
    X12=X12[:,0:2]
    
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
    # ax[0].set_xlim(-4,4)
    # ax[1].set_xlim(-4,4)
    # ax[0].set_ylim(-4,4)
    # ax[1].set_ylim(-4,4)
    ax[0].set_title(str(idx1))
    ax[1].set_title(str(idx2))
    ax[0].axis('equal')
    ax[1].axis('equal')
    if H12 is not None:
        ax[2].plot(X1[:,0],X1[:,1],'b.')
        ax[2].plot(X12[:,0],X12[:,1],'k.')
        # ax[2].set_xlim(-4,4)
        # ax[2].set_ylim(-4,4)
        ax[2].axis('equal')
        
    fig.canvas.draw()
    plt.draw()
    # plt.show()
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
        
        
    
    