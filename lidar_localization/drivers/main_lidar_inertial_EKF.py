# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:46:31 2020

@author: nadur
"""

import pickle as pkl
import numpy as np
import numpy.linalg as nplinalg
import matplotlib.pyplot as plt
from numpy.linalg import multi_dot
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
from fastdist import fastdist
import copy
from lidarprocessing import point2Dprocessing as pt2dproc
from lidarprocessing import point2Dplotting as pt2dplot
import quaternion
import datetime
from scipy.interpolate import UnivariateSpline

dtype = np.float64



 #%%



# import pandas as pd
# accdata=pd.read_csv("Acc_2021-02-19_163407.txt",sep='\t')
# # accdata['TStamp']=pd.to_datetime(accdata['TStamp'])

# accdata.plot(x='TStamp')
# plt.show()

# scanfilepath = 'lidarprocessing/straightLineLIDAR_run2.pkl'
# with open(scanfilepath,'rb') as fh:
#     datasetstd=pkl.load(fh)

# def getscanpts(dataset,idx):
#     # ranges = dataset[i]['ranges']
#     rngs = np.array(dataset[idx]['ranges'])
    
#     angle_min=dataset[idx]['angle_min']
#     angle_max=dataset[idx]['angle_max']
#     angle_increment=dataset[idx]['angle_increment']
#     ths = np.arange(angle_min,angle_max+angle_increment,angle_increment)
#     p=np.vstack([np.cos(ths),np.sin(ths)])
    
#     rngidx = (rngs>= dataset[idx]['range_min']) & (rngs<= dataset[idx]['range_max'])
#     ptset = rngs.reshape(-1,1)*p.T
    
#     X=ptset[rngidx,:]
    
#     return X

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def stepthroughData():
    scanfilepath = 'lidarprocessing/straightLineLIDAR_run2.pkl'
    with open(scanfilepath,'rb') as fh:
        dataset=pkl.load(fh)
        
    # first get the biases
    w=[]
    acc=[]
    for d in dataset['imu'][:10]:
        w.append(d['w'])
        acc.append(d['a'])
    w=np.array(w)
    acc=np.array(acc)
    
    wbias = np.mean(w,axis=0)
    accbias = np.mean(acc,axis=0)
    
    
    wstd = np.std(w,axis=0)
    accstd = np.std(acc,axis=0)
    
    print("accbias = ",accbias, "  accstd = ",accstd)
    print("wbias = ",wbias, "  wstd = ",wstd)
    # A=[]
    # W=[]
    # T=[]
    # for i in range(1,len(dataset['imu'])):
    #     tdelta = dataset['imu'][i]['time']-dataset['imu'][i-1]['time']
    #     dt = tdelta.seconds+1e-6*tdelta.microseconds
        
    #     if dataset['imu'][i]['time']<datetime.datetime(2020, 11, 6, 9, 16, 0, 671177):
    #         continue
        
        
    #     a = np.array(dataset['imu'][i]['a'])-0*accbias
    #     w = np.array(dataset['imu'][i]['w'])-0*wbias
        
    #     A.append(a)
    #     W.append(w)
    #     T.append(dt)
    
    # DT = np.array(T)
    # T = np.cumsum(DT)
    # A = np.array(A)
    # W = np.array(W)
    
    # Aavg = np.zeros_like(A)
    # Navg = 1000
    # Aavg[Navg-1:,0] = moving_average(A[:,0], Navg)
    # Aavg[Navg-1:,1] = moving_average(A[:,1], Navg)
    # Aavg[Navg-1:,2] = moving_average(A[:,2], Navg)
    
    # Wavg = np.zeros_like(W)

    # Wavg[Navg-1:,0] = moving_average(W[:,0], Navg)
    # Wavg[Navg-1:,1] = moving_average(W[:,1], Navg)
    # Wavg[Navg-1:,2] = moving_average(W[:,2], Navg)

    # Alowpass = np.zeros_like(A)
    # Wlowpass = np.zeros_like(W)
    # Alowpass[0]=A[0]
    # Wlowpass[0]=W[0]
    # RC=25
    # alpha = dt/(RC+dt)
    # for i in range(1,A.shape[0]):
    #     Alowpass[i]=alpha*A[i]+(1-alpha)*Alowpass[i-1]
    #     Wlowpass[i]=alpha*W[i]+(1-alpha)*Wlowpass[i-1]
    
    for i in range(1,len(dataset['imu'])-1):
        tdelta = dataset['imu'][i]['time']-dataset['imu'][i-1]['time']
        dt = tdelta.seconds+1e-6*tdelta.microseconds
        
        # a = Aavg[i-1]
        # w = Wavg[i-1]
        
        a = dataset['imu'][i]['a']-1*accbias
        w = dataset['imu'][i]['w']-1*wbias
        
        odom_xy = np.array(dataset['odom'][i]['trans'])
        q=np.array(dataset['odom'][i]['q'])
        odom_q = quaternion.from_float_array(q[::-1])
        odom_euler=quaternion.as_euler_angles(odom_q)
        yield dt,a,w,odom_xy,odom_euler,accstd,wstd


#%% Inertial based EKF
plt.close("all")

A=[]
W=[]
GT=[]
for dt,a,w,odom_xy,odom_euler,accstd,wstd in stepthroughData():
    # print(a,w)
    # time.sleep(0.2)
    A.append(a)
    W.append(w)
    GT.append(dt)

GT = np.array(GT)
GT = np.cumsum(GT)
A = np.array(A)
W = np.array(W)




plt.figure()
plt.plot(GT,A[:,0],'r',label='ax')
plt.plot(GT,A[:,1],'b',label='ay')
plt.plot(GT,A[:,2],'g',label='az')    

# plt.plot(T,Aavg[:,0],'r--',label='axavg')
# plt.plot(T,Aavg[:,1],'b--',label='ayavg')
# plt.plot(T,Aavg[:,2],'g--',label='azavg')    


plt.legend()



plt.figure()
plt.plot(GT,W[:,0],'r',label='wx')
plt.plot(GT,W[:,1],'b',label='wy')
plt.plot(GT,W[:,2],'g',label='wz')    

# plt.plot(T,Wavg[:,0],'r--',label='wxavg')
# plt.plot(T,Wavg[:,1],'b--',label='wyavg')
# plt.plot(T,Wavg[:,2],'g--',label='wzavg')    

plt.legend()


#%%
processedfilepath = 'straightLineNOLIDAR_run2_processedlidar.pkl'
with open(processedfilepath,'rb') as fh:
    lidarprocessed=pkl.load(fh)
    

LT=lidarprocessed['lidarPose']['t']
Lxyz = np.array(lidarprocessed['lidarPose']['trans'])
LgHs = np.array(lidarprocessed['lidarPose']['gHs'])
Lth = np.zeros(len(LT))
for i in range(len(LT)):
    _,Lth[i] = pt2dproc.extractPosAngle(LgHs[i])
    
    
poseLidar = pd.DataFrame({'T':LT,
                          'x':Lxyz[:,0],
                          'y':Lxyz[:,1],
                          'th':Lth
                              })
TT = (poseLidar['T']-poseLidar['T'].iloc[0]).apply(lambda x: x.total_seconds())
poseLidar['TT']=TT

poseLidar.plot(x='TT',y='th')
plt.show()

poseLidar.plot(x='x',y='y')
plt.show()

poseLidar.plot(x='TT',y='x')
plt.show()

poseLidar.plot(x='TT',y='y')
plt.show()


# from scipy.spatial.transform import Rotation as R
# from scipy.spatial.transform import Slerp


eulzyx=np.vstack([poseLidar['th'].values,np.zeros(poseLidar.shape[0]),np.zeros(poseLidar.shape[0])]).T
# key_rots = R.from_euler('zyx', eulzyx)

q=quaternion.from_euler_angles(eulzyx)
ff=quaternion.as_float_array(q)
qdot=quaternion.calculus.spline_derivative(ff,TT)
qdot=quaternion.from_float_array(qdot)
w=np.zeros(len(q))
for i in range(len(q)):
    pp=quaternion.as_float_array(qdot[i]*q[i].inverse())
    w[i]=2*pp[3]
poseLidar['wq'] = w  
    
# slerp = Slerp(TT, key_rots)

Atrue = np.zeros((len(GT),3))


spl = UnivariateSpline(TT, poseLidar['x'], k=2,s=0.001)
poseLidar['sx'] = spl(TT)
splv=spl.derivative()
poseLidar['svx'] = splv(TT)
spla = splv.derivative()
poseLidar['sax'] = spla(TT)
Atrue[:,0] = spla(GT)


spl = UnivariateSpline(TT, poseLidar['y'], k=2,s=0.001)
poseLidar['sy'] = spl(TT)
splv=spl.derivative()
poseLidar['svy'] = splv(TT)
spla = splv.derivative()
poseLidar['say'] = spla(TT)
Atrue[:,1] = spla(GT)


poseLidar['sat'] = poseLidar['sax']*np.cos(poseLidar['th']) + poseLidar['say']*np.sin(poseLidar['th'])
poseLidar['san'] = -poseLidar['sax']*np.sin(poseLidar['th']) + poseLidar['say']*np.cos(poseLidar['th'])




poseLidar.plot(x='TT',y=['x','sx'])
plt.show()

poseLidar.plot(x='TT',y=['svx'])
plt.show()

poseLidar.plot(x='TT',y=['sax'])
plt.show()


poseLidar.plot(x='TT',y=['wq'])
plt.plot(GT,W[:,2],'g',label='wz')   
plt.show()





poseLidar.plot(x='TT',y=['say'])
plt.show()

poseLidar.plot(x='TT',y=['sat'])
plt.show()

plt.figure()
plt.plot(GT,A[:,0],'r',label='ax')  


plt.figure()
plt.plot(GT,A[:,1],'r',label='ay')  


from scipy.io import savemat

savemat("BuildIOModelAccel.mat",{'Atrue':Atrue,'T':GT,'Aaccel':A})
#%% TEST::::::: Pose estimation by keyframe
# plt.close("all")
# poseGraph = nx.DiGraph()
# # Xr=np.zeros((len(dataset),3))
# ri=0


# REL_POS_THRESH=0.3 # meters after which a keyframe is made
# ERR_THRES=1.2


# LOOP_CLOSURE_D_THES=1.5
# LOOP_CLOSURE_POS_THES=4
# LOOP_CLOSURE_ERR_THES=-0.70


# idx1=1000
# previdx_loopclosure = idx1

# scanfilepath = 'lidarprocessing/houseScan_complete.pkl'
# with open(scanfilepath,'rb') as fh:
#     dataset=pkl.load(fh)
    
# for idx in range(idx1,len(dataset['scan'])):

#     X=getscanpts(dataset['scan'],idx)
#     T=dataset['scan'][idx]['time']
    
#     # first frame is automatically a keyframe
#     if len(poseGraph.nodes)==0:
#         Xd,m = pt2dproc.get0meanIcov(X)
#         clf,MU,P,W = pt2dproc.getclf(Xd)
#         H=np.hstack([np.identity(2),np.zeros((2,1))])
#         H=np.vstack([H,[0,0,1]])
#         h=pt2dproc.get2DptFeat(X)
#         poseGraph.add_node(idx,frametype="keyframe",clf=clf,X=X,m_clf=m,time=T,sHg=H,pos=(0,0),h=h,color='g')
#         KeyFrame_prevIdx=idx
#         continue
    
#     # estimate pose to last keyframe
#     KeyFrameClf = poseGraph.nodes[KeyFrame_prevIdx]['clf']
#     m_clf = poseGraph.nodes[KeyFrame_prevIdx]['m_clf']
#     if (idx-KeyFrame_prevIdx)<=1:
#         sHk_prevframe = np.identity(3)
#     else:
#         sHk_prevframe = poseGraph.edges[KeyFrame_prevIdx,idx-1]['H']
#     # assuming sHk_prevframe is very close to sHk
#     st=time.time()
#     sHk,err = pt2dproc.scan2keyframe_match(KeyFrameClf,m_clf,X,sHk=sHk_prevframe)
#     et = time.time()
            
#     print("idx = ",idx," Error = ",err," , and time taken = ",et-st)
#     # now get the global pose to the frame
#     kHg = poseGraph.nodes[KeyFrame_prevIdx]['sHg'] #global pose to the prev keyframe
#     sHg = np.matmul(sHk,kHg) # global pose to the current frame: global to current frame
#     gHs=nplinalg.inv(sHg) # current frame to global
    
#     # check ifyou have to make this the keyframe
#     if err>ERR_THRES or nplinalg.norm(sHk[:2,2])>REL_POS_THRESH or (idx-KeyFrame_prevIdx)>100:
#         print("New Keyframe will now be added")
#         Xd,m = pt2dproc.get0meanIcov(X)
#         clf,MU,P,W = pt2dproc.getclf(Xd)
#         tpos=np.matmul(gHs,np.array([0,0,1])) 

#         h=pt2dproc.get2DptFeat(X)
#         poseGraph.add_node(idx,frametype="keyframe",clf=clf,X=X,m_clf=m,time=T,sHg=sHg,pos=(tpos[0],tpos[1]),h=h,color='g')
#         poseGraph.add_edge(KeyFrame_prevIdx,idx,H=sHk,edgetype="Key2Key",color='k')
        
#         # now delete previous scan data up-until the previous keyframe
#         # this is to save space. but keep 1. Also complete pose estimation to this scan
#         pt2dproc.addedge2midscan(poseGraph,idx,KeyFrame_prevIdx,sHk,keepOtherScans=True)
            
#         # make the current idx as the previous keyframe 
#         KeyFrame_prevIdx = idx
                
#         # detect loop closure and add the edge
#         doLoopClosure = True
#         if doLoopClosure:
#             poseGraph=pt2dproc.detectLoopClosures(poseGraph,idx,LOOP_CLOSURE_D_THES,LOOP_CLOSURE_POS_THES,LOOP_CLOSURE_ERR_THES,returnCopy=False)
            
#             # next do the loop closures after every 25 frames            
#             if idx%25==0 or idx==len(dataset)-1:
#                 res,sHg_updated,sHg_previous=pt2dproc.adjustPoses(poseGraph,idx1,idx)
#                 if res.success:
#                     poseGraph2=pt2dproc.updateGlobalPoses(copy.deepcopy(poseGraph),sHg_updated)
#                     poseGraph = poseGraph2
                        
#     else: #not a keyframe
#         tpos=np.matmul(gHs,np.array([0,0,1]))

#         poseGraph.add_node(idx,frametype="scan",time=T,X=X,sHg=sHg,pos=(tpos[0],tpos[1]),color='r')
#         poseGraph.add_edge(KeyFrame_prevIdx,idx,H=sHk,edgetype="Key2Scan",color='r')
    
    
    
    
    


    
    
    
#     # plotting
#     if idx%25==0 or idx==len(dataset)-1:
#         st = time.time()
#         pt2dplot.plot_keyscan_path(poseGraph,idx1,idx,makeNew=False,plotGraph=False)
#         et = time.time()
#         print("plotting time : ",et-st)
#         plt.show()
#         plt.pause(0.01)
    




# with open("EKF-Inertial-Lidar-Complete-All-timesteps",'wb') as fh:
#     pkl.dump([poseGraph],fh)

#%%
nodekeys = sorted(list(poseGraph.nodes))


