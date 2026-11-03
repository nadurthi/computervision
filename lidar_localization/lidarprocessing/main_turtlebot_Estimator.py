# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 20:46:31 2020

@author: nadur
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_default, qos_profile_sensor_data
from rclpy.qos import QoSReliabilityPolicy

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan, Imu 
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose,PoseStamped

import time
import pickle
import signal
import datetime

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
import codecs
  
from lidarprocessing import turtlebotModels as tbotmd      

#https://quaternion.readthedocs.io/en/latest/
import quaternion


from uq.quadratures import cubatures as quadcub

from uq.uqutils import pdfs as uqutlpdfs
from uq.stats import moments as uqstmom
from uq.filters import kalmanfilter as uqfkf


import datetime
dtype = np.float64

#%%

def UKFprop(t, dt, model, xfk, Pfk, uk, **params):
    
    X, W = quadcub.GaussianSigmaPtsMethodsDict[quadcub.SigmaMethod.UT](xfk, Pfk)
    Xk1 = np.zeros(X.shape)
    for i in range(len(W)):
        _, Xk1[i, :] = model.propforward(t, dt, X[i, :], uk=uk, **params)

    xfk1, Pfk1 = uqstmom.MeanCov(Xk1, W)

    Q = model.processNoise(t, dt, xfk, uk=uk, **params)

    Pfk1 = Pfk1 + Q

    return xfk1, Pfk1
    
def measUpdate( t, dt, xfk, Pfk, model, zk, **params):
    """
    If zk is None, just do pseudo update.

    @param: t
    """

    X, W = quadcub.GaussianSigmaPtsMethodsDict[quadcub.SigmaMethod.UT](xfk, Pfk)

    Z = np.zeros((X.shape[0], len(zk)))
    for i in range(len(W)):
        Z[i, :], isinFOV, L = model.sensormodel(t, dt, X[i, :])

    mz, Pz = uqstmom.MeanCov(Z, W)

    R = sensormodel.measNoise(t, dt, xfk)
    Pz = Pz + R

    Pxz = np.zeros((len(xfk), sensormodel.hn))
    for i in range(len(W)):
        Pxz = Pxz + W[i] * np.outer(X[i, :] - xfk, Z[i, :] - mz)

    pdfz = multivariate_normal(mz, Pz)
    pdfz.isInNsig = lambda x, N: uqutlpdfs.isInNsig(x, mz, Pz, N)

    if zk is None:
        xu, Pu, K = uqfkf.KFfilterer.KalmanDiscreteUpdate(xfk, Pfk, mz, mz, Pz, Pxz)
#            return (xu, Pu, mz,R, Pxz,Pz, K, pdfz, likez )
        return (xu, Pu, mz, R, Pxz, Pz, K, pdfz, None)
    else:
        xu, Pu, K = uqfkf.KFfilterer.KalmanDiscreteUpdate(xfk, Pfk, zk, mz, Pz, Pxz)
        likez = pdfz.pdf(zk)

        return (xu, Pu, mz, R, Pxz, Pz, K, pdfz, likez)
    
 #%%

class TurtlebotEstimator:
    def __init__(self):
        self.xk = None
        self.Pk=None

        self.Lodom={'trans':[],'q':[],'t':[]}
        self.Llidarpose={'trans':[],'q':[],'t':[],'gHs':[]}
        
    
    def save(self,filename):
        print("Saving")
        with open(filename,'wb') as fh:
            pickle.dump({'odom':self.Lodom,'lidarPose':self.Llidarpose},fh)        
            
            
    def odom_listener_callback(self,msg):
        
        T=datetime.datetime.fromtimestamp(msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec)
        
        self.Lodom['trans'].append([msg.pose.pose.position.x,msg.pose.pose.position.y,msg.pose.pose.position.z])
        self.Lodom['t'].append( T )
        self.Lodom['q'].append( [msg.pose.pose.orientation.x,msg.pose.pose.orientation.y,msg.pose.pose.orientation.z,msg.pose.pose.orientation.w] )
    
    def lidar_pose_callback(self,msg):
        T=datetime.datetime.fromtimestamp(msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec)
        
        tpos=[msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        q=np.zeros(4)
        q[0] = msg.pose.orientation.x
        q[1] = msg.pose.orientation.y
        q[2] = msg.pose.orientation.z
        q[3] = msg.pose.orientation.w
        q = quaternion.from_float_array(q)
        gHs = quaternion.as_rotation_matrix(q)
        gHs[0,2]=tpos[0]
        gHs[1,2]=tpos[1]
        
        self.Llidarpose['trans'].append(tpos)
        self.Llidarpose['t'].append(T)
        self.Llidarpose['q'].append(q)
        self.Llidarpose['gHs'].append(gHs)
        # print(gHs)
        
        
    def plotposegraph(self):
        st = time.time()
        Lkeys = list(filter(lambda x: self.poseGraph.nodes[x]['frametype']=="keyframe",self.poseGraph.nodes))
        idx0=min(Lkeys)
        idx1=max(Lkeys)
        self.fig,self.ax,self.figg,self.axgraph=pt2dplot.plot_keyscan_path(self.poseGraph,idx0,idx1,makeNew=False,plotGraph=False)
        et = time.time()
        print("plotting time : ",et-st)

        odotrans = np.array(self.Lodom['trans'])
        self.ax.plot(odotrans[:,0],odotrans[:,1],'m--')

        lidartrans = np.array(self.Llidarpose['trans'])
        self.ax.plot(lidartrans[:,0],lidartrans[:,1],'c--')
        
        
        plt.draw()
        plt.pause(0.01)
        
#%% TEST::::::: Pose estimation by keyframe


# idx1=1000
# previdx_loopclosure = 0 # 

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('turtlebotPlotter')
    ttlest=TurtlebotEstimator()

    node.create_subscription(String,'posegraphclose',ttlest.getPoseGraph,qos_profile_sensor_data)
    node.create_subscription(Odometry,'odom',ttlest.odom_listener_callback)
    node.create_subscription(PoseStamped,'lidarPose',ttlest.lidar_pose_callback,qos_profile_sensor_data)
    
    
    print("ready")
    try:
        while True:
            rclpy.spin_once(node,timeout_sec=1)
            if ttlest.poseGraph is not None:
                print("plotting posegrph")
                ttlest.plotposegraph()
                ttlest.poseGraph = None
                
        # rclpy.spin(node)
    except KeyboardInterrupt:
    	pass
    
    time.sleep(1)
    ttlest.save("turtlebotEstimate.pkl")
    
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    



    
    
    


#%%
