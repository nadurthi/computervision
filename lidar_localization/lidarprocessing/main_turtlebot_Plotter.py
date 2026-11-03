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
import matplotlib
matplotlib.use("TKAgg",warn=False, force=True)

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
import turtlebot_helper as ttlhelp

#https://quaternion.readthedocs.io/en/latest/
import quaternion



import datetime
dtype = np.float64

#%%

from rclpy.duration import Duration
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSLivelinessPolicy
from rclpy.qos import QoSPresetProfiles
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

# qos_closedposegraphPoses_profile = QoSProfile(
#             history=QoSHistoryPolicy.KEEP_LAST,
#             depth=10,
#             reliability=QoSReliabilityPolicy.RELIABLE,
#             durability=QoSDurabilityPolicy.VOLATILE,
#             lifespan=Duration(seconds=4),
#             deadline=Duration(seconds=5),
#             # liveliness=QoSLivelinessPolicy.MANUAL_BY_TOPIC,
#             # liveliness_lease_duration=Duration(nanoseconds=12),
#             avoid_ros_namespace_conventions=True
#         )


 #%%
# Terminology:
    # - GMM or gmm or clf or classifier: gaussian mixture classifer that fits the point cloud
    # - Keyframe contains the gmm classifier
    # - subsequent scans are matched to the previous keyframe and pose is estimated to this key frame
    # - gHs : scanframe to global frame
    # - kHs : scanframe to keyframe
    # - sHk : keyframe to scanframe
    # - for posegraph: nodes carry the sHg transformation matrix from global to current frame
    #                : edges carry the H matrix transformation between the frames.
    
    
# REL_POS_THRESH=0.3 # meters after which a keyframe is made
# ERR_THRES=1.2 # error threshold after which a keyframe is made


# LOOP_CLOSURE_D_THES=1.5 # the histogram threshold for matching
# LOOP_CLOSURE_POS_THES=4 # match two keyframes only if they are within this threshold
# LOOP_CLOSURE_ERR_THES=-0.70 # match two keyframes only if error is less than this threshold


class TurtlebotPlotter:
    def __init__(self):
        self.poseGraph=None
        self.Lodom={'trans':[],'q':[],'t':[]}
        self.Llidarpose={'trans':[],'q':[],'t':[],'gHs':[]}

    def save(self,filename):
        print("Saving")
        with open(filename,'wb') as fh:
            pickle.dump({'odom':self.Lodom,'lidarPose':self.Llidarpose},fh)        
            
    # def processScanPts(self,scanmsg):
    #     Tstamp,T,X = ttlhelp.filter_scanmsg(scanmsg)
        
    #     self.poseData[T]={'X':X}
    #     # return X
        
        
    def getPoseGraph(self,msg):
        unpickled = pkl.loads(codecs.decode(msg.data.encode(), "base64"))
        self.poseGraph,self.params = unpickled
        
        self.plotposegraph()
        # print(self.poseGraph.nodes)
        # print(self.poseData)
        # # replace T in poseData with corresponding idx
        # for idx in self.poseGraph.nodes:
        #     T = self.poseGraph.nodes[idx]['time']
        #     if T in self.poseData:
        #         self.poseData[idx] = copy.deepcopy(self.poseData[T])
        #         del self.poseData[T]
        
        # # delete the unecessary X data in poseData
        # kys = list(self.poseData.keys())
        # posGtimeDict = nx.get_node_attributes(self.poseGraph, "time")
        # for ns in kys:
        #     if ns not in posGtimeDict.values():
        #         self.poseData.pop(ns)
        
        
                
                
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

        self.fig,self.ax,self.figg,self.axgraph=pt2dplot.plot_keyscan_path(self.poseGraph,idx0,idx1,self.params,makeNew=False,skipScanFrame=True,plotGraphbool=False,
                                    forcePlotLastidx=True,plotLastkeyClf=True,plotLoopCloseOnScanPlot=True)
        
        et = time.time()
        print("plotting time : ",et-st)

        # odotrans = np.array(self.Lodom['trans'])
        # self.ax.plot(odotrans[:,0],odotrans[:,1],'m--')

        # lidartrans = np.array(self.Llidarpose['trans'])
        # self.ax.plot(lidartrans[:,0],lidartrans[:,1],'c--')
        
        
        plt.draw()
        plt.pause(0.01)
        
#%% TEST::::::: Pose estimation by keyframe


# idx1=1000
# previdx_loopclosure = 0 # 

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('turtlebotPlotter')
    ttlplt=TurtlebotPlotter()

    node.create_subscription(String,'posegraphplot',ttlplt.getPoseGraph,ttlhelp.qos_closedposegraphPoses_profile)

    
    node.create_subscription(Odometry,'odom',ttlplt.odom_listener_callback,qos_profile_sensor_data)
    node.create_subscription(PoseStamped,'lidarPose',ttlplt.lidar_pose_callback,qos_profile_sensor_data)
    # node.create_subscription(LaserScan,'scan',ttlplt.processScanPts,qos_profile_sensor_data)
    
    
    
    print("ready")
    try:
        while True:
            rclpy.spin_once(node,timeout_sec=1)
            # if ttlplt.poseGraph is not None:
            #     print("plotting posegrph")
            #     ttlplt.plotposegraph()
            #     ttlplt.poseGraph = None
                
        # rclpy.spin(node)
    except KeyboardInterrupt:
    	pass
    
    time.sleep(1)
    # ttlplt.save("straightLineNOLIDAR_run2_processedlidar.pkl")
    
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
    



    
    
    


#%%
