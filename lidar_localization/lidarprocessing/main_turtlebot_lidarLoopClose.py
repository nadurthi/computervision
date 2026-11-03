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
from geometry_msgs.msg import Pose


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
import turtlebot_helper as ttlhelp        
# from turtlebot_helper import params

#https://quaternion.readthedocs.io/en/latest/
import quaternion



import datetime
dtype = np.float64

from rclpy.duration import Duration
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSLivelinessPolicy
from rclpy.qos import QoSPresetProfiles
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy

# qos_closedposegraphPoses_profile = QoSProfile(
#             history=QoSHistoryPolicy.KEEP_LAST,
#             depth=1,
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
    
    

class ProcessLoopClose:
    def __init__(self):
        # self.working=False
        pass
    
    def setlidarloopclose_publisher(self,loopclosedposes_pub):
        self.loopclosedposes_pub=loopclosedposes_pub
    
    # def processScanPts(self,scanmsg):
    #     Tstamp,T,X = ttlhelp.filter_scanmsg(scanmsg)
        
    #     self.poseData[T]={'X':X}
    #     # return X
        
    def getPoseGraph(self,msg):
        unpickled = pkl.loads(codecs.decode(msg.data.encode(), "base64"))
        poseGraph,params = unpickled
        
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
        
        # if self.working is False:
        self.optimizeLoopClosures(copy.deepcopy(poseGraph),copy.deepcopy(params))
    
    
        
    
    def optimizeLoopClosures(self,poseGraph,params):  
        # self.working=True          
        # do the optimization to adjust global poses
        # poseGraph = copy.deepcopy(self.poseGraph)
        
        Lkeys = list(filter(lambda x:poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
        
        if len(Lkeys)>2:
            print("doing loop closure")
            # poseGraph=pt2dproc.detectAllLoopClosures_closebyNodes(poseGraph,params,returnCopy=False,parallel=params['Loop_CLOSURE_PARALLEL'])
            # poseGraph=pt2dproc.LoopCLose_CloseByNodes(poseGraph,params)
            
            poseGraph=pt2dproc.detectAllLoopClosures(poseGraph,params,returnCopy=True,parallel=True) #
            
            Lkeys = list(filter(lambda x: poseGraph.nodes[x]['frametype']=="keyframe",poseGraph.nodes))
            
            res,sHg_updated,sHg_previous=pt2dproc.adjustPoses(poseGraph,Lkeys[0],Lkeys[-1],maxiter=None,algo='trf')
    
            # if res.success:
            #     poseGraph=pt2dproc.updateGlobalPoses(copy.deepcopy(poseGraph),sHg_updated,updateRelPoses=True)              
            # else:
            #     print("opt is failure")
            #     print(res)
            

            
            L=[]
            for n in  poseGraph.nodes:
                poseGraph.nodes[n]['X']=None
                if poseGraph.nodes[n]["frametype"]=="scan":
                    L.append(n)    
            poseGraph.remove_nodes_from(L)
            
            
            # if res.success:
            msg = String()
            # LoopDetectionsDoneDict = nx.get_node_attributes(poseGraph,'LoopDetectDone')
            pickled = codecs.encode(pkl.dumps([poseGraph,sHg_updated]), "base64").decode()
            msg.data = pickled
            self.loopclosedposes_pub.publish(msg)
            print("sending loop closed poses")
        else:
            print("not enought keyframes")
        
        # self.working=False
#%% TEST::::::: Pose estimation by keyframe


# idx1=1000
# previdx_loopclosure = 0 # 

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('turtlebotLoopCloser')
    
    plc=ProcessLoopClose()
    
    loopclosedposes_pub = node.create_publisher(String,'posegraphClosedPoses',ttlhelp.qos_closedposegraphPoses_profile)
    plc.setlidarloopclose_publisher(loopclosedposes_pub)

    node.create_subscription(String,'posegraphclose',plc.getPoseGraph,ttlhelp.qos_closedposegraphPoses_profile)
    
    # node.create_subscription(LaserScan,'scan',plc.processScanPts,qos_profile_sensor_data)
    
    print("ready")
    try:
        # while True:
            # rclpy.spin_once(node,timeout_sec=0.001)
        rclpy.spin(node)
    except KeyboardInterrupt:
    	pass
    

    rclpy.shutdown()


if __name__ == '__main__':
    main()
    



    
    
    


#%%
