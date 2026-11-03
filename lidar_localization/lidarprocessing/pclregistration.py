#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 10:43:45 2021

@author: na0043
"""
import importlib


import pyslam.slam as slam
importlib.reload(slam)

import numpy as np


A=5*np.random.rand(10000,3)
a=np.cos(0*np.pi/6)
b=np.sin(0*np.pi/6)

R=np.array([[a,b,0],[-b,a,0],[0,0,1]])
t=1*np.array([1,2,3])

B=R.dot(A.T).T+t

D={}
D["icp_setMaximumIterations"]=100
D["icp_setMaxCorrespondenceDistance"]=50
D["icp_setRANSACIterations"]=0
D["icp_setRANSACOutlierRejectionThreshold"]=1.5
D["icp_setTransformationEpsilon"]=1e-9
D["icp_setEuclideanFitnessEpsilon"]=1


D["gicp_setMaxCorrespondenceDistance"]=50
D["gicp_setMaximumIterations"]=100
D["gicp_setMaximumOptimizerIterations"]=100
D["gicp_setRANSACIterations"]=0
D["gicp_setRANSACOutlierRejectionThreshold"]=1.5
D["gicp_setTransformationEpsilon"]=1e-9
D["icp_setUseReciprocalCorrespondences"]=0.1

D["ndt_setTransformationEpsilon"]=1e-9
D["ndt_setStepSize"]=0.1
D["ndt_setResolution"]=1
D["ndt_setMaximumIterations"]=100
D["ndt_initialguess_axisangleA"]=0.6931
D["ndt_initialguess_axisangleX"]=0
D["ndt_initialguess_axisangleY"]=0
D["ndt_initialguess_axisangleZ"]=1
D["ndt_initialguess_transX"]=1.7
D["ndt_initialguess_transY"]=0.7
D["ndt_initialguess_transZ"]=0

H=slam.registrations(A,B,D)


