from numba.np.ufunc import parallel
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from numba import njit, prange, jit, cuda
from numba.typed import List, Dict
from numba.core import types
from numpy.core.numeric import identity

float_2Darray = types.float64[:,:]
import heapq
import open3d as o3d
import math
import pickle

import threading
from timeit import repeat


import readData #file for reading/plotting data
import numbaFunctions as nF #file for all of the numba functions
import cudaFunctions as cF
import pointCloudPlotFunctions as cloudPlot
import pykitti

def getKittiOdom(map):
    # Change this to the directory where you store KITTI data
    basedir = "D:/UAH/Grad School/Research/KITTIData/dataset"

    # Specify the dataset to load
    sequence = '00'

    # Load the data. Optionally, specify the frame range to load.
    dataset = pykitti.odometry(basedir, sequence)
    dataset = pykitti.odometry(basedir, sequence, frames=range(0, map, 1))

    # dataset.calib:      Calibration data are accessible as a named tuple
    # dataset.timestamps: Timestamps are parsed into a list of timedelta objects
    # dataset.poses:      List of ground truth poses T_w_cam0
    # dataset.camN:       Generator to load individual images from camera N
    # dataset.gray:       Generator to load monochrome stereo pairs (cam0, cam1)
    # dataset.rgb:        Generator to load RGB stereo pairs (cam2, cam3)
    # dataset.velo:       Generator to load velodyne scans as [x,y,z,reflectance]

    # Grab some data
    pose0 = dataset.poses[0]
    # print(dataset.calib)
    pose0_xyz =np.eye(4)
    # pose0_xyz[0,:] = pose0[2,:] #- pose0[2,3]
    # pose0_xyz[1,:] = -pose0[0,:] #- pose0[0,3]
    # pose0_xyz[2,:] = pose0[1,:]  #- #z is forward
    cameraToLidar = np.eye(4)
    th = np.deg2rad(-90) #about z
    beta = np.deg2rad(0) #about y
    gamma = np.deg2rad(-90) #about x
    Rz = np.array([[np.cos(th), -np.sin(th), 0.0],[np.sin(th), np.cos(th), 0.0],[0.0, 0.0, 1.0]])
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)],[0, 1, 0],[-np.sin(beta), 0, np.cos(beta)]])
    Rx = np.array([[1, 0, 0],[0, np.cos(gamma), -np.sin(gamma)],[0, np.sin(gamma), np.cos(gamma)]]) #https://en.wikipedia.org/wiki/Rotation_matrix
    cameraToLidar[0:3,0:3] = np.matmul(np.matmul(Rz,Ry),Rx)

    # cam = np.eye(4)
    # print(dataset.calib.K_cam0.shape)
    # print(dataset.calib.P_rect_00.shape)
    # cam = np.matmul(dataset.calib.K_cam0,dataset.calib.P_rect_00)
    # print(cam.shape)
    # print(cam)
    calibrationTransform = dataset.calib.T_cam0_velo

    poses = np.zeros((len(dataset.poses),3))
    for i in range(len(dataset.poses)):
        pose = dataset.poses[i]
        poses[i,:] = np.matmul(np.linalg.inv(calibrationTransform),pose[:,3])[0:3]

    return poses, pose0

def plotPickleData(filename, resolution, LidarFile):
    data = pickle.load( open(filename, "rb" ) ) #open saved pickle data
    poseGraph = data['transforms']

    groundTruth,pose0 = getKittiOdom(len(poseGraph))
    

    print("%s Maps" %(len(poseGraph)))

    Xall = []
    startMap = 0
    if startMap == 0:
        endMap = len(poseGraph)
    else:
        endMap = startMap + len(poseGraph)
    scanToGlobal = pose0
    for i in range(startMap,endMap): #load LIDAR data
        dataFile = LidarFile + "%06d"%i + ".bin"
       
        scanData = readData.load_velo_scan(dataFile) #(X,Y,Z,I) where I is intensity
        X = np.ascontiguousarray(scanData[:,0:3])
        SCAN_BOUNDS = np.array([50,20,.5,-2])
        X = nF.scanBoundReduction(X,SCAN_BOUNDS)
        X = np.vstack((X.T,np.ones(X.shape[0])))
        

        localTransform = poseGraph[i-startMap]
        if i >startMap:
            localTransform[0,3] += -resolution[0] #subtract the resolution to get the correct point location
            localTransform[1,3] += -resolution[1]
            localTransform[2,3] += -resolution[2]


        scanToGlobal = np.matmul(scanToGlobal,localTransform)
        transform = scanToGlobal

        t=transform[0:3,3]
        # print("Transformation: X:%s, Y:%s, Z:%s" %(t[0],t[1],t[2]))
        X_Global = np.matmul(scanToGlobal,X)
        # print(X_Global.shape)
        
        # idx = np.where(X_Global[2,:]< -4)
        # X_Global = np.delete(X_Global,idx,axis=1)
        # idx = np.where(X_Global[2,:] > 2)
        # X_Global = np.delete(X_Global,idx,axis=1)
        Xall.append(X_Global[0:3,:].T)

        # cloudPlot.pointCloudSeriesPlot([X[0:3,:].T])
    Xall.append(groundTruth)
    # cloudPlot.pointCloudSeriesAnimation(Xall)
    cloudPlot.pointCloudSeriesPlot(Xall)
    


def main():
    FILENAME = "src/threeDScanMatch/posegraphData_all.txt" #Path of the map files
    LIDAR_DATA_FILE = "D:/UAH/Grad School/Research/KITTIData/dataset/sequences/00/velodyne/" #directory with LIDAR Data
    resolution = np.array([.2,.2,1],dtype=np.float64)
    plotPickleData(FILENAME, resolution, LIDAR_DATA_FILE)


if __name__ == '__main__':
    main()
