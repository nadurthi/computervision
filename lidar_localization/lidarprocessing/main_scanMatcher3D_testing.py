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
import readPosegraph as plotPosegraph

import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)  # Or any other desired level that has a value above that of "INFO".


### GLOBAL VARIABLES -------------------
FILENAME = "src/threeDScanMatch/posegraphData.txt" #file to save the posegraph
LIDAR_DATA_FILE = "D:/UAH/Grad School/Research/KITTIData/dataset/sequences/00/velodyne/" #directory with LIDAR Data
MAP_NUMBER = 4530 #how many scans will be iterated
GPU_CORES = (22,) #streaming multiprocessor cournt, should be ~72 for the 2080 ti

#Scan matching paramters
LMAX = np.array([2,2,1]) #m, max size of keyframe (+-)
THMAX = np.deg2rad(10) #deg, maximum theta of brute force
THSTEP = .5 #deg step of theta
DX_MATCH= np.array([.2,.2,1],dtype=np.float64) #m, minimum  matching size

#Scan Downsample paramters
SCAN_BOUNDS = np.array([50,20,1,-1.25]) #Crop the scan to these X,Y,Ztop,Zbottom bounds [m], between -1.25  and -1.5 seems to remove the ground
SCAN_VOXEL_SIZE = 0.05 #bin scan to this voxel size in [m]
### ------------------------------------


def initilizeBoxes(thL,H,dx,X2,lvl,SolBoxes_init,h,GPU_CORES):
    """Initilizes all of the box errors for the lowest resolution"""
    # count = 0
    Xth= Dict.empty(
        key_type=types.float64,
        value_type=float_2Darray,
    )

    # nthreads = 8
    # t =time.time()
    # func_nb_mt = nF.make_multithread(nF.loopTest, nthreads) #CPU parrellize function
    # print("Function Build Time %s" %(time.time()-t))

    # t = time.time()
    # Xth_cuda,Yth,Zth,  = nF.rotationCuda(thL,X2) #Rotation with CUDA
    # print("Cuda Rotate: %s" %(time.time()-t))

    for th in thL:
        # t = time.time()
        # Xth_th = np.array([Xth_cuda[:,i],Yth[:,i],Zth[:,i]]).T
        # i +=1
        # Xth[th]=Xth_th
        # print("Theta Rot: %s" %(time.time()-t))

        # t = time.time()
        XX = nF.thetaRot(Xth,th,X2)
        Xth[th]=XX
        # print("Theta Numba: %s" %(time.time()-t))

        # h = nF.boxLoop(SolBoxes_init,H,dx,Xth,th,lvl,h) #Numba single thread ~133 seconds

        # t = time.time()
        # constants = [H,dx,Xth,th,lvl]
        # res = func_nb_mt(constants, SolBoxes_init) #CPU Parrellization ~36 seconds
        # print("%s Parrellel Loop time %s:" %(th,time.time()-t))
        # h = np.vstack((h,res))

        # t = time.time()
        boxesLoad = cuda.to_device(SolBoxes_init) #preload for teim save
        costNew = cF.boxesCuda(H,dx,Xth[th],lvl,boxesLoad,th,GPU_CORES,len(SolBoxes_init)) #GPU Parrellization with CUDA ~8 seconds
        h = np.vstack((h,costNew))
        # print("%s CUDA loop time %s:" %(th,time.time()-t))

    hList = list(zip(h[:,0], h[:,1],h[:,2],h[:,3],h[:,4],h[:,5],h[:,6],h[:,7],h[:,8])) #use this if h is a numpy array, creates a list of tuples from numpy array
    
    return Xth, hList

def boxSearch(h,mxLVL,dxs,HLevels,Xth,thmax,GPU_CORES):
    """Attempts to find the highest cost box which is the solution"""
    heapq.heapify(h)
    mainSolbox=()
    count = 0

    maxZ = 0
    maxX = 0
    maxY = 0

    for i in range(len(HLevels)): #determine max size of H
        H=HLevels[i]
        if H.shape[0] > maxX:
            maxX = H.shape[0]
        if H.shape[1] > maxY:
            maxY = H.shape[1]
        if H.shape[2] > maxZ:
            maxZ = H.shape[2]

    HLevelsArray = np.zeros((len(HLevels),maxX,maxY,maxZ)) #creates a 4D matrix to store all the H levels
    for i in range(len(HLevels)):
        H = HLevels[i]
        HLevelsArray[i,0:H.shape[0],0:H.shape[1],0:H.shape[2]] = H #populate H level array

    # t= time.time()
    hLoad = cuda.to_device(HLevelsArray) #pre-load the HLevelsArray to CUDA memory. Takes 0.1 seconds and this way is not done repeatedly
    # print("Load to Cuda %s" %(time.time()-t))
    solutionFound = False
    while(not solutionFound):
        count += 1

        popCount = 50 #250, 50 seems to run a bit faster
        t = time.time()
        hList = []
        maxLevelCount = 0

        maxLevelList = []
        for i in range(popCount):
            (cost,xs,ys,zs,d0,d1,d2,lvl,th)=heapq.heappop(h)

            if lvl==mxLVL and i == 0: #ensure it is the first entry if it is the solution
                mainSolbox = (cost,xs,ys,zs,d0,d1,d2,lvl,th)
                solutionFound = True
                # print("Soluion!")
                break
            elif lvl == mxLVL and i != 0:
                maxLevelList.append((cost,xs,ys,zs,d0,d1,d2,lvl,th)) #track the max level solutions, but do not evaluate them
            else:
                hList.append((cost,xs,ys,zs,d0,d1,d2,lvl,th)) #cue for evaluation

        if not solutionFound:
            cudaList,XthArray,mainSolbox = nF.boxSearchNumbaSetup(Xth,len(hList),hList,dxs,HLevels, mxLVL,thmax,maxLevelCount) #issue with h
            # t2= time.time()
            # cF.boxSearhCudaSetupLaunch(Xth,popCount,hList,dxs,hLoad,mxLVL,thmax,maxLevelCount,GPU_CORES)
            # print(time.time()-t2)
            
            readyTime = time.time() -t
            # print("Ready:%s, max level: %s / %s" %(time.time()-t,maxLevelTest, mxLVL))

            t = time.time()
            costNew = cF.boxeSearchCudaLaunch(cudaList,XthArray,hLoad,GPU_CORES)

            hnew = list(zip(costNew[:,0], costNew[:,1],costNew[:,2],costNew[:,3],costNew[:,4],costNew[:,5],costNew[:,6],costNew[:,7],costNew[:,8])) #use this if h is a numpy array, creates a list of tuples from numpy array
        
            for i in range(popCount- len(maxLevelList)):
                heapq.heappush(h,hnew[i])
            for i in range(len(maxLevelList)): #append any max level solutions to keep them in the heap
                heapq.heappush(h,maxLevelList[i])

        # print("Cuda Solve: %s (prep time: %s), max level: %s / %s count: %s" %(time.time()-t,readyTime, maxLevelTest, mxLVL,popCount*count))
        # print("Cuda Solve: %s (prep time: %s), count: %s" %(time.time()-t,readyTime,popCount*count))

    return mainSolbox, h, count

def save(filename,data):
        t = time.time()
        saveDict = {'transforms':[]}
        saveDict['transforms'] = data
        with open(filename,'wb') as fh:
            pickle.dump(saveDict,fh)  
        
        #print("Saved in %s" %(time.time()-t))   

def binMatcherAdaptive3D(X1,X2,H12,Lmax,thmax,dxMatch,Hfinal,GPU_CORES, thStep, showTimes = False):
    """Runs a many-to-many adaptive matching for 3D data"""
    # print("Running 3D adaptive bin match:")
    # print("Setup:")
    t = time.time()
    thL,H,dx,X2,lvl,SolBoxes_init,h,mxLVL,dxs, HLevels,mn_orig = nF.setup(X1,X2,H12,Lmax,thmax,dxMatch,GPU_CORES,thStep)
    if showTimes:
        print("Setup Time: %s" %(time.time()-t))

    # print("Initilize Boxes:")
    tnp = time.time()
    Xth,h = initilizeBoxes(thL,H,dx,X2,lvl,SolBoxes_init,h,GPU_CORES)
    tnp = time.time() - tnp
    if showTimes:
        print("%s Boxes Initilized in: %s" %(len(h),tnp)) 

   
    # print("Box Search:")
    t = time.time()
    mainSolbox,h, count= boxSearch(h,mxLVL,dxs, HLevels, Xth,thmax,GPU_CORES)
    # mainSolbox,h, count= nF.boxSearchNumba(h,mxLVL,dxs, HLevels, Xth)
    if showTimes:
        print("Solution Time: %s" %(time.time()-t))
    # print(count)

    mn_orig =  np.ascontiguousarray(np.array(mn_orig,dtype=np.float32))
    Htotal21_updt, cost,th = nF.solution(mainSolbox,H12,mn_orig,Hfinal)

    return Htotal21_updt,cost,HLevels,np.rad2deg(th)


def main():
    global FILENAME
    global LIDAR_DATA_FILE
    global MAP_NUMBER
    global GPU_CORES
    global SCAN_BOUNDS
    global SCAN_VOXEL_SIZE
    global LMAX
    global THMAX
    global THSTEP
    global DX_MATCH

 

    poseGraph = []
    T0 = identity(4)
    poseGraph.append(T0)
    transformationSave = []
    transformationSave.append(T0)
    totalTime = time.time() 
    scanToGlobal = np.eye(4)

    for i in range(0,MAP_NUMBER): #max of 5
        print("---------")
        tloadPoints = time.time()
        dataFile = LIDAR_DATA_FILE + "%06d"%i + ".bin"
        scanData = readData.load_velo_scan(dataFile) #(X,Y,Z,I) where I is intensity
        X1FullSize = np.ascontiguousarray(scanData[:,0:3])

        t = time.time()
        X1FullSize = nF.scanBoundReduction(X1FullSize,SCAN_BOUNDS)
        X1 = cloudPlot.downSamplePointCloud(X1FullSize,voxelSize=SCAN_VOXEL_SIZE, doPlot=False)
        # print(X1.shape)
        # print("Scan %s downsampled from %s to %s points in %s" %(i,X1FullSize.shape[0],X1.shape[0], time.time()-t))
 
        dataFile = LIDAR_DATA_FILE + "%06d"%(i+1) + ".bin"
        scanData = readData.load_velo_scan(dataFile) #(X,Y,Z,I) where I is intensity
        X2FullSize= np.ascontiguousarray(scanData[:,0:3])

        t = time.time()
        X2FullSize = nF.scanBoundReduction(X2FullSize,SCAN_BOUNDS)
        X2 = cloudPlot.downSamplePointCloud(X2FullSize,voxelSize=SCAN_VOXEL_SIZE, doPlot=False)
        # print("Scan %s downsampled from %s to %s points in %s" %(i,X2FullSize.shape[0],X2.shape[0], time.time()-t))
   

        # print("Base Scan Points %s, Matched Scan Points %s" %(X1.shape[0],X2.shape[0]))
        print("Running 3D adaptive bin match for scan %s of %s" %(i,MAP_NUMBER))
        downSampleTime = time.time()-tloadPoints
        # print("Points Loaded and Downsampled in %s" %(time.time()-tloadPoints))
        
        # cloudPlot.pointCloudPlot(X1,X2)

        H21 = np.identity(4,dtype=np.float32) #initial transformation guess
        H12 = np.linalg.inv(H21)
        Hfinal = np.identity(4,dtype=np.float32) #initial transformation guess
        t = time.time()
        Hbin21,cost,HLevels2,th = binMatcherAdaptive3D(X1,X2,H12,LMAX,THMAX,DX_MATCH,Hfinal,GPU_CORES,THSTEP)
        print("Adaptive Search Done: %s seconds, (downsample %s)" %(time.time()-t, downSampleTime))

        X2 = np.vstack((X2.T,np.ones(X2.shape[0])))
        X2 = np.matmul(np.linalg.inv(Hbin21),X2)

        # cloudPlot.pointCloudPlot(X1,X2[0:3,:].T)

        Hbin12 = np.linalg.inv(Hbin21) 
        # poseGraph.append(np.matmul(poseGraph[i-1],Hbin12))
        transformationSave.append(Hbin12) #save transformations

        # # livePlot
        # scanToGlobal = cloudPlot.pointCloudUpdateAnimation(X2,DX_MATCH,Hbin12, vis,scanToGlobal)

        
        save(FILENAME,transformationSave)

        R=Hbin12[0:3,0:3]
        t=Hbin12[0:3,3]
        # X22 = R.dot(X2.T).T+t

        # print("Solution Transformation: X:%s, Y:%s, Z:%s, Th: %s" %(t[0],t[1],t[2],th))
        # pointCloudPlot(X1,X22)

    print("Total Computation time for %s scans: %s" %(MAP_NUMBER,time.time()-totalTime))

print("Run readPosegraph.py for plot")
    # plotPosegraph.plotPickleData(FILENAME,dxMatch,LIDAR_DATA_FILE) #read file and plot data


    

if __name__ == '__main__':
    main()
