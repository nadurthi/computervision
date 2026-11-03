import numba as nb
import numpy as np
from numba import njit, prange,jit
from numba import vectorize, float64,guvectorize,int64,double,int32,int64,float32,uintc,boolean, cuda
from numba.typed import List, Dict
from numba.core import types
float_2Darray = types.float64[:,:]
import heapq
import math

import threading
from timeit import repeat
import time

import readData #file for reading/plotting data
import numbaFunctions as nF #file for all of the numba functions

def UpSampleLaunch(Hup,GPU_CORES):
    H=np.zeros((int(np.ceil(Hup.shape[0]/2)),int(np.ceil(Hup.shape[1]/2)),int(np.ceil(Hup.shape[2]/2))),dtype=np.int32) #take previous H and divide size in half
    threads_per_block = (7,7,7) #manually adjusted for time
    blocks = GPU_CORES

    stream = cuda.stream()
    with stream.auto_synchronize():
        Hload = cuda.to_device(H,stream=stream) #load vairables to device
        HupLoad = cuda.to_device(Hup,stream=stream)
        UpsampleMax3DCUDA[blocks, threads_per_block](Hload,HupLoad, H.shape[0],H.shape[1],H.shape[2]) #run operation

    Hload.copy_to_host(H)
    return H


@cuda.jit(device=True)
def checkForOnes(x,y,z,Hup,i,j,k):
    """Checks for a 1 in a shift position"""
    lbx=int(min(2*i+x,Hup.shape[0]-1)) #minimum x coord of the box
    ubx=int(min(2*i+x+1,Hup.shape[0]-1))#max x coord of box, note H is already in index form

    lby=int(min(2*j+y,Hup.shape[1]-1))
    uby=int(min(2*j+y+1,Hup.shape[1]-1))

    lbz=int(min(2*k+z,Hup.shape[2]-1))
    ubz=int(min(2*k+z+1,Hup.shape[2]-1))

    point1 = Hup[lbx,lby,lbz] > 0
    point2 = Hup[ubx,lby,lbz] > 0
    point3 = Hup[lbx,uby,lbz] > 0
    point4 = Hup[ubx,uby,lbz] > 0

    point5 = Hup[lbx,lby,ubz] > 0
    point6 = Hup[ubx,lby,ubz] > 0
    point7 = Hup[lbx,uby,ubz] > 0
    point8 = Hup[ubx,uby,ubz] > 0


    if point1 or point2 or point3 or point4 or point5 or point6 or point7 or point8:
        return True
    else:
        return False


@cuda.jit
def UpsampleMax3DCUDA(H, Hup,H0,H1,H2):
    """Creates Sequentially Lower Resolution Search Spaces"""
    s1, s2, s3 = cuda.grid(3)   # get the thread coordinates in 2D
    d1, d2, d3 = cuda.gridsize(3) 
    # print(s1,s2,s3)

    for i in range(s1, H0, d1): #X
        for j in range(s2, H1, d2): #Y
            for k in range(s3, H2, d3): #Z
                # print(i,j,k)
    
                if checkForOnes(0,0,0,Hup,i,j,k): #base cube
                    H[i,j,k] = 1
                elif checkForOnes(2,0,0,Hup,i,j,k): #shift x, z=0
                    H[i,j,k] = 1
                elif checkForOnes(0,2,0,Hup,i,j,k): #shift y, z=0
                    H[i,j,k] = 1
                elif checkForOnes(2,2,0,Hup,i,j,k): #shift x&y, z = 0
                    H[i,j,k] = 1
                elif checkForOnes(0,0,2,Hup,i,j,k): #shift z, x&y = 0
                    H[i,j,k] = 1
                elif checkForOnes(2,0,2,Hup,i,j,k): #shift x,z y=0
                    H[i,j,k] = 1
                elif checkForOnes(0,2,2,Hup,i,j,k): #shift y,z x=0
                    H[i,j,k] = 1
                elif checkForOnes(2,2,2,Hup,i,j,k): #shift x,y,z
                    H[i,j,k] = 1


#Box Cost Initilize
def boxesCuda(H,dx,X2,lvl,boxesLoad,th,GPU_CORES,cudaCostLength):
    # boxes = SolBoxes_init
    cudaCost = np.zeros([cudaCostLength,9]) #result wanted

    threads_per_block = (23,23) #manually adjusted for time
    blocks = GPU_CORES
    
    stream = cuda.stream()
    with stream.auto_synchronize():
        
        X2Load = cuda.to_device(np.transpose(X2),stream=stream) #load vairables to device
        hCostCuda[blocks, threads_per_block](X2Load,boxesLoad,cudaCost,H,dx,lvl,th) #run operation
   
    return cudaCost

@cuda.jit
def hCostCuda(X2,boxes,cost,H,dx,lvl,th):
  """Calculates cost"""
  s1, s3 = cuda.grid(2)   # get the thread coordinates in 2D
  d1, d3 = cuda.gridsize(2) 

  for i1 in range(s1, X2.shape[1], d1): #per point     
    for i3 in range(s3, len(boxes),d3): #per box
        x1 = X2[0,i1]
        y1 = X2[1,i1]
        z1 = X2[2,i1]

        x = math.floor((x1+ boxes[i3][0])/dx[0]) #Rotation of all points 
        y = math.floor((y1 + boxes[i3][1])/dx[1])
        z = math.floor((z1 + boxes[i3][2])/dx[2])

        c = 0
        if x >= 0 and x < H.shape[0]:
            if y >= 0 and  y < H.shape[1]:
                if z >= 0 and  z < H.shape[2]:
                    c += -H[int(x),int(y),int(z)]
                    cuda.atomic.add(cost,(i3,0),c)
                    
        # result[i,:] = np.array([-cost2-np.random.rand()/1000,xs,ys,zs,d0,d1,d2,lvl,th])
        rowIdx = i3
        cost[rowIdx,1] = boxes[i3][0] 
        cost[rowIdx,2] = boxes[i3][1] 
        cost[rowIdx,3] = boxes[i3][2] 
        cost[rowIdx,4] = boxes[i3][3]
        cost[rowIdx,5] = boxes[i3][4]
        cost[rowIdx,6] = boxes[i3][5]
        cost[rowIdx,7] = lvl
        cost[rowIdx,8] = th


#Box Search
def boxSearhCudaSetupLaunch(Xth,popCount,hList,dxs,hLoad,mxLVL,thmax,maxLevelCount,GPU_CORES):
    # cudaCost = np.zeros([cudaCostLength,9]) #result wanted
    cudaList2 = [(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0) for j in range(popCount*2**3)]
    XthArray = np.zeros((len(Xth[-thmax]),popCount*3))

    threads_per_block = (32*13) #manually adjusted for time
    blocks = (22)


    dxs = [tuple(x) for x in dxs]
    # print(dxs)
    
    stream = cuda.stream()
    with stream.auto_synchronize():
        
        XthLoad = cuda.to_device(XthArray,stream=stream) #load vairables to device
        cudaListLoad = cuda.to_device(cudaList2,stream=stream) #load vairables to device
        hListLoad = cuda.to_device(hList,stream=stream) #load vairables to device
        dxsLoad = cuda.to_device(dxs,stream=stream) #load vairables to device
        # HLevelLoad = cuda.to_device(HLevels,stream=stream) #load vairables to device
        boxSearchCudaSetup[blocks, threads_per_block](XthLoad,cudaListLoad,0,popCount,hListLoad,dxsLoad,hLoad) #run operation

    # XthLoad.copy_to_host(XthArray)
    # cudaListLoad.copy_to_host(cudaList2)

    # return cudaList2, XthArray

@cuda.jit
def boxSearchCudaSetup(XthArray,cudaList2,Xth,popCount,hList,dxs,HLevels):
    s1 = cuda.grid(1)   # get the thread coordinates in 2D
    dgrid = cuda.gridsize(1) 
    
    for i in range(s1,popCount,dgrid):
        # (cost,xs,ys,zs,d0,d1,d2,lvl,th) = hList[i]
        xs = hList[i][0]
        ys = hList[i][1]
        zs = hList[i][2]
        d0 = hList[i][3]
        d1 = hList[i][4]
        d2 = hList[i][5]
        lvl = hList[i][6]
        th = hList[i][7]

        nlvl = int(lvl)+1
        dx=dxs[nlvl]
        H=HLevels[nlvl,:,:,:]
        # Tj= np.array((d0,d1,d2))
        # Oj = np.array((xs,ys,zs))

        # Xg= np.arange(Oj[0],Oj[0]+Tj[0],dx[0])[:2]
        # Yg= np.arange(Oj[1],Oj[1]+Tj[1],dx[1])[:2]
        # Zg = np.arange(Oj[2],Oj[2]+Tj[2],dx[2])[:2]
        
        d0,d1,d2=dx[0],dx[1],dx[2]
        
        # for xs in Xg:
        #     for ys in Yg:
        #         for zs in Zg:
        #             # cudaList.append((xs, ys, zs,H.shape[0],H.shape[1],H.shape[2],dx[0],dx[1],dx[2],d0,d1,d2,nlvl,th))
        #             cudaList2[count] = (xs, ys, zs,float(H.shape[0]),float(H.shape[1]),float(H.shape[2]),dx[0],dx[1],dx[2],d0,d1,d2,float(nlvl),th)
        #             count +=1
        
        cudaList2[8*i+0] = (xs, ys, zs,float(H.shape[0]),float(H.shape[1]),float(H.shape[2]),dx[0],dx[1],dx[2],d0,d1,d2,float(nlvl),th)
        cudaList2[8*i+1] = (xs+d0, ys+d1, zs+d2,float(H.shape[0]),float(H.shape[1]),float(H.shape[2]),dx[0],dx[1],dx[2],d0,d1,d2,float(nlvl),th)
        
        cudaList2[8*i+2] = (xs+d0, ys, zs,float(H.shape[0]),float(H.shape[1]),float(H.shape[2]),dx[0],dx[1],dx[2],d0,d1,d2,float(nlvl),th)
        cudaList2[8*i+3] = (xs, ys+d1, zs,float(H.shape[0]),float(H.shape[1]),float(H.shape[2]),dx[0],dx[1],dx[2],d0,d1,d2,float(nlvl),th)
        cudaList2[8*i+4] = (xs+d0, ys+d1, zs,float(H.shape[0]),float(H.shape[1]),float(H.shape[2]),dx[0],dx[1],dx[2],d0,d1,d2,float(nlvl),th)
        
        cudaList2[8*i+5] = (xs+d0, ys, zs+d2,float(H.shape[0]),float(H.shape[1]),float(H.shape[2]),dx[0],dx[1],dx[2],d0,d1,d2,float(nlvl),th)
        cudaList2[8*i+6] = (xs, ys+d1, zs+d2,float(H.shape[0]),float(H.shape[1]),float(H.shape[2]),dx[0],dx[1],dx[2],d0,d1,d2,float(nlvl),th)
        cudaList2[8*i+7] = (xs+d0, ys+d1, zs+d2,float(H.shape[0]),float(H.shape[1]),float(H.shape[2]),dx[0],dx[1],dx[2],d0,d1,d2,float(nlvl),th)
        
        # topArry = arrayPosition + 3
        # XthArray[:,3*i:3*i +3] = Xth[th]
        # arrayPosition +=3

    # return cudaList2,XthArray





def boxeSearchCudaLaunch(cudaList,XthArray,hLoad,GPU_CORES):
    cudaCost = np.zeros([len(cudaList),9]) #result wanted

    threads_per_block = (23,23) #manually adjusted for time
    blocks = GPU_CORES
    
    stream = cuda.stream()
    with stream.auto_synchronize():
        # t = time.time()
        X2Load = cuda.to_device(XthArray,stream=stream) #load vairables to device
        listLoad = cuda.to_device(cudaList,stream=stream)
        
        
        # print(Hlevels.shape)
        # print("Load time %s" %(time.time()-t))

        boxSearchCuda[blocks, threads_per_block](listLoad,X2Load,hLoad,cudaCost) #run operation
   
    return cudaCost

@cuda.jit
def boxSearchCuda(cudaList,Xth,HLevels,cost):
    s1, s2 = cuda.grid(2)   # get the thread coordinates in 2D
    d1, d2 = cuda.gridsize(2) 

    # cudaList.append((xs, ys, zs,H.shape[0],H.shape[1],H.shape[2],dx[0],dx[1],dx[2],d0,d1,d2,lvl,th))
    for i1 in range(s1, Xth.shape[0],d1): #per point     
        for i2 in range(s2, len(cudaList),d2): #per box
            idx = int(math.floor(i2/(2**3)))
            x1 = Xth[i1,(idx*3)+0]
            y1 = Xth[i1,(idx*3)+1]
            z1 = Xth[i1,(idx*3)+2]
            dx0 = cudaList[i2][6]
            dx1 = cudaList[i2][7]
            dx2 = cudaList[i2][8]

            x = math.floor((x1 + cudaList[i2][0])/dx0) #Rotation of all points 
            y = math.floor((y1 + cudaList[i2][1])/dx1)
            z = math.floor((z1 + cudaList[i2][2])/dx2)

            H0 = cudaList[i2][3]
            H1 = cudaList[i2][4]
            H2 = cudaList[i2][5]
            lvl = cudaList[i2][12]
            H = HLevels[int(lvl),0:int(H0),0:int(H1),0:int(H2)] # H = HLevels[i]
            # H = HLevels[int(lvl),:,:,:] # H = HLevels[i]

            c = 0
            if x >= 0 and x < H.shape[0]:
                if y >= 0 and  y < H.shape[1]:
                    if z >= 0 and  z < H.shape[2]:
                        c += -H[int(x),int(y),int(z)]
                        cuda.atomic.add(cost,(i2,0),c)

            rowIdx = i2
            cost[rowIdx,1] = cudaList[i2][0] #xs
            cost[rowIdx,2] = cudaList[i2][1] #yz
            cost[rowIdx,3] = cudaList[i2][2] #zs
            cost[rowIdx,4] = dx0
            cost[rowIdx,5] = dx1
            cost[rowIdx,6] = dx2
            cost[rowIdx,7] = lvl
            cost[rowIdx,8] = cudaList[i2][13] #th

        # Oj = np.array((xs,ys,zs))
        # cost3=nF.getPointCost3D(H,dx,Xth[th],Oj,Tj) 

        # heapq.heappush(h,(-cost3-np.random.rand()/1000,xs,ys,zs,d0,d1,d2,float(nlvl),th))    


#Rotation Matrix 
def rotationCuda(thL,X2):

    XthCuda = np.zeros([X2.shape[0],thL.shape[0]])
    YthCuda = np.zeros([X2.shape[0],thL.shape[0]])
    ZthCuda = np.zeros([X2.shape[0],thL.shape[0]])

    threads_per_block = (5,5) #manually adjusted for time
    blocks = (22,)
    
    stream = cuda.stream()
    with stream.auto_synchronize():
        ree = time.time()
        X2Load = cuda.to_device(np.transpose(X2),stream=stream) #load vairables to device
        thLoad = cuda.to_device(thL,stream=stream)
        XthLoad = cuda.to_device(XthCuda,stream=stream)
        YthLoad = cuda.to_device(YthCuda,stream=stream)
        ZthLoad = cuda.to_device(ZthCuda,stream=stream)

        rotationMatrixForLoop[blocks, threads_per_block](thLoad,X2Load,XthLoad,YthLoad,ZthLoad) #run operation

    XthLoad.copy_to_host(XthCuda) #return Xth vars to host device
    YthLoad.copy_to_host(YthCuda)
    ZthLoad.copy_to_host(ZthCuda)
    
    return XthCuda,YthCuda,ZthCuda

@cuda.jit
def rotationMatrixForLoop(th,X2,xPrime,yPrime,zPrime):
  s1, s2 = cuda.grid(2)   # get the thread coordinates in 2D
  d1, d2 = cuda.gridsize(2) 

  for i1 in range(s1, X2.shape[1], d1): #per point
    x1 = X2[0,i1]
    y1 = X2[1,i1]
    z1 = X2[2,i1]
    for i2 in range(s2, th.shape[0],d2): #per theta
      #X[th]
        xPrime[i1,i2] = math.cos(th[i2])*x1  - math.sin(th[i2])*y1
        yPrime[i1,i2] = math.sin(th[i2])*x1 + math.cos(th[i2])*y1
        zPrime[i1,i2] = z1

        # Xth[i1,i2] = xPrime
        # Yth[i1,i2] = yPrime

