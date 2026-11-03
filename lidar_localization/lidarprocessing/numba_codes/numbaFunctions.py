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

import cudaFunctions as cF

@jit(int32[:,:,:](float64[:,:], float64[:], float64[:], float64[:]),nopython=True, nogil=True,cache=False) 
def numba_histogram3D(X, xedges,yedges,zedges):
    """bins 3D points into the defined edges and creates a 3D histogram"""
    x_min = np.min(xedges) #get the min and max dimension of the x,y,z edges
    x_max = np.max(xedges)
    nx = len(xedges) 
    
    y_min = np.min(yedges)
    y_max = np.max(yedges)
    ny = len(yedges)

    z_min = np.min(zedges)
    z_max = np.max(zedges)
    nz = len(zedges)

    
    H = np.zeros((nx-1,ny-1,nz-1),dtype=np.int32) #create bins for each edge (the last edge does not get its own bin since it is the edge of the box)
    nxyz = np.array([nx-1,ny-1,nz-1]) #size of H

    dxyz=np.array([x_max-x_min,y_max-y_min,z_max-z_min]) #size of entire box
    xyzmin = np.array([x_min,y_min,z_min]) #list of minimums
    xyzmax = np.array([x_max,y_max,z_max]) #list of maximums
    
    dd = nxyz*((X-xyzmin)/dxyz) #create indicies for X in H

    for i in range(X.shape[0]): #for all rows
        if np.all( (X[i]-xyzmin)>=0) and np.all( (xyzmax-X[i])>=0): #if data point is within search space
            H[int(dd[i][0]),int(dd[i][1]),int(dd[i][2])]+=1   #bin the sample and add one to the samples box

        
    return H


@njit(cache=False, nogil=True)
def UpsampleMax3D(Hup,n):
    """Creates Sequentially Lower Resolution Search Spaces"""
    H=np.zeros((int(np.ceil(Hup.shape[0]/2)),int(np.ceil(Hup.shape[1]/2)),int(np.ceil(Hup.shape[2]/2))),dtype=np.int32) #take previous H and divide size in half
    # print(H.shape[0]*H.shape[1]*H.shape[2])
    for i in range(H.shape[0]): #X
        for j in range(H.shape[1]): #Y
            for k in range(H.shape[2]): #Z
                # t =time.time()
                lbx=max([2*i,0]) #minimum x coord of the box
                ubx=min([2*i+n,Hup.shape[0]-1]) + 1 #max x coord of box, note H is already in index form

                lby=max([2*j,0])
                uby=min([2*j+n,Hup.shape[1]-1]) + 1

                lbz=max([2*k,0])
                ubz=min([2*k+n,Hup.shape[2]-1]) + 1

                H[i,j,k] = np.max( Hup[lbx:ubx+1,lby:uby+1,lbz:ubz+1] ) #identifies maximum value of each new block
    return H


@njit(cache=False, nogil=True)
def UpsampleMax3DParallel(H,dummyVar,Hup,n):
    """Creates Sequentially Lower Resolution Search Spaces"""
    # H=np.zeros((int(np.ceil(Hup.shape[0]/2)),int(np.ceil(Hup.shape[1]/2)),int(np.ceil(Hup.shape[2]/2))),dtype=np.int32) #take previous H and divide size in half
    # print(H.shape[0]*H.shape[1]*H.shape[2])
    for i in range(H.shape[0]): #X
        for j in range(H.shape[1]): #Y
            for k in range(H.shape[2]): #Z
                # t =time.time()
                lbx=max([2*i,0]) #minimum x coord of the box
                ubx=min([2*i+n,Hup.shape[0]-1]) + 1 #max x coord of box, note H is already in index form

                lby=max([2*j,0])
                uby=min([2*j+n,Hup.shape[1]-1]) + 1

                lbz=max([2*k,0])
                ubz=min([2*k+n,Hup.shape[2]-1]) + 1

                H[i,j,k] = np.max( Hup[lbx:ubx,lby:uby,lbz:ubz] ) #identifies maximum value of each new block
    # return H



# @njit(cache=False)
def setup(X11,X22,H12,Lmax,thmax,dxMatch,GPU_CORES,thStep):
    # dxMax is the max resolution allowed
    # Lmax =[xmax,ymax]
    # search window is [-Lmax,Lmax] and [-thmax,thmax]

    mn=np.zeros(3)
    mx=np.zeros(3)
    mn_orig=np.zeros(3)
    mn_orig[0] = np.min(X11[:,0]) #Identify the origin of the base scan such that
    mn_orig[1] = np.min(X11[:,1])
    mn_orig[2] = np.min(X11[:,2])
    
    R=np.ascontiguousarray(H12[0:3,0:3])
    t=H12[0:3,3]
    X222 = R.dot(X22.T).T+t #Transform match scan by intitial guess
    
    X2=X222-mn_orig #Shift matched scan and base scan into same reference frame
    X1=X11-mn_orig
    
    mn[0] = np.min(X1[:,0])
    mn[1] = np.min(X1[:,1])
    mn[2] = np.min(X1[:,2])
    mx[0] = np.max(X1[:,0])
    mx[1] = np.max(X1[:,1])
    mx[2] = np.max(X1[:,2])
    # rmax=np.max(np.sqrt(X2[:,0]**2+X2[:,1]**2))
    
    P = mx-mn
    dxMax=P


    # nnx=np.ceil(np.log2(P[0]))
    # nny=np.ceil(np.log2(P[1]))

    
    xedges=np.arange(mn[0]-dxMatch[0],mx[0]+dxMax[0],dxMatch[0]) #define the edges of the highest resolution search boundaries
    yedges=np.arange(mn[1]-dxMatch[1],mx[1]+dxMax[1],dxMatch[1])
    zedges=np.arange(mn[2]-dxMatch[2],mx[2]+dxMax[2],dxMatch[2])

    if len(xedges)%2==0: #if the number of edges is not divisble by 2, add another row
        xedges=np.hstack((xedges,np.array([xedges[-1]+1*dxMatch[0]])))
    if len(yedges)%2==0:
        yedges=np.hstack((yedges,np.array([yedges[-1]+1*dxMatch[1]])))
    if len(zedges)%2==0:
        zedges=np.hstack((zedges,np.array([zedges[-1]+1*dxMatch[2]])))

    H1match= numba_histogram3D(X1, xedges,yedges,zedges) #bin the base scan at the highest resolution
    
    H1match = np.sign(H1match) #updates histogram to be only 1 or 0. 1 if > 0 and 0 if == 0
    
    # first create multilevel histograms
    HLevels=[H1match]
    dxs = [dxMatch]
    # XYedges=[(xedges,yedges)]

    # nthreads = 4
    # func_nb_mt = make_multithread_upsample(UpsampleMax3DParallel, nthreads) #CPU parrellize function

    flg=0
    for i in range(1,100):
        # print(i)
        dx=2*dxs[i-1] #increase the box size by a factor of 2
        if np.any(dx>dxMax): #if box size is over max threshold, stop
            flg=1
        
        Hup = HLevels[i-1]

        # te = time.time()
        # n= histsmudge = 2 # how much overlap when computing max over adjacent hist for levels
        # H=UpsampleMax3D(Hup,n) #create the next series of levels
        H = cF.UpSampleLaunch(Hup,GPU_CORES) #create the next series of levels, using n = 2
        # print(np.sum(H-H2))
        # print(time.time()-te)

        # te = time.time()
        # constants = [Hup,n]
        # H2 = func_nb_mt(constants, Hup)
        # print(time.time()-te)
        # print(np.sum(H2-H))

        # te = time.time()
        # H2 = cF.UpSampleLaunch(Hup,n)
        # print(time.time()-te)
        # print(np.sum(H2-H))

        HLevels.append(H) #the keyframe of each resolution
        dxs.append(dx) #resolution of square
          
        if flg==1:
            break

    HLevels=HLevels[::-1] #lowest to highest resolution
    dxs=dxs[::-1] 
    # print(time.time()-te)
    # SolBoxes_init=[]
    SolBoxes_init = List() #numba happy format
    Lmax=dxs[0]*(np.floor(Lmax/dxs[0])+1) #adjust user Lmax to fit dx
    for xs in np.arange(-Lmax[0],Lmax[0]+1.5*dxs[0][0],dxs[0][0]): #creates edges of solution boxes
        for ys in np.arange(-Lmax[1],Lmax[1]+1.5*dxs[0][1],dxs[0][1]):
            for zs in np.arange(-Lmax[2],Lmax[2]+1.5*dxs[0][2],dxs[0][2]):
                SolBoxes_init.append( (xs,ys,zs,dxs[0][0],dxs[0][1],dxs[0][2]) )
    mxLVL=len(HLevels)-1


    # #Initialize with all thetas fixed at Max resolution
    lvl=0
    dx=dxs[lvl]
    H=HLevels[lvl]

    thfineRes = thStep*np.pi/180
    thL=np.arange(-thmax,thmax+thfineRes,thfineRes)
    
    h=[(100000.0,00.,0.0,0.0,0.0,0.0,0.0,0.0,0.0)]

    return thL,H,dx,X2,lvl,SolBoxes_init,h,mxLVL,dxs, HLevels,mn_orig


@njit(cache=False, nogil=True)
def getPointCost3D(H,dx,X,Oj,Tj):
    """Calculates the number points that fall in a box"""
    # Tj is the 2D index of displacement
    # X are the points
    # dx is 2D
    # H is the probability histogram
    
    Pn=np.floor((X+Oj)/dx)

    idx1=np.logical_and(Pn[:,0]>=0,Pn[:,0]<H.shape[0])
    idx2=np.logical_and(Pn[:,1]>=0,Pn[:,1]<H.shape[1])
    idx3=np.logical_and(Pn[:,2]>=0,Pn[:,2]<H.shape[2])
    idx=np.logical_and(idx1,idx2)
    idx = np.logical_and(idx,idx3)
    c=0
    Pn=Pn[idx]

    for k in range(Pn.shape[0]):
        c+=H[int(Pn[k,0]),int(Pn[k,1]),int(Pn[k,2])]
        
    return c


@njit(cache=False)
def initilizeBoxes(thL,H,dx,X2,lvl,SolBoxes_init,h):
    """Initilizes all of the box errors for the lowest resolution"""
    # count = 0
    Xth= Dict.empty(
        key_type=types.float64,
        value_type=float_2Darray,
    )
    
    for th in thL:
        Rz = np.array([[np.cos(th), -np.sin(th), 0.0],[np.sin(th), np.cos(th), 0.0],[0.0, 0.0, 1.0]])
        # beta = 0
        # gamma = 0
        # Ry = np.array([[np.cos(beta), 0, np.sin(beta)],[0, 1, 0],[-np.sin(beta), 0, np.cos(beta)]])
        # Rx = np.array([[1, 0, 0],[0, np.cos(gamma), -np.sin(gamma)],[0, np.sin(gamma), np.cos(gamma)]]) #https://en.wikipedia.org/wiki/Rotation_matrix
       
        XX=np.transpose(Rz.dot(X2.T)) #could expand to include roll, pitch, yaw; however, currently only assuming rotation about z
        Xth[th]=XX

        for solbox in SolBoxes_init:  
            # print("Iter: %s of %s" %(count, len(SolBoxes_init)*len(thL)))
            # count += 1
            xs,ys, zs, d0,d1, d2 = solbox
            Tj=np.array((d0,d1,d2))
            Oj = np.array((xs,ys,zs))
            cost2=getPointCost3D(H,dx,Xth[th],Oj,Tj)
            h.append((-cost2-np.random.rand()/1000,xs,ys,zs,d0,d1,d2,lvl,th))

    return Xth, h

@njit(cache=False)
def boxSearchNumba(h,mxLVL,dxs,HLevels,Xth):
    """Attempts to find the highest cost box which is the solution"""
    heapq.heapify(h)
    mainSolbox=()
    count = 0
    while(1):

        (cost,xs,ys,zs,d0,d1,d2,lvl,th)=heapq.heappop(h)

        mainSolbox = (cost,xs,ys,zs,d0,d1,d2,lvl,th)
        
        if lvl==mxLVL:
            break
        
        nlvl = int(lvl)+1
        dx=dxs[nlvl]
        H=HLevels[nlvl]
        Tj=np.array((d0,d1,d2))
        Oj = np.array((xs,ys,zs))

        Xg=np.arange(Oj[0],Oj[0]+Tj[0],dx[0])
        Yg=np.arange(Oj[1],Oj[1]+Tj[1],dx[1])
        Zg = np.arange(Oj[2],Oj[2]+Tj[2],dx[2])
        
        d0,d1,d2=dx[0],dx[1],dx[2]
        Tj=np.array((d0,d1,d2))

        for xs in Xg[:2]:
            for ys in Yg[:2]:
                for zs in Zg[:2]:
                    Oj = np.array((xs,ys,zs))
                    cost3=getPointCost3D(H,dx,Xth[th],Oj,Tj) 
                    heapq.heappush(h,(-cost3-np.random.rand()/1000,xs,ys,zs,d0,d1,d2,float(nlvl),th))
                    
        count +=1
    return mainSolbox, h, count



@njit(cache=False)
def boxSearchNumbaSetup(Xth,popCount,hList,dxs,HLevels,mxLVL,thmax,maxLevelCount):
    # cudaList = [] #[0 for j in range(popCount*2**3)]
    cudaList2 = [(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0) for j in range(popCount*2**3)]
    # arrayPosition = 0
    XthArray = np.zeros((len(Xth[-thmax]),popCount*3))
    # maxLevelTest = 0
    count = 0
    mainSolbox = (0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)
    
    for i in range(popCount):
        (cost,xs,ys,zs,d0,d1,d2,lvl,th) = hList[i]
        
        # if lvl > maxLevelTest:
        #     maxLevelTest = lvl

        nlvl = int(lvl)+1
        dx=dxs[nlvl]
        H=HLevels[nlvl]
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
        XthArray[:,3*i:3*i +3] = Xth[th]
        # arrayPosition +=3

    return cudaList2,XthArray, mainSolbox


# @njit(cache=False)
def solution(mainSolbox,H12,mn_orig,H):
    t=mainSolbox[1:4] #(-cost2-np.random.rand()/1000,xs,ys,zs,d0,d1,d2,lvl,th)
    th = mainSolbox[8]
    cost=-mainSolbox[0]

    Rz = np.array([[np.cos(th), -np.sin(th), 0.0],[np.sin(th), np.cos(th), 0.0],[0.0, 0.0, 1.0]])
    # beta = 0
    # gamma = 0
    # Ry = np.array([[np.cos(beta), 0, np.sin(beta)],[0, 1, 0],[-np.sin(beta), 0, np.cos(beta)]])
    # Rx = np.array([[1, 0, 0],[0, np.cos(gamma), -np.sin(gamma)],[0, np.sin(gamma), np.cos(gamma)]]) #https://en.wikipedia.org/wiki/Rotation_matrix
    
    H[0:3,0:3]=Rz
    H[0:3,3]=t
    # print(t)
    Htotal12 = H.dot(H12) #Original Guess multiplied by solution
    # RT=Htotal12[0:2,0:2]
    tT=Htotal12[0:3,3]
    
    Rs=np.ascontiguousarray(H[0:3,0:3]) #solution Rotation
    ts=H[0:3,3] #solution translation

    t = tT-(Rs.dot(mn_orig)+0*ts)+mn_orig #solution transformed back into original origin (since the orgin was original shifted by mn_orig)

    Htotal12_updt=Htotal12
    Htotal12_updt[0:3,3]=t #final transformation from new location to orgin
    Htotal21_updt = np.linalg.inv(Htotal12_updt) #transformation from origin to new location

    return Htotal21_updt, cost,th


@njit(cache = True)
def thetaRot(Xth,th,X2):
    Rz = np.array([[np.cos(th), -np.sin(th), 0.0],[np.sin(th), np.cos(th), 0.0],[0.0, 0.0, 1.0]])
        # beta = 0
        # gamma = 0
        # Ry = np.array([[np.cos(beta), 0, np.sin(beta)],[0, 1, 0],[-np.sin(beta), 0, np.cos(beta)]])
        # Rx = np.array([[1, 0, 0],[0, np.cos(gamma), -np.sin(gamma)],[0, np.sin(gamma), np.cos(gamma)]]) #https://en.wikipedia.org/wiki/Rotation_matrix
       
    XX=np.transpose(Rz.dot(X2.T)) #could expand to include roll, pitch, yaw; however, currently only assuming rotation about z

    return XX

@njit(cache=False)
def scanBoundReduction(X1,scanBounds):
    """Reduces the size of the scan to be withing the desired bounds"""
    idxX = np.logical_and(X1[:,0]<scanBounds[0],X1[:,0]> -scanBounds[0])
    idxY = np.logical_and(X1[:,1]<scanBounds[1],X1[:,1]> -scanBounds[1])
    idxZ = np.logical_and(X1[:,2]<scanBounds[2],X1[:,2]> scanBounds[3])
    idx=np.logical_and(idxX,idxY)
    idx = np.logical_and(idx,idxZ)

    return X1[idx]


### UNUSED #####

@jit(nopython=True,nogil=True, cache=False)
def boxLoop(SolBoxes_init,H,dx,Xth,th,lvl,h):
    for solbox in SolBoxes_init:
        xs,ys, zs, d0,d1, d2 = solbox
        Tj=np.array((d0,d1,d2))
        Oj = np.array((xs,ys,zs))
        cost2=getPointCost3D(H,dx,Xth[th],Oj,Tj)
        
        # result[i] = -cost2-np.random.rand()/1000
        # result[i,:] = np.array([-cost2-np.random.rand()/1000,xs,ys,zs,d0,d1,d2,lvl,th])
        h.append((-cost2-np.random.rand()/1000,xs,ys,zs,d0,d1,d2,lvl,th))
    return h

@jit(nopython=True,nogil=True, cache=False)
def loopTest(result,SolBoxes_init,H,dx,Xth,th,lvl):
    # i = 0
    # print(h.shape)
    # for solbox in SolBoxes_init:
    for i in range(len(result)):
        solbox = SolBoxes_init[i] 
        # print("Iter: %s of %s" %(count, len(SolBoxes_init)*len(thL)))
        # count += 1
        xs,ys, zs, d0,d1, d2 = solbox
        Tj=np.array((d0,d1,d2))
        Oj = np.array((xs,ys,zs))
        cost2=getPointCost3D(H,dx,Xth[th],Oj,Tj)
        
        # result[i] = -cost2-np.random.rand()/1000
        result[i,:] = np.array([-cost2-np.random.rand()/1000,xs,ys,zs,d0,d1,d2,lvl,th])
        # h.append((-cost2-np.random.rand()/1000,xs,ys,zs,d0,d1,d2,lvl,th))
    # return h

 
def make_multithread(inner_func, numthreads):
    """
    Run the given function inside *numthreads* threads, splitting
    its arguments into equal-sized chunks.
    """
    def func_mt(constants,*args):
        length = len(args[0])
        result = np.zeros((length,9), dtype=np.float64)
        args = (result,) + args
        chunklen = (length + numthreads - 1) // numthreads
        # Create argument tuples for each input chunk
        # chunks = [[arg[i * chunklen:(i + 1) * chunklen] for arg in
        #            args] for i in range(numthreads)]

        # chunks =  [[None]*(len(args)+len(constants))]*numthreads #define list
        chunks = [[0 for i in range(len(args)+len(constants))] for j in range(numthreads)]
        # print(chunks)
        for i in range(numthreads):
            for j in range(len(args)):
                chunks[i][j] = args[j][i * chunklen:(i + 1) * chunklen]
            for k in range(len(constants)): #applies constants
                chunks[i][len(args)+k] = constants[k]

        # Spawn one thread per chunk
        threads = [threading.Thread(target=inner_func, args=chunk)
                   for chunk in chunks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return result
    return func_mt

def make_multithread_upsample(inner_func, numthreads):

    """
    Run the given function inside *numthreads* threads, splitting
    its arguments into equal-sized chunks.
    """
    def func_mt(constants,*args):
        length = len(args[0])
        result = np.zeros((int(np.ceil(args[0].shape[0]/2)),int(np.ceil(args[0].shape[1]/2)),int(np.ceil(args[0].shape[2]/2))),dtype=np.int32)
        args = (result,) + args
        chunklen = (length + numthreads - 1) // numthreads
        # Create argument tuples for each input chunk
        # chunks = [[arg[i * chunklen:(i + 1) * chunklen] for arg in
        #            args] for i in range(numthreads)]

        # chunks =  [[None]*(len(args)+len(constants))]*numthreads #define list
        chunks = [[0 for i in range(len(args)+len(constants))] for j in range(numthreads)]
        # print(chunks)
        for i in range(numthreads):
            for j in range(len(args)):
                chunks[i][j] = args[j][i * chunklen:(i + 1) * chunklen]
            for k in range(len(constants)): #applies constants
                chunks[i][len(args)+k] = constants[k]

        # Spawn one thread per chunk
        threads = [threading.Thread(target=inner_func, args=chunk)
                   for chunk in chunks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return result
    return func_mt


@njit(cache=False)
def buildHLevels(HLevels):
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

    return HLevelsArray