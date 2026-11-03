import numpy as np
from numpy.linalg import multi_dot
import numpy.linalg as nplg
import numpy.linalg as nplinalg

import scipy.linalg as sclg
import matplotlib.pyplot as plt
import scipy.optimize as scopt
import pdb
import numba
from numba import vectorize, float64,guvectorize,int64,double,int32,float32
from numba import njit, prange,jit
plt.close("all")



#%% Kalman Filter

def KalmanDiscreteUpdate(xfk, Pfk, zk, mz, Pz, Pxz):
    K = np.matmul(Pxz, nplinalg.inv(Pz))
    xu = xfk + np.matmul(K, zk - mz)
    Pu = Pfk - multi_dot([K, Pz, K.T])

    return (xu, Pu,K)

def measUpdate(tk, dt,xfk,Pfk,hk,Hk,Rk,zk):
    """
    if zk is None, just do pseudo update
    return (xu, Pu, mz,R, Pxz,Pz, K, likez )
    """

    mz = hk(dt,xfk)
    Pxz = np.matmul(Pfk, Hk.T)
    Pz = multi_dot([Hk, Pfk, Hk.T]) + Rk

    xu, Pu, K = KalmanDiscreteUpdate(xfk, Pfk, zk, mz, Pz, Pxz)
    return xu, Pu

#%% Accelerations in both x and y direction
class CAUMmodel_1:
    def __init__(self,x0,P0):
        self.fn=8 # dimension of state model
        self.hn_inertial = 3
        self.hn_lidar = 3
        self.hn_inertiallidar = 6
        
        self.xk = x0
        self.Pk = P0
        
        self.X=[x0]
        self.PP=[P0]
        
    def f(dt,xk):
        """
        Constant Acceleration uniform motion type 1
    
        """
        # xk= [x,y,v,th,a,w ] 
        T = dt;
        x = xk[0]
        y = xk[1]
        vx = xk[2]
        vy = xk[3]
        ax = xk[4]
        ay = xk[5]
        th = xk[6]
        w = xk[7]
        
        
    
        xk1=np.zeros_like(xk)
        
        xk1[0] = x + vx*dt+0.5*ax*dt**2
        xk1[1] = y + vy*dt+0.5*ay*dt**2
        xk1[2] = vx+ax*dt
        xk1[3] = vy+ay*dt
        xk1[4] = ax
        xk1[5] = ay
        xk1[6] = th+w*dt
        xk1[7] = w
        
        return xk1

    def F(dt,xk):
        """
        Constant Acceleration uniform motion type 1
    
        """
        # xk= [x,y,v,th,a,w ] 
        T = dt;
        x = xk[0]
        y = xk[1]
        vx = xk[2]
        vx = xk[3]
        ax = xk[4]
        ay = xk[5]
        th = xk[6]
        w = xk[7]
        
        
    
        xk1=np.zeros_like(xk)
        
        F=[]
        F.append( [1,0,dt,0,0.5*dt**2,0,0,0] )
        F.append( [0,1,0,dt,0,0.5*dt**2,0,0] )
        F.append( [0,0,1,0,dt,0,0,0] )
        F.append( [0,0,0,1,0,dt,0,0] )
        F.append( [0,0,0,0,1,0,0,0] )
        F.append( [0,0,0,0,0,1,0,0] )
        F.append( [0,0,0,0,0,0,1,dt] )
        F.append( [0,0,0,0,0,0,0,1] )
        
        F=np.array(F)
        
        return F

    def Q(dt,xk):
        Q = np.diag([0.02**2,0.02**2,0.01**2,0.01**2,0.001**2,0.001**2,0.001**2,0.001**2])
        return Q
    
    def prop(self,dt):
        xfk1 = self.f(dt,self.xk)
        Ak =  self.F(dt,self.xk)
        Q = self.Q(dt,self.xk)
        Pfk1 = multi_dot([Ak, self.Pk, Ak.T]) + Q
        
        self.xk = xfk1
        self.Pk = Pfk1
        
        # self.X.append(self.xk)
        # self.PP.append(self.Pk)
        
        return self.xk,self.Pk
    
    def measUpdt(self,t,dt,zk,sensortype='inertial'):
        
        if sensortype=='inertial':
            hk=self.h_inertial
            Hk=self.H_inertial(dt, self.xk)
            Rk=self.R_inertial(dt, self.xk)
            self.xk, self.Pk = measUpdate(t, dt,self.xk,self.Pk,hk,Hk,Rk,zk)
        
        self.X.append(self.xk)
        self.PP.append(self.Pk)
        
        return self.xk,self.Pk
    
        
    def h_inertial(dt,xk):
        """
        Measurement model using IMU only for 
        Constant Acceleration Constant Turn rate model of type 1
        IMU measures accelerations in body frame : (at,an) tangential and normal accelerations
        at is measured along forward direction of robot.
        IMU also measuresangular velocity along the vertical (upward) z direction.
        [at;
         an;
         wz]
        """
        T = dt;
        x = xk[0]
        y = xk[1]
        vx = xk[2]
        vx = xk[3]
        ax = xk[4]
        ay = xk[5]
        th = xk[6]
        w = xk[7]
        
        # R=np.array([[np.cos(th),np.sin(th)],[-np.sin(th),np.cos(th)]])
        R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
        aimu=R.dot([ax,ay])
        h = np.array([aimu[0],aimu[1],w])
    
        return h
    def R_inertial(dt,xk):
        R=np.diag([0.001**2,0.001**2,0.001**2])
        
    def H_inertial(dt,xk):
        """
        Measurement model using IMU only for 
        Constant Acceleration Constant Turn rate model of type 1
        IMU measures accelerations in body frame : (at,an) tangential and normal accelerations
        at is measured along forward direction of robot.
        IMU also measuresangular velocity along the vertical (upward) z direction.
        [at;
         an;
         wz]
        """
        T = dt;
        x = xk[0]
        y = xk[1]
        vx = xk[2]
        vx = xk[3]
        ax = xk[4]
        ay = xk[5]
        th = xk[6]
        w = xk[7]
        
        # R = np.array([[np.cos(th),np.sin(th)],[-np.sin(th),np.cos(th)]])
        # dRdth = np.array([[-np.sin(th),np.cos(th)],[-np.cos(th),-np.sin(th)]])
        
        R = np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
        dRdth = np.array([[-np.sin(th),-np.cos(th)],[np.cos(th),-np.sin(th)]])
        
        aimu=R.dot([ax,ay])
        h = np.array([aimu[0],aimu[1],w])
        hjac = []
        
        thjac=dRdth.dot([ax,ay])
        hjac.append([0,0,0,0,R[0,0],R[0,1],thjac[0],0])
        hjac.append([0,0,0,0,R[1,0],R[1,1],thjac[1],0])
        hjac.append([0,0,0,0,0,0,0,1])
        
        hjac = np.array(hjac)
        
        return hjac

#%%
class CACTmodel_1:
    def __init__(self):
        pass
    
    def f(dt,xk):
        """
        Constant Acceleration Constant Turn rate model of type 1
    
        """
        # xk= [x,y,v,th,a,w ] 
        T = dt;
        x = xk[0]
        y = xk[1]
        v = xk[2]
        th = xk[3]
        a = xk[4]
        w = xk[5]
        
        sn=np.sin(th)
        cs=np.cos(th)
    
        xk1=np.zeros_like(xk)
        
        xk1[0] = x + v*dt*cs+a*cs*dt**2/2-v*w*sn*dt**2/2-a*w*sn*dt**3/3
        xk1[1] = y + v*dt*sn+a*sn*dt**2/2+v*w*cs*dt**2/2+a*w*cs*dt**3/3
        xk1[2] = v+a*dt
        xk1[3] = th+w*dt
        xk1[4] = a
        xk1[5] = w
        

        return xk1
    
    def F(dt,xk):
        """        F = np.array(F)
        Jacobian Constant Acceleration Constant Turn rate model of type 1
    
        """
        # xk= [x,y,v,th,a,w ] 
        T = dt;
        x = xk[0]
        y = xk[1]
        v = xk[2]
        th = xk[3]
        a = xk[4]
        w = xk[5]
        
        sn=np.sin(th)
        cs=np.cos(th)
    
        # xk1=np.zeros_like(xk)
        
        # xk1[0] = x + v*dt*cs+a*cs*dt**2/2-v*w*sn*dt**2/2-a*w*sn*dt**3/3
        # xk1[1] = y + v*dt*sn+a*sn*dt**2/2+v*w*cs*dt**2/2+a*w*cs*dt**3/3
        # xk1[2] = v+a*dt
        # xk1[3] = th+w*dt
        # xk1[4] = a
        # xk1[5] = w
        
        F=[]
        F.append([1,0, 
                  dt*cs-w*sn*dt**2/2,
                  -v*dt*sn-a*sn*dt**2/2-v*w*cs*dt**2/2-a*w*cs*dt**3/3,
                  cs*dt**2/2-w*sn*dt**3/3,
                  -v*sn*dt**2/2-a*sn*dt**3/3])
        F.append([0,1,
                  dt*sn+w*cs*dt**2/2,
                  v*dt*cs+a*cs*dt**2/2-v*w*sn*dt**2/2-a*w*sn*dt**3/3,
                  sn*dt**2/2+w*cs*dt**3/3,
                  v*cs*dt**2/2+a*cs*dt**3/3
            ])
        F.append([0,0,1,0,dt,0])
        F.append([0,0,0,1,0,dt])
        F.append([0,0,0,0,1,0])
        F.append([0,0,0,0,0,1])
        
        F = np.array(F)
        return F

    def Q(dt,xk):
        Q=np.array([[0.1**2,0,0,0,0,0],
                     [0,0.1**2,0,0,0,0],
                     [0,0,0.01**2,0,0,0],
                     [0,0,0,0.001**2,0,0],
                     [0,0,0,0,0.0001**2,0],
                     [0,0,0,0,0,0.0001**2]])

        return Q
    

    def h_inertial(dt,xk):

        """
        Measurement model using IMU only for 
        Constant Acceleration Constant Turn rate model of type 1
        IMU measures accelerations in body frame : (at,an) tangential and normal accelerations
        at is measured along forward direction of robot.
        IMU also measuresangular velocity along the vertical (upward) z direction.
        [at;
         an;
         wz]
        """
        x = xk[0]
        y = xk[1]
        v = xk[2]
        th = xk[3]
        a = xk[4]
        w = xk[5]
        
        h = np.array([a,v*w,w])
    
        return h

    def H_inertial(dt,xk):
        """
        Measurement model using IMU only for 
        Constant Acceleration Constant Turn rate model of type 1
        IMU measures accelerations in body frame : (at,an) tangential and normal accelerations
        at is measured along forward direction of robot.
        IMU also measuresangular velocity along the vertical (upward) z direction.
        [at;
         an;
         wz]
        """
        x = xk[0]
        y = xk[1]
        v = xk[2]
        th = xk[3]
        a = xk[4]
        w = xk[5]
        
        H = []
        H.append([0,0,0,0,1,0 ])
        H.append([0,0,w,0,0,v ])
        H.append([0,0,0,0,0,1 ])
        
        H = np.array(H)
    
        return H

    def h_lidar(dt,xk):
        """
        Measurement model using IMU only for 
        Constant Acceleration Constant Turn rate model of type 1
        IMU measures accelerations in body frame : (at,an) tangential and normal accelerations
        at is measured along forward direction of robot.
        IMU also measuresangular velocity along the vertical (upward) z direction.
        [at;
         an;
         wz]
        """
        x = xk[0]
        y = xk[1]
        v = xk[2]
        th = xk[3]
        a = xk[4]
        w = xk[5]
        
        h = np.array([x,y])
    
        return h

    def H_lidar(dt,xk):
        """
    
        """
        x = xk[0]
        y = xk[1]
        v = xk[2]
        th = xk[3]
        a = xk[4]
        w = xk[5]
        
        H = []
        H.append([1,0,0,0,0,0 ])
        H.append([0,1,0,0,0,0 ])
    
        
        H = np.array(H)
    
        return H

    def h_lidarinertial(dt,xk):
        """
        Measurement model using IMU only for 
        Constant Acceleration Constant Turn rate model of type 1
        IMU measures accelerations in body frame : (at,an) tangential and normal accelerations
        at is measured along forward direction of robot.
        IMU also measuresangular velocity along the vertical (upward) z direction.
        [at;
         an;
         wz]
        Lidar measures [x,y]
        """
        x = xk[0]
        y = xk[1]
        v = xk[2]
        th = xk[3]
        a = xk[4]
        w = xk[5]
        
        h = np.array([x,y,th,a,v*w,w])
    
        return h
    
    def H_lidarinertial(dt,xk):
        """
        Measurement model using IMU only for 
        Constant Acceleration Constant Turn rate model of type 1
        IMU measures accelerations in body frame : (at,an) tangential and normal accelerations
        at is measured along forward direction of robot.
        IMU also measuresangular velocity along the vertical (upward) z direction.
        [at;
         an;
         wz]
        Lidar measures [x,y,th]
        """
        x = xk[0]
        y = xk[1]
        v = xk[2]
        th = xk[3]
        a = xk[4]
        w = xk[5]
        
        H = []
        H.append([1,0,0,0,0,0])
        H.append([0,1,0,0,0,0])
        H.append([0,0,0,1,0,0])
        H.append([0,0,0,0,1,0 ])
        H.append([0,0,w,0,1,v ])
        H.append([0,0,0,0,0,1 ])
        
        H = np.array(H)
    
        return H   

