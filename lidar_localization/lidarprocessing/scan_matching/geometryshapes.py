
import numpy as np
import numpy.linalg as nplnalg
import scipy.linalg as sclnalg

def getIsoTriangle(xc,a,w,th):
    """
    2D iso

    Parameters
    ----------
    xc : TYPE
        Centroid.
    a : TYPE
        large side length.
    w : TYPE
        width of smaller length.    
    th : TYPE
        pointing directions in angle radians measured from standard x-axis.

    Returns
    -------
    triangle vertices

    """
    h = np.sqrt(a**2-w**2/4)
    
    triag = np.array([[-w/2,0],[w/2,0],[0,h],[-w/2,0]])
    centroid = np.mean(triag[0:3,:],axis=0)
    triag = triag - centroid
    th = th - np.pi/2
    R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    for i in range(triag.shape[0]):
        triag[i,:] = np.matmul(R,triag[i,:])+xc
    
    return triag


def getEllipsePoints2D(xyc,a,b,N=100):
    th = np.linspace(0,2*np.pi,N)
    X = np.zeros((N,2))
    for i in range(len(th)):    
        X[i,:]=xyc+[a*np.cos(th[i]),b*np.sin(th[i])];
    
    return X



def getCirclePoints2D(xyc,r,N=100):
    X = getEllipsePoints2D(xyc,r,r,N=N)
    return X

def getSectorPoints2D(xyc,th,r,alpha,N=100):
    """
    

    Parameters
    ----------
    xyc : Center
        DESCRIPTION.
    th : direction
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.
    alpha : sector half angle
        DESCRIPTION.
    N : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    X : TYPE
        DESCRIPTION.

    """
    aa = np.linspace(-alpha,alpha,N)
    X = np.zeros((N,2))
    for i in range(len(aa)):    
        X[i,:]=[r*np.cos(aa[i]),r*np.sin(aa[i])];
    X=np.vstack([X,[0,0],X[0,:]])
    R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    for i in range(X.shape[0]):
        X[i,:] = xyc+np.matmul(R,X[i,:])
    
    return X


def getCovEllipsePoints2D(m,P,nsig=1,N=100):
    mz = np.zeros(m.shape)
    A = sclnalg.sqrtm(P) 
    X = getCirclePoints2D(mz,nsig,N=N)
    
    for i in range(X.shape[0]):    
        X[i,:]=m+np.matmul(A,X[i,:])
    
    return X