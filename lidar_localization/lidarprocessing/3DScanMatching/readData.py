import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))

def plotScan3D(scan):
    """Plot the LIDAR data in 3D"""
    fig = plt.figure()
    ax =  Axes3D(fig)
    print("Plotting...")
    ax.scatter(scan[:,0], scan[:,1], scan[:,2], c = 'blue')
    print("Plot complete")
    ax.set_xlabel('X Distance')
    ax.set_ylabel('Y Distance')
    ax.set_zlabel('Z Distance')
    plt.show()

def plotScan2D(scan):
    """Plot the LIDAR data in 2D """
    fig = plt.figure()

    print("Plotting...")

    plt.scatter(scan[:,0], scan[:,1], scan[:,2], c = 'blue')

    print("Plot complete")

    plt.show(block=False)


def main():
    for i in range(1): #max of 6
        dataFile = "/home/t2salve/turtlebot3_ws/src/src/3DLidarData/velodyneData/" + "00000" + str(i) + ".bin"
        scanData = load_velo_scan(dataFile)
        print(scanData.shape)
        # plotScan3D(scanData)
    plt.show() 
        

if __name__ == '__main__':
    main()
