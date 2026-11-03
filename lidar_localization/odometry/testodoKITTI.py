#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 16:48:05 2020

@author: nagnanamus
"""

import queue
import threading
import open3d as o3d
import pykitti
import time
import cv2
import datetime as dt
import glob
import os
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import itertools as itr
import matplotlib
matplotlib.use('TkAgg')

# %% docs

# - cam0 (monochrome left)
# - cam1 (monochrome right).
# - cam2 (RGB left)
# - cam3 (RGB right)


# %%
# /media/nagnanamus/d0690b96-7f71-44f2-96da-9f7259180ec7/SLAMData/Kitti/visualodo/data_odometry_calib
# Change this to the directory where you store KITTI data
basedir = '/media/nagnanamus/d0690b96-7f71-44f2-96da-9f7259180ec7/SLAMData/Kitti/visualodo/data_odometry_calib/dataset'


def getfilesINfolder(folder):
    # os.path.join(self.sequence_path, 'image_0','*.{}'.format(self.imtype))
    return sorted(glob.glob(folder))


def load_timestamps(timestamp_file):
    """Load timestamps from file."""

    # Read and parse the timestamps
    timestamps = []
    with open(timestamp_file, 'r') as f:
        for line in f.readlines():
            t = dt.timedelta(seconds=float(line))
            timestamps.append(t)

    return timestamps


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


def process_calib_data(data):
    P_rect_00 = np.reshape(data['P0'], (3, 4))
    P_rect_10 = np.reshape(data['P1'], (3, 4))
    P_rect_20 = np.reshape(data['P2'], (3, 4))
    P_rect_30 = np.reshape(data['P3'], (3, 4))

    data['P_rect_00'] = P_rect_00
    data['P_rect_10'] = P_rect_10
    data['P_rect_20'] = P_rect_20
    data['P_rect_30'] = P_rect_30

    # Compute the camera intrinsics
    data['K_cam0'] = P_rect_00[0:3, 0:3]
    data['K_cam1'] = P_rect_10[0:3, 0:3]
    data['K_cam2'] = P_rect_20[0:3, 0:3]
    data['K_cam3'] = P_rect_30[0:3, 0:3]

    # Compute the rectified extrinsics from cam0 to camN
    T1 = np.eye(4)
    T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
    T2 = np.eye(4)
    T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
    T3 = np.eye(4)
    T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

    # Compute the velodyne to rectified camera coordinate transforms
    # calib.txt: Calibration data for the cameras: P0/P1 are the 3x4 projection
    # matrices after rectification. Here P0 denotes the left and P1 denotes the
    # right camera. Tr transforms a point from velodyne coordinates into the
    # left rectified camera coordinate system. In order to map a point X from the
    # velodyne scanner to a point x in the i'th image plane, you thus have to
    # transform it like:

    #   x = Pi * Tr * X
    if 'Tr' in data:
        data['T_cam0_velo'] = np.reshape(data['Tr'], (3, 4))
        data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
        data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
        data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
        data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
        p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
        p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
        p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

        data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

    return data


def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))


def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32)
    return scan.reshape((-1, 4))


def load_poses(pose_file):
    """Load ground truth poses (T_w_cam0) from file."""
    # pose_file = os.path.join(self.pose_path, self.sequence + '.txt')

    # Read and parse the poses
    poses = []
    try:
        with open(pose_file, 'r') as f:
            lines = f.readlines()

            for line in lines:
                T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                T_w_cam0 = T_w_cam0.reshape(3, 4)
                T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                poses.append(T_w_cam0)

    except FileNotFoundError:
        print('Ground truth poses are not available for sequence ' +
              pose_file)

    return poses
# %%
# Folder 'poses':

# The folder 'poses' contains the ground truth poses (trajectory) for the
# first 11 sequences. This information can be used for training/tuning your
# method. Each file xx.txt contains a N x 12 table, where N is the number of
# frames of this sequence. Row i represents the i'th pose of the left camera
# coordinate system (i.e., z pointing forwards) via a 3x4 transformation
# matrix. The matrices are stored in row aligned order (the first entries
# correspond to the first row), and take a point in the i'th coordinate
# system and project it into the first (=0th) coordinate system. Hence, the
# translational part (3x1 vector of column 4) corresponds to the pose of the
# left camera coordinate system in the i'th frame with respect to the first
# (=0th) frame. Your submission results must be provided using the same data
# format.


class KittiOdoData:
    def __init__(self, mainfolder, seq):
        self.seq = seq
        self.mainFolder = mainfolder
        self.calibFolder = os.path.join(
            mainfolder, 'data_odometry_calib/dataset/sequences')
        self.posesFolder = os.path.join(
            mainfolder, 'data_odometry_poses/dataset/poses')
        self.colorFolder = os.path.join(
            mainfolder, 'data_odometry_color/dataset/sequences')
        self.grayFolder = os.path.join(
            mainfolder, 'data_odometry_gray/dataset/sequences')
        self.veloFolder = os.path.join(
            mainfolder, 'data_odometry_velodyne/dataset/sequences')

    def get_nframes(self):
        timestamps = load_timestamps(
            os.path.join(
                self.calibFolder,
                self.seq,
                'times.txt'))
        return len(timestamps)

    def get_calib_main(self):
        data = read_calib_file(
            os.path.join(
                self.calibFolder,
                self.seq,
                'calib.txt'))
        timestamps = load_timestamps(
            os.path.join(
                self.calibFolder,
                self.seq,
                'times.txt'))

        data = process_calib_data(data)

        return data, timestamps

    def get_gray_meta(self):
        data = read_calib_file(
            os.path.join(
                self.grayFolder,
                self.seq,
                'calib.txt'))
        timestamps = load_timestamps(
            os.path.join(
                self.grayFolder,
                self.seq,
                'times.txt'))

        data = process_calib_data(data)

        return data, timestamps

    def get_gray_images_iter(self):
        data, timestamps = self.get_gray_meta()

        folderleft = os.path.join(
            self.grayFolder,
            self.seq,
            'image_0',
            '*.{}'.format('png'))
        folderright = os.path.join(
            self.grayFolder,
            self.seq,
            'image_1',
            '*.{}'.format('png'))

        leftimages = getfilesINfolder(folderleft)
        rightimages = getfilesINfolder(folderright)

        for i in range(len(leftimages)):
            lfimg = os.path.join(
                self.grayFolder,
                self.seq,
                'image_0',
                leftimages[i])
            rtimg = os.path.join(
                self.grayFolder,
                self.seq,
                'image_1',
                rightimages[i])

            yield (i, timestamps[i], lfimg, rtimg)

    def get_color_meta(self):
        data = read_calib_file(
            os.path.join(
                self.colorFolder,
                self.seq,
                'calib.txt'))
        timestamps = load_timestamps(
            os.path.join(
                self.colorFolder,
                self.seq,
                'times.txt'))

        data = process_calib_data(data)

        return data, timestamps

    def get_color_images_iter(self):
        data, timestamps = self.get_color_meta()

        folderleft = os.path.join(
            self.colorFolder,
            self.seq,
            'image_2',
            '*.{}'.format('png'))
        folderright = os.path.join(
            self.colorFolder,
            self.seq,
            'image_3',
            '*.{}'.format('png'))

        leftimages = getfilesINfolder(folderleft)
        rightimages = getfilesINfolder(folderright)

        for i in range(len(leftimages)):
            lfimg = os.path.join(
                self.colorFolder,
                self.seq,
                'image_2',
                leftimages[i])
            rtimg = os.path.join(
                self.colorFolder,
                self.seq,
                'image_3',
                rightimages[i])

            yield (i, timestamps[i], lfimg, rtimg)

    def get_velodyne_iter(self):
        data, timestamps = self.get_calib_main()

        foldervel = os.path.join(
            self.veloFolder,
            self.seq,
            'velodyne',
            '*.{}'.format('bin'))
        veldata = getfilesINfolder(foldervel)

        for i in range(len(veldata)):
            velobinfile = os.path.join(
                self.veloFolder, self.seq, 'velodyne', veldata[i])
            scan = load_velo_scan(velobinfile)
            yield (i, timestamps[i], scan)

    def get_leftcampose_iter(self):
        data, timestamps = self.get_calib_main()

        pose_file = os.path.join(self.posesFolder, self.seq + '.txt')
        poses = load_poses(pose_file)
        for i in range(len(poses)):
            yield (i, timestamps[i], poses[i])

    def velo_scan_X_to_ground0(self, Xvelo):
        x_cam0 = Tr * Xvelo
        x_gnd = T_w_cam0 * x_cam0


# %% testing
basedir = '/media/nagnanamus/d0690b96-7f71-44f2-96da-9f7259180ec7/SLAMData/Kitti/visualodo/'
kod = KittiOdoData(basedir, '00')
data, timestamps = kod.get_calib_main()
Nf = len(timestamps)

poseleftcam_xyztrue = np.zeros((Nf, 3))

T_cam0_velo = data['T_cam0_velo']

fig = plt.figure(3)
# ax = fig.add_subplot(111, projection='3d')

Nl = 100
Xlidarall = np.zeros((Nf * Nl, 3))
lc = 0


def targetfunc(q, quitbool):
    print("in thread")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    P = np.array([])
    while(quitbool):
        try:
            X = q.get(block=False)
        except BaseException:
            X = np.array([])
        if X.shape[0] > 0:
            print("got data")
            if P.shape[0] == 0:
                P = X
            else:
                P = np.vstack((P, X))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(P)
            colors = [[0.5, 0.5, 0.5] for i in range(len(pcd.points))]
            pcd.colors = o3d.utility.Vector3dVector(colors)
            # pcd = pcd.voxel_down_sample(voxel_size=0.1)
            vis.update_geometry(pcd)
            # vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)

        if quitbool.is_set():
            print("quit thread")
            break
    vis.destroy_window()


def targetfunc2(q, quitbool):
    print("in thread")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    P = np.array([])
    while(quitbool):
        try:
            X = q.get(block=False)
        except BaseException:
            X = np.array([])
        if X.shape[0] > 0:
            print("got data")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(X)
            colors = [[0.5, 0.5, 0.5] for i in range(len(pcd.points))]
            pcd.colors = o3d.utility.Vector3dVector(colors)
            # pcd = pcd.voxel_down_sample(voxel_size=0.1)
            vis.add_geometry(pcd)
            # vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)

        if quitbool.is_set():
            print("quit thread")
            break
    vis.destroy_window()


q = queue.Queue()
quitbool = threading.Event()
quitbool.clear()
th = threading.Thread(target=targetfunc2, args=(q, quitbool))
th.start()

for graydata, posedata, veldata in zip(kod.get_gray_images_iter(
), kod.get_leftcampose_iter(), kod.get_velodyne_iter()):
    (i, tk, lfimgfile, rtimgfile) = graydata
    print(i, tk, posedata[1])
    # if i>10:
    #     break

    poseleftcam_xyztrue[i, :] = posedata[2][0:3, 3]

    scan = veldata[2]
    rind_scan = np.random.randint(0, scan.shape[0], Nl)

    Xvel = np.ones((Nl, 4))
    for j in range(Nl):
        Xvel[j, 0:3] = scan[rind_scan[j], 0:3]
        T_w_cam0 = posedata[2]
        Xvel[j, :] = np.matmul(T_w_cam0, np.matmul(T_cam0_velo, Xvel[j, :]))

    Xlidarall[lc:lc + Nl, :] = Xvel[:, 0:3]
    lc = lc + Nl

    q.put(Xvel[:, 0:3])

    plt.figure(2)
    plt.clf()
    plt.plot(poseleftcam_xyztrue[:i, 0], poseleftcam_xyztrue[:i, 2])

    # plt.figure(3)
    # plt.clf()
    # # ax.plot(Xlidarall[:i,0],Xlidarall[:i,1],Xlidarall[:i,2],'k.')
    # plt.plot(Xlidarall[:lc,0],Xlidarall[:lc,2],'k.')
    # plt.plot(poseleftcam_xyztrue[:i,0],poseleftcam_xyztrue[:i,2])

    plt.show()
    plt.pause(0.001)

    lfimg = cv2.imread(lfimgfile)
    rtimg = cv2.imread(rtimgfile)
    h, w, d = lfimg.shape

    img = np.vstack((lfimg, rtimg))

    cv2.imshow('image', img)
    key = cv2.waitKey(10)
    if key == 27:    # Esc key to stop
        break

    # input("Press Enter to continue...")
    # time.sleep(0.5)

time.sleep(1)
quitbool.set()
th.join()

# vis.destroy_window()
# plt.close(2)
# plt.close(3)
cv2.destroyAllWindows()
time.sleep(2)
# plt.close('all')
# %%
# plt.figure()
# plt.plot(poseleftcam_xyztrue[:i,0],poseleftcam_xyztrue[:i,1])
# plt.show()
6 + 6
