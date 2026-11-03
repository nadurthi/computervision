#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:07:09 2020

@author: nagnanamus
"""

import pandas as pd
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
import matplotlib.pyplot as plt
import numpy as np
import itertools as itr
import matplotlib
matplotlib.use('TkAgg')

# %%
# % KITTI TRACKING BENCHMARK DEMONSTRATION
# %
# % This tool displays the images and the object labels for the benchmark and
# % provides an entry point for writing your own interface to the data set.
# % Before running this tool, set root_dir to the directory where you have
# % downloaded the dataset. 'root_dir' must contain the subdirectory
# % 'training', which in turn contains 'image_02', 'label_02' and 'calib'.
# % For more information about the data format, please look into readme.txt.
# %
# % Usage:
# %   SPACE: next frame
# %   '-':   last frame
# %   'x':   +50 frames
# %   'y':   -50 frames
# %   'c':   previous sequence
# %   'v':   next sequence
# %   q:     quit
# %
# % Occlusion Coding:
# %   green:  not occluded
# %   yellow: partly occluded
# %   red:    fully occluded
# %   white:  unknown
# %
# % Truncation Coding:
# %   solid:  not truncated
# %   dashed: truncated


def readLabels(labelpath, seq_idx)

    # % parse input file
    labelfile = os.path.join(labelpath, '{0:04d}.txt'.format(seq_idx))
    df = pd.read_csv(labelfile)
     1    frame        Frame within the sequence where the object appearers
   1    track id     Unique tracking id of this object within this sequence
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Integer (0,1,2) indicating the level of truncation.
                     Note that this is in contrast to the object detection
                     benchmark where truncation is a float in [0,1].
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
                     
    return tracklets
    
    
# % options
root_dir = '/media/nagnanamus/d0690b96-7f71-44f2-96da-9f7259180ec7/SLAMData/Kitti/tracking/mot/allcomb/';
data_set = 'training';

# % set camera
cam = 2; #% 2 = left color camera
seq_idx=1;

cam2path = os.path.join(root_dir,data_set, 'image_{0:02d}'.format(2))
cam3path = os.path.join(root_dir,data_set, 'image_{0:02d}'.format(3))

cam2pathseq = os.path.join(cam2path,'{0:04d}'.format(2))
cam3pathseq = os.path.join(cam2path,'{0:04d}'.format(3))

labelpath = os.path.join(root_dir,data_set, 'label_{0:02d}'.format(2))
calibpath = os.path.join(root_dir,data_set, 'label_{0:02d}'.format(2))


labelfile = os.path.join(labelpath,'{0:04d}.txt'.format( seq_idx) ) 

# %%


# % show data for tracking sequences
nsequences = numel(dir(fullfile(root_dir,data_set, sprintf('image_%02d',cam))))-2;

% get sub-directories
image_dir = fullfile(root_dir,data_set, sprintf('image_%02d/%04d',cam, seq_idx));
label_dir = fullfile(root_dir,data_set, sprintf('label_%02d',cam));
calib_dir = fullfile(root_dir,data_set, 'calib');
P = readCalibration(calib_dir,seq_idx,cam);

% get number of images for this dataset
nimages = length(dir(fullfile(image_dir, '*.png')));

% load labels
tracklets = readLabels(label_dir, seq_idx);

% set up figure
h = visualization('init',image_dir);

% main loop
img_idx=0;
while 1

  % load projection matrix


  % visualization update for next frame
  visualization('update',image_dir,h,img_idx,nimages,data_set);

  % for all annotated tracklets do
  for obj_idx=1:numel(tracklets{img_idx+1})

    % plot 2D bounding box
    drawBox2D(h,tracklets{img_idx+1}(obj_idx));

    % plot 3D bounding box
    [corners,face_idx] = computeBox3D(tracklets{img_idx+1}(obj_idx),P);
    orientation = computeOrientation3D(tracklets{img_idx+1}(obj_idx),P);
    drawBox3D(h, tracklets{img_idx+1}(obj_idx),corners,face_idx,orientation);

  end

  % force drawing and tiny user interface
  try
    waitforbuttonpress; 
  catch
    fprintf('Window closed. Exiting...\n');
    break
  end
  key = get(gcf,'CurrentCharacter');
  switch lower(key)                         
    case 'q',  break;                                 % quit
    case '-',  img_idx = max(img_idx-1,  0);          % previous frame
    case 'x',  img_idx = min(img_idx+50,nimages-1);   % +50 frames
    case 'y',  img_idx = max(img_idx-50,0);           % -50 frames
    case 'v'
      seq_idx   = min(seq_idx+1,nsequences);
      img_idx   = 0;
      image_dir = fullfile(root_dir,data_set, sprintf('image_%02d/%04d',cam, seq_idx));
      label_dir = fullfile(root_dir,data_set, sprintf('label_%02d',cam));
      calib_dir = fullfile(root_dir,data_set, 'calib');
      nimages   = length(dir(fullfile(image_dir, '*.png')));
      tracklets = readLabels(label_dir,seq_idx,nimages);
      P = readCalibration(calib_dir,seq_idx,cam);
    case 'c'
      seq_idx   = max(seq_idx-1,0);
      img_idx   = 0;
      image_dir = fullfile(root_dir,data_set, sprintf('image_%02d/%04d',cam, seq_idx));
      label_dir = fullfile(root_dir,data_set, sprintf('label_%02d',cam));
      calib_dir = fullfile(root_dir,data_set, 'calib');
      nimages   = length(dir(fullfile(image_dir, '*.png')));
      tracklets = readLabels(label_dir,seq_idx,nimages);
      P = readCalibration(calib_dir,seq_idx,cam);
    otherwise, img_idx = min(img_idx+1,  nimages-1);  % next frame
  end
end

% clean up
close all;
