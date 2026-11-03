# -*- coding: utf-8 -*-

import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt 
from torchvision import transforms
import pickle as pkl
import random

class UpOrDownSample(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        

        # swap color axis because
        # numpy image: H x W x C
        # torch series: C X L
        sampleT=[]
        for i in range(2):
            if sample[i].shape[1]==2:
                sample[i] = sample[i].transpose()
            if sample[i].shape[1]<360:
                extridx = np.random.randint(0,sample[i].shape[1], size=360-sample[i].shape[1])
                sample[i] = np.hstack([sample[i],sample[i][:,extridx]])

                
            
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        

        # swap color axis because
        # numpy image: H x W x C
        # torch series: C X L
        sampleT=[]
        for i in range(2):
            if sample[i].shape[1]==2:
                sample[i] = sample[i].transpose()
            sampleT.append(torch.from_numpy(sample[i])) 
            
        return sampleT
    
class RotatePtset:
    def __init__(self,thmin=0,thmax=2*np.pi,sig=0.01):
        self.thmin = thmin
        self.thmax = thmax
        self.sig = sig
        
    def __call__(self,sample):
        for i in range(2):
            theta = np.random.uniform(self.thmin,self.thmax)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
            # print(sample)
            sample[i][:] = sample[i][:].dot(rotation_matrix) # random rotation
            sample[i] += np.random.normal(0,self.sig, size=sample[i].shape) # random jitter
            
        return sample
    

composed = transforms.Compose([RotatePtset(),UpOrDownSample(),ToTensor()])

class lidar3602Dpoints(data.Dataset):
    def __init__(self, filepath,splittype='Train',transform=None):
        self.filepath = filepath
        self.transform = transform
        self.splittype = splittype
        
        with open(filepath,'rb') as fh:
            self.ptsdata = pkl.load(fh)
        
        self.N=len(self.ptsdata)
        self.iterRange = list(range(self.N))
        self.indices = list(range(self.N))
        random.shuffle(self.indices)
        
        n = int(np.floor(0.8*self.N))
        self.trainRange=self.indices[0:n]
        self.testRange=self.indices[n:]
        
    def __len__(self):
        if self.splittype=='Train':
            return len(self.trainRange)
        elif self.splittype=='Test':
            return len(self.testRange)
        elif self.splittype=='Iter':
            return len(self.iterRange)
        else:
            raise NotImplementedError
            
    def __getitem__(self, index):
        
        if self.splittype=='Train':
            idx = self.trainRange[index]
        elif self.splittype=='Test':
            idx = self.testRange[index]
        elif self.splittype=='Iter':
            idx = self.iterRange[index]
        else:
            raise NotImplementedError
            
        rngs = np.array(self.ptsdata[idx]['ranges'])
        
        angle_min=self.ptsdata[idx]['angle_min']
        angle_max=self.ptsdata[idx]['angle_max']
        angle_increment=self.ptsdata[idx]['angle_increment']
        ths = np.arange(angle_min,angle_max+angle_increment,angle_increment)
        p=np.vstack([np.cos(ths),np.sin(ths)])

        
        ptset = rngs.reshape(-1,1)*p.T
        
        safeptsidx = (rngs<=self.ptsdata[idx]['range_max']) & (rngs>=self.ptsdata[idx]['range_min'])
        ptset = ptset[safeptsidx,:]
        
        sample =[ptset.copy().astype(np.float32),ptset.copy().astype(np.float32)]
        
        if self.transform:
            sample = self.transform(sample)
        
        

        return sample

if __name__ == "__main__":
    
    filepath = "../houseScan.pkl"
    
    
    
    ptdataset = lidar3602Dpoints(filepath,splittype='Iter',transform=ToTensor())
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    for i in range(len(ptdataset)):
        print(i,ptdataset[i][0].shape)
        ptset1 = ptdataset[i][0].numpy().T
        ptset2 = ptdataset[i][1].numpy().T
        ax1.cla()
        ax2.cla()
        ax1.plot(ptset1[:,0], ptset1[:,1],'bo')
        ax2.plot(ptset2[:,0], ptset2[:,1],'bo')
        
        plt.tight_layout()
        plt.show()
        plt.pause(0.3)
        
        