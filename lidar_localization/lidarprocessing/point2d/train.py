# -*- coding: utf-8 -*-


import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from dataset import lidar3602Dpoints,composed, ToTensor
from model import PointNetfeat, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils import data as tdata
import matplotlib.pyplot as plt 
from numpy import linalg as LA
import networkx as nx


blue = lambda x: '\033[94m' + x + '\033[0m'
opt={}

opt['manualSeed'] = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt['manualSeed'])
random.seed(opt['manualSeed'])
torch.manual_seed(opt['manualSeed'])


filepath = "../houseScan_std.pkl"

dataset = lidar3602Dpoints(filepath,splittype='Train',transform=composed)

test_dataset = lidar3602Dpoints(filepath,splittype='Test',transform=composed)

opt['batchSize'] = 32
opt['workers'] = 2
opt['nepoch'] = 10

dataloader = tdata.DataLoader(
    dataset,
    batch_size=opt['batchSize'],
    shuffle=True,
    num_workers=int(opt['workers']))

testdataloader = tdata.DataLoader(
        test_dataset,
        batch_size=opt['batchSize'],
        shuffle=True,
        num_workers=int(opt['workers']))


print(len(dataset), len(test_dataset))



classifier = PointNetfeat()
# lossfunc = torch.nn.L1Loss()
lossfunc = torch.nn.MSELoss()

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

if torch.cuda.is_available():
    classifier.cuda()

num_batch = len(dataset) / opt['batchSize'] 

for epoch in range(opt['nepoch']):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        points, target = data
        # target = target[:, 0]
        # points = points.transpose(2, 1)
        if torch.cuda.is_available():
            points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        
        classifier = classifier.eval()
        targetpred, targettrans, targettrans_feat = classifier(target)
        
        loss = lossfunc(pred, targetpred)
        
        loss += feature_transform_regularizer(trans) * 0.001
        # loss += feature_transform_regularizer(trans_feat) * 0.001
        
        loss.backward()
        optimizer.step()
        # pred_choice = pred.data.max(1)[1]lossfunc
        # correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: ' % (epoch, i, num_batch, loss.item()))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data
            # target = target[:, 0]
            # points = points.transpose(2, 1)
            if torch.cuda.is_available():
                points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            targetpred, _, _ = classifier(target)
            loss = lossfunc(pred, targetpred)
            # pred_choice = pred.data.max(1)[1]
            # correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy:' % (epoch, i, num_batch, blue('test'), loss.item()))




# with torch.no_grad():
#     classifier = classifier.eval()
#     total_correct = 0
#     total_testset = 0
#     for i,data in tqdm(enumerate(testdataloader, 0)):
#         points, target = data
#         if torch.cuda.is_available():
#             points, target = points.cuda(), target.cuda()
        
#         pred, _, _ = classifier(points)
#         targetpred, _, _ = classifier(target)

#         loss = lossfunc(pred, targetpred)
    
#         print("final accuracy {}".format(total_correct / float(total_testset)))

#%% plotting
datasetplot = lidar3602Dpoints(filepath,splittype='Train',transform=ToTensor())
classifier = classifier.eval()
fig, ax = plt.subplots(2, 2)
for i in range(len(datasetplot)):
    ptdataset = datasetplot[i]
    ptset1 = ptdataset[0].unsqueeze(0)
    ptset2 = ptdataset[1].unsqueeze(0)
    
    pred1, _, _ = classifier(ptset1)
    pred2, _, _ = classifier(ptset2)
    
    ptset1=ptset1.numpy()[0].T
    ptset2=ptset2.numpy()[0].T
    
    pred1=pred1.detach().numpy()[0]
    pred2=pred2.detach().numpy()[0]
    
    
    ax[0,0].cla()
    ax[0,1].cla()
    ax[1,0].cla()
    ax[1,1].cla()
    
    ax[0,0].plot(ptset1[:,0], ptset1[:,1],'bo')
    ax[0,1].plot(ptset2[:,0], ptset2[:,1],'bo')
    ax[1,0].plot(np.arange(len(pred1)), pred1)
    ax[1,1].plot(np.arange(len(pred2)), pred2)
    
    
    plt.tight_layout()
    plt.show()
    plt.pause(2)
    
    
#%% plotting sequential
datasetplot = lidar3602Dpoints(filepath,splittype='Train',transform=ToTensor())
classifier = classifier.eval()
fig, ax = plt.subplots(2, 2)
for i in range(len(datasetplot)-1):
    ptdataset = datasetplot[i]
    ptset1 = ptdataset[0].unsqueeze(0)
    
    ptdataset = datasetplot[i+1]
    ptset2 = ptdataset[0].unsqueeze(0)
    
    
    pred1, _, _ = classifier(ptset1)
    pred2, _, _ = classifier(ptset2)
    
    ptset1=ptset1.numpy()[0].T
    ptset2=ptset2.numpy()[0].T
    
    pred1=pred1.detach().numpy()[0]
    pred2=pred2.detach().numpy()[0]
    
    
    ax[0,0].cla()
    ax[0,1].cla()
    ax[1,0].cla()
    ax[1,1].cla()
    
    ax[0,0].plot(ptset1[:,0], ptset1[:,1],'bo')
    ax[0,1].plot(ptset2[:,0], ptset2[:,1],'bo')
    ax[1,0].plot(np.arange(len(pred1)), pred1)
    ax[1,1].plot(np.arange(len(pred2)), pred2)
    
    
    plt.tight_layout()
    plt.show()
    plt.pause(3)    
#%% find all similar featured plots
datasetplot = lidar3602Dpoints(filepath,splittype='Iter',transform=ToTensor())
Nd = len(datasetplot)

classifier = classifier.eval()
D=np.zeros((Nd,Nd))
F=[]
with torch.no_grad():
    for i in range(len(datasetplot)):
        print(i)  
        ptdataset1 = datasetplot[i]
        ptset1 = ptdataset1[0].unsqueeze(0)
        pred1, _, _ = classifier(ptset1)
        pred1=pred1.detach().numpy()[0]
        F.append(pred1)



for i in range(len(datasetplot)):
    for j in range(len(datasetplot)):
        D[i,j] = LA.norm(F[i]-F[j],2)
    
    
    
    

G = nx.Graph()    

for i in range(Nd):
    G.add_node(i)

dthres = 0.02
for i in range(Nd):  
    for j in range(i+1,Nd):  
        if D[i,j] <= dthres:
            G.add_edge(i, j)   
        

G.number_of_nodes()    
G.number_of_edges()    

fig, ax = plt.subplots(2, 2)
for i in range(1000,Nd):
    for j in G.neighbors(i):
        
        if np.abs(i-j)<100:
            continue
        print(i,j)
        ptdataset = datasetplot[i]
        ptset1 = ptdataset[0].unsqueeze(0)
        
        ptdataset = datasetplot[j]
        ptset2 = ptdataset[0].unsqueeze(0)
        
        pred1, _, _ = classifier(ptset1)
        pred2, _, _ = classifier(ptset2)
        
        ptset1=ptset1.numpy()[0].T
        ptset2=ptset2.numpy()[0].T
        
        pred1=pred1.detach().numpy()[0]
        pred2=pred2.detach().numpy()[0]
        
        
        ax[0,0].cla()
        ax[0,1].cla()
        ax[1,0].cla()
        ax[1,1].cla()
        
        ax[0,0].plot(ptset1[:,0], ptset1[:,1],'bo')
        ax[0,1].plot(ptset2[:,0], ptset2[:,1],'bo')
        ax[1,0].plot(np.arange(len(pred1)), pred1)
        ax[1,1].plot(np.arange(len(pred2)), pred2)
        
        ax[0,0].set_xlim(-4,4)
        ax[0,1].set_xlim(-4,4)
        
        ax[0,0].set_ylim(-4,4)
        ax[0,1].set_ylim(-4,4)
        
        plt.tight_layout()
        plt.show()
        plt.pause(0.2)
    
    
    
    
    
    
    
    
    
    