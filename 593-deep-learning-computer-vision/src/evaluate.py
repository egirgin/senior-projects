#!/usr/bin/env python
# coding: utf-8

# In[1]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[2]:


import pickle
model_path = 'results/FCN_13.31-12-06-22.pickle' # TODO: change this based on your experiment
with open(model_path, 'rb') as handle:
    model = pickle.load(handle)
print("Num. trainable params: {}".format(count_parameters(model)))


# In[3]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torchmetrics

from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
import SegNet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import os, pickle, datetime, time

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import warnings
warnings.filterwarnings("ignore")


# In[4]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[7]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 256), transforms.InterpolationMode.NEAREST )
])

dataset = datasets.Cityscapes('../cityscapes', split='test', mode='fine', target_type='semantic', target_transform=transform, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

jaccard_iou = torchmetrics.JaccardIndex(num_classes=34).to(device)


# In[8]:


pixelwise_acc = 0
iou = 0
f1_scr = 0
time_hist = []

with torch.no_grad():
    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = (labels.to(device) * 255).long()
        labels = torch.squeeze(labels, dim=1)
        
        sample_start = time.time()
        outputs = model(inputs).to(device)
        pred = torch.argmax(outputs, dim=1)
        sample_end = time.time()
        
        time_hist.append(sample_end - sample_start)
        
        pixelwise_acc += (torch.sum(pred == labels) / torch.numel(pred)).item()

        f1_scr = f1_score(
            labels.detach().cpu().numpy().reshape(-1), 
            pred.detach().cpu().numpy().reshape(-1), 
            average='macro')

        iou += jaccard_iou(pred, labels).detach().cpu().numpy() 


pixelwise_acc = pixelwise_acc / len(dataloader)
iou = iou / len(dataloader)
f1_scr = f1_scr / len(dataloader)

print("Accuracy: {:.2f} | F1: {:.2f} | IoU: {:.2f}".format( 
    pixelwise_acc,
    f1_scr,
    iou
))

mean_sample_time = np.mean(time_hist)

print("Sample FPS: {:.2f}".format(1/mean_sample_time))




# In[ ]:




