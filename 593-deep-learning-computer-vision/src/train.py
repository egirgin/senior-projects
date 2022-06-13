#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

import os, pickle, datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import warnings
warnings.filterwarnings("ignore")


# In[2]:


params = {
    "epochs": 1000,
    "batch_size": 32,
    "lr": 0.01,
    "fcn": False,
    "num_class": 34,
    "img_size": [128, 256],
    "vanilla_bce": False
}


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# In[4]:


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((params["img_size"][0], params["img_size"][1]), transforms.InterpolationMode.NEAREST )
])


trainset = datasets.Cityscapes('../cityscapes', split='train', mode='fine', target_type='semantic', target_transform=transform, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=params["batch_size"], shuffle=True, num_workers=2)

valset = datasets.Cityscapes('../cityscapes', split='val', mode='fine', target_type='semantic', target_transform=transform, transform=transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=2, shuffle=True, num_workers=2)

testset = datasets.Cityscapes('../cityscapes', split='test', mode='fine', target_type='semantic', target_transform=transform, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)


# In[5]:


"""weight_list = [0.9998873433759136, 0.9534936310142599, 0.9862746577109059, 0.9883156950755786, 
0.9865494287142189, 0.9971385463591544, 0.9878501276816091, 0.6730788753878686, 0.9459851070116925, 
0.9937193368070869, 0.9981925718245968, 0.7975315381121891, 0.9941867090040638, 0.9922160487021169, 
0.999912210690078, 0.9971354084630166, 0.9994591743715348, 0.9891240314770771, 0.9999195427022954, 
0.9981569372197633, 0.9951162748439337, 0.8587440367667906, 0.9897223031649025, 0.9643351442070418, 
0.989197895091067, 0.9988032310239731, 0.9379626961164577, 0.9976309089250462, 0.9979141194333312, 
0.9995995695872973, 0.9997915555072087, 0.9979355719781691, 0.9991253063242923, 0.9963304868308447]"""


# In[ ]:


freq_count = {}
median_count = {}
percent_count = {}

all_uniques = []
for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        unique, counts = np.unique(labels.detach().cpu().numpy(), return_counts=True)
            
        unique *= 255
        
        all_uniques += unique.astype(int).tolist()

        for i in range(len(unique)):
            if unique[i] in freq_count.keys():
                freq_count[int(unique[i])] += counts[i]
                median_count[int(unique[i])].append(counts[i])
                percent_count[int(unique[i])].append(1 / counts[i]) # 1/percent
            else:
                freq_count[int(unique[i])] = counts[i]
                median_count[int(unique[i])] = [counts[i]]
                percent_count[int(unique[i])] = [1 / counts[i]]

                

all_uniques = np.unique(all_uniques)


# In[ ]:


all_pixels = params["img_size"][0]*params["img_size"][1]*params["batch_size"]*len(trainloader)

weight_list = []            
for i in range(params["num_class"]):
    if i in all_uniques:
        weight_list.append(
            (all_pixels-freq_count[i]) / all_pixels
            #np.median(median_count[i]) / np.sum(median_count[i])   
            #np.mean(percent_count[i])
        )
    else:
        print("There is an error!")
        weight_list.append(0)
print(weight_list)


# In[5]:


img, smnt = trainset[0]
fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(img.permute(1,2,0))
ax[1].imshow(smnt.permute(1,2,0))


# In[6]:


if params["fcn"]:
    vgg_model = VGGNet(requires_grad=True, remove_fc=True).to(device)
    model = FCN32s(pretrained_net=vgg_model, n_class=params["num_class"]).to(device)
else:
    model = SegNet.SegNet().to(device)
    
#torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', pretrained=True)

if params["vanilla_bce"]:
    criterion = nn.CrossEntropyLoss()
    
else:
    criterion = nn.CrossEntropyLoss(torch.Tensor(weight_list).to(device))
    print("Using weighted BCE...")
optimizer = optim.Adam(model.parameters(), lr=params["lr"])
jaccard_iou = torchmetrics.JaccardIndex(num_classes=params["num_class"]).to(device)


# In[7]:


loss_hist = []
acc_hist = []
iou_hist = []
f1_hist = []
for epoch in range(params["epochs"]):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = (labels.to(device) * 255).long()

        labels = torch.squeeze(labels, dim=1) # since grayscale has 1 channel remove that dim
        #print(torch.unique(labels*255))
        
        #labels = one_hot(labels)
        
        # clear the parameter gradients
        optimizer.zero_grad()        

        # forward + backward + optimize
        outputs = model(inputs).to(device)
        softmax = torch.nn.Softmax(dim=1)
        outputs = softmax(outputs)
        #outputs = torch.max(outputs, dim=1)[1]

        # calculate loss
        loss = criterion(outputs, labels)

        # backprop 
        loss.backward()

        # update weights
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        #print(loss.item())
        loss_hist.append(loss.item())
        
    # Validation
    
    pixelwise_acc = 0
    iou = 0
    f1_scr = 0
    with torch.no_grad():
        for i, data in enumerate(valloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = (labels.to(device) * 255).long()
            labels = torch.squeeze(labels, dim=1)

            outputs = model(inputs).to(device)
            pred = torch.argmax(outputs, dim=1)

            
            pixelwise_acc += (torch.sum(pred == labels) / torch.numel(pred)).item()
            
            f1_scr = f1_score(
                labels.detach().cpu().numpy().reshape(-1), 
                pred.detach().cpu().numpy().reshape(-1), 
                average='macro')
            
            iou += jaccard_iou(pred, labels).detach().cpu().numpy() 
            
    acc_hist.append(pixelwise_acc / len(valloader))
    iou_hist.append(iou / len(valloader))
    f1_hist.append(f1_scr / len(valloader))
    print("Epochs: {} | Loss: {:.5f} | Accuracy: {:.2f} | F1: {:.2f} | IoU: {:.2f}".format(
        epoch, 
        running_loss/len(trainloader), 
        acc_hist[-1],
        f1_hist[-1],
        iou_hist[-1]
    ))
    running_loss = 0.0
    
    plt.cla()
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(24,5))
    ax[0,0].imshow(inputs[0].permute(1,2,0).detach().cpu().numpy())
    ax[0,1].imshow(labels[0:1].permute(1,2,0).detach().cpu().numpy())
    ax[0,2].imshow(pred[0:1].permute(1,2,0).detach().cpu().numpy()/255)
    ax[1,0].imshow(inputs[1].permute(1,2,0).detach().cpu().numpy())
    ax[1,1].imshow(labels[1:].permute(1,2,0).detach().cpu().numpy())
    ax[1,2].imshow(pred[1:].permute(1,2,0).detach().cpu().numpy()/255)
    
    os.makedirs("results/", exist_ok=True)
    plt.savefig("results/segmentation_sample_{}.png".format(epoch))
    
    plt.close()
    plt.cla()
    plt.plot(loss_hist)

    plt.title("Weighted BCE Loss")
    plt.xlabel("Iterations")
    plt.ylabel("BCELoss")

    #plt.show()
    plt.savefig("results/pixelloss.png")
    
    
    plt.close()
    plt.cla()
    plt.plot(acc_hist)

    plt.title("Pixelwise Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    #plt.show()
    plt.savefig("results/pixelacc.png")
    
    plt.close()
    plt.cla()
    plt.plot(f1_hist)

    plt.title("Macro Avg. F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")

    #plt.show()
    plt.savefig("results/f1_score.png")

    plt.close()
    plt.cla()
    plt.plot(iou_hist)

    plt.title("Macro Avg. IoU")
    plt.xlabel("Epochs")
    plt.ylabel("Jaccard Index")

    #plt.show()
    plt.savefig("results/iou.png")
    
    if epoch%10 == 0:
        sign = datetime.datetime.fromtimestamp(datetime.datetime.now().timestamp()).strftime('%H.%M-%d-%m-%y')
        if params["fcn"]:
            with open('results/FCN_{}.pickle'.format(sign), 'wb') as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('results/SEGNET_{}.pickle'.format(sign), 'wb') as handle:
                pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


print('Finished Training')


# In[ ]:




