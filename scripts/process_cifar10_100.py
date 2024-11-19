import os
import numpy as np
import torch
from torchvision import datasets,transforms
from sklearn.utils import shuffle
import gzip
import pickle


data={}
taskcla=[]
size=[3,32,32]


mean=[x/255 for x in [125.3,123.0,113.9]]
std=[x/255 for x in [63.0,62.1,66.7]]

# CIFAR10
dat={}
dat['train']=datasets.CIFAR10('../dat/',train=True,download=True,
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
dat['test']=datasets.CIFAR10('../dat/',train=False,download=True,
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
data[0]={}
data[0]['name']='cifar10'
data[0]['ncla']=10
data[0]['train']={'x': [],'y': []}
data[0]['test']={'x': [],'y': []}
for s in ['train','test']:
    loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
    for image,target in loader:
        data[0][s]['x'].append(image)
        data[0][s]['y'].append(target.numpy()[0])


# CIFAR100
dat={}

dat['train']=datasets.CIFAR100('../dat/',train=True,download=True,
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
dat['test']=datasets.CIFAR100('../dat/',train=False,download=True,
                                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
for n in range(1,11):
    data[n]={}
    data[n]['name']='cifar100'
    data[n]['ncla']=10
    data[n]['train']={'x': [],'y': []}
    data[n]['test']={'x': [],'y': []}
for s in ['train','test']:
    loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
    for image,target in loader:
        task_idx = target.numpy()[0] // 10 + 1
        data[task_idx][s]['x'].append(image.numpy())
        data[task_idx][s]['y'].append(target.numpy()[0]%10)

train = {}
test = {}
for i in range(11):
    train[i] = data[i]['train']
    test[i] = data[i]['test']

with gzip.open(f'data/split_cifar10_100.pkl.gz', 'wb') as f:
    pickle.dump((train, test), f)

