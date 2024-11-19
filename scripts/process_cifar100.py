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

# CIFAR100
dat={}

dat['train']=datasets.CIFAR100('../dat/',train=True,download=True,
                            transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
dat['test']=datasets.CIFAR100('../dat/',train=False,download=True,
                            transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std)]))
for n in range(10):
    data[n]={}
    data[n]['name']='cifar100'
    data[n]['ncla']=10
    data[n]['train']={'x': [],'y': []}
    data[n]['test']={'x': [],'y': []}
for s in ['train','test']:
    loader=torch.utils.data.DataLoader(dat[s],batch_size=1,shuffle=False)
    for image,target in loader:
        task_idx = target.numpy()[0] // 10
        data[task_idx][s]['x'].append(image.numpy())
        data[task_idx][s]['y'].append(target.numpy()[0]%10)

train = {}
test = {}
for i in range(10):
    train[i] = data[i]['train']
    test[i] = data[i]['test']

with gzip.open(f'data/split_cifar100.pkl.gz', 'wb') as f:
    pickle.dump((train, test), f)

