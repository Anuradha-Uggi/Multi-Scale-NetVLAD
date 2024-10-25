import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image
import os
import cv2
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

root_dir = '/media/amulya/Expansion/Anni-SrinadhsDGX/Mapillary/msls/train_val/'
if not exists(root_dir):
    raise FileNotFoundError('root_dir is hardcoded, please adjust to point to Pittsburth dataset')


def input_transform():
    return transforms.Compose([
        transforms.Resize((320,240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])      
    
class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, input_transform=input_transform(), onlyDB=False):
        super().__init__()

        self.input_transform = input_transform
        self.Path = os.listdir(root_dir)        
        self.dataset = 'gsv_cities'
        self.images=[]
        for P in self.Path:
            for D in sorted(os.listdir(root_dir+P+'/')):
                self.images.append(root_dir+P+'/'+D)
        #for D in sorted(os.listdir(root_dir+self.Path[0]+'/')):
         #   self.images.append(root_dir+self.Path[0]+'/'+D)    
    def __getitem__(self, index):
        img = Image.open(self.images[index])
        if self.input_transform:
            img = self.input_transform(img)
        return img, index

    def __len__(self):
        return len(self.images)
#P = WholeDatasetFromStruct()        
#print('[len of files]', len(P), np.array(P[0]).shape)   
