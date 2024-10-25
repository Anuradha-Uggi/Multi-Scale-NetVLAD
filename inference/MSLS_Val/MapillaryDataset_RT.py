from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
# NOTE: you need to download the mapillary_sls dataset from  https://github.com/FrederikWarburg/mapillary_sls
# make sure the path where the mapillary_sls validation dataset resides on your computer is correct.
# the folder named train_val should reside in DATASET_ROOT path (that's the only folder you need from mapillary_sls)
# I hardcoded the groundtruth for image to image evaluation, otherwise it would take ages to run the groundtruth script at each epoch.
DATASET_ROOT = '/home/anuradha/msls/'

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception('Please make sure the path to mapillary_sls dataset is correct')

if not path_obj.joinpath('train_val'):
    raise Exception(f'Please make sure the directory train_val from mapillary_sls dataset is situated in the directory {DATASET_ROOT}')

class MSLS(Dataset):
    def __init__(self, input_transform1 = None, input_transform2 = None, input_transform3 = None, gallery = None):
        
        self.input_transform1 = input_transform1
        self.input_transform2 = input_transform2
        self.input_transform3 = input_transform3
        
        # hard coded reference image names, this avoids the hassle of listing them at each epoch.
        self.dbImages = np.load('./datasets/msls_val/msls_val_dbImages.npy')
        # hard coded query image names.
        self.qImages = np.load('./datasets/msls_val/msls_val_qImages.npy')
        
        # hard coded index of query images
        self.qIdx = np.load('./datasets/msls_val/msls_val_qIdx.npy')
        
        # hard coded groundtruth (correspondence between each query and its matches)
        self.pIdx = np.load('./datasets/msls_val/msls_val_pIdx.npy', allow_pickle=True)
        
        # concatenate reference images then query images so that we can use only one dataloader
        
        if gallery=='qry':
            self.images = self.qImages[self.qIdx]
        elif gallery=='db':
            self.images = self.dbImages
        elif gallery==None:    
            self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))
        # we need to keeo the number of references so that we can split references-queries 
        # when calculating recall@K
        self.numDb = len(self.dbImages)
        self.numQ = len(self.qIdx)
    
    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT+self.images[index])
        if self.input_transform1 and self.input_transform2 and self.input_transform3:
            img1 = self.input_transform1(img)
            img2 = self.input_transform2(img)
            img3 = self.input_transform3(img)

        return img1, img2, img3, index

    def __len__(self):
        return len(self.images)
