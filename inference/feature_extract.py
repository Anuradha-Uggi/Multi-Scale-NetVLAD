from __future__ import print_function
import sys
import pytorch_lightning as pl
import argparse
import configparser
import os
import random
from os.path import join, isfile, exists
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F
import numpy as np

from PCA_Tools.training_tools.tools import pca
from datasets_original_bb import PlaceDataset
from tqdm.auto import tqdm
sys.path.append('./..')
from models import helper_inf as helper
from MSLS_Val.MapillaryDataset import MSLS  

class Weight(nn.Module):
 ###This finction is to define and initialize the scaling weights: W1 and W2 
    def __init__(self, dim=1):
        super().__init__()
        self.W = nn.Parameter(torch.rand(2))
        
    def forward(self, loss1, loss2):
        self.normalized_weights = F.softmax(self.W, dim=0)
        weighted_loss1 = self.normalized_weights[0] * loss1 
        weighted_loss2 = self.normalized_weights[1] * loss2
        weighted_loss = None
        return weighted_loss, weighted_loss1, weighted_loss2

class VPRModel(pl.LightningModule):
    def __init__(self,
                #---- Backbone
                backbone_arch='resnet50',
                pretrained=True,
                layers_to_freeze=2,
                layers_to_crop=[],
                
                #---- Aggregator
                agg_arch='ConvAP', #CosPlace, NetVLAD, MS-NetVLAD, GeM
                agg_config_pool1={},
                agg_config_pool2={},
                init_path = True,
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config_pool1 = agg_config_pool1
        self.agg_config_pool2 = agg_config_pool2
        self.init_path = init_path
        
        self.backbone = helper.get_backbone(self.encoder_arch, self.pretrained, self.layers_to_freeze, self.layers_to_crop)
        self.aggregator = helper.get_aggregator(self.agg_arch, self.agg_config_pool1, self.agg_config_pool2, self.init_path)
        self.aggregator1 = self.aggregator[0]
        self.aggregator2 = self.aggregator[1]
        #self.weights = Weight()
    def forward(self, x):
        x = self.backbone(x)
        #_, weighted_feat1, weighted_feat2 = self.weights(x[0], x[1])
        P1 = self.aggregator1(x[0])        
        P2 = self.aggregator2(x[1])
        return P1, P2

def Model(opt, config, checkpoint, device):
    model = VPRModel(
           backbone_arch='resnet50',
           pretrained=True,
           layers_to_freeze=1,
           layers_to_crop=[4],         
           agg_arch='MSNV',
           agg_config_pool1={'num_clusters' : opt.num_clusters_pool1,
                       'dim' : 512,
                       'vladv2' : opt.vladv2},
           agg_config_pool2={'num_clusters' : opt.num_clusters_pool2,
                       'dim' : 1024,
                       'vladv2' : opt.vladv2}, )                  
    
    append_pca_layer=True    
    if append_pca_layer:
        num_pcs = int(config['global_params']['num_pcs'])
        netvlad_output_dim1 = 512; netvlad_output_dim2 = 1024; ''' These are hardcoded for now for ResNet50. Could be automated later'''
        if config['global_params']['pooling'].lower() == 'netvlad' or config['global_params']['pooling'].lower() == 'patchnetvlad':
            netvlad_output_dim1 *= int(config['global_params']['num_clusters1'])
            netvlad_output_dim2 *= int(config['global_params']['num_clusters2'])

        pca_conv1 = nn.Conv2d(netvlad_output_dim1, num_pcs, kernel_size=(1, 1), stride=1, padding=0)
        pca_conv2 = nn.Conv2d(netvlad_output_dim2, num_pcs, kernel_size=(1, 1), stride=1, padding=0)
        model.add_module('WPCA_L1', nn.Sequential(*[pca_conv1, Flatten(), L2Norm(dim=-1)]))
        model.add_module('WPCA_L2', nn.Sequential(*[pca_conv2, Flatten(), L2Norm(dim=-1)]))
    model.load_state_dict(checkpoint)#state_dict['state_dict'])
    model.eval()
    return model


class Flatten(nn.Module):
    def forward(self, input_data):
        return input_data.view(input_data.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)      

def get_pca_encoding_pool1(model, vlad_encoding):
    pca_encoding = model.WPCA_L1(vlad_encoding.unsqueeze(-1).unsqueeze(-1))
    return pca_encoding 

def get_pca_encoding_pool2(model, vlad_encoding):
    pca_encoding = model.WPCA_L2(vlad_encoding.unsqueeze(-1).unsqueeze(-1))
    return pca_encoding

def msnv_feature_extract(eval_set, model, device, opt, config):
    K = [64, 16] 
    enc_dim = [512, 1024]
    output_global_features_filename1 = join(opt.output_features_dir, 'db_globalfeats_pool1.npy')
    output_global_features_filename2 = join(opt.output_features_dir, 'db_globalfeats_pool2.npy')
    pool_size = int(config['global_params']['num_pcs'])
    
    test_data_loader = DataLoader(dataset=eval_set, num_workers=int(config['global_params']['threads']),
                                  batch_size=int(config['feature_extract']['cacheBatchSize']),
                                  shuffle=False, pin_memory=(not opt.nocuda))
    model=model.to(device)
    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        feat_pool1 = np.empty((len(eval_set), pool_size), dtype=np.float32) 
        feat_pool2 = np.empty((len(eval_set), pool_size), dtype=np.float32)        
                       
        for iteration, (input_data, indices) in \
                enumerate(tqdm(test_data_loader, position=1, leave=False, desc='Test Iter'.rjust(15)), 1):
            indices_np = indices.detach().numpy()
            input_data = input_data.to(device)
            image_encoding_pool1, image_encoding_pool2 = model.forward(input_data)
            vlad_global_pca_pool1 = get_pca_encoding_pool1(model, image_encoding_pool1)    
            vlad_global_pca_pool2 = get_pca_encoding_pool2(model, image_encoding_pool2)            
            feat_pool1[indices_np, :] = vlad_global_pca_pool1.detach().cpu().numpy()
            feat_pool2[indices_np, :] = vlad_global_pca_pool2.detach().cpu().numpy()
    np.save(output_global_features_filename1, feat_pool1)
    np.save(output_global_features_filename2, feat_pool2) 


def input_transform(resize=(224, 224)):
    if resize[0] > 0 and resize[1] > 0:
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

def main():
    parser = argparse.ArgumentParser(description='Patch-NetVLAD-Feature-Extract')
    parser.add_argument('--config_path', type=str, default='performance_bb.ini',
                        help='File name (with extension) to an ini file that stores most of the configuration data')    
    parser.add_argument('--query_file_path', type=str, default='./dataset_names/dataset_imagenames/pitts30k_imageNames_query.txt',
                        help='Full path (with extension) to a text file that stores the save location and name of all images in the dataset folder')                    
    parser.add_argument('--index_file_path', type=str, default='./dataset_names/dataset_imagenames/pitts30k_imageNames_index.txt',
                        help='Full path (with extension) to a text file that stores the save location and name of all images in the dataset folder')
    
    parser.add_argument('--dataset_root_dir', type=str, default='/home/',
                        help='If the files in dataset_file_path are relative, use dataset_root_dir as prefix.')
    parser.add_argument('--ground_truth_path', type=str, default='./dataset_names/dataset_gt_files/pitts30k_test.npz')
    parser.add_argument('--model_path', type=str, default='./../LOGS-32/resnet50/lightning_logs/version_1/checkpoints/resnet50_epoch(03)_step(2504)_R1[0.7818]_R5[0.8856].ckpt')
    parser.add_argument('--arch', type=str, default='resnet50', 
                        help='basenetwork to use', choices=['resnet50','vgg16', 'alexnet'])   
    parser.add_argument('--output_features_dir', type=str, default='./p30k/')   
    parser.add_argument('--msls', action='store_true', help='None')
    parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
    parser.add_argument('--num_clusters_pool1', type=int, default=64, help='Number of NetVlad clusters. Default=64')
    parser.add_argument('--num_clusters_pool2', type=int, default=16, help='Number of NetVlad clusters. Default=16')
    parser.add_argument('--append_pca_layer', action='store_false', help='Number of NetVlad clusters. Default=64')
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')
    opt = parser.parse_args()
    print(opt)
    
    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")    
    opt.device = device = torch.device("cuda" if cuda else "cpu")
    
    print("=> loading checkpoint '{}'".format(opt.model_path))
    checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
    
    config['global_params']['num_clusters1'] = str(checkpoint['state_dict']['aggregator1.centroids'].shape[0])
    config['global_params']['num_clusters2'] = str(checkpoint['state_dict']['aggregator2.centroids'].shape[0])
    
    model=Model(opt, config, checkpoint['state_dict'], device)
    
    if not opt.msls:
       print('===> Non-MSLS activated')
       qImgs = PlaceDataset(None, opt.query_file_path, opt.dataset_root_dir, opt.ground_truth_path, config['feature_extract'], None, opt)
       dbImgs = PlaceDataset(None, opt.index_file_path, opt.dataset_root_dir, opt.ground_truth_path, config['feature_extract'], None, opt)
    
       qfeats = msnv_feature_extract(qImgs, model, device, opt, config, 'qry')
       dbfeats = msnv_feature_extract(dbImgs, model, device, opt, config, 'db')
    
    elif opt.msls:
       print('===> MSLS activated')
       my_transform = input_transform()
       val_dataset = MSLS(my_transform, None)
       qImgs = MSLS(my_transform, 'qry')
       dbImgs = MSLS(my_transform, 'db')
       
       qfeats = msnv_feature_extract(qImgs, model, device, opt, config, 'qry')
       dbfeats = msnv_feature_extract(dbImgs, model, device, opt, config, 'db')     
    
if __name__ == "__main__":
    main()
        
