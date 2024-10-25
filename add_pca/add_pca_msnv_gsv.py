#!/usr/bin/env python



from __future__ import print_function
import sys
import pytorch_lightning as pl
import argparse
import configparser
import os
import random
from os.path import join, isfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler

import numpy as np

from PCA_Tools.training_tools.tools import pca
from datasets_original import input_transform
from PCA_Tools.tools import PATCHNETVLAD_ROOT_DIR

from tqdm.auto import tqdm

from PCA_Tools.training_tools.msls import MSLS, ImagesFromList
from PCA_Tools.tools.datasets import PlaceDataset
sys.path.append('../gsv-cities/')
from models import helper_inf as helper

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

    def forward(self, x):
        x = self.backbone(x)
        P1 = self.aggregator1(x[0])        
        P2 = self.aggregator2(x[1])
        return [P1, P2]

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
    model.load_state_dict(checkpoint)
    model.eval()
    return model
    
class Flatten(nn.Module):
    def forward(self, input_data):
        return input_data.view(input_data.size(0), -1)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input_data):
        return F.normalize(input_data, p=2, dim=self.dim)    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Patch-NetVLAD-add-pca')

    parser.add_argument('--config_path', type=str, default='./PCA_Tools/configs/train.ini',
                        help='File name (with extension) to an ini file that stores most of the configuration data for patch-netvlad')
    parser.add_argument('--resume_path', type=str, default='./checkpoints/model.ckpt',
                        help='Full path and name (with extension) to load checkpoint from, for resuming training.')
    parser.add_argument('--dataset_root_dir', type=str, default='/home/anuradha/MyResearch@Sindhu/DataSets/Gsv-Cities/Images/',
                        help='Root directory of dataset')
    parser.add_argument('--dataset_choice', type=str, default='gsv-cities', help='choice of mapillary or pitts or gsv-cities, for PCA',
                        choices=['mapillary', 'pitts', 'gsv-cities'])
    parser.add_argument('--threads', type=int, default=6, help='Number of threads for each data loader to use')
    parser.add_argument('--arch', type=str, default='resnet50', 
                        help='basenetwork to use', choices=['vgg16', 'alexnet', 'resnet50'])   
    parser.add_argument('--mode', type=str, default='train', help='Mode', choices=['train', 'test', 'cluster'])  
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')
    parser.add_argument('--num_clusters_pool1', type=int, default=64, help='Number of NetVlad clusters. Default=64')
    parser.add_argument('--num_clusters_pool2', type=int, default=16, help='Number of NetVlad clusters. Default=16')
    parser.add_argument('--vladv2', action='store_true', help='Use VLAD v2')
    
    opt = parser.parse_args()
    print(opt)

    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)
    encoder_dim = [512, 1024]
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    random.seed(int(config['train']['seed']))
    np.random.seed(int(config['train']['seed']))
    torch.manual_seed(int(config['train']['seed']))
    if cuda:
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(int(config['train']['seed']))

    print('===> Building model')

    #encoder_dim, encoder = get_backend()

    if opt.resume_path: # must resume for PCA
        if isfile(opt.resume_path):
            print("=> loading checkpoint '{}'".format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path, map_location=lambda storage, loc: storage)            
            config['global_params']['num_clusters1'] = str(checkpoint['state_dict']['aggregator1.centroids'].shape[0])
            config['global_params']['num_clusters2'] = str(checkpoint['state_dict']['aggregator2.centroids'].shape[0])
            print('=> Number of clusters:', config['global_params']['num_clusters1'], '&', config['global_params']['num_clusters2'])
            ## Model loading    
            model = Model(opt, config, checkpoint['state_dict'], device) 

            print("=> loaded checkpoint '{}'".format(opt.resume_path, ))
        else:
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(opt.resume_path))
    else:
        raise ValueError("Need an existing checkpoint in order to run PCA")

    model = model.to(device)

    pool_size1 = encoder_dim[0]; pool_size2 = encoder_dim[1];
    if config['global_params']['pooling'].lower() == 'netvlad':
        pool_size1 *= int(config['global_params']['num_clusters1'])
        pool_size2 *= int(config['global_params']['num_clusters2'])

    print('===> Loading PCA dataset(s)')

    nFeatures = 10000
    if opt.dataset_choice == 'mapillary':
        exlude_panos_training = not config['train'].getboolean('includepanos')

        pca_train_set = MSLS(opt.dataset_root_dir, mode='test', cities='train',
                             transform=input_transform(),
                             bs=int(config['train']['cachebatchsize']), threads=opt.threads,
                             margin=float(config['train']['margin']),
                             exclude_panos=exlude_panos_training)

        pca_train_images = pca_train_set.dbImages
    elif opt.dataset_choice == 'pitts' or 'gsv-cities':
        dataset_file_path = 'gsv_cities.txt'
        pca_train_set = PlaceDataset(None, dataset_file_path, opt.dataset_root_dir, None, config['train'])
        pca_train_images = pca_train_set.images
        
    else:
        raise ValueError('Unknown dataset choice: ' + opt.dataset_choice)

    if nFeatures > len(pca_train_images):
        nFeatures = len(pca_train_images)

    sampler = SubsetRandomSampler(np.random.choice(len(pca_train_images), nFeatures, replace=False))

    data_loader = DataLoader(
        dataset=ImagesFromList(pca_train_images, transform=input_transform()),
        num_workers=opt.threads, batch_size=int(config['train']['cachebatchsize']), shuffle=False,
        pin_memory=cuda,
        sampler=sampler)

    print('===> Do inference to extract features and save them.')

    model.eval()
    with torch.no_grad():
        tqdm.write('====> Extracting Features')

        dbFeat_pool1 = np.empty((len(data_loader.sampler), pool_size1))
        dbFeat_pool2 = np.empty((len(data_loader.sampler), pool_size2))
        print('Compute', len(dbFeat_pool1), 'features')

        for iteration, (input_data, indices) in enumerate(tqdm(data_loader)):
            input_data = input_data.to(device)
            vlad_encoding_pool1, vlad_encoding_pool2 = model.forward(input_data) #model.pool(image_encoding)
            out_vectors1, out_vectors2 = vlad_encoding_pool1.detach().cpu().numpy(), vlad_encoding_pool2.detach().cpu().numpy()
              
            # this allows for randomly shuffled inputs
            for idx, out_vector in enumerate(out_vectors1):
                dbFeat_pool1[iteration * data_loader.batch_size + idx, :] = out_vector
                dbFeat_pool2[iteration * data_loader.batch_size + idx, :] = out_vectors2[idx]

            #del input_data, image_encoding, vlad_encoding
            del input_data, vlad_encoding_pool1, vlad_encoding_pool2

    print('===> Compute PCA, takes a while')
    num_pcs = int(config['global_params']['num_pcs'])
    u1, lams1, mu1 = pca(dbFeat_pool1, num_pcs); u2, lams2, mu2 = pca(dbFeat_pool2, num_pcs);     
    u1 = u1[:, :num_pcs]; u2 = u2[:, :num_pcs];    
    lams1 = lams1[:num_pcs]; lams2 = lams2[:num_pcs];

    print('===> Add PCA Whiten')
    u1 = np.matmul(u1, np.diag(np.divide(1., np.sqrt(lams1 + 1e-9))))
    u2 = np.matmul(u2, np.diag(np.divide(1., np.sqrt(lams2 + 1e-9))))
    pca_str1 = 'WPCA_L1'; pca_str2 = 'WPCA_L2'

    utmu1 = np.matmul(u1.T, mu1); utmu2 = np.matmul(u2.T, mu2);

    pca_conv1 = nn.Conv2d(pool_size1, num_pcs, kernel_size=(1, 1), stride=1, padding=0)
    pca_conv2 = nn.Conv2d(pool_size2, num_pcs, kernel_size=(1, 1), stride=1, padding=0)
    # noinspection PyArgumentList
    pca_conv1.weight = nn.Parameter(torch.from_numpy(np.expand_dims(np.expand_dims(u1.T, -1), -1)))
    pca_conv2.weight = nn.Parameter(torch.from_numpy(np.expand_dims(np.expand_dims(u2.T, -1), -1)))
    # noinspection PyArgumentList
    pca_conv1.bias = nn.Parameter(torch.from_numpy(-utmu1))
    pca_conv2.bias = nn.Parameter(torch.from_numpy(-utmu2))

    model.add_module(pca_str1, nn.Sequential(*[pca_conv1, Flatten(), L2Norm(dim=-1)]))
    model.add_module(pca_str2, nn.Sequential(*[pca_conv2, Flatten(), L2Norm(dim=-1)]))

    save_path = opt.resume_path.replace(".ckpt", "_WPCA_L1L2" + str(num_pcs) + ".ckpt")

    torch.save({'num_pcs': num_pcs, 'state_dict': model.state_dict()}, save_path)

    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    # memory after runs

    print('Done')
