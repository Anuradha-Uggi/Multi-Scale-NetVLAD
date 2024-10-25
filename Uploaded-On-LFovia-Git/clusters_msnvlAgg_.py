import pytorch_lightning as pl
from models import helper
import h5py
from math import log10, ceil
import random, shutil, json
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
import torch
import faiss

class NetVLAD_Clusters(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                #---- Backbone
                backbone_arch='resnet50',
                pretrained=True,
                layers_to_freeze=2,
                layers_to_crop=[],               
                faiss_gpu=False
                 ):
        super().__init__()
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.faiss_gpu = faiss_gpu       
        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
        
    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        return x

def get_Cluster_NV(gsv_whole_set, opt, device):
    cuda = not opt.nocuda
    cluster_set = gsv_whole_set
    encoder_dim = [512,1024]    
    num_clusters = [opt.num_clusters_pool1, opt.num_clusters_pool2] 
    model = NetVLAD_Clusters(
           #---- Encoder
           backbone_arch='resnet50',
           pretrained=True,
           layers_to_freeze=2,
           layers_to_crop=[4], 
           faiss_gpu=False
       ).to(device)
    nDescriptors = 50000
    nPerImage = 50
    nIm = ceil(nDescriptors/nPerImage)

    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False, 
                pin_memory=cuda,
                sampler=sampler)

    if not exists(join(opt.dataPath, 'centroids')):
        makedirs(join(opt.dataPath, 'centroids'))

    initcache = join(opt.dataPath, 'centroids', opt.arch + '_' + cluster_set.dataset + '_' + str(opt.num_clusters_pool1) + '_' + str(opt.num_clusters_pool2) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = [h5.create_dataset("descriptors%i"%cc,[nDescriptors, encoder_dim[cc]], dtype=np.float32) for cc in range(len(encoder_dim))]            

            for iteration, (input, indices) in enumerate(data_loader, 1):
                input = input.to(device)
                print('input_shape in clusters:\t', input.shape)
                image_desc = model.forward(input)
                image_descs = [image_desc[kk].view(input.size(0), encoder_dim[kk], -1).permute(0, 2, 1) for kk in range(len(encoder_dim))]

                batchix = (iteration-1)*opt.cacheBatchSize*nPerImage
                for it, image_descriptors in enumerate(image_descs):
                    for ix in range(image_descriptors.size(0)):
                        # sample different location for each image in batch
                        sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                        startix = batchix + ix*nPerImage
                        dbFeat[it][startix:startix+nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()
                    del sample    
                    if iteration % 50 == 0 or len(data_loader) <= 10:
                        print("==> Batch ({}/{})".format(iteration, 
                              ceil(nIm/opt.cacheBatchSize)), flush=True)
                del input, image_desc
        
        print('====> Clustering..')
        niter = 100
        kmeans = [faiss.Kmeans(encoder_dim[cnt], num_clusters[cnt], niter=niter, verbose=False) for cnt in range(len(num_clusters))]
        [kmeans[cnnt].train(dbFeat[cnnt][...]) for cnnt in range(len(num_clusters))]
       
        [print('====> Storing centroids for Layer:',bb,';', kmeans[bb].centroids.shape) for bb in range(len(num_clusters))]
        [h5.create_dataset('centroids%i'%bb, data=kmeans[bb].centroids) for bb in range(len(num_clusters))]
        print('====> Done!')
      
