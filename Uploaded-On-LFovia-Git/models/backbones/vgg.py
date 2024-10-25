from collections import namedtuple
import torch
from torchvision import models 
import netvlad

class VGG(torch.nn.Module):
    def __init__(self,
                 model_name='resnet50',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[],
                 ):
        super(VGG, self).__init__()
        encoder=models.vgg16(pretrained=True)
        layers=list(encoder.features.children())[:-2]
        for l in layers[:-5]: 
            for p in l.parameters():
                p.requires_grad = False
        self.slice1 = torch.nn.Sequential(*layers[0:4])
        self.slice2 = torch.nn.Sequential(*layers[4:9])
        self.slice3 = torch.nn.Sequential(*layers[9:16])
        self.slice4 = torch.nn.Sequential(*layers[16:23])
        self.slice5 = torch.nn.Sequential(*layers[23:29])
        
    def forward(self,X):    
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h    
        h = self.slice5(h)
        h_relu5_3 = h
    
        #vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        #out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        vgg_outputs = namedtuple("VggOutputs", ['relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu4_3, h_relu5_3)

        return out

