import numpy as np
from models import aggregators
from models.aggregators import netvlad
from models import backbones


def get_backbone(backbone_arch='resnet50',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[],):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): . Defaults to 'resnet50'.
        pretrained (bool, optional): . Defaults to True.
        layers_to_freeze (int, optional): . Defaults to 2.
        layers_to_crop (list, optional): This is mostly used with ResNet where 
                                         we sometimes need to crop the last 
                                         residual block (ex. [4]). Defaults to [].

    Returns:
        nn.Module: the backbone as a nn.Model object
    """
    if 'resnet' in backbone_arch.lower():
        return backbones.ResNet(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
    
    elif 'vgg' in backbone_arch.lower():
        return backbones.VGG(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)    

    elif 'efficient' in backbone_arch.lower():
        if '_b' in backbone_arch.lower():
            return backbones.EfficientNet(backbone_arch, pretrained, layers_to_freeze+2)
        else:
            return backbones.EfficientNet(model_name='efficientnet_b0',
                                          pretrained=pretrained, 
                                          layers_to_freeze=layers_to_freeze)
            
    elif 'swin' in backbone_arch.lower():
        return backbones.Swin(model_name='swinv2_base_window12to16_192to256_22kft1k', 
                              pretrained=pretrained, 
                              layers_to_freeze=layers_to_freeze)

def get_aggregator(agg_arch='ConvAP', agg_config_pool1={}, agg_config_pool2={}, init_path=True):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """
    
    if 'cosplace' in agg_arch.lower():
        assert 'in_dim' in agg_config
        assert 'out_dim' in agg_config
        return aggregators.CosPlace(**agg_config)

    elif 'gem' in agg_arch.lower():
        if agg_config == {}:
            agg_config['p'] = 3
        else:
            assert 'p' in agg_config
        return aggregators.GeMPool(**agg_config)
    
    elif 'convap' in agg_arch.lower():
        assert 'in_channels' in agg_config
        return aggregators.ConvAP(**agg_config)   
    
    elif 'mixvpr' in agg_arch.lower():
        assert 'in_channels' in agg_config
        assert 'out_channels' in agg_config
        assert 'in_h' in agg_config
        assert 'in_w' in agg_config
        assert 'mix_depth' in agg_config
        return aggregators.MixVPR(**agg_config)
    
    elif 'netvlad' in agg_arch.lower(): 
        assert 'num_clusters' in agg_config
        assert 'dim' in agg_config
        assert 'vladv2' in agg_config
        #assert 'weight_file' in agg_config        
        net_vlad = netvlad.NetVLAD(**agg_config)
        return netvlad.initialize_weights(init_path, net_vlad)
        
    elif 'msnv' in agg_arch.lower(): 
        assert 'num_clusters' in agg_config_pool1; assert 'num_clusters' in agg_config_pool2;
        assert 'dim' in agg_config_pool1; assert 'dim' in agg_config_pool2; 
        assert 'vladv2' in agg_config_pool1; assert 'vladv2' in agg_config_pool2; 

        net_vlad_pool1 = netvlad.NetVLAD(**agg_config_pool1)
        net_vlad_pool2 = netvlad.NetVLAD(**agg_config_pool2)
        return netvlad.initialize_weights_msnv(init_path, net_vlad_pool1, net_vlad_pool2)     
        
        
