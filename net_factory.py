# This file reads the network name from config file and returns the desired networks


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.distributions import Normal, Independent
from config import cfg
from collections import OrderedDict

from models.networks import Discrim, NLayerDiscriminator, MultiscaleDiscriminator, GANLoss, VGGLoss

def count_trainable_parameters(model):  # to count trainable parameters
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_network(name, machine, need_discrim=False, discrim_name=None, need_GAN_loss=False, GAN_loss_name=None, need_feature_loss = False, need_TA=False):

    all_networks = []

    # networks
    if name == 'ours_full':
        from models.gusnav_full import gusnav
        net = gusnav(latent_dim=cfg.model.latent_dim)        

    else:
        raise ValueError('Model not available')

    all_networks.append(net)
    print('network trainable parameters: ', count_trainable_parameters(net))

    # discriminator
    if need_discrim:
        if discrim_name == 'MultiscaleDiscriminator':
            discrim = MultiscaleDiscriminator(input_nc=3)
        elif discrim_name == 'Vanilla':
            discrim = Discrim()
        elif discrim_name == 'patch':
            discrim = NLayerDiscriminator(input_nc=3)

        all_networks.append(discrim)

    # GAN Loss
    if need_GAN_loss:
        if GAN_loss_name == 'LS':
            Gan_loss = GANLoss(use_lsgan=True)
        elif GAN_loss_name == 'BCE':
            Gan_loss = GANLoss(use_lsgan=False)

        all_networks.append(Gan_loss)

    # feature loss
    if need_feature_loss:
        vgg_loss = VGGLoss([0])
        all_networks.append(vgg_loss)

    # transient attribute loss
    if need_TA:
            from models.trans_attr import TA_ResNet
            TA_net = TA_ResNet()
            all_networks.append(TA_net)
            
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for k in all_networks:
        k.to(device)

    final_networks_list = []

    # attach to CUDA and DaraParallel, as specified in settings
    if len(cfg.train.device_ids)>1:
        for j in all_networks:
            j = torch.nn.DataParallel(j, device_ids=cfg.train.device_ids)
            final_networks_list.append(j)
    else:
        for j in all_networks:
            final_networks_list.append(j)

    # return all network modules
    return final_networks_list