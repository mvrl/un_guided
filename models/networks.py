## Various networks modules are added here

# Proper attribution and source links are provided with modules

# List of networks:
# Discriminator (full image)
# Patch Discriminator

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import functools

# These modules are taken from CycleGAN and Pix2Pix in PyTorch by Zhu et al.
# source: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

class Discrim(nn.Module):
    def __init__(self):
        super(Discrim, self).__init__()
        self.pano_conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4,4),stride=(2,2), padding=(1,1))
        torch.nn.init.xavier_uniform_(self.pano_conv1.weight)
        self.pano_bn1 = nn.BatchNorm2d(64)

        self.pano_conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,4),stride=(2,2), padding=(1,1))
        torch.nn.init.xavier_uniform_(self.pano_conv2.weight)
        self.pano_bn2 = nn.BatchNorm2d(128)

        self.pano_conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(4,4),stride=(2,2), padding=(1,1))
        torch.nn.init.xavier_uniform_(self.pano_conv3.weight)
        self.pano_bn3 = nn.BatchNorm2d(256)

        self.pano_conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(4,4),stride=(2,2), padding=(1,1))
        torch.nn.init.xavier_uniform_(self.pano_conv4.weight)
        self.pano_bn4 = nn.BatchNorm2d(512)

        self.pano_conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(4,4),stride=(2,2), padding=(1,1))
        torch.nn.init.xavier_uniform_(self.pano_conv5.weight)
        self.pano_bn5 = nn.BatchNorm2d(512)

        #self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

        self.fc0 = nn.Linear(512, 256) 
        self.bn0 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(256, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64, 1)

    def forward(self, img):
        y = self.pano_conv1(img)        
        y = self.pano_bn1(y)
        y = F.relu(y)

        y = self.pano_conv2(y)
        y = self.pano_bn2(y)
        y = F.relu(y)

        y = self.pano_conv3(y)
        y = self.pano_bn3(y)
        y = F.relu(y)

        y = self.pano_conv4(y)
        y = self.pano_bn4(y)
        y = F.relu(y)

        y = self.pool(y)

        y = self.pano_conv5(y)
        y = self.pano_bn5(y)
        y = F.relu(y)

        y = self.pool(y)

        y = y.view(img.shape[0], -1)

        y = self.fc0(y)
        y = self.bn0(y)
        y = F.relu(y)

        y = self.fc1(y)
        y = self.bn1(y)
        y = F.relu(y)

        y = self.fc2(y)
        y = torch.sigmoid(y)

        return y

# Patch Discriminator from Selection GAN
# https://github.com/Ha0Tang/SelectionGAN/blob/master/models/networks.py

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        import functools
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

# LS GAN loss: https://arxiv.org/pdf/1611.04076.pdf
# Code from Selection GAN repo
# https://github.com/Ha0Tang/SelectionGAN/blob/master/models/networks.py

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)



## Multi scale discriminator
# https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
# Defines the PatchGAN discriminator with the specified arguments.
# Same as before, taken from Pix2Pix
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator_HD(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
                
        inp_list = []
        for i in range(num_D):
            if i==0:
                inp_list.append(input)
            else:
                inp_list.append(self.downsample(inp_list[i-1].clone().detach()))
                
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            
            result.append(self.singleD_forward(model, inp_list[i]))
        return result

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

