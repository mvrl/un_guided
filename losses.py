# Starting to move new loss functions here

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl, MultivariateNormal
import numpy as np
from utils import init_weights,init_weights_orthogonal_normal, l2_regularisation

## MMD Loss
# https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
# https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb

def tv_loss(img):
    # source: https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/style_transfer_pytorch.py
    # modifications by me
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.mean(torch.pow(img[:,:,:,:-1] - img[:,:,:,1:], 2))
    h_variance = torch.mean(torch.pow(img[:,:,:-1,:] - img[:,:,1:,:], 2))
    loss = h_variance + w_variance
    return loss

def edge_loss(pred, target):
    # rough idea from: https://github.com/shaoanlu/faceswap-GAN/blob/master/networks/losses.py#L95
    # modifications by me
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    pred_dx = pred[:,:,:,:-1] - pred[:,:,:,1:]  # torch.abs
    pred_dy = pred[:,:,:-1,:] - pred[:,:,1:,:]
    
    target_dx = target[:,:,:,:-1] - target[:,:,:,1:]
    target_dy = target[:,:,:-1,:] - target[:,:,1:,:]
    
    loss_dx = torch.mean( torch.abs(pred_dx - target_dx) )
    loss_dy = torch.mean( torch.abs(pred_dy - target_dy) )
    
    loss = loss_dx + loss_dy
    return loss


def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)/float(dim)
    return torch.exp(-kernel_input)


def compute_mmd_simple(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


def compute_kernel_simple(x, y):
    dim = x.size(1)
    kernel_input = (x - y).pow(2).mean() / float(dim)
    return torch.exp(-kernel_input)


def MMD_loss(x, y):

    #x_kernel = compute_kernel_simple(x, x)
    #y_kernel = compute_kernel_simple(y, y)
    #xy_kernel = compute_kernel_simple(x, y)
    #mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

    return mmd

#def Batch_KL_Unit_Gaussian(distribution):
#    sample = distribution.rsample()
def Batch_KL_Unit_Gaussian(sample):

    # compute batch-wise mean and sigma
    sample_mu = sample.mean(dim=0)
    sample_sigma = sample.std(dim=0)

    batch_distribution = Independent(Normal(sample_mu, sample_sigma), 1)

    unit_gaussian = Independent( Normal(torch.zeros_like(sample_mu), torch.ones_like(sample_sigma)   )  ,1)

    loss = kl.kl_divergence(unit_gaussian, batch_distribution)
    #loss = kl.kl_divergence(batch_distribution, unit_gaussian)

    return loss

def KL_distributions(mean1, sigma1, mean2, sigma2):
    # KL divergence between 2 distributions
    distribution1 = Independent( Normal(  mean1,  sigma1), 1) 
    distribution2 = Independent( Normal(  mean2,  sigma2), 1) 
                        
    kl_loss = kl.kl_divergence(distribution1, distribution2)
    
    return kl_loss
    

def strict_KL(distribution):
    # computes KL divergence with unit Gaussian
    
    # compute the dimension
    #distribution_mean = distribution.mean

    unit_gaussian = Independent( Normal(  torch.zeros_like(distribution.mean), torch.ones_like(distribution.mean) ) , 1)

    loss = kl.kl_divergence(unit_gaussian, distribution)
    #loss = kl.kl_divergence(batch_distribution, unit_gaussian)

    return loss

def log_prob_modified(distribution, sample):
    var = distribution.stddev**2
    #log_scale = distribution.stddev.log()

    log_prob = -( (sample - distribution.mean) ** 2) / (2 * var)
    return log_prob

####Mixture of Gaussians Approximation
def MoG(gaussians):
    means = gaussians.mean
    #print('means:', means.shape)
    variances = torch.sum(gaussians.variance,dim=0)  # .to(device)
    size = means.shape[0]
    b = (torch.sum(means, dim=0)/size)           #.to(device)
    b_temp = torch.mm(b.unsqueeze(1) ,  b.unsqueeze(0))
    B = ( (torch.diag(variances)/size + torch.mm(means.transpose(0,1),means)/size - b_temp ) )#.unsqueeze(0)
    # b = b.unsqueeze(0)
    #print('b:', b.shape)
    #print('covariance B:', b.shape)
    return MultivariateNormal(b, covariance_matrix=B)

def MoG_KL_Unit_Gaussian(distribution):
    collapsed_multivariate = MoG(distribution)   # Mixture of Gaussian modeling
    
    unit_cov = torch.eye(collapsed_multivariate.mean.shape[-1]).cuda()  # .unsqueeze(0)
    #print('cov matrix shape:', unit_cov.shape)
    unit_Gaussian = MultivariateNormal(torch.zeros_like(collapsed_multivariate.mean),  unit_cov )
    
    loss = kl.kl_divergence(unit_Gaussian, collapsed_multivariate)
    
    return loss
    
def EMD(mean1, sigma1, mean2, sigma2):
    var1 = sigma1*sigma1
    var2 = sigma2*sigma2
    m = torch.norm(mean1 - mean2, dim=1).pow(2)
    trace = torch.sum(var1 + var2 - 2*(var1*var2)**.5, dim=1)
    trace = torch.max(trace, torch.tensor([0.0]).cuda())
    return (m + trace)**.5
    

def TA_loss(TA_net, image1, image2):
    TA_1 = TA_net(image1)
    TA_2 = TA_net(image2)
    return F.mse_loss(TA_1, TA_2)