# Saves image results on the disk. This code is for clean data without fusion

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

from config import cfg
from data_factory import get_dataset
from net_factory import get_network
from torch.distributions import Normal, Independent
from torchvision.utils import save_image

def un_normalize(normalized, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # normalized is of size HxWx3
    normalized[:,:,0] = normalized[:,:,0]*std[0] + mean[0]
    normalized[:,:,1] = normalized[:,:,1]*std[1] + mean[1]
    normalized[:,:,2] = normalized[:,:,2]*std[2] + mean[2]
    return normalized


def save_batch_images(fused_image, predicted_image, target_image, out_dir=None, normalized=False):

    ctr = 0
    num_fig = min(fused_image.shape[0], 12)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print('overwriting results in ' + out_dir)

    for k in range(num_fig):
        plt.figure()
        if normalized:
            plt.imshow(un_normalize(fused_image[k, :, :, :].permute(1,2,0).detach().cpu()))  # un_normalize
        else:
            plt.imshow(fused_image[k, :, :, :].permute(1,2,0).detach().cpu())  # un_normalize
        plt.axis('off')
        fname1 = str(str(ctr) + '_source' + '.jpg')  # naming ans saving
        plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight', pad_inches=0.0)
        plt.close()

        plt.figure()
        plt.imshow((predicted_image[k, :, :, :].permute(1,2,0).detach().cpu()))
        plt.axis('off')
        fname1 = str(str(ctr) + '_predicted' + '.jpg')  # naming ans saving
        plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight', pad_inches=0.0)
        plt.close()

        plt.figure(dpi=300)
        a = target_image[k, :, :, :].permute(1,2,0).cpu().detach()
        plt.axis('off')
        plt.imshow((a))
        fname1 = str(str(ctr) + '_target' + '.jpg')  # naming ans saving
        plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight', pad_inches=0.0)
        plt.close()


        #"""
        plt.subplot(3, 1, 1)
        plt.imshow((fused_image[k, :, :, :].permute(1,2,0).detach().cpu()))
        plt.axis('off')
        plt.title('source')

        plt.subplot(3, 1, 2)
        plt.imshow((predicted_image[k, :, :, :].permute(1,2,0).detach().cpu()))
        plt.axis('off')
        plt.title('predicted')

        plt.subplot(3, 1, 3)
        plt.imshow((target_image[k, :, :, :].permute(1,2,0).detach().cpu()))
        plt.axis('off')
        plt.title('target')

        fname1 = str(str(ctr) + '_0_combined' + '.jpg')  # naming ans saving
        plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight', pad_inches=0.0)
        plt.close()

        ctr += 1
        plt.clf()

    return 0.0

def apply_dropout(m):   # keep drop out during validation/testing
    if type(m) == nn.Dropout:
        m.train()

if __name__ == "__main__":
    out_dir = cfg.train.out_dir
    
    model_best = False       # if True, the checkpoint with best val loss is selected
                             # if False, the last checkpoint is used

    mode = 'test_different'               
                                # test_same :       test set, style image from the same scene
                                # test_different:   test set, style image from a different scene
                                # test_unguided: unguided synthesis, only a source image is provided

    if mode == 'test_different':
        folder_name = 'images_guided_diff_'
        max_batches = 5
        images_per_scene = 10
    elif mode == 'test_same':
        folder_name = 'images_guided_same_'
        max_batches = 3
        images_per_scene = 10
    elif mode == 'test_unguided':
        folder_name = 'images_unguided_'
        
        cfg.train.batch_size = 1
        images_per_scene = 50
        scenes_done = []
        max_batches = 100
    else:
        raise ValueError('Unknown inference mode:', mode)
        
    mode_str = 'best' if model_best else 'end'
    folder_name += mode_str

    if not os.path.exists(out_dir):
        raise ValueError(
            'The folder with a trained model does not exist. Make sure to set the correct folder variable cfg.train.out_dir in config.py')

    if not os.path.exists(os.path.join(out_dir,folder_name)):
        os.makedirs(os.path.join(out_dir, folder_name))
    else:
        print('overwriting results in: ', os.path.join(out_dir, folder_name))

    cfg.train.shuffle = True

    ds_test = get_dataset(mode)
    print('Data loaders have been prepared!')

    ## Load netoworks
    net, discrim, gan_loss, vgg_loss = get_network(name=cfg.model.name, machine=cfg.train.machine,
                                                   need_discrim=True, discrim_name=cfg.model.discrim_name,
                                                   need_GAN_loss=True, GAN_loss_name=cfg.model.GAN_loss_name,
                                                   need_feature_loss=True)
    
    if model_best:
        checkpoint = torch.load(os.path.join(out_dir, "trained_model_dict.pth"))   # best
    else:
        checkpoint = torch.load(os.path.join(out_dir, "trained_model_dict_end.pth"))    # last
    
    net.load_state_dict(checkpoint['model_state_dict'])

    net.eval()

    ctr = 0
    with torch.no_grad():
        for i, data in enumerate(ds_test, 0):
            if i>max_batches:
                break
            # reading images
            source_image = data[0].cuda()
            target_image = data[1].cuda()
            style_image = data[2].cuda()
            
            # guided synthesis
            if mode == 'test_different' or mode=='test_same':
                training = True
                style_image = data[2].cuda()
                predicted_image = []
                for j in range(images_per_scene):
                    return_list = net(source_image, style_image, training)  # full model
                    if not isinstance(return_list, tuple):
                        predicted_image.append(return_list)
                    else:
                        predicted_image.append(return_list[0])
                
            # unguided synthesis
            else:                                   
                training = False
                dist = Independent(Normal(torch.zeros(1, 8), torch.ones(1, 8)), 1)
                feedback_vector = dist.sample()

                scene_id = data[1]
                
                if scene_id[0] not in scenes_done:
                    scenes_done.append(scene_id[0])
                    print('new scene: ', scene_id[0])
                else:
                    continue
                
                predicted_image = []
                for j in range(images_per_scene):
                        
                    predicted_image1 = net(source=source_image, style=None, training=training)
                    predicted_image.append(predicted_image1)

            my_dpi = 1200

            for k in range(target_image.shape[0]):

                if mode == 'test_unguided': 
                    plt.figure(dpi=300)
                    plt.subplot(3, 4, 1)
                    plt.axis('off')
                    plt.imshow(source_image[k, :, :, :].permute(1,2,0).detach().cpu().numpy())
                    
                    for j in range(11):
                        plt.subplot(3, 4, j+2)
                        plt.axis('off')
                        plt.imshow(predicted_image[j][0, :, :, :].permute(1,2,0).detach().cpu())
                    fname1 = str(str(ctr) + '_result' + '.jpg')  # naming and saving
                    plt.savefig(os.path.join(out_dir, folder_name, fname1), bbox_inches='tight')
                    plt.close()
                    
                    fname1 = os.path.join(out_dir, folder_name, str(str(ctr) + '_source' + '.jpg'))
                    save_image(source_image[k, :, :, :].cpu(), fname1)

                    for j in range(len(predicted_image)):
                        fname1 = os.path.join(out_dir, folder_name, str(str(ctr) + '_pred_'+ str(j) + '.jpg'))
                        save_image(predicted_image[j][0, :, :, :].cpu(), fname1)

                else:
                    # Guided
                    
                    # Image results
                    plt.figure(dpi=300)
                    plt.subplot(3, 3, 1)
                    plt.axis('off')
                    plt.imshow(source_image[k, :, :, :].permute(1,2,0).detach().cpu().numpy())
                    plt.title('source')
                    plt.subplot(3, 3, 2)
                    plt.axis('off')
                    plt.imshow(style_image[k, :, :, :].permute(1,2,0).detach().cpu().numpy())
                    plt.title('style')
                    plt.subplot(3, 3, 3)
                    plt.axis('off')
                    plt.imshow(target_image[k, :, :, :].permute(1,2,0).detach().cpu().numpy())
                    plt.title('target')
                    
                    for j in range(6):
                        plt.subplot(3, 3, j+4)
                        plt.axis('off')
                        plt.imshow(predicted_image[j][k, :, :, :].permute(1,2,0).detach().cpu())
                    fname1 = str(str(ctr) + '_result' + '.jpg')  # naming and saving
                    plt.savefig(os.path.join(out_dir, folder_name, fname1), bbox_inches='tight')
                    plt.close()
                    
                    # Individual images
                    fname1 = os.path.join(out_dir, folder_name, str(str(ctr) + '_source' + '.jpg'))
                    save_image(source_image[k, :, :, :].cpu(), fname1)

                    fname1 = os.path.join(out_dir, folder_name, str(str(ctr) + '_style' + '.jpg'))
                    save_image(style_image[k, :, :, :].cpu(), fname1)
                            
                    for j in range(len(predicted_image)):
                        fname1 = os.path.join(out_dir, folder_name, str(str(ctr) + '_pred_'+ str(j) + '.jpg'))
                        save_image(predicted_image[j][k, :, :, :].cpu(), fname1)
                    
                    fname1 = os.path.join(out_dir, folder_name, str(str(ctr) + '_target' + '.jpg'))
                    save_image(target_image[k, :, :, :].cpu(), fname1)

                ctr += 1
    print('Finished saving images....')
