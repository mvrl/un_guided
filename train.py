# This is the main training file
# Training is done with reconstruction + GAN loss

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.autograd import Variable

from config import cfg
from data_factory import get_dataset
import numpy as np
from datetime import datetime
import os

from torch.distributions import Normal, Independent

from net_factory import get_network

from losses import TA_loss, log_prob_modified, MoG_KL_Unit_Gaussian, KL_distributions, edge_loss

from models.networks import set_requires_grad


def apply_dropout(m):   # keep drop out during validation/testing
    if type(m) == nn.Dropout:
        m.train()

def main():
    torch.autograd.set_detect_anomaly(True)

    out_dir = cfg.train.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print('Folder already exists. Are you sure you want to overwrite results?')

    print('Configuration:', cfg)

    ## getting the dataset

    # Make sure the dataset is appropriate for training:
    if cfg.data.random_target:
        raise ValueError('Training is not possible with cfg,random_target=True! Set it to false for training...')

    cfg.data.style_flipped = True
    # Training data loader
    cfg.train.mode = 'train'
    ds_train = get_dataset(cfg.train.mode)

    # validation data loader
    cfg.train.mode = 'val'
    ds_val = get_dataset(cfg.train.mode)
    print('Data loaders have been prepared!')

    ## Getting the model
    cfg.train.mode = 'train'

    ## Load netoworks
    net, discrim, gan_loss, vgg_loss, TA_net = get_network(name=cfg.model.name, machine=cfg.train.machine,
                                         need_discrim=True, discrim_name=cfg.model.discrim_name,
                                         need_GAN_loss=True, GAN_loss_name=cfg.model.GAN_loss_name,
                                         need_feature_loss=True, need_TA=True)


    chpk_name = 'data/TA_dict.pth'
    TA_net.load_state_dict(torch.load(chpk_name))
    TA_net.eval()
    
    print('Network loaded. Starting training...')

    criterion = torch.nn.L1Loss()

    param = list(net.parameters()) 
    if cfg.train.train_from_cpk:
        if not os.path.exists(cfg.train.continue_from):
            raise ValueError('To continue training, there must be a trained model in ' + cfg.train.continue_from)
        
        optim = torch.optim.Adam(param, lr=cfg.train.learning_rate, weight_decay=cfg.train.l2_reg)
        
        # load from trained 
        checkpoint = torch.load(os.path.join(cfg.train.continue_from, "trained_model_dict.pth"))
        net.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        for g in optim.param_groups:
            g['lr'] = cfg.train.learning_rate
        print('resuming training from the checkpoint in: ', cfg.train.continue_from)
    else:
        optim = torch.optim.Adam(param, lr=cfg.train.learning_rate, weight_decay=cfg.train.l2_reg)

    optim_d = torch.optim.Adam(discrim.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.l2_reg)

    # learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=cfg.train.lr_decay_every, gamma=cfg.train.lr_decay)

    ep = cfg.train.num_epochs # number of epochs

    # loss logs
    loss_train = 9999.0*np.ones(ep)
    temp_train_loss = 0
    loss_val = 9999.0*np.ones(ep)
    acc_val_log = 0*np.ones(ep)


    pp = False

    lamda_p = cfg.train.lambda_p   # loss term for unguided conditions
    lamda_q = cfg.train.lambda_p
    lamda_pq  = cfg.train.lambda_pq    
    
    # training the network
    for epoch in range(ep):
        running_loss = 0.0
        running_ctr = 0

        # switch model to training mode, clear gradient accumulators
        net.train()
        discrim.train()

        temp_train_loss = 0
        t1 = datetime.now()
        training = True

        for i, data in enumerate(ds_train, 0):
            #if i>20:
            #    break

            # reading images
            source_image = data[0].cuda()
            target_image = data[1].cuda()
            style_image = data[2].cuda()

            predicted_image, unguided_mean, unguided_sigma, posterior_mean, posterior_sigma, posterior_sample  = net(source_image, style_image, training)
                        
            unguided_distribution = Independent( Normal(unguided_mean, unguided_sigma), 1)
            
            # discriminator optimization
            optim_d.zero_grad()
            set_requires_grad(discrim, True)
            
            discrim_loss_real = gan_loss(discrim(target_image), target_is_real=True)
            discrim_loss_fake = gan_loss(discrim(predicted_image.detach()), target_is_real=False)
            
            discrim_loss = discrim_loss_fake + discrim_loss_real

            # discriminator optimization step
            discrim_loss.mean().backward()
            optim_d.step()
            
            ## network optimization
            set_requires_grad(discrim, False)
            optim.zero_grad()
            
            net_reconst_loss = 1.0*criterion(predicted_image, target_image) # reconstruction loss
            feature_loss = vgg_loss(predicted_image, target_image)      # feature loss
            ta_loss = 5.0*TA_loss(TA_net, predicted_image, target_image)
            e_loss = edge_loss(predicted_image, target_image.detach())

            pred_fake = discrim(Variable(predicted_image))
            net_gan_loss = cfg.train_lamda_gan*gan_loss(pred_fake, target_is_real=True)    # GAN loss

            unguided_loss = lamda_p * MoG_KL_Unit_Gaussian(unguided_distribution)

            likelihood_posterior_loss = (lamda_pq)*-log_prob_modified(unguided_distribution, posterior_sample).mean()   # entropy regularized log-likelihood (ours)
            
            loss = net_gan_loss.mean() + net_reconst_loss + unguided_loss + likelihood_posterior_loss + feature_loss.mean() + e_loss  + ta_loss.mean() 

            # discrim optimization step
            loss.backward()
            optim.step()
            
            temp_train_loss += loss.item()
            # print statistics
            running_loss += loss.item()
            running_ctr += 1

            if i %25 ==0:
                t2 = datetime.now()
                delta = t2 - t1
                t_print = delta.total_seconds()
                print('[%d, %5d out of %5d] loss: %f, time = %f' %
                      (epoch + 1, i + 1, len(ds_train) , running_loss / running_ctr,  t_print ))
                running_loss = 0.0
                running_ctr = 0
                t1 = t2

        temp_train_loss /= len(ds_train)

        # at the end of every epoch, calculating val loss
        net.eval()
        net.apply(apply_dropout)

        discrim.eval()

        val_loss = 0

        training = False
        with torch.no_grad():
            for i, data in enumerate(ds_val, 0):
                #if i>20:
                #    break
                
                # reading images
                source_image = data[0].cuda()
                target_image = data[1].cuda()
                style_image = data[2].cuda()

                # posterior for training
                predicted_image = net(source_image, style_image, training)

                pred_fake = discrim(Variable(predicted_image))
                net_gan_loss = cfg.train_lamda_gan*gan_loss(pred_fake, target_is_real=True)
                
                net_reconst_loss = criterion(predicted_image, target_image)
                feature_loss = vgg_loss(predicted_image, target_image)      # feature loss
                ta_loss = 5.0*TA_loss(TA_net, predicted_image, target_image)
                e_loss = edge_loss(predicted_image, target_image.detach())
                
                loss = net_reconst_loss + feature_loss.mean() + net_gan_loss.mean() + e_loss + ta_loss.mean()

                # val loss
                val_loss +=  loss.item()


            # print statistics
            val_loss = val_loss /len(ds_val)
            print('End of epoch ' + str(epoch + 1) + '. Train loss is ' + str(temp_train_loss) + '. Val loss is ' + str(val_loss))

            # Model check point
            if val_loss < np.min(loss_val, axis=0):
               model_path = os.path.join(out_dir, "trained_model_dict.pth")
               # torch.save(net.state_dict(), model_path)
                
               torch.save({'epoch': epoch, 
                'model_state_dict': net.state_dict(), 
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss
                }, model_path)


               print('Model saved at epoch ' + str(epoch+1))

            # saving losses
            loss_val[epoch] = val_loss
            loss_train[epoch] = temp_train_loss

            # Save last batch of images
            dir_name = os.path.join(out_dir, str('ep_' + str(epoch)))
            save_batch_images(source_image, predicted_image, target_image, dir_name)


        # Step the LR scheduler
        scheduler.step()  # update learning rate

    print('Training finished')
    
    # saving model
    model_path = os.path.join(out_dir, "trained_model_dict_end.pth")
    torch.save({'epoch': epoch, 
                'model_state_dict': net.state_dict(), 
                'optimizer_state_dict': optim.state_dict(),
                'loss': loss
                }, model_path)

    # saving quality net at the end

    # Saving logs
    log_name = os.path.join(out_dir, "logging.txt")
    with open(log_name, 'w') as result_file:
        result_file.write('Logging... \n')
        result_file.write('Validation loss ')
        result_file.write(str(loss_val))
        result_file.write('\nTraining loss  ')
        result_file.write(str(loss_train))
        result_file.write('\nValidation accuracy  ')
        result_file.write(str(acc_val_log))

    print('Model saved')

    # saving loss curves
    a = loss_val
    b = loss_train
    print(a[0:epoch])

    plt.figure()
    plt.plot(b[0:epoch])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training loss'])
    fname1 = str('loss_train.png')
    plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight')

    plt.figure()
    plt.plot(a[0:epoch])
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Validation Loss'])
    fname1 = str('loss_test.png')
    plt.savefig(os.path.join(out_dir, fname1), bbox_inches='tight')

    print('All done!')

if __name__ == '__main__':
    from visualize import save_batch_images
    main()
