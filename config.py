# This file contains configuration for training and evaluation

from easydict import EasyDict as edict

cfg = edict()

## MODEL
cfg.model = edict()

# Main network
cfg.model.name = 'ours_full'

cfg.model.latent_dim = 32                  # length of latent (or style) vector
    
## Discriminator
cfg.model.need_discrim = True
cfg.model.discrim_name ='patch'            # options:
                                           # MultiscaleDiscriminator
                                           # Vanilla
                                           # patch

## GAN loss
cfg.model.need_GAN_loss = True
cfg.model.GAN_loss_name = 'LS'             # options
                                           # LS: least squares, as in the LS-GAN paper
                                           # BCE

## Feature loss
cfg.model.need_feature_loss = True         # currently, it only returns a VGG loss

## DATA
cfg.data = edict()
cfg.data.name = 'train_val'
                                           # 'train_val': for training and validation
                                           # 'prior': test set for evaluation of the prior net
                                           # 'posterior': # test set. eval_trained_posterior.py sepcifies if is
                                                       # same-scene or different-scene evaluation


cfg.data.image_scale = 0.5                 # deprecated now. scale factor used to resize the image
cfg.data.image_size = (256, 256)           #  (256, 256) or (320, 320),  # using PIL convention i.e. (W, H)
cfg.data.style_flipped = True


cfg.data.random_target = False             # IMPORTANT: if True, network cannot be trained!!!!

cfg.train = edict()


cfg.train.train_from_cpk = False            # True implies load model and optim dict and continue training
cfg.train.continue_from = './outputs/'     # required only if the training is to be continued

cfg.model.pretrained_ae = False              # True: load model from autoencoder training

cfg.train.machine = 'local'                # local or LCC

cfg.data.root_dir = 'data/GUSNAV'
cfg.train.batch_size = 24  # batch size
cfg.train.learning_rate = 1.2 * 1e-4   # initial learning rate for adam.
cfg.train.device_ids = [0, 1]          # specify device IDs for multi-GPU training

## Training details
# Loss weights
cfg.train.lambda_p = 0.1
cfg.train.lambda_q = 0.1
cfg.train.lambda_pq = 0.1
cfg.train_lamda_gan = 2.0

# optimization settings
cfg.train.mode = 'train'                   # details of modes listed in data_factory 
cfg.train.shuffle = True                   # shuffle training samples
cfg.train.num_epochs = 50                  # number of training epochs
cfg.train.num_workers = 12                 # workers for data loading
cfg.train.l2_reg = 1.0*1e-5                # L2 regularization
cfg.train.lr_decay = 0.9                   # LR decay, this is the scale factor, applied to the current LR
cfg.train.lr_decay_every = 5               # Apply LR decay after this many epochs

cfg.train.out_dir = './outputs/1'       # [225] Ours 512x512; adjusting loss weights