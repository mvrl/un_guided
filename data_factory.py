# All the dataloaders are implemented in this file.

import torch
import PIL
from PIL import Image
from config import cfg
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_function
import random
from natsort import natsorted
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from skimage import transform as sk_transform
from skimage.io import imread


import csv
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
class dataset_train_val(Dataset):
    # This is used during training
    
    def __init__(self, data_folder, split_dir, mode):

        fname_source = mode +'_source.txt'
        fname_target = mode +'_target.txt'
        
        self.image_size = cfg.data.image_size
        self.mode = mode
        self.flip_style = cfg.data.style_flipped

        full_name_target = os.path.join(split_dir, fname_target) # files placed in the code repo
        full_name_source = os.path.join(split_dir, fname_source)

        # source list
        with open(full_name_source, 'r') as myfile:
            self.full_list_source = [j[:-1] for j in myfile]

        # target list
        with open(full_name_target, 'r') as myfile:
            self.full_list_target = [j[:-1] for j in myfile]

        self.root_dir = os.path.join(data_folder)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet
        self.crop = transforms.RandomCrop(size=self.image_size, pad_if_needed=True, padding_mode='reflect')
        
        self.style_flip = transforms.RandomHorizontalFlip(p=0.5)

        self.random_target = cfg.data.random_target
        if self.random_target and mode=='train':
            raise ValueError('Random target style image... this setting CANNOT be used for training!!!!')

    def __len__(self):
        return len(self.full_list_target)

    def __getitem__(self, idx):
        # read image names
        target_name = self.full_list_target[idx]

        scene = target_name.split('/')[1]   # read scene name from the target scene

        # find a source image from the same scene as the target scene
        while True:
            rand_ind = np.random.randint(len(self.full_list_source))
            source_name = self.full_list_source[rand_ind]
            scene_now = source_name.split('/')[1]
            if scene_now == scene:
                break

        # read images
        target_image = Image.open(os.path.join(self.root_dir, target_name))  # open image
        source_img = Image.open(os.path.join(self.root_dir, source_name))  # open image

        # Random cropping
        image_w, image_h = source_img.size
        image_aspect = image_w/image_h
        
        if self.mode == 'train':
            # step 1: first resize so that height is 1.5*target height, maintaining the aspect ratio
            target_h = int(1.25*self.image_size[1])
            target_w = int(image_aspect * target_h)
            if target_w < self.image_size[0]:
                target_w = int(1.25*self.image_size[0])
                target_h = int( target_w /image_aspect )

            target_image = target_image.resize((target_w, target_h), PIL.Image.LANCZOS) 
            source_img = source_img.resize((target_w, target_h), PIL.Image.LANCZOS)
            
            # random flip
            if random.random() > 0.5:
                source_img = transforms_function.hflip(source_img)
                target_image = transforms_function.hflip(target_image)

        else:
            # step 1: first resize so that height is desired, maintaining the aspect ratio
            target_h = int(self.image_size[1])
            target_w = int(image_aspect * target_h)
            if target_w < self.image_size[0]:
                target_w = int(1.25*self.image_size[0])
                target_h = int( target_w /image_aspect )

            target_image = target_image.resize((target_w, target_h), PIL.Image.LANCZOS) 
            source_img = source_img.resize((target_w, target_h), PIL.Image.LANCZOS)

        # step 2: random crop to the desired size
        i, j, h, w = transforms.RandomCrop.get_params(target_image, output_size=
                                                     (cfg.data.image_size[1],cfg.data.image_size[0]))

        source_img = transforms_function.crop(source_img, i, j, h, w)   # apply transform
        target_image = transforms_function.crop(target_image, i, j, h, w)   # apply the same transform
            
        # should the target image be flipped to be used as a style image?
        if self.flip_style:
            style_image = transforms_function.hflip(target_image)
        else:
            style_image = target_image
        
        # Convert to tensor
        target_image = self.to_tensor(target_image)  # convert to tensor
        source_img = self.to_tensor(source_img)  # convert to tensor
        style_image = self.to_tensor(style_image)

        return source_img, target_image, style_image



class dataset_unguided(Dataset):
    # This is the dataset for unguided evaluation
    def __init__(self, data_folder, split_dir, mode):

        fname_source = 'source_same_scene.txt' 
        fname_target = 'target_same_scene.txt'

        full_name_target = os.path.join(split_dir, fname_target)  # files placed in the code repo
        full_name_source = os.path.join(split_dir, fname_source)

        # source list
        with open(full_name_source, 'r') as myfile:
            self.full_list_source = [j[:-1] for j in myfile]

        # target list
        with open(full_name_target, 'r') as myfile:
            self.full_list_target = [j[:-1] for j in myfile]

        self.image_size = cfg.data.image_size
        self.root_dir = os.path.join(data_folder)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet
        self.crop = transforms.RandomCrop(size=self.image_size, pad_if_needed=True, padding_mode='reflect')
        self.crop_final = transforms.CenterCrop(size=self.image_size)

        self.scene_ids = ['24_hour_Timelapse_of_the_Gardiner_Expressway', '00000270', '00000573', '911_2012_Timelapse', '00008728', '011711_TL_10', '00017609', '00017632', '00017659', '00017664', '00018478', '00018505', '00018962', '44973127', '90000009', 'Clear_Lake_Time_Lapse', 'First_time_lapse', 'SEATTLE_DOWNTOWN_AT_NIGHT___CITY_LIGHTS', 'SEATTLE_STOCK_FOOTAGE']

    def __len__(self):
        return len(self.full_list_source)

    def __getitem__(self, idx):
        # read image names
        source_name = self.full_list_source[idx]
        source_img = Image.open(os.path.join(self.root_dir, source_name))  # open image

        # resize image
        image_w, image_h = source_img.size
        image_aspect = image_w/image_h
        
        target_h = int(self.image_size[1]) 
        target_w = int(image_aspect * target_h)
        if target_w < self.image_size[0]:
            target_w = int(1.25*self.image_size[0])
            target_h = int( target_w /image_aspect )
            
        source_img = source_img.resize((target_w, target_h), PIL.Image.LANCZOS)
        
        # center crop
        source_img = self.crop_final(source_img)
        
        # convert to tensor
        source_img = self.to_tensor(source_img)  # convert to tensor

        scene_now = source_name.split('/')[1]
        
        target_names_this_scene = [i for i in self.full_list_target if '_'.join(i.split('/')[1].split('_')) == scene_now]
        n_images_this_scene = len(target_names_this_scene)  # number of images of a scene
        
        
        scene_id = self.scene_ids.index(scene_now)

        return source_img, scene_id, n_images_this_scene


class dataset_guided(Dataset):
    #  this dataset is for posterior evaluation
    def __init__(self, data_folder, split_dir, mode):

        if mode=='test_different':
            fname_source = 'source_different_scene.txt'  # source
            fname_stlye = 'style_different_scene.txt'  # style
            fname_target = 'target_different_scene.txt'  # target
            
        elif mode=='test_same':
            fname_source = 'source_same_scene.txt'  # source
            fname_stlye = 'target_same_scene.txt'  # style
            fname_target = 'target_same_scene.txt'  # target
        
        elif mode=='time_lapse':
            fname_source = 'source_time_lapse.txt'  # source
            fname_stlye = 'target_time_lapse.txt'  # style
            fname_target = 'target_time_lapse.txt'  # target            

        self.mode = mode
        self.flip_style = cfg.data.style_flipped
        print('flipping style:', self.flip_style)
        
        self.image_size = cfg.data.image_size
        self.root_dir = os.path.join(data_folder)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet
        self.crop = transforms.RandomCrop(size=self.image_size, pad_if_needed=True, padding_mode='reflect')
        self.crop_final = transforms.CenterCrop(size=self.image_size)

        full_name_source = os.path.join(split_dir, fname_source)
        full_name_style = os.path.join(split_dir, fname_stlye)
        full_name_target = os.path.join(split_dir, fname_target) # files placed in the code repo

        # source list
        with open(full_name_source, 'r') as myfile:
            self.full_list_source = [j[:-1] for j in myfile]

        # style list
        with open(full_name_style, 'r') as myfile:
            self.full_list_style = [j[:-1] for j in myfile]

        # target list
        with open(full_name_target, 'r') as myfile:
            self.full_list_target = [j[:-1] for j in myfile]

        self.random_target = cfg.data.random_target

    def __len__(self):
        return len(self.full_list_target)

    def __getitem__(self, idx):
        # Read image names
        source_name = self.full_list_source[idx]
        style_name = self.full_list_style[idx]
        target_name = self.full_list_target[idx]

        # read images
        source_img = Image.open(os.path.join(self.root_dir, source_name))    # open image
        style_img = Image.open(os.path.join(self.root_dir, style_name))      # open image
        target_image = Image.open(os.path.join(self.root_dir, target_name))  # open image

        ## resizing and cropping to get to the final size
        # compute aspect ratios
        image_w_source, image_h_source = source_img.size   # source image
        image_aspect_source = image_w_source/image_h_source
        
        image_w_style, image_h_style = style_img.size   # style image
        image_aspect_style = image_w_style/image_h_style

        # resize source and target
        target_h = int(self.image_size[1])
        target_w = int(image_aspect_source * target_h)
        if target_w < self.image_size[0]:
            target_w = int(1.25*self.image_size[0])
            target_h = int( target_w /image_aspect_source )

        # source and target have the same size, resizing them with same settings
        target_image = target_image.resize((target_w, target_h), PIL.Image.LANCZOS) 
        source_img = source_img.resize((target_w, target_h), PIL.Image.LANCZOS)
        
        # resize style image
        target_h = int(self.image_size[1])
        target_w = int(image_aspect_style * target_h)
        if target_w < self.image_size[0]:
            target_w = int(1.25*self.image_size[0])
            target_h = int( target_w /image_aspect_style )

        style_img = style_img.resize((target_w, target_h), PIL.Image.LANCZOS) 
        
        # center crop all images
        source_img = self.crop_final(source_img)
        target_image = self.crop_final(target_image)
        style_img = self.crop_final(style_img)
                
        # should the target image be flipped to be used as a style image?
        if self.flip_style:
            style_img = transforms_function.hflip(style_img)            
        
        # Convert to tensor
        source_img = self.to_tensor(source_img)  # convert to tensor
        style_img = self.to_tensor(style_img)  # convert to tensor
        target_image = self.to_tensor(target_image)  # convert to tensor

        return source_img, target_image, style_img


def get_dataset(mode):
    # Get dataset object by its mode

    # Here is the full list of dataset modes:
    
    # [Training modes]
    # train: training set
    # val: new val set of unseen scenes
    
    # [Evaluation modes]
    # test_same : test set, guided same-scene
    # test_different: test set, guided cross-scene
    # test_unguided: test set, unguided synthesis
    # time_lapse: for time-lapse generation 

    data_folder = cfg.data.root_dir     # set data directory

    if mode == 'train':      
        split_dir = 'split_files'
        ds = dataset_train_val(data_folder, split_dir, 'train')
    elif mode == 'val':      
        split_dir = 'split_files'
        ds = dataset_train_val(data_folder, split_dir, 'val')
    
    elif mode == 'test_unguided':    # for evaluation of prior images
        split_dir = 'split_files/guided_same_scene'
        ds = dataset_unguided(data_folder, split_dir, mode)
    
    elif mode == 'test_same':
        split_dir = 'split_files/guided_same_scene'
        ds = dataset_guided(data_folder, split_dir, mode)
    elif mode == 'test_different':
        split_dir = 'split_files/guided_different_scene'
        ds = dataset_guided(data_folder, split_dir, mode)
    elif mode == 'time_lapse':
        split_dir = 'split_files/time_lapse'
        ds = dataset_guided(data_folder, split_dir)
            
    else:
        raise ValueError('This dataset is not available:', cfg.data.name)

    # prepare the PyTorch data loader
    ds_final = torch.utils.data.DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle, num_workers=cfg.train.num_workers, drop_last=True, worker_init_fn=worker_init_fn)

    return ds_final
