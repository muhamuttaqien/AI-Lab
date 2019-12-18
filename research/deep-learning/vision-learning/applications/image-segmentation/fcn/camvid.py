import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import utils

class CamVid(torch.utils.data.Dataset):
    """
    CamVid dataset from https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid/
    """
    
    # training dataset root directories
    train_dir = 'train'
    train_label_dir = 'trainannot'
    
    # validation dataset root directories
    valid_dir = 'val'
    valid_label_dir = 'valannot'
    
    # test dataset root directories
    test_dir = 'test'
    test_label_dir = 'testannot'
    
    # images extension
    img_extension = '.png'
    
    # default encoding for pixel value, class name and class color
    from collections import OrderedDict
    
    color_encoding = OrderedDict([
        ('sky', (128, 128, 128)),
        ('building', (128, 0, 0)),
        ('pole', (192, 192, 128)),
        ('road_marking', (255, 69, 0)),
        ('road', (128, 64, 128)),
        ('pavement', (60, 40, 222)),
        ('tree', (128, 128, 0)),
        ('sign_symbol', (192, 128, 128)),
        ('fence', (64, 64, 128)),
        ('car', (64, 0, 128)),
        ('pedestrian', (64, 64, 0)),
        ('bicyclist', (0, 128, 192)),
        ('unlabeled', (0, 0, 0))
    ])
    
    def __init__(self, 
                 root_dir, 
                 mode='train', 
                 data_transform=None, 
                 label_transform=None, 
                 loader=utils.pil_loader):
        
        self.root_dir = root_dir
        self.mode = mode
        self.data_transform = data_transform
        self.label_transform = label_transform
        self.loader = loader
        
        # get the training data and labels filepaths
        if self.mode.lower() == 'train':
            self.train_data = utils.get_files(os.path.join(root_dir, self.train_dir), 
                                                           extension_filter=self.img_extension)
            
            self.train_labels = utils.get_files(os.path.join(root_dir, self.train_label_dir), 
                                                             extension_filter=self.img_extension)
            
        # get the validation data and labels filepaths
        elif self.mode.lower() == 'valid':
            self.valid_data = utils.get_files(os.path.join(root_dir, self.valid_dir), 
                                                           extension_filter=self.img_extension)
            
            self.valid_labels = utils.get_files(os.path.join(root_dir, self.valid_label_dir), 
                                                             extension_filter=self.img_extension)
            
        # get the test data and labels filepaths
        elif self.mode.lower() == 'test':
            self.test_data = utils.get_files(os.path.join(root_dir, self.test_dir), 
                                                          extension_filter=self.img_extension)
            
            self.test_labels = utils.get_files(os.path.join(root_dir, self.test_label_dir), 
                                                            extension_filter=self.img_extension)
        
        else:
            raise RuntimeError('Unexpected dataset mode. Supported modes are: train, valid and test')
        
    def __getitem__(self, index):
        
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[index]
            
        elif self.mode.lower() == 'valid':
            data_path, label_path = self.valid_data[index], self.valid_labels[index]
        
        elif self.mode.lower() == 'test':
            data_path, label_path = self.test_data[index], self.test_labels[index]
        
        else:
            raise RuntimeError('Unexpected dataset mode. Supported modes are: train, valid and test')
        
        image, label = self.loader(data_path, label_path)
        
        if self.data_transform is not None:
            image = self.data_transform(image)
        
        if self.label_transform is not None:
            label = self.label_transform(label)
            
        return image, label
    
    def __len__(self):
        
        if self.mode.lower() == 'train':
            return len(self.train_data)
        
        elif self.mode.lower() == 'valid':
            return len(self.valid_data)
        
        elif self.mode.lower() == 'test':
            return len(self.test_data)
        else:
            raise RuntimeError('Unexpected dataset mode. Supported modes are: train, valid and test')