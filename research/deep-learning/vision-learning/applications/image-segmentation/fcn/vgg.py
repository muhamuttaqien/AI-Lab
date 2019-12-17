#!/usr/bin/env python
# coding: utf-8

# # Visual Geometry Group (VGG)

# In[1]:


import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.vgg import VGG


# In[2]:


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}


# In[3]:


cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# In[4]:


def make_layers(cfg, batch_norm=False):
    
    layers = []
    in_channels = 3
    
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    
    return nn.Sequential(*layers)


# ## Build VGG Architecture

# In[5]:


class VGGNet(VGG):
    
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        
        super().__init__(make_layers(cfg[model]))
        
        self.ranges = ranges[model]
        
        if pretrained: exec(f'self.load_state_dict(models.{model}(pretrained=True).state_dict())')
            
        if not requires_grad:
            for param in super().parameters(): param.requires_grad = False
                
        if remove_fc: del self.classifier
            
        if show_params:
            for name, param in self.named_parameters(): print(name, param.size())
    
    def forward(self, x):
        
        output = {}
        
        # get the output of each max-pooling layer (5 max-pool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output[f'x{idx+1}'] = x
            
        return output


# ---
