#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


# In[2]:


class Conv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super(Conv, self).__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
            
    def forward(self, x):
        return self.conv_layer(x)


# In[3]:


class Down(nn.Module):
    
    def __init__(self, kernel_size=2):
        
        super(Down, self).__init__()
        
        self.down_layer = nn.Sequential(nn.MaxPool2d(kernel_size))
        
    def forward(self, x):
        x = self.down_layer(x)
        return x


# In[4]:


class Up(nn.Module):
    
    def __init__(self, in_channels, middle_channels, out_channels):
        
        super(Up, self).__init__()
        
        self.up_layer = nn.Sequential(Conv(in_channels, middle_channels), 
                                      nn.ConvTranspose2d(middle_channels, out_channels, 
                                                         kernel_size=3, stride=2, padding=1, output_padding=1))
    
    def forward(self, x):
        return self.up_layer(x)


# In[5]:


class OutputConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super(OutputConv, self).__init__()
        
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.conv_layer(x)
        return(x)


# ---
