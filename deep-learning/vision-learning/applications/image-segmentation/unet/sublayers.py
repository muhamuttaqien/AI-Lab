#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[2]:


class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super(DoubleConv, self).__init__()
        
        self.double_conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv_layer(x)


# In[3]:


class Down(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super(Down, self).__init__()
        
        self.down_layer = nn.Sequential(nn.MaxPool2d(2),
                                        DoubleConv(in_channels, out_channels))
        
    def forward(self, x):
        x = self.down_layer(x)
        return x


# In[4]:


class Up(nn.Module):
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        
        super(Up, self).__init__()
        
        if bilinear:
            self.up_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up_layer = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
        self.conv_layer = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1, x2):
        
        x1 = self.up_layer(x1)
        
        x_diff = x2.size()[2] - x1.size()[2]
        y_diff = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [x_diff // 2, x_diff - x_diff // 2,
                        y_diff // 2, y_diff - y_diff // 2])
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv_layer(x)
        return x


# In[5]:


class OutputConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        
        super(OutputConv, self).__init__()
        
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.conv_layer(x)
        return(x)


# ---
