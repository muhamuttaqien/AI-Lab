#!/usr/bin/env python
# coding: utf-8

# # U-Net

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[3]:


from sublayers import DoubleConv, Down, Up, OutputConv


# ## Build U-Net Architecture

# In[ ]:


class UNet(nn.Module):
    
    def __init__(self, n_channels, n_classes, bilinear=True):
        
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        self.input_layer = DoubleConv(n_channels, 64)
        self.down1_layer = Down(64, 128)
        self.down2_layer = Down(128, 256)
        self.down3_layer = Down(256, 512)
        self.down4_layer = Down(512, 512)
        
        self.up1_layer = Up(1024, 256, bilinear)
        self.up2_layer = Up(512, 128, bilinear)
        self.up3_layer = Up(256, 64, bilinear)
        self.up4_layer = Up(128, 64, bilinear)
        self.output_layer = OutputConv(64, n_classes)

    def forward(self, x):
        
        x1 = self.input_layer(x)
        x2 = self.down1_layer(x1)
        x3 = self.down2_layer(x2)
        x4 = self.down3_layer(x3)
        x5 = self.down4_layer(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        logits = self.output_layer
        return logits


# ---
