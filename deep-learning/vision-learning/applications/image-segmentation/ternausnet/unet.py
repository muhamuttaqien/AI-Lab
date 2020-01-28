#!/usr/bin/env python
# coding: utf-8

# # U-Net (VGG11, VGG16)

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


# In[2]:


from sublayers import Conv, Down, Up_11, Up_16, OutputConv


# ## Build [U-Net](https://github.com/ternaus/TernausNet/blob/master/unet_models.py) Architecture

# In[3]:


class UNet_11(nn.Module):
    
    def __init__(self, num_classes, num_filters=32, pretrained=False):
    
        super(UNet_11, self).__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.encoder = models.vgg11(pretrained=pretrained).features
        
        self.conv1_layer = self.encoder[0]
        self.relu = self.encoder[1]
        self.conv2_layer = self.encoder[3]
        self.conv3s_layer = self.encoder[6]
        self.conv3_layer = self.encoder[8]
        self.conv4s_layer = self.encoder[11]
        self.conv4_layer = self.encoder[13]
        self.conv5s_layer = self.encoder[16]
        self.conv5_layer = self.encoder[18]
        
        self.down = Down(kernel_size=2)
        
        self.bottleneck_layer = Up_11(num_filters*8*2, num_filters*8*2, num_filters*8)
        
        self.up1_layer = Up_11(num_filters*(16+8), num_filters*8*2, num_filters*8)
        self.up2_layer = Up_11(num_filters*(16+8), num_filters*8*2, num_filters*4)
        self.up3_layer = Up_11(num_filters*(8+4), num_filters*4*2, num_filters*2)
        self.up4_layer = Up_11(num_filters*(4+2), num_filters*2*2, num_filters)
        self.up5_layer = Conv(num_filters*(2+1), num_filters)
        
        self.output_layer = OutputConv(num_filters, num_classes)

    def forward(self, x):
        
        x1 = self.relu(self.conv1_layer(x))
        x2 = self.relu(self.conv2_layer(self.down(x1)))
        x3s = self.relu(self.conv3s_layer(self.down(x2)))
        x3 = self.relu(self.conv3_layer(x3s))
        x4s = self.relu(self.conv4s_layer(self.down(x3)))
        x4 = self.relu(self.conv4_layer(x4s))
        x5s = self.relu(self.conv5s_layer(self.down(x4)))
        x5 = self.relu(self.conv5_layer(x5s))
        
        x_bottleneck = self.bottleneck_layer(self.down(x5))
        
        x = self.up1_layer(torch.cat([x_bottleneck, x5], 1))
        x = self.up2_layer(torch.cat([x, x4], 1))
        x = self.up3_layer(torch.cat([x, x3], 1))
        x = self.up4_layer(torch.cat([x, x2], 1))
        x = self.up5_layer(torch.cat([x, x1], 1))
        
        logits = self.output_layer(x)
        return logits


# In[4]:


class UNet_16(nn.Module):
    
    def __init__(self, num_classes, num_filters=32, pretrained=False):
        
        super(UNet_16, self).__init__()
        self.num_classes = num_classes
        self.num_filters = num_filters

        self.encoder = models.vgg11(pretrained=pretrained).features
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv1_layer = nn.Sequential(self.encoder[0],
                                         self.relu,
                                         self.encoder[2],
                                         self.relu)
        
        self.conv2_layer = nn.Sequential(self.encoder[5],
                                         self.relu,
                                         self.encoder[7],
                                         self.relu)
        
        self.conv3_layer = nn.Sequential(self.encoder[10],
                                         self.relu,
                                         self.encoder[12],
                                         self.relu,
                                         self.encoder[14],
                                         self.relu)
        
        self.conv4_layer = nn.Sequential(self.encoder[17],
                                         self.relu,
                                         self.encoder[19],
                                         self.relu,
                                         self.encoder[21],
                                         self.relu)
        
        self.conv5_layer = nn.Sequential(self.encoder[24],
                                         self.relu,
                                         self.encoder[26],
                                         self.relu,
                                         self.encoder[28],
                                         self.relu)
        
        self.down = Down(kernel_size=2)
        
        self.bottleneck_layer = Up_16(512, num_filters*8*2, num_filters*8)
        
        self.up1_layer = Up_16(512+num_filters*8, num_filters*8*2, num_filters*8)
        self.up2_layer = Up_16(512+num_filters*8, num_filters*8*2, num_filters*8)
        self.up3_layer = Up_16(256+num_filters*8, num_filters*4*2, num_filters*2)
        self.up4_layer = Up_16(128+num_filters*2, num_filters*2*2, num_filters)
        self.up5_layer = Conv(64+num_filters, num_filters)
        
        self.output_layer = OutputConv(num_filters, num_classes)
        
    def forward(self, x):

        x1 = self.conv1_layer(x)
        x2 = self.conv2_layer(self.down(x1))
        x3 = self.conv3_layer(self.down(x2))
        x4 = self.conv4_layer(self.down(x3))
        x5 = self.conv5_layer(self.down(x4))
        
        x_bottleneck = self.bottleneck_layer(self.down(x5))
        
        x = self.up1_layer(torch.cat([x_bottleneck, x5], 1))
        x = self.up2_layer(torch.cat([x, x4], 1))
        x = self.up3_layer(torch.cat([x, x3], 1))
        x = self.up4_layer(torch.cat([x, x2], 1))
        x = self.up5_layer(torch.cat([x, x1], 1))
        
        if self.num_classes > 1:
            logits = F.log_softmax(self.output_layer(x))
        else:
            logits = self.output_layer(x)
            
        return logits


# ---
