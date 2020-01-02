#!/usr/bin/env python
# coding: utf-8

# # Fully Convolutional Network(FCN)

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


# ## Build FCN32s, FCN16s & FCN8s Architecture

# In[2]:


class FCN32s(nn.Module):
    
    def __init__(self, pretrained_net, n_class):
        
        super(FCN32s, self).__init__()
        
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
        
    def forward(self, x):
        
        output = self.pretrained_net(x)
        x5 = output['x5'] # size: (N, 512, x.H/32, x.W/32)
        
        score = self.bn1(self.relu(self.deconv1(x5)))    # size: (N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score))) # size: (N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score))) # size: (N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score))) # size: (N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score))) # size: (N, 32, x.H, x.W)
        score = self.classifier(score)                   # size: (N, n_class, x.H/1, x.W/1)
        
        return score # size: (N, n_class, x.H/1, x.W/1)


# In[3]:


class FCN16s(nn.Module):
    
    def __init__(self, pretrained_net, n_class):
        
        super(FCN16s, self).__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
        
    def forward(self, x):
        
        output = self.pretrained_net(x)
        x5 = output['x5'] # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4'] # size=(N, 512, x.H/16, x.W/16)
        
        score = self.relu(self.deconv1(x5))              # size: (N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                     # element-wise add, size: (N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score))) # size: (N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score))) # size: (N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score))) # size: (N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score))) # size: (N, 32, x.H, x.W)
        score = self.classifier(score)                   # size: (N, n_class, x.H/1, x.W/1)
        
        return score # size: (N, n_class, x.H/1, x.W/1)


# In[4]:


class FCN8s(nn.Module):
    
    def __init__(self, pretrained_net, n_class):
        
        super(FCN8s, self).__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
        
    def forward(self, x):
        
        output = self.pretrained_net(x)
        x5 = output['x5'] # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4'] # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3'] # size=(N, 256, x.H/8, x.W/8)
        
        score = self.relu(self.deconv1(x5))              # size: (N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                     # element-wise add, size: (N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))           # size: (N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)                     # element-wise add, size: (N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score))) # size: (N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score))) # size: (N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score))) # size: (N, 32, x.H, x.W)
        score = self.classifier(score)                   # size: (N, n_class, x.H/1, x.W/1)
        
        return score # size: (N, n_class, x.H/1, x.W/1)


# In[5]:


class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_class):

        super(FCNs, self).__init__()
        
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        
        output = self.pretrained_net(x)
        x5 = output['x5'] # size: (N, 512, x.H/32, x.W/32)
        x4 = output['x4'] # size: (N, 512, x.H/16, x.W/16)
        x3 = output['x3'] # size: (N, 256, x.H/8,  x.W/8)
        x2 = output['x2'] # size: (N, 128, x.H/4,  x.W/4)
        x1 = output['x1'] # size: (N, 64, x.H/2,  x.W/2)

        score = self.bn1(self.relu(self.deconv1(x5)))    # size: (N, 512, x.H/16, x.W/16)
        score = score + x4                               # element-wise add, size: (N, 512, x.H/16, x.W/16)
        score = self.bn2(self.relu(self.deconv2(score))) # size: (N, 256, x.H/8, x.W/8)
        score = score + x3                               # element-wise add, size: (N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score))) # size: (N, 128, x.H/4, x.W/4)
        score = score + x2                               # element-wise add, size: (N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score))) # size: (N, 64, x.H/2, x.W/2)
        score = score + x1                               # element-wise add, size: (N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score))) # size: (N, 32, x.H, x.W)
        score = self.classifier(score)                   # size: (N, n_class, x.H/1, x.W/1)

        return score # size: (N, n_class, x.H/1, x.W/1)


# ---
