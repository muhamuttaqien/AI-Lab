import sys
import math
from tqdm import tqdm
import torch
import torch.nn as nn

class RBMNet(nn.Module):
    
    def __init__(self, visible_units=256, hidden_units=64, k=2,
                       learning_rate=1e-5, learning_rate_decay=False,
                       xavier_init=False, increase_to_cd_k=False, use_gpu=False):
        
        super(RBMNet, self).__init__()
        self.desc = 'RBM Network'
        self.visible_units = visible_units
        self.hidden_units = hidden_units
        self.k = k
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.xavier_init = xavier_init
        self.increase_to_cd_k = increase_to_cd_k
        self.use_gpu = use_gpu
        self.batch_size = 16
        
        if not self.xavier_init:
            self.W = torch.randn(self.visible_units, self.hidden_units) * 0.01
        else:
            self.xavier_value = torch.sqrt(torch.FloatTensor([1.0 / (self.visible_units + self.hidden_units)]))
            self.W = -self.xavier_value + torch.rand(self.visible_units, self.hidden_units) * (2 * self.xavier_value)
        self.h_bias = torch.zeros(self.hidden_units)
        self.v_bias = torch.zeros(self.visible_units)
        
    def bernoulli_sampling(self, prob):
        sample = torch.distributions.Bernoulli(prob).sample()
        return sample
    
    def visible_to_hidden(self, v):
        
        h_prob = torch.add(torch.matmul(v, self.W), self.h_bias)
        h_prob = torch.sigmoid(h_prob)
        sample_h_prob = self.bernoulli_sampling(h_prob)
        
        return h_prob, sample_h_prob
    
    def hidden_to_visible(self, h):
        
        v_prob = torch.add(torch.matmul(h, self.W.transpose(0, 1)), self.v_bias)
        v_prob = torch.sigmoid(v_prob)
        sample_v_prob = self.bernoulli_sampling(v_prob)
        
        return v_prob, sample_v_prob
    
    def reconstruction_error(self, data):
        return self.contrastive_divergence(data, False)
    
    def reconstruct(self, x, n_gibbs):
        v = x
        for i in range(n_gibbs):
            h_prob, h = self.visible_to_hidden(v)
            v_prob, v = self.hidden_to_visible(h_prob)
        
        return v_prob, v
    
    def contrastive_divergence(self, x, n_gibbs_sampling_steps=1, lr=0.001, train=True):
        
        # positive phase
        positive_h_prob, positive_h = self.visible_to_hidden(x)
        positive_connectivity = torch.matmul(x.t(), positive_h)
        
        # negative phase
        hidden_activations = positive_h
        for i in range(n_gibbs_sampling_steps):
            negative_v_prob, _ = self.hidden_to_visible(hidden_activations)
            negative_h_prob, hidden_connectivity = self.visible_to_hidden(negative_v_prob)
        negative_connectivity = torch.matmul(negative_v_prob.t(), negative_h_prob)
        
        if train:
        
            batch_size = self.batch_size
            
            gradients = (positive_connectivity - negative_connectivity)
            gradients_update = gradients/ batch_size
            v_bias_update = torch.sum(x - negative_v_prob, dim=0)/ batch_size
            h_bias_update = torch.sum(positive_h_prob - negative_h_prob, dim=0)/ batch_size
            
            self.W += lr * gradients_update
            self.v_bias += lr * v_bias_update
            self.h_bias += lr * h_bias_update
            
        error = torch.mean(torch.sum((x - negative_v_prob)**2, dim=0))
        
        return error, torch.sum(torch.abs(gradients_update))
    
    def forward(self, x):
        return self.visible_to_hidden(x)
    
    def step(self, x, epoch, n_epochs):
        
        if self.increase_to_cd_k:
            n_gibbs_sampling_steps = int(math.ceil((epoch/n_epochs) * self.k))
        else:
            n_gibbs_sampling_steps = self.k
            
        if self.learning_rate_decay:
            lr = self.learning_rate / epoch
        else:
            lr = self.learning_rate
        
        return self.contrastive_divergence(x, n_gibbs_sampling_steps, lr, train=True)
    
    def train(self, train_loader, n_epochs=50, batch_size=16):
        
        self.batch_size = batch_size
        if(isinstance(train_loader, torch.utils.data.DataLoader)):
            train_loader = train_loader
        else:
            train_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size)
            
        for epoch in range(1, n_epochs+1):
            epoch_error = 0.0
            n_batches = int(len(train_loader))
            
            cost_ = torch.FloatTensor(n_batches, 1)
            gradient_ = torch.FloatTensor(n_batches, 1)
            
            for i, (x, _) in tqdm(enumerate(train_loader), ascii=True,
                                  desc='RBM Net is fitting', file=sys.stdout):
                
                x = x.view(len(x), self.visible_units)
                
                if self.use_gpu:
                    x = x.cuda()
                cost_[i-1], gradient_[i-1] = self.step(x, epoch, n_epochs)
                
            print('Epoch: {}, Average Cost: {}, STD Cost: {}, Average Gradient: {}, STD Gradient: {}'.format(epoch,
                                                                                                             torch.mean(cost_),
                                                                                                             torch.std(cost_),
                                                                                                             torch.mean(gradient_),
                                                                                                             torch.std(gradient_)))
        return