import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# batch normalization prevents the range of values in the layers changing too much
# this will support the model to train faster and has better ability to generalize
class Norm(nn.Module):
    
    def __init__(self, d_model, eps=1e-6):
        super(Norm, self).__init__()
        
        self.size = d_model
        self.eps = eps
        
        # create two learnable parameters to calibrate normalization
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
                          / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    
def calculate_attention(q, k, v, d_k, mask=None, dropout=None):
    
    # here will perform dot product of the q vector with the k vector of the respective word the network is scoring
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # before we perform Softmax, we apply our mask and hence reduce values where the input is padding (so does the decoder)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

# Multi-headed attention layer, each input is split into multiple heads 
# this allows the network to simultaneously attend to different subsections of each embedding
class MultiHeadedSelfAttention(nn.Module):
    
    def __init__(self, heads, embedding_dim, dropout=0.1):
        super(MultiHeadedSelfAttention, self).__init__()
        
        self.d_model = embedding_dim
        self.heads = heads
        self.d_k = self.d_model // self.heads
        
        # this will be three matrices (Wq, Wk and Wv) trained during the training process
        self.q_linear = nn.Linear(self.d_model, self.d_model)
        self.k_linear = nn.Linear(self.d_model, self.d_model)
        self.v_linear = nn.Linear(self.d_model, self.d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # additional weights matrix (WO) trained jointly with the network
        self.concatenate_layer = nn.Linear(self.d_model, self.d_model)
        
    def forward(self, q, k, v, mask=None):
        
        batch_size = q.size(0)
        
        # create a Query vector, a Key vector, and a Value vector by performing linear operation and split into N heads
        q = self.q_linear(q).view(batch_size, -1, self.heads, self.d_k)
        k = self.k_linear(k).view(batch_size, -1, self.heads, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.heads, self.d_k)
        
        # transpose to get dimensions batch_size * N * seq_len * d_k
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        
        # calculate attention scores
        z_scores = calculate_attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        z_scores = z_scores.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        output = self.concatenate_layer(z_scores)
        
        return output

# the feed-forward layer simply deepens our network 
# this will employ linear layers to analyze patterns in the attention layers output
# the various paths can be executed in parallel while flowing through the feed-forward layer (no dependencies)
class FeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.fc1_layer = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.fc2_layer = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        x = F.relu(self.fc1_layer(x))
        x = self.dropout(x)
        x = self.fc2_layer(x)
        
        return x
    