import torch
import torch.nn as nn
import torch.nn.functional as F

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
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
        
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.d_model = d_model
        self.d_k = embedding_dim // heads
        self.heads = heads
        
        self.q_linear = nn.Linear(self.d_model , self.d_model )
        self.k_linear = nn.Linear(self.d_model , self.d_model )
        self.v_linear = nn.Linear(self.d_model , self.d_model )
        
        self.dropout = nn.Dropout(dropout)
        self.fc_layer = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, q, k, v, mask=None):
        
        batch_size = q.size(0)
        
        # perform linear operation and split into N heads
        q = self.q_linear(q).view(batch_size, -1, self.heads, self.d_k)
        k = self.k_linear(k).view(batch_size, -1, self.heads, self.d_k)
        v = self.v_linear(v).view(batch_size, -1, self.heads, self.d_k)
        
        # transpose to get dimensions batch_size * N * seq_len * d_k
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        
        # calculate attention score
        scores = calculate_attention(q, k, v, self.d_k, mask, self.dropout)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc_layer(concat)
        
        return output
    
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
    