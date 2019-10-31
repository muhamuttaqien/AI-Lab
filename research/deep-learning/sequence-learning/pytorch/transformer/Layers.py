import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.norm = Norm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention_layer = MultiHeadSelfAttention(heads, d_model, dropout=dropout)
        self.ff_layer = FeedForward(d_model, dropout=dropout)
    
    def forward(self, x, mask):
        
        x = self.norm(x)
        x = x + self.dropout(self.attention_layer(x, x, x, mask))
        x = self.norm(x)
        x = x + self.dropout(self.ff_layer(x))
        
        return x
    
class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.norm = Norm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.attention1_layer = MultiHeadSelfAttention(heads, d_model, dropout=dropout)
        self.attention2_layer = MultiHeadSelfAttention(heads, d_model, dropout=dropout)
        self.ff_layer = FeedForward(d_model, dropout=dropout)
        
    def forward(self, x, encoder_outputs, source_mask, target_mask):
        
        x = self.norm(x)
        x = x + self.dropout(self.attention1_layer(x, x, x, target_mask))
        
        x = self.norm(x)
        x = x + self.dropout(self.attention2_layer(x, encoder_outputs, encoder_outputs, source_mask))
        
        x = self.norm(x)
        x = x + self.dropout(self.ff_layer(x))
        return x
    