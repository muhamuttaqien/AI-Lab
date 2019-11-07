import torch
import torch.nn as nn
import torch.nn.functional as F

from Sublayers import Norm, MultiHeadedSelfAttention, FeedForward

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        
        self.norm = Norm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.attention_layer = MultiHeadedSelfAttention(heads, d_model, dropout=dropout)
        self.ffnn_layer = FeedForward(d_model, dropout=dropout)
    
    def forward(self, x, mask, device):
        
        x = x.to(device)
        mask = mask.to(device)
        
        x = self.norm(x)
        x = x + self.dropout(self.attention_layer(x, x, x, mask)) # perform residual connection then followed by normalization
        x = self.norm(x)
        x = x + self.dropout(self.ffnn_layer(x))
        
        return x
    
class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, heads, dropout=0.1):
        super(DecoderLayer, self).__init__()
        
        self.norm = Norm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.attention_layer = MultiHeadedSelfAttention(heads, d_model, dropout=dropout)
        self.encoder_decoder_attention_layer = MultiHeadedSelfAttention(heads, d_model, dropout=dropout)
        self.ffnn_layer = FeedForward(d_model, dropout=dropout)
        
    def forward(self, x, encoder_outputs, source_mask, target_mask, device):
        
        x = x.to(device)
        encoder_outputs = encoder_outputs.to(device)
        source_mask = source_mask.to(device)
        target_mask = target_mask.to(device)
        
        x = self.norm(x)
        x = x + self.dropout(self.attention_layer(x, x, x, target_mask)) # perform residual connection
        
        x = self.norm(x)
        x = x + self.dropout(self.encoder_decoder_attention_layer(x, encoder_outputs, encoder_outputs, source_mask))
        
        x = self.norm(x)
        x = x + self.dropout(self.ffnn_layer(x))
        return x
    