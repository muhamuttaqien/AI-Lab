import math
import torch
import torch.nn as nn 

class PositionalEncoder(nn.Module):
    
    def __init__(self, d_model, max_seq_len=200, dropout=0.1):
        super(PositionalEncoder, self).__init__()
        
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # create a 2-d constant matrix of position-specific values
        # Pos refers to the order in the sentence
        # i refers to the position along the embedding vector dimension
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/ d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1))/ d_model)))
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        
        # make embedding relatively larger and keep the positional encoding relatively smaller
        x = x * math.sqrt(self.d_model)
        
        # add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        
        if x.is_cuda: pe.cuda()
        x = x + pe
        x = self.dropout(x)
        
        return x