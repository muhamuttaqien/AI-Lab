import torch
import torch.nn as nn 

class Embedder(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super(Embedder, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        
    def forward(self, x):
        x = self.embedding_layer(x)
        return x
    