import torch
import torch.nn as nn

from RBM import RBMNet

class DBNNet(nn.Module):
    
    def __init__(self, visible_units=256, hidden_units=[64, 100],
                 k=2, learning_rate=1e-5, learning_rate_decay=False,
                 xavier_init=False, increase_to_cd_k=False, use_gpu=False):
        
        super(DBNNet, self).__init__()
        self.n_layers = len(hidden_units)
        self.rbm_layers = []
        self.rbm_nodes = []
        
        for i in range(self.n_layers):
            input_size = 0
            if i==0:
                input_size = visible_units
            else:
                input_size = hidden_units[i-1]
            
            # creating different RBM layers
            rbm = RBMNet(visible_units=input_size,
                         hidden_units=hidden_units[i],
                         k=k,
                         learning_rate=learning_rate,
                         learning_rate_decay=learning_rate_decay,
                         xavier_init=xavier_init,
                         increase_to_cd_k=increase_to_cd_k,
                         use_gpu=use_gpu)
            
            self.rbm_layers.append(rbm)
            
        self.W_rec = [nn.Parameter(self.rbm_layers[i].W.data.clone()) for i in range(self.n_layers-1)]
        self.W_gen = [nn.Parameter(self.rbm_layers[i].W.data) for i in range(self.n_layers-1)]
        
        self.bias_rec = [nn.Parameter(self.rbm_layers[i].h_bias.data.clone()) for i in range(self.n_layers-1)]
        self.bias_gen = [nn.Parameter(self.rbm_layers[i].v_bias.data) for i in range(self.n_layers-1)]
        
        self.W_mem = nn.Parameter(self.rbm_layers[-1].W.data)
        self.v_bias_mem = nn.Parameter(self.rbm_layers[-1].v_bias.data)
        self.h_bias_mem = nn.Parameter(self.rbm_layers[-1].h_bias.data)
        
        for i in range(self.n_layers-1):
            self.register_parameter('W_rec%i'%i, self.W_rec[i])
            self.register_parameter('W_gen%i'%i, self.W_gen[i])
            self.register_parameter('bias_rec%i'%i, self.bias_rec[i])
            self.register_parameter('bias_gen%i'%i, self.bias_gen[i])
            
    def forward(self, x):
    
        v = x
        for i in range(len(self.rbm_layers)):
            v = v.view((v.shape[0], -1)).type(torch.FloatTensor)
            h_prob, h = self.rbm_layers[i].visible_to_hidden(v)
            
        return h_prob, h
    
    def reconstruct(self, x):
        
        h = x
        for i in range(len(self.rbm_layers)):
            h = h.view((h.shape[0], -1)).type(torch.FloatTensor)
            h_prob, h = self.rbm_layers[i].visible_to_hidden(h) # reverse operation

        v = h
        for i in range(len(self.rbm_layers)-1,-1,-1):
            v = v.view((v.shape[0], -1)).type(torch.FloatTensor)
            v_prob, v = self.rbm_layers[i].hidden_to_visible(v) # reverse operation
            
        return v_prob, v
    
    def train_static(self, x, y, n_epochs=50, batch_size=10):
        
        temp = x
        
        for i in range(len(self.rbm_layers)):
            
            print('Training the RBM layer index: {}'.format(i+1))
            
            x_tensor = temp.type(torch.FloatTensor)
            y_tensor = y.type(torch.FloatTensor)
            datasets = torch.utils.data.TensorDataset(x_tensor, y_tensor)
            data_loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, drop_last=True)
            
            self.rbm_layers[i].train(data_loader, n_epochs, batch_size)
            v = temp.view((temp.shape[0], -1)).type(torch.FloatTensor)
            v_prob, v = self.rbm_layers[i].forward(v)
        
        return
    
    def train_ith(self, x, y, n_epochs, batch_size, ith_layer):
        
        if(ith_layer-1>len(self.rbm_layers) or ith_layer<=0):
            print('Layer index is out of the range!')
            return
    
        ith_layer = ith_layer-1
        v = x.view((x.shape[0], -1)).type(torch.FloatTensor)
        
        for ith in range(ith_layer):
            v_prob, v = self.rbm_layers[ith].forward(v)
            
        temp = v
        x_tensor = temp.type(torch.FloatTensor)
        y_tensor = y.type(torch.FloatTensor)
        datasets = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        data_loader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, drop_last=True)
        self.rbm_layers[ith_layer].train(data_loader, n_epochs, batch_size)
        
        return