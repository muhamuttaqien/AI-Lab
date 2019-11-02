import numpy as np
import spacy
import torch
from torch.autograd import Variable

def show_plot_evaluation(points, n):
    
    points = moving_average(points, n)
    
    plt.figure(figsize=(10, 6))
    plt.plot(points)
    plt.savefig('./images/plot_evaluation_of_network.png')
    plt.show()
    
def create_no_peak_mask(size, device):
    
    no_peak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    no_peak_mask = Variable(torch.from_numpy(no_peak_mask) == 0)
    if device == 0:
        no_peak_mask = no_peak_mask.cuda()
    
    return no_peak_mask

# the purposes of these masks are:
# in the encoder and decoder is to zero attention outputs wherever there is just padding in the input sentences
# in the decoder is to prevent the decoder 'peaking' ahead at the rest of the translated sentence when predicting the next word

def create_masks(source, target, source_pad, target_pad, device):
    
    source_mask = (source != source_pad).unsqueeze(-2)
    
    if target is not None:
        target_mask = (target != target_pad).unsqueeze(-2)
        size = target.size(1) # get seq_len for matrix
        no_peak_mask = create_no_peak_mask(size, device)
        if target.is_cuda:
            no_peak_mask.cuda()
        target_mask = target_mask & no_peak_mask
    
    else:
        target_mask = None
    
    return source_mask, target_mask
    