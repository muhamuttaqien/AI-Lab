import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import spacy
from nltk.corpus import wordnet

import torch
import torch.nn.functional as F
from torch.autograd import Variable

def init_vars(sentence, model, source_field, target_field, k, max_length, device):
    
    init_token = target_field.vocab.stoi['<SOS>']
    source_mask = (sentence != source_field.vocab.stoi['<PAD>']).unsqueeze(-2)
    e_output = model.encoder(sentence, source_mask)
    
    outputs = torch.LongTensor([[init_token]])
    outputs.to(device)
    
    target_mask = create_no_peak_mask(1, device)
    output = model.output_layer(model.decoder(outputs, e_output, source_mask, target_mask))
    output = F.softmax(output, dim=-1)
    
    probs, index = output[:, -1].data.topk(k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(k, max_length).long()
    outputs.to(device)
    
    outputs[:, 0] = init_token
    outputs[:, 1] = index[0]
    
    e_outputs = torch.zeros(k, e_output.size(-2), e_output.size(-1))
    e_outputs.to(device)
    
    e_outputs[:, :] = e_output[0]
    
    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, output, log_scores, i, k):
    
    probs, index = output[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_index = log_probs.view(-1).topk(k)
    
    row = k_index // k
    col = k_index % k
    
    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = index[row, col]
    
    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores

def beam_search(sentence, model, source_field, target_field, max_length, device):
    
    K = 3
        
    outputs, e_outputs, log_scores = init_vars(sentence, model, source_field, target_field, K, max_length, device)
    eos_token = target_field.vocab.stoi['<EOS>']
    source_mask = (sentence != source_field.vocab.stoi['<PAD>']).unsqueeze(-2)
    
    index = None
    for i in range(2, max_length):
        
        target_mask = create_no_peak_mask(i, device)
        output = model.output_layer(model.decoder(outputs[:,:i], e_outputs, source_mask, target_mask))
        output = F.softmax(output, dim=-1)
        
        outputs, log_scores = k_best_outputs(outputs, output, log_scores, i, K)
        
        ones = (outputs == eos_token).nonzero() # occurrences of end symbols for all input sentences
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).to(device)
        
        for vector in ones:
            i = vector[0]
            if sentence_lengths[i] == 0: # first end symbol has not been found yet
                sentence_lengths[i] = vector[1] # position of first end symbol
                
        num_finished_sentences = len([s for s in sentence_lengths if s > 0])
        
        if num_finished_sentences == K:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, index = torch.max(log_scores * div, 1)
            index = index.data[0]
            break
            
    if index is None:
        length = (outputs[0] == eos_token).nonzero()[0]
        return ' '.join(target_field.vocab.itos[token] for token in outputs[0][1:length])
    else:
        length = (outputs[index] == eos_token).nonzero()[0]
        return ' '.join(target_field.vocab.itos[token] for token in outputs[index][1:length])

def get_synonym(word, source_field):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if source_field.vocab.stoi[l.name()] != 0:
                return source_field.vocab.stoi[l.name()]
            
    return 0

def multiple_replace(dict, text):
    
    # create a regular expression from the dictionary keys
    regex = re.compile('(%s)' % '|'.join(map(re.escape, dict.keys())))
    
    # for each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start(): mo.end()]], text)

def translate_sentence(sentence, model, source_field, target_field, max_length, device):
    
    model.eval()
    indexed = []
    sentence = source_field.preprocess(sentence)
    
    for token in sentence:
        if source_field.vocab.stoi[token] != 0:
            indexed.append(source_field.vocab.stoi[token])
        else:
            indexed.append(get_synonym(token, source_field))
            
    sentence = Variable(torch.LongTensor([indexed]))
    sentence.to(device)
    
    sentence = beam_search(sentence, model, source_field, target_field, max_length, device)
    
    return multiple_replace({ ' ?' : '?', ' !':'!', ' .':'.', '\' ':'\'', ' ,': ',' }, sentence)

def translate_text(text, model, source_field, target_field, max_length, device):
    sentences = text.lower().split('.')
    translated = []
    
    for sentence in sentences:
        translated.append(translate_sentence(sentence + '.', model, source_field, target_field, max_length, device).capitalize())
        
    return (' '.join(translated))

def moving_average(a, n):
    
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def show_plot_evaluation(points, n):
    
    points = moving_average(points, n)
    
    plt.figure(figsize=(10, 6))
    plt.plot(points)
    if not os.path.exists('./images/'): os.makedirs('./images/')
    plt.savefig('./images/plot_evaluation_of_network.png')
    plt.show()

# to prevent the first output predictions from being able to see by model
# when the mask is applied in our attention function, each prediction will only be able to make use of the sentence up until the word it is predicting
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
            target_mask = target_mask.cuda()
            no_peak_mask = no_peak_mask.cuda()
        target_mask = target_mask & no_peak_mask
    
    else:
        target_mask = None
    
    return source_mask, target_mask
    