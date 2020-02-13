import re
import math
import time
import matplotlib.pyplot as plt

import torch

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    
    return s

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filter_pair(p, max_length):
    return len(p[0].split(' ')) < max_length and \
           len(p[1].split(' ')) < max_length and \
           p[1].startswith(eng_prefixes)
        
def filter_pairs(pairs, max_length):
    return [pair for pair in pairs if filter_pair(pair, max_length)]

def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensor_from_sentence(lang, sentence, eos_token):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(eos_token)
    
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensor_from_pair(pair, eos_token):
    input_tensor = tensor_from_sentence(input_lang, pair[0], eos_token)
    target_tensor = tensor_from_sentence(target_lang, pair[1], eos_token)
    
    return (input_tensor, target_tensor)

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' %(m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    
    return ret[n - 1:] / n

def show_plot_evaluation(points, n):
    
    points = moving_average(points, n)
    
    plt.figure(figsize=(10, 6))
    plt.plot(points)
    plt.savefig('./images/plot_evaluation_of_network.png')
    plt.show()
    