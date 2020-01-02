import re
import math
import time
import unicodedata
import matplotlib.pyplot as plt

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r'([.!?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s).strip()
    
    return s

# returns true if both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filter_pair(p, max_length):
    # input sequences need to preserve the last word for EOS token
    return len(p[0].split(' ')) < max_length and \
           len(p[1].split(' ')) < max_length

# filter pairs
def filter_pairs(pairs, max_length):
    return [pair for pair in pairs if filter_pair(pair, max_length)]

def indexes_from_sentence(vocab, sentence, eos_token):
    return [vocab.word2index[word] for word in sentence.split(' ')] + [eos_token]

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
    