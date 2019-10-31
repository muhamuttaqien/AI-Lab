import spacy
import re
from torchtext import data

class Tokenizer(object):
    
    def __init__(self, lang):
        self.nlp = spacy.load(lang)
            
    def tokenize(self, sentence):
        sentence = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok.text for tok in self.nlp.tokenizer(sentence) if tok.text != " "]
    
global max_source_in_batch, max_target_in_batch

def batch_size_fn(new, count, sofar):
    
    # keep augmenting batch and calculate total number of tokens + padding
    global max_source_in_batch, max_target_in_batch
    if count == 1:
        max_source_in_batch = 0
        max_target_in_batch = 0
        
    max_source_in_batch = max(max_source_in_batch, len(new.SOURCE))
    max_target_in_batch = max(max_target_in_batch, len(new.TARGET) + 2)
    
    source_elements = count * max_source_in_batch
    target_elements = count * max_target_in_batch
    
    return max(source_elements, target_elements)
    
class Iterator(data.Iterator):
    
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(sorted(p, key=self.sort_key), self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
                        
            self.batches = pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size, self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))
