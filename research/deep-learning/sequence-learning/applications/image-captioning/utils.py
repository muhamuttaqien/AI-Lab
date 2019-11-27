import os
import pickle
from PIL import Image
from tqdm import tqdm_notebook
from collections import Counter
from pycocotools.coco import COCO

import nltk
import torch
import matplotlib.pyplot as plt

def resize_image_due_to_pytorch_issue(images, size=224):
    
    images = np.resize(images, (images.shape[0], images.shape[1], size, size))
    return images

def load_image(image_path, crop_size, transform=None):
    
    image = Image.open(image_path)
    image = image.resize([crop_size, crop_size], Image.LANCZOS)
    
    if transform is not None: image = transform(image).unsqueeze(0)
        
    return image

def resize_image(image, size):
    
    image = image.resize(size, Image.ANTIALIAS)
    return image

def resize_images(image_path, output_path, image_size):
    
    size = [image_size, image_size]
    
    if not os.path.exists(output_path): os.makedirs(output_path)
        
    images = os.listdir(image_path)
    num_images = len(images)
    
    print('Resizing and saving the images...')
    
    for index, image in enumerate(tqdm_notebook(images)):
        
        if os.path.exists(output_path): continue
            
        with open(os.path.join(image_path, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_path, image), img.format)
                
def build_vocabulary(Vocabulary, min_word_count, caption_path, vocabulary_path):
    
    coco = COCO(caption_path)
    counter = Counter()
    
    print('Tokenizing the captions...')
    
    ids = coco.anns.keys()
    for i, index in enumerate(tqdm_notebook(ids)):
        caption = str(coco.anns[index]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
            
    # if the word frequency is less than 'min_word_count', then the word is discarded
    words = [word for word, count in counter.items() if count >= min_word_count]
    
    # create a vocab wrapper and add some special tokens
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unknown>')
    
    # add the words to the vocabulary
    for index, word in enumerate(words): vocab.add_word(word)
    
    # save vocabulary into pickle format
    with open(vocabulary_path, 'wb') as f:
        pickle.dump(vocab, f)

    return vocab

def collate_fn(data):
    
    # create mini-batch tensors from the list of tuples (image, caption)
    
    # sort a data list by caption length (descending order)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    
    # merge images (from tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)
    
    # merge captions (from tuple of 1D tensor to 2D tensor)
    lengths = [len(caption) for caption in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    
    for index, caption in enumerate(captions):
        end = lengths[index]
        targets[index, :end] = caption[:end]
        
    return images, targets, lengths

def get_data_loader(COCODataset, image_path, coco_path, vocab, transform, batch_size, shuffle, num_workers):
    
    coco = COCODataset(image_path=image_path, coco_path=coco_path, vocab=vocab, transform=transform)
    
    # this data loader will return (images, captions, length) for each iteration
    # images: a tensor of shape (batch_size, 3, 224, 224), 
    # captions: a tensor of shape (batch_size, padded_length)
    # lengths: a list indicating valid length for each caption
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    
    return data_loader

def show_plot_evaluation(points, n):
    
    points = moving_average(points, n)
    
    plt.figure(figsize=(10, 6))
    plt.plot(points)
    plt.savefig('./images/plot_evaluation_of_network.png')
    plt.show()
    