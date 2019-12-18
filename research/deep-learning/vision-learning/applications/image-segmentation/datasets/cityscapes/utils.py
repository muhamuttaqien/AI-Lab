import os
import numpy as np
from PIL import Image

def one_hot_encode(num_classes, labels):
    
    batch_size, _, h, w = labels.size()
    map_classes = np.unique(labels)
    
    targets = np.zeros((batch_size, num_classes, h, w))
    
    for i in range(batch_size):
        for c in range(num_classes):
            targets[i][c][labels[i,0] == map_classes[c]] = 1

    return num_classes, targets

def get_files(folder, name_filter=None, extension_filter=None):
    
    if not os.path.isdir(folder): raise RuntimeError(f"\"{folder}\" is not a folder.")
        
    if name_filter is None: name_check = lambda filename: True
    else: name_check = lambda filename: name_filter in filename
        
    if extension_filter is None: extension_check = lambda filename: True
    else: extension_check = lambda filename: filename.endswith(extension_filter)
        
    filtered_files = []
    
    for path, _, files in os.walk(folder):
        
        files.sort()
        
        for file in files:
            if name_check(file) and extension_check(file):
                full_path = os.path.join(path, file)
                filtered_files.append(full_path)
    
    return filtered_files

def pil_loader(data_path, label_path):
    
    data = Image.open(data_path)
    label = Image.open(label_path)
    
    return data, label

def remap(image, old_values, new_values):
    
    assert isinstance(image, Image.Image) or isinstance(image, np.ndarray), "image must be of type PIL.Image or numpy.ndarray"
    
    assert type(old_values) is tuple, "old_values must be of type tuple"

    assert type(new_values) is tuple, "new_values must be of type tuple"
    
    assert len(old_values) == len(new_values), "new_values and old_values must have the same length"
    
    temp = np.zeros_like(image)
    for old, new in zip(old_values, new_values):
        
        if new != 0: 
            temp[image == old] = new
            
    return Image.fromarray(temp)
          
