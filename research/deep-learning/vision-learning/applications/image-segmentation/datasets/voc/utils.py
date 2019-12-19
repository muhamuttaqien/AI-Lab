import os
import numpy as np
from PIL import Image

def one_hot_encode(num_classes, label):
    
    _, h, w = label.size()
    map_classes = np.unique(label)
    num_map_classes = len(map_classes)

    target = np.zeros((num_classes, h, w))
    for c in range(num_map_classes):
        target[c][label[0] == map_classes[c]] = 1

    return target

def one_hot_encode_for_sanity_check(num_classes, labels):
    
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

def remove_unsegmented_images(IMAGE_DIR, LABEL_DIR):

    image_list = [image.replace('.jpg', '') for image in os.listdir(IMAGE_DIR)]
    label_list = [label.replace('.png', '') for label in os.listdir(LABEL_DIR)]

    for image in image_list:
        if image not in label_list:
            os.remove(f'{IMAGE_DIR}{image}.jpg')
            