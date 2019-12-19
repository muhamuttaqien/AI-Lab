import os
import numpy as np
from PIL import Image

def calculate_iou(pred, label, n_class):
    
    iou = []
    for i in range(n_class):
        pred_index = pred == i
        label_index = label == i
        
        intersection = pred_index[label_index].sum()
        union = pred_index.sum() + label_index.sum() - intersection
        
        if union == 0:
            iou.append(float('nan')) # if there is no ground truth, do not include in evaluation
        else:
            iou.append(float(intersection) / max(union, 1))
    
    return iou

def calculate_accuracy(pred, label):
    
    correct = (pred == label).sum()
    total = (label == label).sum()
    
    return correct / total

def one_hot_encode(label):
    
    _, h, w = label.size()
    map_classes = np.unique(label)
    num_classes = len(map_classes)

    target = np.zeros((12, h, w))
    for c in range(num_classes):
        target[c][label[0] == map_classes[c]] = 1

    return target

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
