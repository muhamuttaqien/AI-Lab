import os
from PIL import Image

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
