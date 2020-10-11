import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


class Dataset:
    def __init__(self, path, get_labels, get_image_files, label_enc, size=(700,300)):
        self.path = path
        self.image_files = get_image_files(self.path)  #  list of image file path
        self.labels = {p.stem: get_labels(p) for p in self.image_files}  #  dict of image file to label json dict
        self.label_enc = label_enc
        self.max_len = {
            'company': 64,
            'address': 160,
            'date': 16,
            'total': 16
        }
        self.tfms = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(**imagenet_stats)
        ])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        img_path = self.image_files[item]
        img = Image.open(img_path).convert('RGB')
        img = self.tfms(img)
    
        targets = self.labels[img_path.stem].copy()
        for k, v in targets.items():
            # adding 1 to the encoded labels to reserve 0 for blank
            v = self.label_enc.transform(list(v)) + 1
            padding_len = self.max_len[k] - len(v)
            v = np.append(v, np.zeros(padding_len))
            targets[k] = v            
        
        return {
            'images': img,
            'company': torch.tensor(targets['company'], dtype=torch.long),
            'address': torch.tensor(targets['address'], dtype=torch.long),
            'date': torch.tensor(targets['date'], dtype=torch.long),
            'total': torch.tensor(targets['total'], dtype=torch.long),
            
        }

