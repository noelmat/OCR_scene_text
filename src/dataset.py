import torch
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


class Dataset:
    def __init__(self, path, get_labels, get_image_files, label_enc):
        self.path = path
        self.image_files = get_image_files(self.path)  #  list of image file path
        self.labels = {p.stem: get_labels(p) for p in self.image_files}  #  dict of image file to label json dict
        self.label_enc = label_enc
        self.tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(**imagenet_stats)
        ])
        
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, item):
        img_path = self.image_files[item]
        img = Image.open(img_path)
        img = self.tfms(img)
    
        targets = self.labels[img_path.stem]
        for k, v in targets.items():
            targets[k] = self.label_enc.transforms(list(v))
        
        return {
            'images': img,
            'company': targets['company'],
            'address': targets['address'],
            'date': targets['date'],
            'total': targets['total'],
            
        }

