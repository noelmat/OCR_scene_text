import torch
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}


class Dataset:
    def __init__(self, path, get_labels, get_image_files):
        self.path = path
        self.image_files = get_image_files(self.path) #  list of image file path
        self.labels = get_labels(self.image_files) #  dict of image file to label json dict
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
        return {
            'images': img,
            'targets': targets
        }   