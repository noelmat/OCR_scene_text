import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
imagenet_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


class Dataset:
    def __init__(self, image_paths, get_labels,  label_enc, ds_type='train', size=(700, 300)):
        self.image_paths = image_paths
        self.get_labels = get_labels
        self.label_enc = label_enc
        self.max_len = {"company": 50, "address": 160, "date": 16, "total": 16}
        self.tfms = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(**imagenet_stats),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img_path = self.image_paths[item]
        img = Image.open(img_path).convert("RGB")
        img = self.tfms(img)
        
        
        targets = self.get_labels(img_path)
        for k, v in targets.items():
            # adding 1 to the encoded labels to reserve 0 for blank
            v = self.label_enc.transform(list(v)) + 1
            v_len = len(v)
            padding_len = self.max_len[k] - v_len -1
            new_arr = np.zeros(self.max_len[k])
            new_arr[0] = v_len
            new_arr[1:v_len+1]=v
            targets[k] = new_arr

        return {
            "images": img,
            "company": torch.tensor(targets["company"], dtype=torch.long),
            "address": torch.tensor(targets["address"], dtype=torch.long),
            "date": torch.tensor(targets["date"], dtype=torch.long),
            "total": torch.tensor(targets["total"], dtype=torch.long),
        }
