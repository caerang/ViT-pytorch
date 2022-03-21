import os
from PIL import Image
from typing import Optional, Callable

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode


class TinyImagenet200(Dataset):
    def __init__(self,
                 annotations_file: str,
                 img_dir: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotations_file, delimiter='\t')
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # forced to read in rbg because there exist gray image.
        # TODO
        #  - filter gray image in sampler or collate_fn
        # shape of image: [C, H, W]
        image = read_image(img_path, mode=ImageReadMode.RGB)
        image = image.permute(1, 2, 0)
        image = Image.fromarray(image.numpy())
        
        # class label id (string type)
        label = self.img_labels.iloc[idx, 1]

        # read_imate returns torch.Tensor type of image
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

class LabelTransform:
    def __init__(self, label_file: str = None):
        label_data = pd.read_csv(label_file, header=None, delimiter='\t')
        self.idx_to_label_name = []
        for label_name in label_data.iloc[:, 1]:
            self.idx_to_label_name.append(label_name)
            
        self.label_id_to_idx = {}
        for idx, label_id in enumerate(label_data.iloc[:, 0]):
            self.label_id_to_idx[label_id] = idx
    
    def __call__(self, label_id):
        return self.label_id_to_idx[label_id]

    def label_name(self, idx):
        assert idx < len(self.idx_to_label_name)
        return self.idx_to_label_name[idx]
