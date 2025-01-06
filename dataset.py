import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Tuple
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader

'''
Two folders with each having 110 folders of patient brain slices and masks for each slices
'''

class BrainImageSegmentationDataset(Dataset):
    def __init__(self, base_dir = "data/kaggle_3m", data_file = "data/kaggle_3m/data.csv", transform = None):
        self.base_dir = base_dir
        self.transform = transform
        # data as image mask pairs
        self.data_image_mask_pairs = []

        for patient_dir in os.listdir(base_dir):
            if not patient_dir.endswith(".csv") and not patient_dir.endswith(".md"):
                patient_dir = os.path.join(base_dir, patient_dir)
                #print(patient_dir)
                for filename in os.listdir(patient_dir):
                    if filename.endswith(".tif") and not filename.endswith("_mask.tif"):
                        image_path = os.path.join(patient_dir, filename)
                        mask_path = image_path.replace(".tif", "_mask.tif")
                        if os.path.exists(mask_path):
                            self.data_image_mask_pairs.append((image_path, mask_path))

    def __len__(self):
        return len(self.data_image_mask_pairs)
    
    def __getitem__(self, index):
        #Loading the image and mask as arrays
        image_path, mask_path = self.data_image_mask_pairs[index]
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))

        #Normalizing
        image = image/255.0
        mask = mask / 255.0
        mask = (mask > 0.5).astype(np.float32)

        #Converting to tensors
        image = torch.tensor(image, dtype = torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype = torch.float32).unsqueeze(0)

        #Apply transform

        return image, mask
