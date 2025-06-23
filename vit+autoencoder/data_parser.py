import os
import cv2
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from collections import defaultdict
import pandas as pd
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision import transforms

# do horiz flip
def horizontal_flip(img, p=0.5):
    if random.random() < p:
        return cv2.flip(img, 1)
    return img

# do vertical flip
def vertical_flip(img, p=0.5):
    if random.random() < p:
        return cv2.flip(img, 0)
    return img

# do random rotation on image and warp to fix it
def random_rotation(img, max_angle=30):
    rows, columns = img.shape[:2]
    angle = random.uniform(-max_angle, max_angle)

    mat = cv2.getRotationMatrix2D((columns/2, rows/2), angle, 1)
    return cv2.warpAffine(img, mat, (columns, rows))

# adjusting of brightness
def adjust_brightness(img, factor_range=(0.5, 1.5)):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    factor = random.uniform(*factor_range)
    hsv[...,2] = np.clip(hsv[...,2] * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# apply random gaussian noise
def add_gaussian_noise(img, sigma_limit=25):
    sigma = random.uniform(0, sigma_limit)
    gauss = np.random.normal(0, sigma, img.shape).astype(np.uint8)
    return cv2.add(img, gauss)

no_images = 220 

class DataParser:
    def __init__(self, path, target_per_class=no_images):
        self.path = path
        self.target_per_class = target_per_class
    
    def read_images(self, img_size=(224, 224), augmentations=None):
        data = defaultdict(list)
        folders = os.listdir(self.path)
        
        for folder in folders:
            folder_path = os.path.join(self.path, folder)
            files = os.listdir(folder_path)
            
            for file in files:
                file_path = os.path.join(folder_path, file)
                img = cv2.imread(file_path)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    data[folder].append(img)

        balanced_data = []
        for label, images in data.items():
            while len(images) < self.target_per_class:
                img = random.choice(images) 
                augmented_img = self.random_aug(img, augmentations)
                images.append(augmented_img)
        
            balanced_data.extend([{"image": img, "label": label} for img in images[:self.target_per_class]])
        
        return pd.DataFrame(balanced_data)
    
    def random_aug(self, img, augmentations):
        if not augmentations:
            return img
        
        if 'flip' in augmentations:
            if random.random() < augmentations['flip']['h_prob']:
                img = horizontal_flip(img, p=1)
            if random.random() < augmentations['flip']['v_prob']:
                img = vertical_flip(img, p=1)
        
        if 'rotate' in augmentations and random.random() < augmentations['rotate']['apply_prob']:
            img = random_rotation(img, max_angle=augmentations['rotate']['max_angle'])
        
        if 'brightness' in augmentations and random.random() < augmentations['brightness']['apply_prob']:
            img = adjust_brightness(img, factor_range=augmentations['brightness']['factor_range'])
        
        if 'noise' in augmentations and random.random() < augmentations['noise']['apply_prob']:
            img = add_gaussian_noise(img, sigma_limit=augmentations['noise']['sigma_limit'])
        return img

    
class BrainTumorDataset(Dataset):
    def __init__(self, dataframe, label_map, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label_map = label_map  
        self.counter = 0

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img = self.dataframe.iloc[idx]["image"]
        label = self.dataframe.iloc[idx]["label"]
        label = self.label_map[label]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        mean = torch.tensor([0.485, 0.456, 0.406])  
        std = torch.tensor([0.229, 0.224, 0.225])
        img = (img - mean[:, None, None]) / std[:, None, None]

        img /= 255.0

        if self.transform:
            img = self.transform(img)
        if self.counter == 0:
            img_np = img.numpy()  
            img_np = np.squeeze(img_np) 

            img_pil = transforms.ToPILImage()(img)
            img_pil.save(f"transformed_image_pil{self.counter}.png") 
            self.counter += 1

        return img, label
    

def split_dataset(df, train_size=0.8, val_size=0.2, test_size=0.00, random_state=42):

    # Split into train and temp (val + test)
    train_df, temp_df = train_test_split(df, train_size=train_size, stratify=df["label"], random_state=random_state)

    # # Split temp into val and test
    # val_df, test_df = train_test_split(temp_df, test_size=test_size / (val_size + test_size), 
    #                                    stratify=temp_df["label"], random_state=random_state)

    return train_df, temp_df#, test_df