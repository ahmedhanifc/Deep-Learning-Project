from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import os
import numpy as np

import matplotlib.pyplot as plt
from src.config import TRANSFORM_CONFIG

class WasteClassificationDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        '''
            This initializes the dataset
            Args:
                root_dir (str): Directly with subfolders
                transform (callable): For transforming images like Augmentation

            We pass in data/train or data/test
        '''

        self.root_dir = root_dir # To know where to find images
        self.transform = transform

        self.classes = []
        all_items = os.listdir(root_dir) # in our case, its paper, metal, glass, plastic.
        
        # getting all directories in /train for
        for item in all_items:
            item_path = os.path.join(root_dir, item)

            if os.path.isdir(item_path):
                self.classes.append(item)

        self.classes = sorted(self.classes) #folder names as classes

        self.class_to_idx = {}
        for idx, class_name in enumerate(self.classes):
            self.class_to_idx[class_name] = idx
        # e.g. {'glass': 0, 'metal': 1, 'paper': 2, 'plastic': 3} -- Alphabetically

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            all_files = os.listdir(class_dir)
            for img_name in all_files:
                if img_name.endswith(".jpg"):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        '''
            return number of samples in the dataset
        '''
        return len(self.samples)

    def __getitem__(self, idx):
        '''
            Return one sample at given index
        '''
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        # the above opens image file and converts to RGB

        if self.transform:
            image = self.transform(image)
            #we transform here so we can apply different transformations
        
        return image, label
    
    def plot_stuff(self, num_samples_per_class=4, show_transformed=True):
        """
        Plot sample images from each class in a grid
        
        Args:
            num_samples_per_class: Number of images to show per class
            show_transformed: If True, shows transformed images (as used in training)
                             If False, shows original images
        """
        num_classes = len(self.classes)
        fig, axes = plt.subplots(num_classes, num_samples_per_class, 
                                figsize=(num_samples_per_class * 2, num_classes * 2))
        
        # If only one class, make axes 2D
        if num_classes == 1:
            axes = axes.reshape(1, -1)
        
        # Use constants from TRANSFORM_CONFIG
        mean = np.array(TRANSFORM_CONFIG["normalize_mean"])
        std = np.array(TRANSFORM_CONFIG["normalize_std"])
        
        for class_idx, class_name in enumerate(self.classes):
            # Get samples for this class
            class_samples = [s for s in self.samples if s[1] == class_idx]
            
            # Randomly select samples
            if len(class_samples) > num_samples_per_class:
                selected = np.random.choice(len(class_samples), num_samples_per_class, replace=False)
                selected_samples = [class_samples[i] for i in selected]
            else:
                selected_samples = class_samples
                # If we have fewer samples than requested, pad with repeats
                while len(selected_samples) < num_samples_per_class:
                    selected_samples.extend(class_samples[:num_samples_per_class - len(selected_samples)])
                    selected_samples = selected_samples[:num_samples_per_class]
            
            # Plot each sample
            for col, (img_path, label) in enumerate(selected_samples):
                if show_transformed:
                    # Get transformed image (as it would be used in training)
                    sample_idx = self.samples.index((img_path, label))
                    image, _ = self.__getitem__(sample_idx)
                    image_np = image.numpy().transpose((1, 2, 0))
                    # Denormalize
                    image_np = image_np * std + mean
                    image_np = np.clip(image_np, 0, 1)
                else:
                    # Show original image
                    image = Image.open(img_path).convert("RGB")
                    image_np = np.array(image) / 255.0
                
                ax = axes[class_idx, col] if num_classes > 1 else axes[col]
                ax.imshow(image_np)
                ax.set_title(f'{class_name}\n(y={class_idx})', fontsize=10)
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()