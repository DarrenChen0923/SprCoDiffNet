import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class SpringbackDataset(Dataset):
    def __init__(self, data_dir, grid_size, transform=None):
        """
        Init dataset, load specifi grid size data
        Args:
            data_dir (str)
            grid_size (str)
            transform (callable, optional): None
        """
        self.image_dir = os.path.join(data_dir, grid_size, "images")
        self.label_dir = os.path.join(data_dir, grid_size, "labels")
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        self.transform = transform
        
        # Debugging: print the number of image files found
        print(f"Found {len(self.image_files)} image files in {self.image_dir}")
           
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {self.image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # replace prefix to get the correpsonding label file.
        label_name = img_name.replace('image', 'label').replace('.jpg', '.txt')
        label_path = os.path.join(self.label_dir, label_name)
        
        # read image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # read label
        with open(label_path, 'r') as f:
            label = float(f.read().strip())
        
        label = torch.tensor(label, dtype=torch.float32)
        
        return image, label
