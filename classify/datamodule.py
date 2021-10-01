import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import DatasetFolder, ImageFolder
from torchvision import transforms
from typing import *
import torch



def transform_fn(train=False, size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
    
    if train:                                 
        return transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ])


class ClassifyDataset(ImageFolder):
    def __init__(self, root, transforms, train=True, split_val=0.2):
        super(ClassifyDataset, self).__init__(root, transforms)
        self.train = train
        self.split_val = split_val
        self._execute_split()
        
    def _execute_split(self):
        total_valid = int(len(self.samples) * self.split_val)
        total_train = len(self.samples) - total_valid
        self.train_samples, self.valid_samples = random_split(
            self.samples, [total_train, total_valid],
            generator=torch.Generator().manual_seed(42)
        )
        
        if self.train:
            self.samples = self.train_samples
        else:
            self.samples = self.valid_samples

    @property
    def idx_to_class(self):
        i2c = {}
        for key, val in self.class_to_idx.items():
            i2c[val]=key
        return i2c

        
class ClassifyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 16, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_transform = transform_fn(train=True)
        self.valid_transform = transform_fn(train=False)
        
    def setup(self, stage: Optional[str] = None):
        self.classify_trainset = ClassifyDataset(self.data_dir, transforms=self.train_transform, train=True)
        self.classify_validset = ClassifyDataset(self.data_dir, transforms=self.train_transform, train=True)

    def train_dataloader(self):
        return DataLoader(self.classify_trainset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.classify_validset, batch_size=self.batch_size, num_workers=self.num_workers)
        