import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

# Note - you must have torchvision installed for this example
from torchvision.datasets import DatasetFolder
from torchvision import transforms
from typing import *
import torch


class ClassifyDataset(DatasetFolder):
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

        
class ClassifyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 16, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.workers = num_workers
        
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize,
        ])
        
        self.valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])
        
    def setup(self, stage: Optional[str] = None):
        self.classify_trainset = ClassifyDataset(self.data_dir, transforms=self.train_transform, train=True)
        self.classify_validset = ClassifyDataset(self.data_dir, transforms=self.train_transform, train=True)

    def train_dataloader(self):
        return DataLoader(self.classify_trainset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.classify_validset, batch_size=self.batch_size, num_workers=self.num_workers)
        