"""
Custom PyTorch Datasets

Patterns for creating custom datasets.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class NumpyDataset(Dataset):
    """Dataset from NumPy arrays."""
    
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y.dtype == np.int64 else torch.FloatTensor(y)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class CSVDataset(Dataset):
    """Dataset from CSV file."""
    
    def __init__(self, csv_path, target_column, transform=None):
        import pandas as pd
        
        df = pd.read_csv(csv_path)
        self.X = torch.FloatTensor(df.drop(columns=[target_column]).values)
        self.y = torch.FloatTensor(df[target_column].values)
        self.transform = transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


def create_dataloaders(X_train, y_train, X_val, y_val, batch_size=32):
    """Create train and validation dataloaders."""
    train_dataset = NumpyDataset(X_train, y_train)
    val_dataset = NumpyDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    dataset = NumpyDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for batch_x, batch_y in loader:
        print(f"Batch shape: X={batch_x.shape}, y={batch_y.shape}")
        break
