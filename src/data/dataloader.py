import torch
import torch.optim as optim
import pytorch_lightning as pl
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Custom dataset for hail prediction
class HailDataset(Dataset):
    """Dataset for hail prediction from radar and meteorological data"""
    
    def __init__(self, h5_file, input_seq_len=4, output_seq_len=18, transform=None):
        """
        Initialize the dataset
        
        Args:
            h5_file: Path to H5 file containing radar and meteorological data
            input_seq_len: Length of input sequence
            output_seq_len: Length of output sequence
            transform: Optional transformations to apply
        """
        self.h5_file = h5py.File(h5_file, 'r')
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.transform = transform
        
        # Get metadata
        self.num_samples = len(self.h5_file['radar'])
        self.sample_indices = np.arange(self.num_samples - (input_seq_len + output_seq_len) + 1)
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        # Get the starting index
        start_idx = self.sample_indices[idx]
        
        # Get input and target sequences
        input_seq = []
        for i in range(self.input_seq_len):
            radar = self.h5_file['radar'][start_idx + i]
            if self.transform:
                radar = self.transform(radar)
            input_seq.append(radar)
        
        target_seq = []
        for i in range(self.output_seq_len):
            radar = self.h5_file['radar'][start_idx + self.input_seq_len + i]
            if self.transform:
                radar = self.transform(radar)
            target_seq.append(radar)
        
        # Get metadata for physics-informed loss
        metadata = {}
        if 'temperature' in self.h5_file:
            metadata['temperature'] = np.array(