"""
Data loading utilities for the hybrid physics-informed hailstorm prediction model.
"""

import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import random


class HailDataset(Dataset):
    """
    Dataset for hail prediction using radar and meteorological data
    """
    
    def __init__(
        self,
        h5_file,
        input_seq_len=4,
        output_seq_len=18,
        split='train',
        split_ratio=(0.7, 0.15, 0.15),
        random_seed=42,
        transform=None,
        target_transform=None,
        include_metadata=True
    ):
        """
        Initialize the dataset
        
        Args:
            h5_file: Path to the HDF5 file containing data
            input_seq_len: Length of input sequence
            output_seq_len: Length of output sequence
            split: Data split to use ('train', 'val', or 'test')
            split_ratio: Ratio for train/val/test splits
            random_seed: Random seed for reproducibility
            transform: Transform to apply to inputs
            target_transform: Transform to apply to targets
            include_metadata: Whether to include metadata for physics-informed loss
        """
        self.h5_file = h5_file
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.split = split
        self.split_ratio = split_ratio
        self.transform = transform
        self.target_transform = target_transform
        self.include_metadata = include_metadata
        
        # Open the HDF5 file
        self.h5 = h5py.File(h5_file, 'r')
        
        # Get available fields
        self.available_fields = list(self.h5.keys())
        
        # Ensure required fields are available
        if 'radar' not in self.available_fields:
            raise ValueError(f"Required field 'radar' not found in {h5_file}")
        
        # Get sequence length
        self.total_seq_len = self.h5['radar'].shape[0]
        
        # Check if we have enough data
        if self.total_seq_len < input_seq_len + output_seq_len:
            raise ValueError(f"Not enough data in {h5_file} for specified sequence lengths")
        
        # Create indices for the dataset
        self.indices = self._create_indices(random_seed)
    
    def _create_indices(self, random_seed=42):
        """
        Create indices for train/val/test splits
        
        Args:
            random_seed: Random seed for reproducibility
            
        Returns:
            List of indices for the current split
        """
        # Number of available sequences
        num_sequences = self.total_seq_len - (self.input_seq_len + self.output_seq_len) + 1
        
        # Create all possible start indices
        all_indices = list(range(num_sequences))
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        random.shuffle(all_indices)
        
        # Split indices according to split_ratio
        train_size = int(num_sequences * self.split_ratio[0])
        val_size = int(num_sequences * self.split_ratio[1])
        
        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:train_size + val_size]
        test_indices = all_indices[train_size + val_size:]
        
        # Return indices for the current split
        if self.split == 'train':
            return train_indices
        elif self.split == 'val':
            return val_indices
        elif self.split == 'test':
            return test_indices
        else:
            raise ValueError(f"Invalid split: {self.split}")
    
    def __len__(self):
        """Return the length of the dataset"""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (inputs, targets, metadata)
        """
        # Get the start index
        start_idx = self.indices[idx]
        
        # Extract input sequence
        inputs = self.h5['radar'][start_idx:start_idx + self.input_seq_len]
        
        # Extract target sequence
        targets = self.h5['radar'][start_idx + self.input_seq_len:start_idx + self.input_seq_len + self.output_seq_len]
        
        # Apply transforms if available
        if self.transform:
            inputs = self.transform(inputs)
        if self.target_transform:
            targets = self.target_transform(targets)
        
        # Convert to torch tensors
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()
        
        # If we don't need metadata, return inputs and targets
        if not self.include_metadata:
            return inputs, targets
        
        # Extract metadata for physics-informed loss
        metadata = {}
        
        # Add time steps
        metadata['time_steps'] = torch.arange(
            start_idx, 
            start_idx + self.input_seq_len + self.output_seq_len
        ).float()
        
        # Add other available fields
        fields_to_extract = [
            'temperature', 'density', 'velocity', 'pressure',
            'qv', 'qc', 'qr', 'qh', 'vertical_velocity', 'cape',
            'wave_function', 'hail_probability'
        ]
        
        for field in fields_to_extract:
            if field in self.available_fields:
                field_data = self.h5[field][start_idx:start_idx + self.input_seq_len + self.output_seq_len]
                metadata[field] = torch.from_numpy(field_data).float()
        
        return inputs, targets, metadata


def create_dataloaders(
    h5_file,
    batch_size=16,
    input_seq_len=4,
    output_seq_len=18,
    num_workers=4,
    pin_memory=True,
    include_metadata=True
):
    """
    Create dataloaders for train, validation, and test splits
    
    Args:
        h5_file: Path to the HDF5 file containing data
        batch_size: Batch size for dataloaders
        input_seq_len: Length of input sequence
        output_seq_len: Length of output sequence
        num_workers: Number of workers for dataloaders
        pin_memory: Whether to pin memory for dataloaders
        include_metadata: Whether to include metadata for physics-informed loss
        
    Returns:
        Dictionary of dataloaders for train, val, and test splits
    """
    # Create datasets for each split
    train_dataset = HailDataset(
        h5_file=h5_file,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        split='train',
        include_metadata=include_metadata
    )
    
    val_dataset = HailDataset(
        h5_file=h5_file,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        split='val',
        include_metadata=include_metadata
    )
    
    test_dataset = HailDataset(
        h5_file=h5_file,
        input_seq_len=input_seq_len,
        output_seq_len=output_seq_len,
        split='test',
        include_metadata=include_metadata
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


class HDF5Creator:
    """
    Utility class for creating HDF5 files from meteorological data
    """
    
    def __init__(self, output_path, mode='w'):
        """
        Initialize the HDF5 creator
        
        Args:
            output_path: Path to save the HDF5 file
            mode: File mode ('w' for write, 'a' for append)
        """
        self.output_path = output_path
        self.mode = mode
        self.h5 = None
    
    def __enter__(self):
        """Open the HDF5 file when entering context"""
        self.h5 = h5py.File(self.output_path, self.mode)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the HDF5 file when exiting context"""
        if self.h5 is not None:
            self.h5.close()
    
    def add_dataset(self, name, data, compression="gzip", chunks=True):
        """
        Add a dataset to the HDF5 file
        
        Args:
            name: Name of the dataset
            data: Data to add (numpy array)
            compression: Compression type
            chunks: Whether to use chunking
        """
        if self.h5 is None:
            raise ValueError("HDF5 file not opened, use with context manager")
        
        # Create dataset
        self.h5.create_dataset(
            name,
            data=data,
            compression=compression,
            chunks=chunks if chunks else None
        )
    
    def add_attributes(self, dataset_name, attributes):
        """
        Add attributes to a dataset
        
        Args:
            dataset_name: Name of the dataset
            attributes: Dictionary of attributes
        """
        if self.h5 is None:
            raise ValueError("HDF5 file not opened, use with context manager")
        
        # Get dataset
        if dataset_name not in self.h5:
            raise ValueError(f"Dataset {dataset_name} not found")
        
        dataset = self.h5[dataset_name]
        
        # Add attributes
        for key, value in attributes.items():
            dataset.attrs[key] = value


def prepare_radar_data(
    input_data,
    output_path,
    include_meteorological=True,
    include_derived=True,
    thresholds=None
):
    """
    Prepare radar data for hail prediction
    
    Args:
        input_data: Path to input data (can be a directory or a file)
        output_path: Path to save processed data
        include_meteorological: Whether to include meteorological variables
        include_derived: Whether to include derived variables
        thresholds: Dictionary of thresholds for classification
        
    Returns:
        Path to processed data
    """
    # Default thresholds
    if thresholds is None:
        thresholds = {
            'hail': 55.0,  # dBZ threshold for hail
            'rain': 35.0,  # dBZ threshold for rain
        }
    
    # Check if input is a directory or a file
    if os.path.isdir(input_data):
        # Find all radar files in the directory
        radar_files = []
        for root, _, files in os.walk(input_data):
            for file in files:
                if file.endswith(('.nc', '.h5', '.hdf5', '.HDF5', '.grib', '.grib2')):
                    radar_files.append(os.path.join(root, file))
        
        if not radar_files:
            raise ValueError(f"No radar files found in {input_data}")
        
        # Process each file
        with HDF5Creator(output_path) as creator:
            # Process radar data
            radar_data = process_radar_files(radar_files)
            creator.add_dataset('radar', radar_data)
            
            # Add meteorological variables if requested
            if include_meteorological:
                met_vars = process_meteorological_data(radar_files, radar_data.shape[0])
                for name, data in met_vars.items():
                    creator.add_dataset(name, data)
            
            # Add derived variables if requested
            if include_derived:
                derived_vars = compute_derived_variables(radar_data, met_vars if include_meteorological else None)
                for name, data in derived_vars.items():
                    creator.add_dataset(name, data)
                
                # Add hail probability based on thresholds
                hail_prob = (radar_data > thresholds['hail']).astype(np.float32)
                creator.add_dataset('hail_probability', hail_prob)
    
    else:
        # Process single file
        with HDF5Creator(output_path) as creator:
            # Load radar data
            radar_data = load_radar_data(input_data)
            creator.add_dataset('radar', radar_data)
            
            # Add meteorological variables if requested
            if include_meteorological:
                met_vars = load_meteorological_data(input_data)
                for name, data in met_vars.items():
                    creator.add_dataset(name, data)
            
            # Add derived variables if requested
            if include_derived:
                derived_vars = compute_derived_variables(radar_data, met_vars if include_meteorological else None)
                for name, data in derived_vars.items():
                    creator.add_dataset(name, data)
                
                # Add hail probability based on thresholds
                hail_prob = (radar_data > thresholds['hail']).astype(np.float32)
                creator.add_dataset('hail_probability', hail_prob)
    
    return output_path


def process_radar_files(radar_files):
    """
    Process multiple radar files
    
    Args:
        radar_files: List of radar files
        
    Returns:
        Processed radar data
    """
    # This is a simplified implementation
    # In practice, this would involve:
    # 1. Reading each file
    # 2. Extracting radar reflectivity data
    # 3. Preprocessing (normalization, filtering, etc.)
    # 4. Combining data from multiple files
    
    # Example implementation using h5py
    all_data = []
    
    for file in radar_files:
        # Load data from file
        data = load_radar_data(file)
        all_data.append(data)
    
    # Combine data from multiple files
    combined_data = np.concatenate(all_data, axis=0)
    
    return combined_data


def load_radar_data(radar_file):
    """
    Load radar data from a file
    
    Args:
        radar_file: Path to radar file
        
    Returns:
        Radar data
    """
    # This is a simplified implementation
    # In practice, this would involve:
    # 1. Reading the file format (NetCDF, HDF5, GRIB, etc.)
    # 2. Extracting radar reflectivity data
    # 3. Preprocessing (normalization, filtering, etc.)
    
    # Example implementation using h5py
    if radar_file.endswith(('.h5', '.hdf5', '.HDF5')):
        with h5py.File(radar_file, 'r') as f:
            # Assuming radar data is stored in a dataset named 'reflectivity'
            if 'reflectivity' in f:
                data = f['reflectivity'][:]
            else:
                # Try to find a suitable dataset
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset) and len(f[key].shape) >= 3:
                        data = f[key][:]
                        break
                else:
                    raise ValueError(f"No suitable radar data found in {radar_file}")
    
    elif radar_file.endswith(('.nc')):
        # Example for NetCDF files, requires netCDF4 package
        try:
            import netCDF4 as nc
            with nc.Dataset(radar_file, 'r') as f:
                # Assuming radar data is stored in a variable named 'reflectivity'
                if 'reflectivity' in f.variables:
                    data = f.variables['reflectivity'][:]
                else:
                    # Try to find a suitable variable
                    for var_name in f.variables:
                        var = f.variables[var_name]
                        if len(var.shape) >= 3:
                            data = var[:]
                            break
                    else:
                        raise ValueError(f"No suitable radar data found in {radar_file}")
        except ImportError:
            raise ImportError("netCDF4 package required for reading NetCDF files")
    
    elif radar_file.endswith(('.grib', '.grib2')):
        # Example for GRIB files, requires eccodes package
        try:
            import eccodes as ec
            # This is a simplified example, GRIB processing is more complex in practice
            data = []
            with open(radar_file, 'rb') as f:
                while True:
                    gid = ec.codes_grib_new_from_file(f)
                    if gid is None:
                        break
                    # Check if this message contains radar data
                    param = ec.codes_get(gid, 'paramId')
                    if param in [19, 21]:  # Common radar parameters in GRIB
                        values = ec.codes_get_values(gid)
                        data.append(values.reshape(ec.codes_get(gid, 'Ny'), ec.codes_get(gid, 'Nx')))
                    ec.codes_release(gid)
            data = np.array(data)
        except ImportError:
            raise ImportError("eccodes package required for reading GRIB files")
    
    else:
        raise ValueError(f"Unsupported file format: {radar_file}")
    
    # Ensure data shape is [time, height, width]
    if len(data.shape) == 3:
        pass  # Already in the correct shape
    elif len(data.shape) == 4:
        # If shape is [time, channel, height, width]
        # Extract the first channel
        data = data[:, 0, :, :]
    elif len(data.shape) == 2:
        # If shape is [height, width]
        # Add a time dimension
        data = data[np.newaxis, :, :]
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    
    # Add channel dimension if not present
    data = data[:, np.newaxis, :, :] if len(data.shape) == 3 else data
    
    return data


def process_meteorological_data(radar_files, num_time_steps):
    """
    Process meteorological data from multiple files
    
    Args:
        radar_files: List of radar files
        num_time_steps: Number of time steps in radar data
        
    Returns:
        Dictionary of meteorological variables
    """
    # This is a simplified implementation
    # In practice, this would involve:
    # 1. Reading meteorological data from files (temperature, humidity, etc.)
    # 2. Preprocessing (interpolation to radar grid, etc.)
    # 3. Combining data from multiple files
    
    # Example implementation
    # Generate synthetic meteorological variables
    height, width = 256, 256  # Example spatial dimensions
    
    # Create variables with realistic ranges
    temperature = 270 + 30 * np.random.rand(num_time_steps, 1, height, width)  # 270-300 K
    pressure = 800 + 200 * np.random.rand(num_time_steps, 1, height, width)    # 800-1000 hPa
    qv = 0.005 + 0.015 * np.random.rand(num_time_steps, 1, height, width)      # 0.005-0.02 kg/kg
    vertical_velocity = -5 + 30 * np.random.rand(num_time_steps, 1, height, width)  # -5 to 25 m/s
    
    # Create velocity field (u, v components)
    u = -10 + 20 * np.random.rand(num_time_steps, 1, height, width)  # -10 to 10 m/s
    v = -10 + 20 * np.random.rand(num_time_steps, 1, height, width)  # -10 to 10 m/s
    velocity = np.concatenate([u, v], axis=1)
    
    # Create density field
    density = pressure / (287.0 * temperature)  # Ideal gas law (p = rho*R*T)
    
    return {
        'temperature': temperature.astype(np.float32),
        'pressure': pressure.astype(np.float32),
        'qv': qv.astype(np.float32),
        'vertical_velocity': vertical_velocity.astype(np.float32),
        'velocity': velocity.astype(np.float32),
        'density': density.astype(np.float32)
    }


def load_meteorological_data(data_file):
    """
    Load meteorological data from a file
    
    Args:
        data_file: Path to data file
        
    Returns:
        Dictionary of meteorological variables
    """
    # This is a simplified implementation
    # In practice, this would involve:
    # 1. Reading the file format (NetCDF, HDF5, GRIB, etc.)
    # 2. Extracting meteorological variables
    # 3. Preprocessing (interpolation to radar grid, etc.)
    
    # Example implementation using h5py
    variables = {}
    
    if data_file.endswith(('.h5', '.hdf5', '.HDF5')):
        with h5py.File(data_file, 'r') as f:
            # Check for meteorological variables
            for var_name in ['temperature', 'pressure', 'qv', 'vertical_velocity', 'velocity', 'density']:
                if var_name in f:
                    variables[var_name] = f[var_name][:]
    
    # If no variables were found, generate synthetic data
    if not variables:
        # Get dimensions from radar data
        with h5py.File(data_file, 'r') as f:
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset) and len(f[key].shape) >= 3:
                    data_shape = f[key].shape
                    break
            else:
                raise ValueError(f"No suitable data found in {data_file}")
        
        # Generate synthetic data
        num_time_steps = data_shape[0]
        height, width = data_shape[-2], data_shape[-1]
        
        # Create variables with realistic ranges
        temperature = 270 + 30 * np.random.rand(num_time_steps, 1, height, width)  # 270-300 K
        pressure = 800 + 200 * np.random.rand(num_time_steps, 1, height, width)    # 800-1000 hPa
        qv = 0.005 + 0.015 * np.random.rand(num_time_steps, 1, height, width)      # 0.005-0.02 kg/kg
        vertical_velocity = -5 + 30 * np.random.rand(num_time_steps, 1, height, width)  # -5 to 25 m/s
        
        # Create velocity field (u, v components)
        u = -10 + 20 * np.random.rand(num_time_steps, 1, height, width)  # -10 to 10 m/s
        v = -10 + 20 * np.random.rand(num_time_steps, 1, height, width)  # -10 to 10 m/s
        velocity = np.concatenate([u, v], axis=1)
        
        # Create density field
        density = pressure / (287.0 * temperature)  # Ideal gas law (p = rho*R*T)
        
        variables = {
            'temperature': temperature.astype(np.float32),
            'pressure': pressure.astype(np.float32),
            'qv': qv.astype(np.float32),
            'vertical_velocity': vertical_velocity.astype(np.float32),
            'velocity': velocity.astype(np.float32),
            'density': density.astype(np.float32)
        }
    
    return variables


def compute_derived_variables(radar_data, met_vars=None):
    """
    Compute derived variables from radar and meteorological data
    
    Args:
        radar_data: Radar data
        met_vars: Dictionary of meteorological variables
        
    Returns:
        Dictionary of derived variables
    """
    # This is a simplified implementation
    # In practice, this would involve:
    # 1. Computing derived variables (CAPE, shear, etc.)
    # 2. Computing microphysical variables (qc, qr, qh, etc.)
    
    # Example implementation
    num_time_steps, _, height, width = radar_data.shape
    
    # Initialize derived variables
    derived_vars = {}
    
    # Compute derived variables based on radar data
    # Rain water mixing ratio (qr) based on Z-R relationship
    # Z = 200 * R^1.6, where Z is in mm^6/m^3 and R is in mm/h
    # Convert dBZ to Z: Z = 10^(dBZ/10)
    Z = 10 ** (np.clip(radar_data, 0, 60) / 10)
    R = (Z / 200) ** (1/1.6)  # Rain rate in mm/h
    qr = R / 3600 * 0.001  # Convert to kg/kg (approximate)
    derived_vars['qr'] = qr.astype(np.float32)
    
    # If meteorological variables are available
    if met_vars is not None:
        # Compute CAPE (simplified)
        temperature = met_vars['temperature']
        pressure = met_vars['pressure']
        qv = met_vars['qv']
        
        # Simplified CAPE calculation based on temperature and humidity
        # In practice, CAPE calculation is more complex
        virtual_temp = temperature * (1 + 0.61 * qv)
        environmental_temp = np.mean(virtual_temp, axis=(2, 3), keepdims=True)
        temperature_excess = virtual_temp - environmental_temp
        cape = 9.81 * np.maximum(0, temperature_excess) * 1000  # Simplified
        derived_vars['cape'] = cape.astype(np.float32)
        
        # Compute cloud water mixing ratio (qc) (simplified)
        # Based on temperature and relative humidity
        e_sat = 6.112 * np.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))
        rh = qv * pressure / e_sat
        qc = 0.001 * np.maximum(0, rh - 0.95)  # Simplified
        derived_vars['qc'] = qc.astype(np.float32)
        
        # Compute hail mixing ratio (qh) (simplified)
        # Based on radar reflectivity and temperature
        freezing_level = temperature < 273.15
        hail_mask = (radar_data > 45) & freezing_level
        qh = 0.001 * hail_mask * (radar_data - 45) / 15.0  # Simplified
        derived_vars['qh'] = qh.astype(np.float32)
    
    return derived_vars