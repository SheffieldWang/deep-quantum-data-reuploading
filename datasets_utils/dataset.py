import os
import torch
from torch.utils.data import Dataset, DataLoader
import jax.numpy as jnp
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, n_qubits, n_layers, n_samples, data_type, is_train=True,n_test=10000):
        """
        Initialize the quantum dataset

        Args:
            n_qubits (int): Number of qubits
            n_layers (int): Number of layers
            n_samples (int): Number of samples
            data_type (str): Type of data
            is_train (bool): Whether to use the training set, defaults to True
        """
        # Load the corresponding data
        base_dir = '../datasets'
        if is_train == False:
            n_samples = n_test
        # Load the corresponding data
        prefix = "train" if is_train else "test"
        data_path = os.path.join(base_dir, data_type, f"x_{prefix}_qubit_{n_qubits}_layer_{n_layers}_sample_{n_samples}.npy")
        targets_path = os.path.join(base_dir, data_type, f"y_{prefix}_qubit_{n_qubits}_layer_{n_layers}_sample_{n_samples}.npy")
        self.data = jnp.load(data_path)
        self.targets = jnp.load(targets_path)
        self.class_idx = None
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def numpy_collate(batch):
    """Convert batch data to jax.array format"""
    if isinstance(batch[0], tuple):
        return tuple(map(numpy_collate, zip(*batch)))
    return jnp.array(batch)

def get_quantum_dataloaders(n_qubits, n_layers, n_samples,data_type,batch_size,n_test=10000):
    """
    Get the DataLoader for the quantum dataset
    
    Args:
        n_qubits (int): Number of qubits in the quantum circuit
        n_layers (int): Number of layers in the quantum circuit
        n_samples (int): Number of samples in the training dataset
        data_type (str): Type of dataset to load (e.g. 'linear', 'nonlinear')
        batch_size (int): Size of each batch
        n_test (int): Number of samples in the test dataset, defaults to 10000
    
    Returns:
        tuple: (train_loader, test_loader) containing DataLoader objects for training and testing
    """
    train_dataset = CustomDataset(n_qubits, n_layers, n_samples, data_type, is_train=True,n_test=n_test)
    test_dataset = CustomDataset(n_qubits, n_layers, n_samples, data_type, is_train=False,n_test=n_test)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=numpy_collate
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=numpy_collate
    )
    
    return train_loader, test_loader

def get_dataloaders_with_class_idx(n_qubits, n_layers, n_samples, data_type,batch_size=32,class_idx=1):
    """
    Get filtered dataloaders containing only samples from a specific class.
    
    Args:
        n_qubits (int): Number of qubits in the quantum circuit
        n_layers (int): Number of layers in the quantum circuit 
        n_samples (int): Total number of samples in the dataset
        data_type (str): Type of dataset to load
        batch_size (int): Size of each batch (default: 32)
        class_idx (int): Index of class to filter for (default: 1)
        
    Returns:
        tuple: (train_loader, test_loader) containing only samples from specified class
    """
    
    train_loader, test_loader = get_quantum_dataloaders(n_qubits, n_layers, n_samples,data_type,batch_size)


    
    # Create filtered training dataset
    train_mask = train_loader.dataset.targets == class_idx
    filtered_train_dataset = CustomDataset(n_qubits, n_layers, n_samples, data_type, is_train=True)
    filtered_train_dataset.data = train_loader.dataset.data[train_mask]
    filtered_train_dataset.targets = train_loader.dataset.targets[train_mask]
    filtered_train_dataset.class_idx = class_idx
    
    # Create filtered test dataset
    test_mask = test_loader.dataset.targets == class_idx
    filtered_test_dataset = CustomDataset(n_qubits, n_layers, n_samples, data_type, is_train=False)
    filtered_test_dataset.data = test_loader.dataset.data[test_mask]
    filtered_test_dataset.targets = test_loader.dataset.targets[test_mask]
    filtered_test_dataset.class_idx = class_idx
    
    train_loader_class = DataLoader(filtered_train_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=numpy_collate)
    
    test_loader_class = DataLoader(filtered_test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=numpy_collate)
    
    return train_loader_class, test_loader_class  


