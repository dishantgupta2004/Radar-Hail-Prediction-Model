"""
Training script for the hybrid physics-informed hailstorm prediction model.

This module handles the training process, including data loading, loss computation,
optimization, and logging/checkpointing.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.integration.hybrid_model import HybridPhysicsInformedHailModel
from src.data.dataloader import HailDataset
from src.pidl.physics_loss import create_physics_loss
from src.utils.metrics import compute_metrics
from src.utils.visualization import visualize_predictions


class PhysicsInformedHailTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training the hybrid physics-informed hailstorm prediction model
    """
    
    def __init__(
        self,
        model_config=None,
        loss_config=None,
        optimizer_config=None,
        data_config=None,
        use_wandb=False
    ):
        """
        Initialize the trainer
        
        Args:
            model_config: Configuration dictionary for the model
            loss_config: Configuration dictionary for the loss function
            optimizer_config: Configuration dictionary for the optimizer
            data_config: Configuration dictionary for data loading
            use_wandb: Whether to use Weights & Biases for logging
        """
        super(PhysicsInformedHailTrainer, self).__init__()
        
        # Default configurations
        if model_config is None:
            model_config = {
                'input_channels': 1,
                'output_channels': 1,
                'forecast_steps': 18,
                'latent_dim': 128,
                'use_dgmr': True
            }
        
        if loss_config is None:
            loss_config = {
                'lambda_data': 1.0,
                'lambda_phys': 0.1,
                'lambda_kl': 0.01,
                'physics_loss_type': 'weighted',
                'lambda_adv': 1.0,
                'lambda_cont': 1.0,
                'lambda_helm': 1.0,
                'hail_weight': 2.0
            }
        
        if optimizer_config is None:
            optimizer_config = {
                'lr': 1e-4,
                'weight_decay': 1e-5,
                'scheduler': 'cosine',
                'warmup_epochs': 10,
                'max_epochs': 100
            }
        
        if data_config is None:
            data_config = {
                'input_seq_len': 4,
                'output_seq_len': 18,
                'batch_size': 16,
                'num_workers': 4
            }
        
        # Save configurations
        self.model_config = model_config
        self.loss_config = loss_config
        self.optimizer_config = optimizer_config
        self.data_config = data_config
        self.use_wandb = use_wandb
        
        # Initialize model
        self.model = HybridPhysicsInformedHailModel(
            input_channels=model_config['input_channels'],
            output_channels=model_config['output_channels'],
            forecast_steps=model_config['forecast_steps'],
            latent_dim=model_config['latent_dim'],
            use_dgmr=model_config['use_dgmr']
        )
        
        # Initialize physics-informed loss
        self.physics_loss = create_physics_loss(
            loss_type=loss_config['physics_loss_type'],
            lambda_adv=loss_config['lambda_adv'],
            lambda_cont=loss_config['lambda_cont'],
            lambda_helm=loss_config['lambda_helm'],
            hail_weight=loss_config.get('hail_weight', 2.0)
        )
        
        # Initialize data loss
        self.data_loss = nn.MSELoss()
        
        # Save hyperparameters for logging
        self.save_hyperparameters()
    
    def forward(self, x, num_samples=1):
        """Forward pass"""
        return self.model(x, num_samples)
    
    def compute_loss(self, predictions, targets, metadata=None):
        """
        Compute the combined loss
        
        Args:
            predictions: Dictionary of model outputs
            targets: Ground truth targets
            metadata: Dictionary of metadata for physics-informed loss
            
        Returns:
            Dictionary of loss components and total loss
        """
        # Extract model outputs
        pred = predictions['prediction']
        mu, logvar = predictions['latent_params']
        
        # Data fidelity loss
        data_loss = self.data_loss(pred, targets)
        
        # Physics-informed loss
        phys_loss, phys_components = self.physics_loss(pred, metadata)
        
        # KL divergence loss
        kl_loss = self.model.compute_kl_loss(mu, logvar)
        
        # Total loss
        total_loss = (
            self.loss_config['lambda_data'] * data_loss +
            self.loss_config['lambda_phys'] * phys_loss +
            self.loss_config['lambda_kl'] * kl_loss
        )
        
        # Compile loss components
        loss_dict = {
            'total_loss': total_loss,
            'data_loss': data_loss,
            'physics_loss': phys_loss,
            'kl_loss': kl_loss
        }
        
        # Add physics components if available
        if isinstance(phys_components, dict):
            for key, value in phys_components.items():
                loss_dict[f'physics_{key}'] = value
        
        return loss_dict
    
    def training_step(self, batch, batch_idx):
        """
        Training step
        
        Args:
            batch: Tuple of (inputs, targets, metadata)
            batch_idx: Batch index
            
        Returns:
            Loss dictionary
        """
        inputs, targets, metadata = batch
        
        # Forward pass
        predictions = self(inputs)
        
        # Compute loss
        loss_dict = self.compute_loss(predictions, targets, metadata)
        
        # Log metrics
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, on_step=True, on_epoch=True, prog_bar=True)
        
        # Compute and log validation metrics periodically
        if batch_idx % 100 == 0:
            metrics = compute_metrics(predictions['prediction'], targets)
            for key, value in metrics.items():
                self.log(f'train/{key}', value, on_step=True, on_epoch=True)
        
        # Visualize predictions periodically
        if self.global_step % 500 == 0:
            fig = visualize_predictions(
                inputs[0], targets[0], predictions['prediction'][0], 
                predictions['uncertainty'][0] if 'uncertainty' in predictions else None
            )
            
            # Log figure
            if self.use_wandb and self.logger:
                import wandb
                self.logger.experiment.log({
                    "train/visualization": wandb.Image(fig)
                })
        
        return loss_dict['total_loss']
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step
        
        Args:
            batch: Tuple of (inputs, targets, metadata)
            batch_idx: Batch index
            
        Returns:
            Loss dictionary
        """
        inputs, targets, metadata = batch
        
        # Forward pass with multiple samples for uncertainty estimation
        predictions = self(inputs, num_samples=10)
        
        # Compute loss
        loss_dict = self.compute_loss(predictions, targets, metadata)
        
        # Compute metrics
        metrics = compute_metrics(predictions['prediction'], targets)
        
        # Log metrics
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, on_epoch=True, prog_bar=True)
        
        for key, value in metrics.items():
            self.log(f'val/{key}', value, on_epoch=True)
        
        # Visualize predictions for a few samples
        if batch_idx == 0:
            figs = []
            for i in range(min(4, len(inputs))):
                fig = visualize_predictions(
                    inputs[i], targets[i], predictions['prediction'][i], 
                    predictions['uncertainty'][i] if 'uncertainty' in predictions else None
                )
                figs.append(fig)
            
            # Log figures
            if self.use_wandb and self.logger:
                import wandb
                for i, fig in enumerate(figs):
                    self.logger.experiment.log({
                        f"val/visualization_{i}": wandb.Image(fig)
                    })
        
        return loss_dict['total_loss']
    
    def test_step(self, batch, batch_idx):
        """
        Test step
        
        Args:
            batch: Tuple of (inputs, targets, metadata)
            batch_idx: Batch index
            
        Returns:
            Loss dictionary
        """
        inputs, targets, metadata = batch
        
        # Forward pass with multiple samples for uncertainty estimation
        predictions = self(inputs, num_samples=20)
        
        # Compute loss
        loss_dict = self.compute_loss(predictions, targets, metadata)
        
        # Compute metrics
        metrics = compute_metrics(predictions['prediction'], targets)
        
        # Log metrics
        for key, value in loss_dict.items():
            self.log(f'test/{key}', value)
        
        for key, value in metrics.items():
            self.log(f'test/{key}', value)
        
        return {
            'loss': loss_dict['total_loss'],
            'metrics': metrics,
            'predictions': predictions,
            'targets': targets,
            'inputs': inputs
        }
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        # Initialize optimizer
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.optimizer_config['lr'],
            weight_decay=self.optimizer_config['weight_decay']
        )
        
        # Initialize scheduler
        scheduler_type = self.optimizer_config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.optimizer_config['max_epochs'] - self.optimizer_config['warmup_epochs']
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return optimizer
        
        # Add warmup
        if self.optimizer_config.get('warmup_epochs', 0) > 0:
            warmup_scheduler = optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: min(1.0, epoch / self.optimizer_config['warmup_epochs'])
            )
            
            return [optimizer], [
                {
                    'scheduler': warmup_scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                    'strict': True,
                    'monitor': 'val/total_loss',
                },
                {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                    'strict': True,
                    'monitor': 'val/total_loss',
                }
            ]
        
        return [optimizer], [scheduler]


def train_model(
    data_path,
    output_dir,
    model_config=None,
    loss_config=None,
    optimizer_config=None,
    data_config=None,
    use_wandb=False
):
    """
    Train the hybrid physics-informed hailstorm prediction model
    
    Args:
        data_path: Path to input data
        output_dir: Directory to save outputs
        model_config: Configuration dictionary for the model
        loss_config: Configuration dictionary for the loss function
        optimizer_config: Configuration dictionary for the optimizer
        data_config: Configuration dictionary for data loading
        use_wandb: Whether to use Weights & Biases for logging
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Default data configuration
    if data_config is None:
        data_config = {
            'input_seq_len': 4,
            'output_seq_len': 18,
            'batch_size': 16,
            'num_workers': 4
        }
    
    # Load datasets
    train_dataset = HailDataset(
        h5_file=data_path,
        input_seq_len=data_config['input_seq_len'],
        output_seq_len=data_config['output_seq_len'],
        split='train'
    )
    
    val_dataset = HailDataset(
        h5_file=data_path,
        input_seq_len=data_config['input_seq_len'],
        output_seq_len=data_config['output_seq_len'],
        split='val'
    )
    
    test_dataset = HailDataset(
        h5_file=data_path,
        input_seq_len=data_config['input_seq_len'],
        output_seq_len=data_config['output_seq_len'],
        split='test'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config['batch_size'],
        shuffle=True,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=data_config['batch_size'],
        shuffle=False,
        num_workers=data_config['num_workers'],
        pin_memory=True
    )
    
    # Initialize PyTorch Lightning model
    model = PhysicsInformedHailTrainer(
        model_config=model_config,
        loss_config=loss_config,
        optimizer_config=optimizer_config,
        data_config=data_config,
        use_wandb=use_wandb
    )
    
    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'checkpoints'),
        filename='hail-prediction-{epoch:02d}-{val/total_loss:.4f}',
        monitor='val/total_loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/total_loss',
        min_delta=0.00,
        patience=20,
        verbose=True,
        mode='min'
    )
    
    # Configure logger
    if use_wandb:
        logger = WandbLogger(
            name=f"hail-prediction-{time.strftime('%Y%m%d-%H%M%S')}",
            project="hail-prediction",
            save_dir=os.path.join(output_dir, 'logs')
        )
    else:
        logger = TensorBoardLogger(
            save_dir=os.path.join(output_dir, 'logs'),
            name="hail-prediction"
        )
    
    # Initialize trainer
    max_epochs = optimizer_config.get('max_epochs', 100) if optimizer_config else 100
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10,
        precision=16 if torch.cuda.is_available() else 32,  # Use mixed precision if available
        gradient_clip_val=1.0
    )
    
    # Train the model
    trainer.fit(model, train_loader, val_loader)
    
    # Test the model
    trainer.test(model, test_loader)
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
    
    # Return the trained model
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train the hybrid physics-informed hailstorm prediction model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases for logging")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--input_seq_len", type=int, default=4, help="Input sequence length")
    parser.add_argument("--output_seq_len", type=int, default=18, help="Output sequence length")
    parser.add_argument("--lambda_data", type=float, default=1.0, help="Weight for data fidelity loss")
    parser.add_argument("--lambda_phys", type=float, default=0.1, help="Weight for physics-informed loss")
    parser.add_argument("--lambda_kl", type=float, default=0.01, help="Weight for KL divergence loss")
    
    args = parser.parse_args()
    
    # Create configurations
    model_config = {
        'input_channels': 1,
        'output_channels': 1,
        'forecast_steps': args.output_seq_len,
        'latent_dim': 128,
        'use_dgmr': True
    }
    
    loss_config = {
        'lambda_data': args.lambda_data,
        'lambda_phys': args.lambda_phys,
        'lambda_kl': args.lambda_kl,
        'physics_loss_type': 'weighted',
        'lambda_adv': 1.0,
        'lambda_cont': 1.0,
        'lambda_helm': 1.0,
        'hail_weight': 2.0
    }
    
    optimizer_config = {
        'lr': args.learning_rate,
        'weight_decay': 1e-5,
        'scheduler': 'cosine',
        'warmup_epochs': 10,
        'max_epochs': args.max_epochs
    }
    
    data_config = {
        'input_seq_len': args.input_seq_len,
        'output_seq_len': args.output_seq_len,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers
    }
    
    # Train the model
    train_model(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_config=model_config,
        loss_config=loss_config,
        optimizer_config=optimizer_config,
        data_config=data_config,
        use_wandb=args.use_wandb
    )