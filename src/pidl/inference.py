"""
Inference script for the hybrid physics-informed hailstorm prediction model.

This module handles the prediction process, including loading a trained model,
running inference, and visualizing results.
"""

import os
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.integration.hybrid_model import HybridPhysicsInformedHailModel
from src.utils.metrics import compute_metrics
from src.utils.visualization import visualize_predictions, create_uncertainty_map


def load_model(model_path, model_config=None, device='cuda'):
    """
    Load a trained model from checkpoint
    
    Args:
        model_path: Path to model checkpoint
        model_config: Configuration dictionary for the model
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Default configurations
    if model_config is None:
        model_config = {
            'input_channels': 1,
            'output_channels': 1,
            'forecast_steps': 18,
            'latent_dim': 128,
            'use_dgmr': True
        }
    
    # Initialize model
    model = HybridPhysicsInformedHailModel(
        input_channels=model_config['input_channels'],
        output_channels=model_config['output_channels'],
        forecast_steps=model_config['forecast_steps'],
        latent_dim=model_config['latent_dim'],
        use_dgmr=model_config['use_dgmr']
    )
    
    # Load state dict
    if device == 'cuda' and torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path))
        model = model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.cpu()
    
    # Set to evaluation mode
    model.eval()
    
    return model


def prepare_input(data, input_seq_len=4, device='cuda'):
    """
    Prepare input data for the model
    
    Args:
        data: Input data (can be a numpy array, h5py dataset, or file path)
        input_seq_len: Length of input sequence
        device: Device to load the data on
        
    Returns:
        Prepared input tensor
    """
    # Handle different input types
    if isinstance(data, str):
        # Load from file
        if data.endswith('.h5') or data.endswith('.hdf5'):
            with h5py.File(data, 'r') as f:
                # Assuming 'radar' is the dataset name
                if 'radar' in f:
                    data = f['radar'][:]
                else:
                    # Try to get the first dataset
                    for key in f.keys():
                        if isinstance(f[key], h5py.Dataset):
                            data = f[key][:]
                            break
    
    # Convert to numpy array if not already
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Extract input sequence
    if len(data.shape) == 5:  # [batch, time, channel, height, width]
        inputs = data[:, :input_seq_len]
    elif len(data.shape) == 4:  # [time, channel, height, width]
        inputs = data[:input_seq_len][np.newaxis, ...]
    elif len(data.shape) == 3:  # [time, height, width]
        inputs = data[:input_seq_len][np.newaxis, ..., np.newaxis]
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    
    # Convert to tensor
    inputs = torch.from_numpy(inputs).float()
    
    # Move to device
    if device == 'cuda' and torch.cuda.is_available():
        inputs = inputs.cuda()
    
    return inputs


def predict(model, inputs, num_samples=20, device='cuda'):
    """
    Run inference with the model
    
    Args:
        model: Trained model
        inputs: Input tensor or data
        num_samples: Number of samples to generate for uncertainty estimation
        device: Device to run inference on
        
    Returns:
        Dictionary of model outputs
    """
    # Prepare input if needed
    if not isinstance(inputs, torch.Tensor):
        inputs = prepare_input(inputs, device=device)
    
    # Move to device
    if device == 'cuda' and torch.cuda.is_available() and not inputs.is_cuda:
        inputs = inputs.cuda()
    
    # Run inference
    with torch.no_grad():
        predictions = model(inputs, num_samples=num_samples)
    
    # Convert tensors to numpy arrays
    for key in predictions:
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy()
        elif isinstance(predictions[key], tuple) and all(isinstance(item, torch.Tensor) for item in predictions[key]):
            predictions[key] = tuple(item.cpu().numpy() for item in predictions[key])
        elif isinstance(predictions[key], list) and all(isinstance(item, torch.Tensor) for item in predictions[key]):
            predictions[key] = [item.cpu().numpy() for item in predictions[key]]
    
    return predictions


def evaluate(model, test_data, targets=None, num_samples=20, device='cuda'):
    """
    Evaluate model performance on test data
    
    Args:
        model: Trained model
        test_data: Test data (inputs)
        targets: Ground truth targets (optional)
        num_samples: Number of samples to generate for uncertainty estimation
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation results
    """
    # Prepare input
    inputs = prepare_input(test_data, device=device)
    
    # Run inference
    predictions = predict(model, inputs, num_samples=num_samples, device=device)
    
    # Compute metrics if targets are provided
    metrics = {}
    if targets is not None:
        # Prepare targets
        if not isinstance(targets, torch.Tensor):
            if isinstance(targets, str):
                # Load from file
                with h5py.File(targets, 'r') as f:
                    # Assuming 'radar' is the dataset name
                    if 'radar' in f:
                        targets = f['radar'][:]
                    else:
                        # Try to get the first dataset
                        for key in f.keys():
                            if isinstance(f[key], h5py.Dataset):
                                targets = f[key][:]
                                break
            targets = torch.from_numpy(np.array(targets)).float()
        
        # Compute metrics
        metrics = compute_metrics(predictions['prediction'], targets)
    
    # Return results
    return {
        'predictions': predictions,
        'metrics': metrics
    }


def batch_predict(model, data_path, output_dir, batch_size=16, num_samples=20, device='cuda'):
    """
    Run batch prediction on a dataset and save results
    
    Args:
        model: Trained model
        data_path: Path to input data
        output_dir: Directory to save outputs
        batch_size: Batch size for prediction
        num_samples: Number of samples to generate for uncertainty estimation
        device: Device to run prediction on
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    with h5py.File(data_path, 'r') as f:
        # Assuming 'radar' is the dataset name
        if 'radar' in f:
            data = f['radar'][:]
        else:
            # Try to get the first dataset
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    data = f[key][:]
                    break
    
    # Get data dimensions
    num_samples_total = data.shape[0] if len(data.shape) >= 4 else 1
    
    # Initialize arrays for results
    prediction_mean = []
    prediction_std = []
    
    # Run batch prediction
    for i in tqdm(range(0, num_samples_total, batch_size)):
        # Extract batch
        batch_end = min(i + batch_size, num_samples_total)
        if len(data.shape) >= 4:
            batch_data = data[i:batch_end]
        else:
            batch_data = data[np.newaxis, ...]
        
        # Prepare input
        inputs = prepare_input(batch_data, device=device)
        
        # Run inference
        with torch.no_grad():
            predictions = model(inputs, num_samples=num_samples)
        
        # Store results
        prediction_mean.append(predictions['prediction'].cpu().numpy())
        prediction_std.append(predictions['uncertainty'].cpu().numpy())
        
        # Visualize first sample in each batch
        if i % (batch_size * 10) == 0:
            sample_idx = 0 if len(inputs.shape) > 4 else 0
            vis_dir = os.path.join(output_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            fig = visualize_predictions(
                inputs[sample_idx].cpu().numpy() if len(inputs.shape) > 4 else inputs.cpu().numpy(),
                None,
                predictions['prediction'][sample_idx].cpu().numpy() if len(predictions['prediction'].shape) > 4 else predictions['prediction'].cpu().numpy(),
                predictions['uncertainty'][sample_idx].cpu().numpy() if len(predictions['uncertainty'].shape) > 4 else predictions['uncertainty'].cpu().numpy()
            )
            
            plt.savefig(os.path.join(vis_dir, f'prediction_{i}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Concatenate results
    prediction_mean = np.concatenate(prediction_mean, axis=0)
    prediction_std = np.concatenate(prediction_std, axis=0)
    
    # Save results
    with h5py.File(os.path.join(output_dir, 'predictions.h5'), 'w') as f:
        f.create_dataset('prediction_mean', data=prediction_mean)
        f.create_dataset('prediction_std', data=prediction_std)
    
    print(f"Predictions saved to {os.path.join(output_dir, 'predictions.h5')}")


def create_ensemble_visualization(predictions, output_path, cmap='viridis'):
    """
    Create visualization of ensemble predictions to show uncertainty
    
    Args:
        predictions: Dictionary of model outputs
        output_path: Path to save visualization
        cmap: Colormap for visualization
    """
    # Extract data
    ensemble = predictions['samples']
    mean = predictions['prediction']
    std = predictions['uncertainty']
    
    # Create figure
    fig, axes = plt.subplots(3, len(mean[0]), figsize=(len(mean[0]) * 3, 9))
    
    # Plot mean prediction
    for t in range(len(mean[0])):
        axes[0, t].imshow(mean[0, t, 0], cmap=cmap)
        axes[0, t].set_title(f'T+{t+1}')
        axes[0, t].axis('off')
    axes[0, 0].set_ylabel('Mean Prediction')
    
    # Plot ensemble samples
    num_samples_to_show = min(3, len(ensemble))
    for s in range(num_samples_to_show):
        for t in range(len(ensemble[s][0])):
            if t < len(mean[0]):
                axes[1, t].imshow(ensemble[s][0, t, 0], cmap=cmap, alpha=0.5)
                axes[1, t].axis('off')
    axes[1, 0].set_ylabel('Ensemble Samples\n(Overlay)')
    
    # Plot uncertainty (standard deviation)
    for t in range(len(std[0])):
        uncertainty_map = create_uncertainty_map(mean[0, t, 0], std[0, t, 0])
        axes[2, t].imshow(uncertainty_map, cmap='viridis')
        axes[2, t].axis('off')
    axes[2, 0].set_ylabel('Uncertainty\n(Standard Deviation)')
    
    # Save figure
    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Ensemble visualization saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with the hybrid physics-informed hailstorm prediction model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to input data")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for prediction")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to generate for uncertainty estimation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    parser.add_argument("--forecast_steps", type=int, default=18, help="Number of forecast steps")
    parser.add_argument("--input_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--output_channels", type=int, default=1, help="Number of output channels")
    
    args = parser.parse_args()
    
    # Create model configuration
    model_config = {
        'input_channels': args.input_channels,
        'output_channels': args.output_channels,
        'forecast_steps': args.forecast_steps,
        'latent_dim': 128,
        'use_dgmr': True
    }
    
    # Load model
    model = load_model(
        model_path=args.model_path,
        model_config=model_config,
        device=args.device
    )
    
    # Run batch prediction
    batch_predict(
        model=model,
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        device=args.device
    )