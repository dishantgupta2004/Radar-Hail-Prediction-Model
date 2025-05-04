"""
Visualization utilities for the hybrid physics-informed hailstorm prediction model.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import io
import os


def create_colormap(name='hail'):
    """
    Create custom colormaps for radar and hail visualization
    
    Args:
        name: Name of the colormap ('hail', 'radar', or 'uncertainty')
        
    Returns:
        Matplotlib colormap
    """
    if name.lower() == 'hail':
        # Custom colormap for hail (green to yellow to red)
        colors = [
            (0.0, 0.0, 0.3, 0.0),      # Transparent for no hail
            (0.0, 0.5, 0.0, 0.5),      # Dark green for low probability
            (0.0, 1.0, 0.0, 0.7),      # Green
            (1.0, 1.0, 0.0, 0.8),      # Yellow
            (1.0, 0.5, 0.0, 0.9),      # Orange
            (1.0, 0.0, 0.0, 1.0)       # Red for high probability
        ]
        cmap = LinearSegmentedColormap.from_list('hail', colors)
        
    elif name.lower() == 'radar':
        # Custom colormap for radar reflectivity (blue to red)
        colors = [
            (0.0, 0.0, 0.0, 0.0),      # Transparent for no echo
            (0.0, 0.0, 0.5, 0.5),      # Dark blue for light rain
            (0.0, 0.0, 1.0, 0.6),      # Blue
            (0.0, 1.0, 1.0, 0.7),      # Cyan
            (0.0, 1.0, 0.0, 0.8),      # Green
            (1.0, 1.0, 0.0, 0.9),      # Yellow
            (1.0, 0.5, 0.0, 0.95),     # Orange
            (1.0, 0.0, 0.0, 1.0),      # Red for heavy precipitation
            (0.5, 0.0, 0.5, 1.0)       # Purple for extreme echo
        ]
        cmap = LinearSegmentedColormap.from_list('radar', colors)
        
    elif name.lower() == 'uncertainty':
        # Custom colormap for uncertainty (blue to purple)
        colors = [
            (0.0, 0.0, 0.0, 0.0),      # Transparent for no uncertainty
            (0.0, 0.0, 0.5, 0.5),      # Dark blue for low uncertainty
            (0.0, 0.5, 1.0, 0.7),      # Light blue
            (0.5, 0.0, 1.0, 0.8),      # Purple
            (1.0, 0.0, 1.0, 0.9),      # Magenta
            (1.0, 0.0, 0.5, 1.0)       # Pink for high uncertainty
        ]
        cmap = LinearSegmentedColormap.from_list('uncertainty', colors)
        
    else:
        # Default to viridis
        cmap = plt.cm.viridis
    
    return cmap


def normalize_data(data, vmin=None, vmax=None, scale_type='linear'):
    """
    Normalize data for visualization
    
    Args:
        data: Input data
        vmin: Minimum value for normalization
        vmax: Maximum value for normalization
        scale_type: Type of scaling ('linear', 'log', or 'symlog')
        
    Returns:
        Normalized data and normalization object
    """
    # Set default value range
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    
    # Create normalization
    if scale_type.lower() == 'log':
        # For logarithmic scaling (e.g., rainfall)
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=max(vmax, 1e-5))
    elif scale_type.lower() == 'symlog':
        # For symmetric logarithmic scaling (e.g., diverging data)
        norm = mcolors.SymLogNorm(linthresh=0.01, linscale=0.01, vmin=vmin, vmax=vmax)
    else:
        # Linear scaling
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    return norm


def visualize_predictions(inputs, targets, predictions, uncertainty=None, figsize=(15, 10)):
    """
    Visualize model inputs, targets, and predictions
    
    Args:
        inputs: Input data [time, channel, height, width] or [channel, height, width]
        targets: Target data [time, channel, height, width] or [channel, height, width]
        predictions: Predicted data [time, channel, height, width] or [channel, height, width]
        uncertainty: Uncertainty data [time, channel, height, width] or [channel, height, width]
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Create colormaps
    radar_cmap = create_colormap('radar')
    hail_cmap = create_colormap('hail')
    uncertainty_cmap = create_colormap('uncertainty')
    
    # Create normalization
    radar_norm = normalize_data(None, vmin=0, vmax=65)  # dBZ range
    hail_norm = normalize_data(None, vmin=0, vmax=1)    # Probability range
    uncertainty_norm = normalize_data(None, vmin=0, vmax=0.5)  # Uncertainty range
    
    # Determine number of time steps
    if inputs is not None:
        if len(inputs.shape) == 4:
            input_timesteps = inputs.shape[0]
        else:
            input_timesteps = 1
    else:
        input_timesteps = 0
    
    if predictions is not None:
        if len(predictions.shape) == 4:
            pred_timesteps = predictions.shape[0]
        else:
            pred_timesteps = 1
    else:
        pred_timesteps = 0
    
    if targets is not None:
        if len(targets.shape) == 4:
            target_timesteps = targets.shape[0]
        else:
            target_timesteps = 1
    else:
        target_timesteps = 0
    
    # Determine number of rows and timesteps to display
    num_rows = 2 if targets is not None else 1
    if uncertainty is not None:
        num_rows += 1
    
    timesteps_to_display = max(input_timesteps, pred_timesteps)
    
    # Create figure
    fig, axes = plt.subplots(num_rows, timesteps_to_display, figsize=figsize)
    
    # If only one row, wrap it in a list
    if num_rows == 1:
        axes = [axes]
    
    # If only one column, wrap it in a list
    if timesteps_to_display == 1:
        axes = [[ax] for ax in axes]
    
    # Plot inputs and predictions
    for t in range(timesteps_to_display):
        # Plot input/target
        if inputs is not None and t < input_timesteps:
            if len(inputs.shape) == 4:
                input_data = inputs[t, 0]
            else:
                input_data = inputs[0]
            
            im = axes[0][t].imshow(input_data, cmap=radar_cmap, norm=radar_norm)
            axes[0][t].set_title(f'Input t={t}')
            axes[0][t].axis('off')
            
            # Add colorbar
            if t == input_timesteps - 1:
                divider = make_axes_locatable(axes[0][t])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(im, cax=cax, label='Reflectivity (dBZ)')
        
        # Plot target if available
        if targets is not None and t < target_timesteps:
            if len(targets.shape) == 4:
                target_data = targets[t, 0]
            else:
                target_data = targets[0]
            
            im = axes[1][t].imshow(target_data, cmap=radar_cmap, norm=radar_norm)
            axes[1][t].set_title(f'Target t+{t+1}')
            axes[1][t].axis('off')
            
            # Add colorbar
            if t == target_timesteps - 1:
                divider = make_axes_locatable(axes[1][t])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(im, cax=cax, label='Reflectivity (dBZ)')
        
        # Plot prediction
        if predictions is not None and t < pred_timesteps:
            if len(predictions.shape) == 4:
                pred_data = predictions[t, 0]
            else:
                pred_data = predictions[0]
            
            row_idx = 1 if targets is None else 2
            im = axes[row_idx-1][t].imshow(pred_data, cmap=radar_cmap, norm=radar_norm)
            axes[row_idx-1][t].set_title(f'Prediction t+{t+1}')
            axes[row_idx-1][t].axis('off')
            
            # Add colorbar
            if t == pred_timesteps - 1:
                divider = make_axes_locatable(axes[row_idx-1][t])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(im, cax=cax, label='Reflectivity (dBZ)')
        
        # Plot uncertainty if available
        if uncertainty is not None and t < pred_timesteps:
            row_idx = num_rows - 1
            if len(uncertainty.shape) == 4:
                uncertainty_data = uncertainty[t, 0]
            else:
                uncertainty_data = uncertainty[0]
            
            im = axes[row_idx][t].imshow(uncertainty_data, cmap=uncertainty_cmap, norm=uncertainty_norm)
            axes[row_idx][t].set_title(f'Uncertainty t+{t+1}')
            axes[row_idx][t].axis('off')
            
            # Add colorbar
            if t == pred_timesteps - 1:
                divider = make_axes_locatable(axes[row_idx][t])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(im, cax=cax, label='Uncertainty')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig


def create_uncertainty_map(prediction, uncertainty, threshold=0.5):
    """
    Create an uncertainty map that combines prediction and uncertainty
    
    Args:
        prediction: Prediction data
        uncertainty: Uncertainty data
        threshold: Threshold for prediction
        
    Returns:
        RGB image representing uncertainty
    """
    # Normalize data to [0, 1]
    pred_norm = np.clip(prediction / 65.0, 0, 1)  # Assuming dBZ range of 0-65
    uncertainty_norm = np.clip(uncertainty, 0, 1)
    
    # Create an RGB image
    rgb = np.zeros((prediction.shape[0], prediction.shape[1], 4), dtype=np.float32)
    
    # Red channel: high prediction, high uncertainty (potentially false positives)
    rgb[:, :, 0] = pred_norm * uncertainty_norm
    
    # Green channel: high prediction, low uncertainty (confident predictions)
    rgb[:, :, 1] = pred_norm * (1 - uncertainty_norm)
    
    # Blue channel: low prediction, high uncertainty (potentially false negatives)
    rgb[:, :, 2] = (1 - pred_norm) * uncertainty_norm
    
    # Alpha channel: make background transparent
    rgb[:, :, 3] = np.maximum(pred_norm, uncertainty_norm)
    
    return rgb


def create_animated_forecast(predictions, output_path, fps=2, dpi=100, cmap='radar', vmin=0, vmax=65):
    """
    Create an animated GIF of the forecast
    
    Args:
        predictions: Prediction data [time, channel, height, width]
        output_path: Path to save the animation
        fps: Frames per second
        dpi: Dots per inch (resolution)
        cmap: Colormap to use
        vmin: Minimum value for color scaling
        vmax: Maximum value for color scaling
    """
    # Check input dimensions
    if len(predictions.shape) == 4:
        time_steps, channels, height, width = predictions.shape
    else:
        time_steps, height, width = predictions.shape
        channels = 1
        predictions = predictions.reshape(time_steps, 1, height, width)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get colormap
    if cmap == 'radar':
        cmap = create_colormap('radar')
    elif cmap == 'hail':
        cmap = create_colormap('hail')
    else:
        cmap = plt.get_cmap(cmap)
    
    # Create normalization
    norm = normalize_data(None, vmin=vmin, vmax=vmax)
    
    # Initialize plot
    im = ax.imshow(predictions[0, 0], cmap=cmap, norm=norm)
    title = ax.set_title(f'Forecast t+1')
    colorbar = plt.colorbar(im, ax=ax)
    
    # Update function for animation
    def update(frame):
        im.set_array(predictions[frame, 0])
        title.set_text(f'Forecast t+{frame+1}')
        return [im, title]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=time_steps, interval=1000/fps, blit=True)
    
    # Save animation
    anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    
    # Close figure
    plt.close(fig)


def create_multi_panel_animation(inputs, predictions, uncertainty=None, output_path=None, fps=2, dpi=100):
    """
    Create a multi-panel animation showing inputs, predictions, and uncertainty
    
    Args:
        inputs: Input data [time, channel, height, width]
        predictions: Prediction data [time, channel, height, width]
        uncertainty: Uncertainty data [time, channel, height, width]
        output_path: Path to save the animation (if None, return animation object)
        fps: Frames per second
        dpi: Dots per inch (resolution)
        
    Returns:
        Animation object (if output_path is None) or None
    """
    # Check input dimensions
    if len(inputs.shape) == 4:
        input_steps, _, height, width = inputs.shape
    else:
        input_steps, height, width = inputs.shape
        inputs = inputs.reshape(input_steps, 1, height, width)
    
    if len(predictions.shape) == 4:
        forecast_steps, _, _, _ = predictions.shape
    else:
        forecast_steps, _, _ = predictions.shape
        predictions = predictions.reshape(forecast_steps, 1, height, width)
    
    # Determine number of rows
    num_rows = 2 if uncertainty is None else 3
    
    # Create figure
    fig, axes = plt.subplots(num_rows, 1, figsize=(8, num_rows * 4))
    
    # Get colormaps
    radar_cmap = create_colormap('radar')
    uncertainty_cmap = create_colormap('uncertainty')
    
    # Create normalizations
    radar_norm = normalize_data(None, vmin=0, vmax=65)
    uncertainty_norm = normalize_data(None, vmin=0, vmax=0.5)
    
    # Initialize plots
    im_input = axes[0].imshow(inputs[-1, 0], cmap=radar_cmap, norm=radar_norm)
    im_pred = axes[1].imshow(predictions[0, 0], cmap=radar_cmap, norm=radar_norm)
    
    title_input = axes[0].set_title('Latest Observation')
    title_pred = axes[1].set_title('Forecast t+1')
    
    axes[0].set_axis_off()
    axes[1].set_axis_off()
    
    if uncertainty is not None:
        if len(uncertainty.shape) == 4:
            im_uncertainty = axes[2].imshow(uncertainty[0, 0], cmap=uncertainty_cmap, norm=uncertainty_norm)
        else:
            im_uncertainty = axes[2].imshow(uncertainty.reshape(forecast_steps, 1, height, width)[0, 0], 
                                           cmap=uncertainty_cmap, norm=uncertainty_norm)
        title_uncertainty = axes[2].set_title('Uncertainty t+1')
        axes[2].set_axis_off()
    
    # Add colorbars
    plt.colorbar(im_input, ax=axes[0], label='Reflectivity (dBZ)')
    plt.colorbar(im_pred, ax=axes[1], label='Reflectivity (dBZ)')
    if uncertainty is not None:
        plt.colorbar(im_uncertainty, ax=axes[2], label='Uncertainty')
    
    # Update function for animation
    def update(frame):
        outputs = []
        
        # Always show the last input frame
        outputs.extend([im_input, title_input])
        
        # Update prediction
        if frame < forecast_steps:
            im_pred.set_array(predictions[frame, 0])
            title_pred.set_text(f'Forecast t+{frame+1}')
        outputs.extend([im_pred, title_pred])
        
        # Update uncertainty if available
        if uncertainty is not None:
            if frame < forecast_steps:
                if len(uncertainty.shape) == 4:
                    im_uncertainty.set_array(uncertainty[frame, 0])
                else:
                    im_uncertainty.set_array(uncertainty.reshape(forecast_steps, 1, height, width)[frame, 0])
                title_uncertainty.set_text(f'Uncertainty t+{frame+1}')
            outputs.extend([im_uncertainty, title_uncertainty])
        
        return outputs
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=forecast_steps, interval=1000/fps, blit=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save animation if output path is provided
    if output_path is not None:
        anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
        plt.close(fig)
        return None
    
    return anim


def create_ensemble_visualization_animation(predictions, uncertainty, output_path=None, fps=2, dpi=100, num_samples=5):
    """
    Create an animation showing ensemble members, mean prediction, and uncertainty
    
    Args:
        predictions: Dictionary with 'prediction' (mean) and 'samples' (list of ensemble members)
        uncertainty: Uncertainty data [time, channel, height, width]
        output_path: Path to save the animation (if None, return animation object)
        fps: Frames per second
        dpi: Dots per inch (resolution)
        num_samples: Number of ensemble members to show
        
    Returns:
        Animation object (if output_path is None) or None
    """
    # Extract data
    mean_prediction = predictions['prediction']
    ensemble_members = predictions['samples']
    
    # Limit number of samples to display
    if len(ensemble_members) > num_samples:
        ensemble_members = ensemble_members[:num_samples]
    
    # Check input dimensions
    if len(mean_prediction.shape) == 4:
        forecast_steps, _, height, width = mean_prediction.shape
    else:
        forecast_steps, height, width = mean_prediction.shape
        mean_prediction = mean_prediction.reshape(forecast_steps, 1, height, width)
    
    # Determine number of rows and columns
    num_rows = len(ensemble_members) + 2  # ensemble members + mean + uncertainty
    
    # Create figure
    fig, axes = plt.subplots(num_rows, 1, figsize=(8, num_rows * 3))
    
    # Get colormaps
    radar_cmap = create_colormap('radar')
    uncertainty_cmap = create_colormap('uncertainty')
    
    # Create normalizations
    radar_norm = normalize_data(None, vmin=0, vmax=65)
    uncertainty_norm = normalize_data(None, vmin=0, vmax=0.5)
    
    # Initialize plots
    im_plots = []
    title_plots = []
    
    # Mean prediction
    im_mean = axes[0].imshow(mean_prediction[0, 0], cmap=radar_cmap, norm=radar_norm)
    title_mean = axes[0].set_title('Mean Prediction t+1')
    axes[0].set_axis_off()
    plt.colorbar(im_mean, ax=axes[0], label='Reflectivity (dBZ)')
    im_plots.append(im_mean)
    title_plots.append(title_mean)
    
    # Uncertainty
    if uncertainty is not None:
        if len(uncertainty.shape) == 4:
            im_uncertainty = axes[1].imshow(uncertainty[0, 0], cmap=uncertainty_cmap, norm=uncertainty_norm)
        else:
            im_uncertainty = axes[1].imshow(uncertainty.reshape(forecast_steps, 1, height, width)[0, 0], 
                                           cmap=uncertainty_cmap, norm=uncertainty_norm)
        title_uncertainty = axes[1].set_title('Uncertainty t+1')
        axes[1].set_axis_off()
        plt.colorbar(im_uncertainty, ax=axes[1], label='Uncertainty')
        im_plots.append(im_uncertainty)
        title_plots.append(title_uncertainty)
    
    # Ensemble members
    ensemble_plots = []
    ensemble_titles = []
    for i, member in enumerate(ensemble_members):
        if len(member.shape) == 4:
            im_member = axes[i+2].imshow(member[0, 0], cmap=radar_cmap, norm=radar_norm)
        else:
            im_member = axes[i+2].imshow(member.reshape(forecast_steps, 1, height, width)[0, 0], 
                                        cmap=radar_cmap, norm=radar_norm)
        title_member = axes[i+2].set_title(f'Ensemble Member {i+1} t+1')
        axes[i+2].set_axis_off()
        plt.colorbar(im_member, ax=axes[i+2], label='Reflectivity (dBZ)')
        ensemble_plots.append(im_member)
        ensemble_titles.append(title_member)
    
    # Update function for animation
    def update(frame):
        outputs = []
        
        # Update mean prediction
        im_mean.set_array(mean_prediction[frame, 0])
        title_mean.set_text(f'Mean Prediction t+{frame+1}')
        outputs.extend([im_mean, title_mean])
        
        # Update uncertainty
        if uncertainty is not None:
            if len(uncertainty.shape) == 4:
                im_uncertainty.set_array(uncertainty[frame, 0])
            else:
                im_uncertainty.set_array(uncertainty.reshape(forecast_steps, 1, height, width)[frame, 0])
            title_uncertainty.set_text(f'Uncertainty t+{frame+1}')
            outputs.extend([im_uncertainty, title_uncertainty])
        
        # Update ensemble members
        for i, (im_member, title_member, member) in enumerate(zip(ensemble_plots, ensemble_titles, ensemble_members)):
            if len(member.shape) == 4:
                im_member.set_array(member[frame, 0])
            else:
                im_member.set_array(member.reshape(forecast_steps, 1, height, width)[frame, 0])
            title_member.set_text(f'Ensemble Member {i+1} t+{frame+1}')
            outputs.extend([im_member, title_member])
        
        return outputs
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=forecast_steps, interval=1000/fps, blit=True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save animation if output path is provided
    if output_path is not None:
        anim.save(output_path, writer='pillow', fps=fps, dpi=dpi)
        plt.close(fig)
        return None
    
    return anim


def plot_verification_metrics(metrics, output_path=None):
    """
    Plot verification metrics over time
    
    Args:
        metrics: Dictionary of metrics over time
        output_path: Path to save the plot (if None, show the plot)
    """
    # Extract metrics
    times = metrics.get('times', np.arange(len(next(iter(metrics.values())))))
    metrics_to_plot = ['rmse', 'csi', 'pod', 'far', 'hss', 'iou']
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot metrics
    for i, metric_name in enumerate(metrics_to_plot):
        if metric_name in metrics:
            axes[i].plot(times, metrics[metric_name], marker='o')
            axes[i].set_title(f'{metric_name.upper()}')
            axes[i].set_xlabel('Lead Time (hours)')
            
            # Set y-axis limits based on metric
            if metric_name in ['rmse']:
                axes[i].set_ylim(bottom=0)
            elif metric_name in ['csi', 'pod', 'far', 'hss', 'iou']:
                axes[i].set_ylim(0, 1)
            
            # Add grid
            axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    return fig


def plot_performance_comparison(models_metrics, metric_name='csi', output_path=None):
    """
    Plot performance comparison between different models
    
    Args:
        models_metrics: Dictionary of metrics for different models
        metric_name: Name of the metric to plot
        output_path: Path to save the plot (if None, show the plot)
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot metrics for each model
    for model_name, metrics in models_metrics.items():
        times = metrics.get('times', np.arange(len(metrics[metric_name])))
        ax.plot(times, metrics[metric_name], marker='o', label=model_name)
    
    # Set labels and title
    ax.set_xlabel('Lead Time (hours)')
    ax.set_ylabel(metric_name.upper())
    ax.set_title(f'{metric_name.upper()} Comparison')
    
    # Set y-axis limits based on metric
    if metric_name in ['rmse']:
        ax.set_ylim(bottom=0)
    elif metric_name in ['csi', 'pod', 'far', 'hss', 'iou']:
        ax.set_ylim(0, 1)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show plot
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    return fig