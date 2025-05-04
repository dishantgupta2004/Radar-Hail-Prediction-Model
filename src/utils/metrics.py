"""
Evaluation metrics for the hybrid physics-informed hailstorm prediction model.
"""

import numpy as np
import torch
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim


def to_numpy(tensor):
    """
    Convert tensor to numpy array
    
    Args:
        tensor: Input tensor
        
    Returns:
        Numpy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def compute_rmse(predictions, targets):
    """
    Compute Root Mean Square Error
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        RMSE value
    """
    # Convert tensors to numpy arrays
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    # Compute RMSE
    rmse = np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten()))
    
    return rmse


def compute_mae(predictions, targets):
    """
    Compute Mean Absolute Error
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        MAE value
    """
    # Convert tensors to numpy arrays
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    # Compute MAE
    mae = mean_absolute_error(targets.flatten(), predictions.flatten())
    
    return mae


def compute_correlation(predictions, targets):
    """
    Compute Pearson correlation coefficient
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        Correlation coefficient
    """
    # Convert tensors to numpy arrays
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    # Compute correlation
    corr, _ = pearsonr(targets.flatten(), predictions.flatten())
    
    return corr


def compute_ssim(predictions, targets):
    """
    Compute Structural Similarity Index
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        SSIM value
    """
    # Convert tensors to numpy arrays
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    # Reshape if needed
    if len(predictions.shape) > 2:
        # If we have multiple channels/timesteps, compute SSIM for each and average
        ssim_values = []
        for i in range(predictions.shape[0]):
            for j in range(predictions.shape[1]):
                ssim_values.append(ssim(predictions[i, j], targets[i, j], data_range=targets[i, j].max() - targets[i, j].min()))
        ssim_val = np.mean(ssim_values)
    else:
        # Single image
        ssim_val = ssim(predictions, targets, data_range=targets.max() - targets.min())
    
    return ssim_val


def compute_contingency_metrics(predictions, targets, threshold=0.5):
    """
    Compute contingency table based metrics (POD, FAR, CSI, HSS)
    
    Args:
        predictions: Predicted values
        targets: Target values
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metrics
    """
    # Convert tensors to numpy arrays
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    # Binarize data
    pred_binary = (predictions > threshold).astype(int)
    target_binary = (targets > threshold).astype(int)
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(target_binary.flatten(), pred_binary.flatten(), labels=[0, 1]).ravel()
    
    # Compute metrics
    # Probability of Detection (POD) or Hit Rate
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # False Alarm Ratio (FAR)
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Critical Success Index (CSI) or Threat Score
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    
    # Heidke Skill Score (HSS)
    expected_correct = ((tp + fn) * (tp + fp) + (tn + fn) * (tn + fp)) / (tp + tn + fp + fn)
    hss = (tp + tn - expected_correct) / (tp + tn + fp + fn - expected_correct) if (tp + tn + fp + fn - expected_correct) > 0 else 0
    
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall (same as POD)
    recall = pod
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Intersection over Union (IoU)
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    return {
        'pod': pod,
        'far': far,
        'csi': csi,
        'hss': hss,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }


def compute_fractions_skill_score(predictions, targets, threshold=0.5, neighborhood_size=3):
    """
    Compute Fractions Skill Score (FSS)
    
    Args:
        predictions: Predicted values
        targets: Target values
        threshold: Threshold for binary classification
        neighborhood_size: Size of neighborhood for fraction calculation
        
    Returns:
        FSS value
    """
    # Convert tensors to numpy arrays
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    # Binarize data
    pred_binary = (predictions > threshold).astype(float)
    target_binary = (targets > threshold).astype(float)
    
    # Initialize FSS for each sample
    fss_values = []
    
    # Process each sample independently
    for i in range(predictions.shape[0]):
        # Get single sample
        if len(predictions.shape) > 3:  # Multiple timesteps
            pred_sample = pred_binary[i]
            target_sample = target_binary[i]
        else:
            pred_sample = pred_binary[i:i+1]
            target_sample = target_binary[i:i+1]
        
        # Compute neighborhood fractions
        pred_fractions = np.zeros_like(pred_sample)
        target_fractions = np.zeros_like(target_sample)
        
        # For each timestep
        for t in range(pred_sample.shape[0]):
            # Apply convolution to compute fractions
            from scipy.ndimage import uniform_filter
            pred_fractions[t] = uniform_filter(pred_sample[t], size=neighborhood_size, mode='constant')
            target_fractions[t] = uniform_filter(target_sample[t], size=neighborhood_size, mode='constant')
        
        # Compute FSS
        numer = np.mean((pred_fractions - target_fractions)**2)
        denom = np.mean(pred_fractions**2) + np.mean(target_fractions**2)
        
        fss = 1.0 - numer / denom if denom > 0 else 0.0
        fss_values.append(fss)
    
    return np.mean(fss_values)


def compute_continuous_ranked_probability_score(predictions_ensemble, targets, bins=10):
    """
    Compute Continuous Ranked Probability Score (CRPS)
    
    Args:
        predictions_ensemble: Ensemble of predictions
        targets: Target values
        bins: Number of bins for probability calculation
        
    Returns:
        CRPS value
    """
    # Convert tensors to numpy arrays
    predictions_ensemble = [to_numpy(p) for p in predictions_ensemble]
    targets = to_numpy(targets)
    
    # Stack ensemble predictions
    ensemble = np.stack(predictions_ensemble, axis=0)
    
    # Compute CRPS
    crps_values = []
    
    # Create bin edges
    min_val = min(np.min(ensemble), np.min(targets))
    max_val = max(np.max(ensemble), np.max(targets))
    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # For each grid point
    for i in range(targets.shape[0]):
        for j in range(targets.shape[1]):
            for k in range(targets.shape[2]):
                for l in range(targets.shape[3]):
                    # Get ensemble predictions and observation for this point
                    ens_point = ensemble[:, i, j, k, l]
                    obs_point = targets[i, j, k, l]
                    
                    # Compute CDF for ensemble
                    ensemble_cdf = np.zeros(bins + 1)
                    for m in range(len(ens_point)):
                        ensemble_cdf += (ens_point[m] <= bin_edges).astype(float)
                    ensemble_cdf /= len(ens_point)
                    
                    # Compute CDF for observation (step function)
                    obs_cdf = (bin_edges >= obs_point).astype(float)
                    
                    # Compute CRPS as integral of squared difference between CDFs
                    crps_point = np.mean((ensemble_cdf - obs_cdf)**2)
                    crps_values.append(crps_point)
    
    return np.mean(crps_values)


def compute_spatial_metrics(predictions, targets, thresholds=None):
    """
    Compute spatial verification metrics for different thresholds
    
    Args:
        predictions: Predicted values
        targets: Target values
        thresholds: List of thresholds for binary classification
        
    Returns:
        Dictionary of metrics for each threshold
    """
    if thresholds is None:
        thresholds = [0.1, 0.5, 10.0, 35.0, 45.0]  # Default thresholds (in dBZ for radar)
    
    # Convert tensors to numpy arrays
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    # Initialize dictionary for metrics
    all_metrics = {}
    
    # Compute metrics for each threshold
    for threshold in thresholds:
        metrics = compute_contingency_metrics(predictions, targets, threshold=threshold)
        for metric_name, metric_value in metrics.items():
            all_metrics[f"{metric_name}_{threshold}"] = metric_value
    
    return all_metrics


def compute_metrics(predictions, targets, ensemble=None, thresholds=None):
    """
    Compute comprehensive evaluation metrics
    
    Args:
        predictions: Predicted values
        targets: Target values
        ensemble: Ensemble of predictions for probabilistic metrics
        thresholds: List of thresholds for binary classification
        
    Returns:
        Dictionary of metrics
    """
    # Initialize metrics dictionary
    metrics = {}
    
    # Continuous metrics
    metrics['rmse'] = compute_rmse(predictions, targets)
    metrics['mae'] = compute_mae(predictions, targets)
    metrics['correlation'] = compute_correlation(predictions, targets)
    metrics['ssim'] = compute_ssim(predictions, targets)
    
    # Spatial contingency metrics (using default threshold of 0.5)
    contingency_metrics = compute_contingency_metrics(predictions, targets)
    metrics.update(contingency_metrics)
    
    # Fractions Skill Score
    metrics['fss'] = compute_fractions_skill_score(predictions, targets)
    
    # Probabilistic metrics (if ensemble is provided)
    if ensemble is not None:
        metrics['crps'] = compute_continuous_ranked_probability_score(ensemble, targets)
    
    # Compute metrics for different thresholds
    if thresholds is not None:
        threshold_metrics = compute_spatial_metrics(predictions, targets, thresholds)
        metrics.update(threshold_metrics)
    
    return metrics


def compute_metrics_over_time(predictions, targets, ensemble=None, thresholds=None):
    """
    Compute metrics for each timestep
    
    Args:
        predictions: Predicted values [time, batch, channel, height, width]
        targets: Target values [time, batch, channel, height, width]
        ensemble: Ensemble of predictions for probabilistic metrics
        thresholds: List of thresholds for binary classification
        
    Returns:
        Dictionary of metrics over time
    """
    # Convert tensors to numpy arrays
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    # Compute metrics for each timestep
    time_metrics = {
        'rmse': [],
        'mae': [],
        'correlation': [],
        'ssim': [],
        'pod': [],
        'far': [],
        'csi': [],
        'hss': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'iou': [],
        'fss': []
    }
    
    # Add CRPS if ensemble is provided
    if ensemble is not None:
        time_metrics['crps'] = []
    
    # Compute metrics for each timestep
    num_timesteps = predictions.shape[0]
    for t in range(num_timesteps):
        # Get predictions and targets for this timestep
        pred_t = predictions[t]
        target_t = targets[t]
        
        # Compute metrics
        metrics = compute_metrics(pred_t, target_t, 
                                 ensemble[t] if ensemble is not None else None,
                                 thresholds=None)  # Omit threshold metrics for simplicity
        
        # Store metrics
        for metric_name, metric_value in metrics.items():
            if metric_name in time_metrics:
                time_metrics[metric_name].append(metric_value)
    
    return time_metrics


def compute_metrics_by_leadtime(predictions, targets, leadtimes, thresholds=None):
    """
    Compute metrics grouped by lead time
    
    Args:
        predictions: Predicted values
        targets: Target values
        leadtimes: List of lead times (in minutes)
        thresholds: List of thresholds for binary classification
        
    Returns:
        Dictionary of metrics by lead time
    """
    # Convert tensors to numpy arrays
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    # Initialize metrics dictionary
    metrics_by_leadtime = {lt: {} for lt in leadtimes}
    
    # Compute metrics for each lead time
    for i, lt in enumerate(leadtimes):
        # Get predictions and targets for this lead time
        if len(predictions.shape) > 4:  # [batch, time, channel, height, width]
            pred_lt = predictions[:, i]
            target_lt = targets[:, i]
        else:  # [time, channel, height, width]
            pred_lt = predictions[i:i+1]
            target_lt = targets[i:i+1]
        
        # Compute metrics
        metrics = compute_metrics(pred_lt, target_lt, thresholds=thresholds)
        
        # Store metrics
        metrics_by_leadtime[lt] = metrics
    
    return metrics_by_leadtime


def compute_extreme_value_metrics(predictions, targets, percentile=95):
    """
    Compute metrics focused on extreme values
    
    Args:
        predictions: Predicted values
        targets: Target values
        percentile: Percentile threshold for extreme values
        
    Returns:
        Dictionary of extreme value metrics
    """
    # Convert tensors to numpy arrays
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    # Calculate threshold for extreme values
    threshold = np.percentile(targets, percentile)
    
    # Create masks for extreme values
    extreme_mask = targets > threshold
    
    # Masked predictions and targets
    pred_extreme = predictions[extreme_mask]
    target_extreme = targets[extreme_mask]
    
    # Check if there are extreme values
    if len(pred_extreme) == 0:
        return {
            'extreme_rmse': np.nan,
            'extreme_mae': np.nan,
            'extreme_correlation': np.nan,
            'extreme_bias': np.nan
        }
    
    # Compute metrics for extreme values
    extreme_rmse = np.sqrt(mean_squared_error(target_extreme, pred_extreme))
    extreme_mae = mean_absolute_error(target_extreme, pred_extreme)
    extreme_corr, _ = pearsonr(target_extreme, pred_extreme)
    extreme_bias = np.mean(pred_extreme - target_extreme)
    
    return {
        'extreme_rmse': extreme_rmse,
        'extreme_mae': extreme_mae,
        'extreme_correlation': extreme_corr,
        'extreme_bias': extreme_bias
    }


def compute_hail_specific_metrics(predictions, targets, hail_threshold=55.0, storm_mask=None):
    """
    Compute hail-specific verification metrics
    
    Args:
        predictions: Predicted values (radar reflectivity)
        targets: Target values (radar reflectivity)
        hail_threshold: dBZ threshold for hail classification
        storm_mask: Optional mask to focus on storm regions
        
    Returns:
        Dictionary of hail-specific metrics
    """
    # Convert tensors to numpy arrays
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    # Binary classification based on hail threshold
    pred_hail = predictions > hail_threshold
    target_hail = targets > hail_threshold
    
    # Apply storm mask if provided
    if storm_mask is not None:
        storm_mask = to_numpy(storm_mask)
        pred_hail = pred_hail & storm_mask
        target_hail = target_hail & storm_mask
    
    # Compute basic contingency metrics
    contingency_metrics = compute_contingency_metrics(pred_hail, target_hail, threshold=0.5)
    
    # Compute additional hail-specific metrics
    
    # Hail size estimation error (if predictions represent reflectivity)
    # Convert dBZ to estimated hail size using simplified relationship
    # Hail Size (mm) ≈ (dBZ - 40) / 1.5, for dBZ > 40
    def dbz_to_hail_size(dbz):
        return np.maximum(0, (dbz - 40) / 1.5)
    
    pred_size = dbz_to_hail_size(predictions)
    target_size = dbz_to_hail_size(targets)
    
    # Compute size errors only in regions with hail
    size_mask = target_hail | pred_hail
    if np.any(size_mask):
        size_rmse = np.sqrt(mean_squared_error(
            target_size[size_mask], 
            pred_size[size_mask]
        ))
        size_mae = mean_absolute_error(
            target_size[size_mask], 
            pred_size[size_mask]
        )
    else:
        size_rmse = np.nan
        size_mae = np.nan
    
    # Combine metrics
    hail_metrics = {
        'hail_size_rmse': size_rmse,
        'hail_size_mae': size_mae,
        **{f'hail_{k}': v for k, v in contingency_metrics.items()}
    }
    
    return hail_metrics


def compute_storm_object_metrics(predictions, targets, min_size=10, connectivity=2):
    """
    Compute object-based metrics for storm cells
    
    Args:
        predictions: Predicted values
        targets: Target values
        min_size: Minimum size (in pixels) for a storm object
        connectivity: Pixel connectivity for labeling (1=4-connected, 2=8-connected)
        
    Returns:
        Dictionary of object-based metrics
    """
    # Convert tensors to numpy arrays
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    try:
        from skimage.measure import label, regionprops
    except ImportError:
        raise ImportError("scikit-image is required for object-based metrics")
    
    # Initialize metrics
    object_metrics = {
        'object_pod': [],
        'object_far': [],
        'object_csi': [],
        'centroid_rmse': [],
        'area_error': [],
        'intensity_error': []
    }
    
    # Process each timestep
    for t in range(predictions.shape[0]):
        # Binary classification using reflectivity threshold (35 dBZ is typical for storms)
        storm_threshold = 35.0
        pred_binary = predictions[t, 0] > storm_threshold
        target_binary = targets[t, 0] > storm_threshold
        
        # Label storm objects
        pred_labels = label(pred_binary, connectivity=connectivity)
        target_labels = label(target_binary, connectivity=connectivity)
        
        # Get region properties
        pred_regions = regionprops(pred_labels)
        target_regions = regionprops(target_labels)
        
        # Filter small objects
        pred_regions = [r for r in pred_regions if r.area >= min_size]
        target_regions = [r for r in target_regions if r.area >= min_size]
        
        # Object counts
        n_pred = len(pred_regions)
        n_target = len(target_regions)
        
        if n_target == 0:
            # No target objects, all predicted objects are false alarms
            if n_pred > 0:
                object_metrics['object_pod'].append(0.0)
                object_metrics['object_far'].append(1.0)
                object_metrics['object_csi'].append(0.0)
            continue
            
        if n_pred == 0:
            # No predicted objects, all target objects are misses
            object_metrics['object_pod'].append(0.0)
            object_metrics['object_far'].append(0.0)
            object_metrics['object_csi'].append(0.0)
            continue
        
        # Create object matching matrix based on IoU
        match_matrix = np.zeros((n_target, n_pred))
        for i, target_obj in enumerate(target_regions):
            target_mask = target_labels == target_obj.label
            for j, pred_obj in enumerate(pred_regions):
                pred_mask = pred_labels == pred_obj.label
                # Calculate IoU
                intersection = np.sum(target_mask & pred_mask)
                union = np.sum(target_mask | pred_mask)
                iou = intersection / union if union > 0 else 0
                match_matrix[i, j] = iou
        
        # Match objects based on maximum IoU
        match_threshold = 0.1  # Minimum IoU for a match
        matches = []
        
        # Find best matches
        while np.max(match_matrix) > match_threshold:
            i, j = np.unravel_index(np.argmax(match_matrix), match_matrix.shape)
            matches.append((i, j))
            match_matrix[i, :] = 0  # Remove target object
            match_matrix[:, j] = 0  # Remove predicted object
        
        # Calculate object-based metrics
        hits = len(matches)
        misses = n_target - hits
        false_alarms = n_pred - hits
        
        # Object-based contingency metrics
        object_pod = hits / n_target
        object_far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else 0
        object_csi = hits / (hits + misses + false_alarms) if (hits + misses + false_alarms) > 0 else 0
        
        object_metrics['object_pod'].append(object_pod)
        object_metrics['object_far'].append(object_far)
        object_metrics['object_csi'].append(object_csi)
        
        # Calculate centroid distance, area error, and intensity error for matches
        centroid_distances = []
        area_errors = []
        intensity_errors = []
        
        for target_idx, pred_idx in matches:
            target_obj = target_regions[target_idx]
            pred_obj = pred_regions[pred_idx]
            
            # Centroid distance (in pixels)
            target_centroid = np.array(target_obj.centroid)
            pred_centroid = np.array(pred_obj.centroid)
            distance = np.sqrt(np.sum((target_centroid - pred_centroid)**2))
            centroid_distances.append(distance)
            
            # Area error (relative)
            area_error = abs(pred_obj.area - target_obj.area) / target_obj.area
            area_errors.append(area_error)
            
            # Intensity error (mean value within object)
            target_mask = target_labels == target_obj.label
            pred_mask = pred_labels == pred_obj.label
            target_intensity = np.mean(targets[t, 0][target_mask])
            pred_intensity = np.mean(predictions[t, 0][pred_mask])
            intensity_error = abs(pred_intensity - target_intensity)
            intensity_errors.append(intensity_error)
        
        # Store metrics for this timestep
        if centroid_distances:
            object_metrics['centroid_rmse'].append(np.mean(centroid_distances))
            object_metrics['area_error'].append(np.mean(area_errors))
            object_metrics['intensity_error'].append(np.mean(intensity_errors))
    
    # Average metrics across timesteps
    for key in object_metrics:
        if object_metrics[key]:
            object_metrics[key] = np.mean(object_metrics[key])
        else:
            object_metrics[key] = np.nan
    
    return object_metrics


def compute_comprehensive_metrics(predictions, targets, ensemble=None, metadata=None):
    """
    Compute a comprehensive set of verification metrics
    
    Args:
        predictions: Predicted values
        targets: Target values
        ensemble: Ensemble of predictions for probabilistic metrics
        metadata: Additional metadata (e.g., leadtimes, storm masks)
        
    Returns:
        Dictionary of all metrics
    """
    # Convert tensors to numpy arrays
    predictions = to_numpy(predictions)
    targets = to_numpy(targets)
    
    # Initialize metrics dictionary
    all_metrics = {}
    
    # Basic metrics
    basic_metrics = compute_metrics(predictions, targets)
    all_metrics.update(basic_metrics)
    
    # Metrics over time
    if len(predictions.shape) >= 4 and predictions.shape[0] > 1:
        time_metrics = compute_metrics_over_time(predictions, targets, ensemble)
        all_metrics['time_metrics'] = time_metrics
    
    # Leadtime metrics
    if metadata is not None and 'leadtimes' in metadata:
        leadtime_metrics = compute_metrics_by_leadtime(
            predictions, targets, metadata['leadtimes']
        )
        all_metrics['leadtime_metrics'] = leadtime_metrics
    
    # Extreme value metrics
    extreme_metrics = compute_extreme_value_metrics(predictions, targets)
    all_metrics.update(extreme_metrics)
    
    # Hail-specific metrics
    hail_metrics = compute_hail_specific_metrics(
        predictions, targets, 
        storm_mask=metadata.get('storm_mask', None) if metadata is not None else None
    )
    all_metrics.update(hail_metrics)
    
    # Object-based metrics
    try:
        object_metrics = compute_storm_object_metrics(predictions, targets)
        all_metrics.update(object_metrics)
    except ImportError:
        # Skip object metrics if scikit-image is not available
        pass
    
    return all_metrics