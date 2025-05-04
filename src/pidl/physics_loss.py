"""
Physics-based loss functions for the hybrid hail prediction model.

This module implements the physics-informed loss components that enforce
physical consistency in the predictions, based on fundamental atmospheric
equations relevant to hailstorm formation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PhysicsLoss(nn.Module):
    """
    Physics-informed loss module incorporating atmospheric equations
    """
    
    def __init__(self, lambda_adv=1.0, lambda_cont=1.0, lambda_helm=1.0):
        """
        Initialize the physics-informed loss module
        
        Args:
            lambda_adv: Weight for advection-diffusion loss
            lambda_cont: Weight for continuity equation loss
            lambda_helm: Weight for Helmholtz equation loss
        """
        super(PhysicsLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.lambda_cont = lambda_cont
        self.lambda_helm = lambda_helm
    
    def advection_diffusion_loss(self, phi, u, t, kappa=0.01):
        """
        Computes the advection-diffusion loss:
        ∂φ/∂t + u·∇φ = κ∇²φ
        
        Args:
            phi: Scalar field (temperature, humidity) (B, T, C, H, W)
            u: Velocity field (B, T, 2, H, W) - (u_x, u_y)
            t: Time steps
            kappa: Diffusion coefficient
            
        Returns:
            Advection-diffusion residual loss
        """
        # Create grid for computing spatial derivatives
        batch_size, time_steps, channels, height, width = phi.shape
        
        # Compute temporal gradient: ∂φ/∂t
        dphi_dt = (phi[:, 1:] - phi[:, :-1]) / (t[1:] - t[:-1]).view(-1, 1, 1, 1, 1)
        
        # Compute spatial gradients using finite differences
        # Central difference scheme for interior points
        # ∇φ = (∂φ/∂x, ∂φ/∂y)
        dphi_dx = (phi[:, :-1, :, :, 2:] - phi[:, :-1, :, :, :-2]) / 2.0
        dphi_dy = (phi[:, :-1, :, 2:, :] - phi[:, :-1, :, :-2, :]) / 2.0
        
        # Compute Laplacian (∇²φ)
        # ∇²φ = ∂²φ/∂x² + ∂²φ/∂y²
        d2phi_dx2 = (phi[:, :-1, :, :, 2:] + phi[:, :-1, :, :, :-2] - 2*phi[:, :-1, :, :, 1:-1])
        d2phi_dy2 = (phi[:, :-1, :, 2:, :] + phi[:, :-1, :, :-2, :] - 2*phi[:, :-1, :, 1:-1, :])
        
        # Reshape for proper alignment of spatial derivatives
        dphi_dx = dphi_dx[:, :, :, 1:-1, :]
        dphi_dy = dphi_dy[:, :, :, :, 1:-1]
        
        laplacian = d2phi_dx2[:, :, :, 1:-1, :] + d2phi_dy2[:, :, :, :, 1:-1]
        
        # Compute advection term (u·∇φ)
        u_x = u[:, :-1, 0:1, 1:-1, 1:-1]  # x-component of velocity
        u_y = u[:, :-1, 1:2, 1:-1, 1:-1]  # y-component of velocity
        
        advection = u_x * dphi_dx + u_y * dphi_dy
        
        # Compute the advection-diffusion residual: ∂φ/∂t + u·∇φ - κ∇²φ
        residual = dphi_dt[:, :, :, 1:-1, 1:-1] + advection - kappa * laplacian
        
        # Return the mean squared residual
        return torch.mean(residual**2)
    
    def continuity_loss(self, rho, u, t):
        """
        Computes the continuity equation loss (mass conservation):
        ∂ρ/∂t + ∇·(ρu) = 0
        
        Args:
            rho: Air density (B, T, 1, H, W)
            u: Velocity field (B, T, 2, H, W) - (u_x, u_y)
            t: Time steps
            
        Returns:
            Continuity equation residual loss
        """
        # Compute temporal gradient: ∂ρ/∂t
        drho_dt = (rho[:, 1:] - rho[:, :-1]) / (t[1:] - t[:-1]).view(-1, 1, 1, 1, 1)
        
        # Compute divergence of momentum (∇·(ρu))
        # ∇·(ρu) = ∂(ρu_x)/∂x + ∂(ρu_y)/∂y
        rho_u_x = rho[:, :-1] * u[:, :-1, 0:1]
        rho_u_y = rho[:, :-1] * u[:, :-1, 1:2]
        
        div_rho_u_x = (rho_u_x[:, :, :, :, 2:] - rho_u_x[:, :, :, :, :-2]) / 2.0
        div_rho_u_y = (rho_u_y[:, :, :, 2:, :] - rho_u_y[:, :, :, :-2, :]) / 2.0
        
        # Reshape for proper alignment
        div_rho_u = div_rho_u_x[:, :, :, 1:-1, :] + div_rho_u_y[:, :, :, :, 1:-1]
        
        # Compute the continuity residual: ∂ρ/∂t + ∇·(ρu) = 0
        residual = drho_dt[:, :, :, 1:-1, 1:-1] + div_rho_u
        
        return torch.mean(residual**2)
    
    def helmholtz_loss(self, psi, k=1.0):
        """
        Computes the Helmholtz equation loss:
        ∇²ψ + k²ψ = 0
        
        Args:
            psi: Wave function (B, T, C, H, W)
            k: Wave number
            
        Returns:
            Helmholtz equation residual loss
        """
        # Compute Laplacian (∇²ψ)
        # ∇²ψ = ∂²ψ/∂x² + ∂²ψ/∂y²
        d2psi_dx2 = (psi[:, :, :, :, 2:] + psi[:, :, :, :, :-2] - 2*psi[:, :, :, :, 1:-1])
        d2psi_dy2 = (psi[:, :, :, 2:, :] + psi[:, :, :, :-2, :] - 2*psi[:, :, :, 1:-1, :])
        
        laplacian = d2psi_dx2[:, :, :, 1:-1, :] + d2psi_dy2[:, :, :, :, 1:-1]
        
        # Compute the Helmholtz residual: ∇²ψ + k²ψ = 0
        residual = laplacian + k**2 * psi[:, :, :, 1:-1, 1:-1]
        
        return torch.mean(residual**2)
    
    def forward(self, predictions, metadata):
        """
        Compute the total physics-informed loss
        
        Args:
            predictions: Model predictions (B, T, C, H, W)
            metadata: Dictionary containing required physical variables
                - temperature: Temperature field (B, T, C, H, W)
                - density: Air density (B, T, 1, H, W)
                - velocity: Velocity field (B, T, 2, H, W)
                - wave_function: Wave function for Helmholtz equation (B, T, C, H, W)
                - time_steps: Time steps array
        
        Returns:
            Total physics-informed loss
        """
        # Extract required variables from metadata
        temperature = metadata.get('temperature', None)
        density = metadata.get('density', None)
        velocity = metadata.get('velocity', None)
        wave_function = metadata.get('wave_function', None)
        time_steps = metadata.get('time_steps', None)
        
        # Initialize total loss
        total_loss = 0.0
        component_losses = {}
        
        # Add advection-diffusion loss if variables are available
        if temperature is not None and velocity is not None and time_steps is not None:
            adv_loss = self.advection_diffusion_loss(temperature, velocity, time_steps)
            total_loss += self.lambda_adv * adv_loss
            component_losses['advection_diffusion'] = adv_loss.item()
        
        # Add continuity loss if variables are available
        if density is not None and velocity is not None and time_steps is not None:
            cont_loss = self.continuity_loss(density, velocity, time_steps)
            total_loss += self.lambda_cont * cont_loss
            component_losses['continuity'] = cont_loss.item()
        
        # Add Helmholtz loss if variables are available
        if wave_function is not None:
            helm_loss = self.helmholtz_loss(wave_function)
            total_loss += self.lambda_helm * helm_loss
            component_losses['helmholtz'] = helm_loss.item()
        
        return total_loss, component_losses


class WeightedPhysicsLoss(PhysicsLoss):
    """
    Extended physics loss with spatially-adaptive weighting
    for focusing on regions of high hail probability
    """
    
    def __init__(self, lambda_adv=1.0, lambda_cont=1.0, lambda_helm=1.0, hail_weight=2.0):
        """
        Initialize the weighted physics loss
        
        Args:
            lambda_adv: Weight for advection-diffusion loss
            lambda_cont: Weight for continuity equation loss
            lambda_helm: Weight for Helmholtz equation loss
            hail_weight: Additional weight for regions with high hail probability
        """
        super(WeightedPhysicsLoss, self).__init__(lambda_adv, lambda_cont, lambda_helm)
        self.hail_weight = hail_weight
    
    def compute_weights(self, hail_prob):
        """
        Compute spatial weights based on hail probability
        
        Args:
            hail_prob: Hail probability field (B, T, 1, H, W)
            
        Returns:
            Spatial weights with higher values in potential hail regions
        """
        # Base weight of 1.0 everywhere
        weights = torch.ones_like(hail_prob)
        
        # Add extra weight to regions with high hail probability
        high_prob_regions = (hail_prob > 0.5).float()
        weights = weights + high_prob_regions * self.hail_weight
        
        return weights
    
    def forward(self, predictions, metadata):
        """
        Compute the weighted physics-informed loss
        
        Args:
            predictions: Model predictions (B, T, C, H, W)
            metadata: Dictionary containing required physical variables
                - hail_probability: Hail probability field (B, T, 1, H, W)
                - (Other fields as in parent class)
        
        Returns:
            Weighted physics-informed loss
        """
        # Get base physics loss
        total_loss, component_losses = super().forward(predictions, metadata)
        
        # Apply weighting if hail probability is available
        hail_prob = metadata.get('hail_probability', None)
        if hail_prob is not None:
            weights = self.compute_weights(hail_prob)
            
            # Recompute the weighted total loss
            weighted_loss = total_loss * torch.mean(weights)
            component_losses['weighting_factor'] = torch.mean(weights).item()
            
            return weighted_loss, component_losses
        
        return total_loss, component_losses


# Factory function to create the appropriate physics loss
def create_physics_loss(loss_type="standard", **kwargs):
    """
    Factory function to create physics loss modules
    
    Args:
        loss_type: Type of physics loss ('standard' or 'weighted')
        **kwargs: Additional arguments for the specific loss type
        
    Returns:
        Instantiated physics loss module
    """
    if loss_type.lower() == "weighted":
        return WeightedPhysicsLoss(**kwargs)
    else:
        return PhysicsLoss(**kwargs)