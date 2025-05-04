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
    
    def __init__(self, lambda_adv=1.0, lambda_cont=1.0, lambda_helm=1.0, lambda_micro=1.0):
        """
        Initialize the physics-informed loss module
        
        Args:
            lambda_adv: Weight for advection-diffusion loss
            lambda_cont: Weight for continuity equation loss
            lambda_helm: Weight for Helmholtz equation loss
            lambda_micro: Weight for microphysics equation loss
        """
        super(PhysicsLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.lambda_cont = lambda_cont
        self.lambda_helm = lambda_helm
        self.lambda_micro = lambda_micro
    
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
    
    def microphysics_loss(self, qr, qc, qv, qh, T, p):
        """
        Computes the microphysics equations loss for hail formation:
        This simplified version focuses on the key processes:
        - Conversion of cloud water to rain (autoconversion)
        - Accretion of cloud water by rain
        - Freezing of rainwater to form hail
        - Melting of hail
        
        Args:
            qr: Rainwater mixing ratio (B, T, 1, H, W)
            qc: Cloud water mixing ratio (B, T, 1, H, W)
            qv: Water vapor mixing ratio (B, T, 1, H, W)
            qh: Hail mixing ratio (B, T, 1, H, W)
            T: Temperature (B, T, 1, H, W)
            p: Pressure (B, T, 1, H, W)
            
        Returns:
            Microphysics residual loss
        """
        # Constants
        autoconv_rate = 0.001  # Autoconversion rate [s^-1]
        accretion_rate = 0.01  # Accretion rate [s^-1]
        freezing_threshold = 273.15  # Freezing temperature [K]
        freezing_rate = 0.005  # Freezing rate [s^-1]
        melting_rate = 0.002  # Melting rate [s^-1]
        
        # Compute saturation vapor pressure (simplified Bolton formula)
        e_sat = 6.112 * torch.exp(17.67 * (T - 273.15) / (T - 29.65))
        
        # Compute relative humidity
        rh = qv * p / e_sat
        
        # Autoconversion (cloud water to rain)
        auto_conv = autoconv_rate * qc * torch.relu(qc - 0.0005)
        
        # Accretion (rain collecting cloud water)
        accretion = accretion_rate * qr * qc
        
        # Freezing (rain to hail, enhanced when T < freezing_threshold)
        freezing = freezing_rate * qr * torch.relu(freezing_threshold - T)
        
        # Melting (hail to rain, enhanced when T > freezing_threshold)
        melting = melting_rate * qh * torch.relu(T - freezing_threshold)
        
        # Growth of hail by riming (collection of supercooled water)
        riming = accretion_rate * qh * qc * torch.relu(freezing_threshold - T)
        
        # Compute rates of change for the different water species
        dqc_dt = -auto_conv - accretion - riming
        dqr_dt = auto_conv + accretion - freezing + melting
        dqh_dt = freezing - melting + riming
        
        # Compute residuals (difference between predicted changes and physical changes)
        # We compare the actual changes in the prediction with what would be expected from the microphysics
        # Using finite differences to approximate the time derivatives
        
        # Get actual changes from model predictions
        dqc_dt_pred = torch.zeros_like(qc[:, 1:])
        dqr_dt_pred = torch.zeros_like(qr[:, 1:])
        dqh_dt_pred = torch.zeros_like(qh[:, 1:])
        
        if qc.shape[1] > 1:
            dqc_dt_pred = (qc[:, 1:] - qc[:, :-1])
        if qr.shape[1] > 1:
            dqr_dt_pred = (qr[:, 1:] - qr[:, :-1])
        if qh.shape[1] > 1:
            dqh_dt_pred = (qh[:, 1:] - qh[:, :-1])
        
        # Compute residuals
        qc_residual = dqc_dt_pred - dqc_dt[:, :-1] if qc.shape[1] > 1 else torch.zeros_like(dqc_dt)
        qr_residual = dqr_dt_pred - dqr_dt[:, :-1] if qr.shape[1] > 1 else torch.zeros_like(dqr_dt)
        qh_residual = dqh_dt_pred - dqh_dt[:, :-1] if qh.shape[1] > 1 else torch.zeros_like(dqh_dt)
        
        # Compute the total residual
        residual = torch.mean(qc_residual**2 + qr_residual**2 + qh_residual**2)
        
        return residual
    
    def adiabatic_loss(self, T, p, w):
        """
        Computes the adiabatic process loss:
        dT/dt + w * g/cp = 0 (dry adiabatic lapse rate)
        
        Args:
            T: Temperature (B, T, 1, H, W)
            p: Pressure (B, T, 1, H, W)
            w: Vertical velocity (B, T, 1, H, W)
            
        Returns:
            Adiabatic process residual loss
        """
        # Constants
        g = 9.81  # Gravitational acceleration [m/s^2]
        cp = 1004.0  # Specific heat capacity of air at constant pressure [J/kg/K]
        R = 287.0  # Gas constant for dry air [J/kg/K]
        
        # Calculate lapse rate (adiabatic cooling rate due to rising air)
        gamma_d = g / cp  # Dry adiabatic lapse rate [K/m]
        
        # Calculate expected temperature change due to vertical motion (simplified)
        dT_dt_expected = -w * gamma_d
        
        # Calculate actual temperature change from model predictions
        dT_dt_actual = torch.zeros_like(T[:, 1:])
        if T.shape[1] > 1:
            dT_dt_actual = (T[:, 1:] - T[:, :-1])
        
        # Compute residual
        residual = dT_dt_actual - dT_dt_expected[:, :-1] if T.shape[1] > 1 else torch.zeros_like(dT_dt_expected)
        
        return torch.mean(residual**2)
    
    def convective_instability_loss(self, T, p, qv):
        """
        Computes the convective instability loss based on CAPE:
        Simplified model for convective available potential energy,
        which is a key indicator for the potential of hail formation.
        
        Args:
            T: Temperature (B, T, 1, H, W)
            p: Pressure (B, T, 1, H, W)
            qv: Water vapor mixing ratio (B, T, 1, H, W)
            
        Returns:
            Convective instability residual loss
        """
        # Constants
        g = 9.81  # Gravitational acceleration [m/s^2]
        cp = 1004.0  # Specific heat capacity of air at constant pressure [J/kg/K]
        R = 287.0  # Gas constant for dry air [J/kg/K]
        p0 = 1000.0  # Reference pressure [hPa]
        
        # Calculate potential temperature
        theta = T * (p0 / p) ** (R / cp)
        
        # Calculate equivalent potential temperature (simplified)
        L_v = 2.5e6  # Latent heat of vaporization [J/kg]
        theta_e = theta * torch.exp(L_v * qv / (cp * T))
        
        # Vertical gradient of theta_e (simplified)
        dtheta_e_dz = torch.zeros_like(theta_e[:, :, :, 1:, :])
        if theta_e.shape[3] > 1:
            dtheta_e_dz = (theta_e[:, :, :, 1:, :] - theta_e[:, :, :, :-1, :])
        
        # Negative gradient indicates potential instability
        instability = -torch.relu(-dtheta_e_dz)
        
        # In hail-conducive environments, the instability should align with the predicted hail
        # The loss checks if the regions of high instability align with predicted hail occurrence
        # This is a simplified approximation; in reality, this relationship is more complex
        
        # Assuming the model prediction (output) includes a channel for hail probability
        # Model output should have high hail probability in regions of high instability
        
        return torch.mean(instability**2)
    
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
                - qr: Rainwater mixing ratio (B, T, 1, H, W)
                - qc: Cloud water mixing ratio (B, T, 1, H, W)
                - qv: Water vapor mixing ratio (B, T, 1, H, W)
                - qh: Hail mixing ratio (B, T, 1, H, W)
                - pressure: Pressure (B, T, 1, H, W)
                - vertical_velocity: Vertical velocity (B, T, 1, H, W)
        
        Returns:
            Total physics-informed loss and component losses
        """
        # Extract required variables from metadata
        temperature = metadata.get('temperature', None)
        density = metadata.get('density', None)
        velocity = metadata.get('velocity', None)
        wave_function = metadata.get('wave_function', None)
        time_steps = metadata.get('time_steps', None)
        qr = metadata.get('qr', None)
        qc = metadata.get('qc', None)
        qv = metadata.get('qv', None)
        qh = metadata.get('qh', None)
        pressure = metadata.get('pressure', None)
        vertical_velocity = metadata.get('vertical_velocity', None)
        
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
        
        # Add microphysics loss if variables are available
        if all(var is not None for var in [qr, qc, qv, qh, temperature, pressure]):
            micro_loss = self.microphysics_loss(qr, qc, qv, qh, temperature, pressure)
            total_loss += self.lambda_micro * micro_loss
            component_losses['microphysics'] = micro_loss.item()
        
        # Add adiabatic loss if variables are available
        if all(var is not None for var in [temperature, pressure, vertical_velocity]):
            adiabatic_loss = self.adiabatic_loss(temperature, pressure, vertical_velocity)
            total_loss += self.lambda_micro * adiabatic_loss  # Using same weight as microphysics
            component_losses['adiabatic'] = adiabatic_loss.item()
        
        # Add convective instability loss if variables are available
        if all(var is not None for var in [temperature, pressure, qv]):
            instability_loss = self.convective_instability_loss(temperature, pressure, qv)
            total_loss += self.lambda_micro * instability_loss  # Using same weight as microphysics
            component_losses['instability'] = instability_loss.item()
        
        return total_loss, component_losses


class WeightedPhysicsLoss(PhysicsLoss):
    """
    Extended physics loss with spatially-adaptive weighting
    for focusing on regions of high hail probability
    """
    
    def __init__(self, lambda_adv=1.0, lambda_cont=1.0, lambda_helm=1.0, lambda_micro=1.0, hail_weight=2.0):
        """
        Initialize the weighted physics loss
        
        Args:
            lambda_adv: Weight for advection-diffusion loss
            lambda_cont: Weight for continuity equation loss
            lambda_helm: Weight for Helmholtz equation loss
            lambda_micro: Weight for microphysics equation loss
            hail_weight: Additional weight for regions with high hail probability
        """
        super(WeightedPhysicsLoss, self).__init__(lambda_adv, lambda_cont, lambda_helm, lambda_micro)
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
        if hail_prob is None and isinstance(predictions, dict):
            # Try to extract from predictions if not in metadata
            hail_prob = predictions.get('prediction', None)
        
        if hail_prob is not None:
            weights = self.compute_weights(hail_prob)
            
            # Recompute the weighted total loss
            weighted_loss = total_loss * torch.mean(weights)
            component_losses['weighting_factor'] = torch.mean(weights).item()
            
            return weighted_loss, component_losses
        
        return total_loss, component_losses


class HailPredictionPhysicsLoss(WeightedPhysicsLoss):
    """
    Specialized physics loss for hail prediction, incorporating additional
    hail-specific physical processes and variables
    """
    
    def __init__(
        self, 
        lambda_adv=1.0, 
        lambda_cont=1.0, 
        lambda_helm=1.0, 
        lambda_micro=1.0,
        lambda_updraft=1.0,
        lambda_hail_growth=1.0,
        hail_weight=2.0
    ):
        """
        Initialize the hail prediction physics loss
        
        Args:
            lambda_adv: Weight for advection-diffusion loss
            lambda_cont: Weight for continuity equation loss
            lambda_helm: Weight for Helmholtz equation loss
            lambda_micro: Weight for microphysics equation loss
            lambda_updraft: Weight for updraft intensity loss
            lambda_hail_growth: Weight for hail growth process loss
            hail_weight: Additional weight for regions with high hail probability
        """
        super(HailPredictionPhysicsLoss, self).__init__(
            lambda_adv, lambda_cont, lambda_helm, lambda_micro, hail_weight
        )
        self.lambda_updraft = lambda_updraft
        self.lambda_hail_growth = lambda_hail_growth
    
    def updraft_intensity_loss(self, vertical_velocity, cape, hail_prob):
        """
        Computes the loss for the relationship between updraft intensity, 
        convective available potential energy (CAPE), and hail probability
        
        Args:
            vertical_velocity: Vertical velocity (B, T, 1, H, W)
            cape: Convective available potential energy (B, T, 1, H, W)
            hail_prob: Hail probability (B, T, 1, H, W)
            
        Returns:
            Updraft intensity relationship loss
        """
        # Strong updrafts are necessary for hail formation
        # Updraft intensity generally scales with sqrt(2*CAPE)
        expected_w_max = torch.sqrt(2.0 * cape)
        
        # Create a mask for regions with significant hail probability
        hail_mask = (hail_prob > 0.3).float()
        
        # In hail regions, vertical velocity should be close to expected maximum
        velocity_deficit = torch.relu(expected_w_max - vertical_velocity) * hail_mask
        
        # Compute loss
        loss = torch.mean(velocity_deficit**2)
        
        return loss
    
    def hail_growth_process_loss(self, qh, qc, qr, qv, T, w):
        """
        Computes the loss for hail growth processes, focusing on:
        - Wet growth regime (above freezing level)
        - Dry growth regime (below freezing level)
        - Impact of updraft strength on hail size
        
        Args:
            qh: Hail mixing ratio (B, T, 1, H, W)
            qc: Cloud water mixing ratio (B, T, 1, H, W)
            qr: Rain water mixing ratio (B, T, 1, H, W)
            qv: Water vapor mixing ratio (B, T, 1, H, W)
            T: Temperature (B, T, 1, H, W)
            w: Vertical velocity (B, T, 1, H, W)
            
        Returns:
            Hail growth process loss
        """
        # Constants
        freezing_temp = 273.15  # Freezing temperature [K]
        wet_growth_threshold = 0.01  # Threshold for wet growth regime
        
        # Determine growth regime
        is_freezing = (T < freezing_temp).float()
        
        # Wet growth regime (riming and collection of supercooled water)
        wet_growth_rate = qc * w * is_freezing
        
        # Dry growth regime (deposition of water vapor)
        # Simplified formula based on vapor density gradient
        e_sat_ice = 6.112 * torch.exp(22.46 * (T - 273.15) / (T - 0.53))
        vapor_gradient = qv - e_sat_ice
        dry_growth_rate = vapor_gradient * is_freezing
        
        # Combined growth rate
        growth_rate = wet_growth_rate + dry_growth_rate
        
        # Growth time (residence time in updraft)
        # Simplified as proportional to updraft strength
        growth_time = torch.clamp(10.0 / (w + 1e-6), min=0.0, max=100.0)
        
        # Expected hail content based on growth processes
        expected_qh = growth_rate * growth_time
        
        # Loss based on the difference between modeled and expected hail content
        loss = torch.mean((qh - expected_qh)**2)
        
        return loss
    
    def forward(self, predictions, metadata):
        """
        Compute the total hail-specific physics-informed loss
        
        Args:
            predictions: Model predictions (B, T, C, H, W)
            metadata: Dictionary containing required physical variables
                - All variables from parent class
                - cape: Convective available potential energy (B, T, 1, H, W)
        
        Returns:
            Total hail-specific physics-informed loss
        """
        # Get base weighted physics loss
        total_loss, component_losses = super().forward(predictions, metadata)
        
        # Extract additional variables
        vertical_velocity = metadata.get('vertical_velocity', None)
        cape = metadata.get('cape', None)
        hail_prob = metadata.get('hail_probability', None)
        qh = metadata.get('qh', None)
        qc = metadata.get('qc', None)
        qr = metadata.get('qr', None)
        qv = metadata.get('qv', None)
        temperature = metadata.get('temperature', None)
        
        # If hail_probability is not in metadata, try to get from predictions
        if hail_prob is None and isinstance(predictions, dict):
            hail_prob = predictions.get('prediction', None)
        elif hail_prob is None and isinstance(predictions, torch.Tensor):
            hail_prob = predictions
        
        # Add updraft intensity loss if variables are available
        if all(var is not None for var in [vertical_velocity, cape, hail_prob]):
            updraft_loss = self.updraft_intensity_loss(vertical_velocity, cape, hail_prob)
            total_loss += self.lambda_updraft * updraft_loss
            component_losses['updraft_intensity'] = updraft_loss.item()
        
        # Add hail growth process loss if variables are available
        if all(var is not None for var in [qh, qc, qr, qv, temperature, vertical_velocity]):
            growth_loss = self.hail_growth_process_loss(qh, qc, qr, qv, temperature, vertical_velocity)
            total_loss += self.lambda_hail_growth * growth_loss
            component_losses['hail_growth'] = growth_loss.item()
        
        return total_loss, component_losses


# Factory function to create the appropriate physics loss
def create_physics_loss(loss_type="standard", **kwargs):
    """
    Factory function to create physics loss modules
    
    Args:
        loss_type: Type of physics loss ('standard', 'weighted', or 'hail')
        **kwargs: Additional arguments for the specific loss type
        
    Returns:
        Instantiated physics loss module
    """
    if loss_type.lower() == "hail":
        return HailPredictionPhysicsLoss(**kwargs)
    elif loss_type.lower() == "weighted":
        return WeightedPhysicsLoss(**kwargs)
    else:
        return PhysicsLoss(**kwargs)