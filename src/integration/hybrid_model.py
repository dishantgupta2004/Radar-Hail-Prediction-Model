"""
Hybrid Transformer-Physics-GenAI Framework for physics-informed hailstorm prediction.

This module implements the core hybrid model that integrates transformer-based deep learning
with physics-informed neural operators and generative components for hailstorm prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from dgmr.dgmr import DGMR
from dgmr.common import ContextConditioningStack, LatentConditioningStack
from dgmr.generators import Generator, Sampler


class FourierLayer(nn.Module):
    """
    Fourier Neural Operator layer for solving PDEs
    """
    def __init__(self, in_channels, out_channels, modes=32):
        super(FourierLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply
        
        # Complex weights for Fourier space multiplication
        self.weights = nn.Parameter(
            torch.FloatTensor(in_channels, out_channels, self.modes, self.modes, 2)
        )
        nn.init.xavier_normal_(self.weights)
        
        # Linear transform in physical space
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def compl_mul2d(self, input, weights):
        """Complex multiplication in Fourier space"""
        # Convert weights to complex
        weights_complex = torch.complex(weights[..., 0], weights[..., 1])
        
        # Complex multiplication
        return torch.einsum("bixy,ioxy->boxy", input, weights_complex)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x, norm="ortho")
        
        # Prepare output array
        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1, 
            dtype=torch.cfloat, device=x.device
        )
        
        # Apply weight only to low frequency components
        out_ft[:, :, :self.modes, :self.modes] = self.compl_mul2d(
            x_ft[:, :, :self.modes, :self.modes], 
            self.weights
        )
        
        # Return to physical space
        x_fourier = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm="ortho")
        
        # Combine with linear transform in physical space
        x_linear = self.conv(x)
        
        return x_fourier + x_linear


class FNO(nn.Module):
    """
    Fourier Neural Operator for learning the solution operator of PDEs
    """
    def __init__(self, in_channels, out_channels, hidden_dim=64, num_layers=4, modes=32):
        super(FNO, self).__init__()
        self.num_layers = num_layers
        
        # Lifting layer
        self.fc0 = nn.Linear(in_channels, hidden_dim)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer(hidden_dim, hidden_dim, modes) for _ in range(num_layers)
        ])
        
        # Non-linearities
        self.activations = nn.ModuleList([
            nn.GELU() for _ in range(num_layers)
        ])
        
        # Projection layer
        self.fc1 = nn.Linear(hidden_dim, out_channels)
        
    def forward(self, x):
        """
        Forward pass through FNO
        
        Args:
            x: Input tensor [batch, channels, height, width]
        
        Returns:
            Output tensor [batch, out_channels, height, width]
        """
        # Lift to higher dimension
        x = x.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [batch, hidden_dim, height, width]
        
        # Apply Fourier layers
        for i in range(self.num_layers):
            x = self.fourier_layers[i](x)
            x = self.activations[i](x)
        
        # Project back to output dimension
        x = x.permute(0, 2, 3, 1)  # [batch, height, width, hidden_dim]
        x = self.fc1(x)
        x = x.permute(0, 3, 1, 2)  # [batch, out_channels, height, width]
        
        return x


class LatentVariableModule(nn.Module):
    """
    Variational latent space module for capturing uncertainty
    """
    def __init__(self, input_dim, latent_dim=128):
        super(LatentVariableModule, self).__init__()
        
        # Encoder network (for computing distribution parameters)
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Determine flattened size (this will depend on your input dimensions)
        # For input of shape [B, C, H, W], the output of encoder will have shape [B, 256*H'*W']
        # where H' = H/8 and W' = W/8 due to the three stride-2 convolutions
        test_input = torch.zeros(1, input_dim, 64, 64)  # Adjust size as needed
        test_output = self.encoder(test_input)
        flattened_size = test_output.shape[1]
        
        # Linear layers for mean and log variance
        self.fc_mu = nn.Linear(flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(flattened_size, latent_dim)
        
        # Decoder network
        self.decoder_fc = nn.Linear(latent_dim, flattened_size)
        
        # Calculate the feature map size after flattening
        self.feature_size = (test_output.shape[0], 256, int(np.sqrt(flattened_size/256)), int(np.sqrt(flattened_size/256)))
        
        self.decoder = nn.Sequential(
            # Assuming the feature map is of shape [B, 256, H/8, W/8]
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Assuming normalized inputs in [0, 1]
        )
        
    def encode(self, x):
        """Encode inputs to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def decode(self, z):
        """Decode from latent space to observation space"""
        h = F.relu(self.decoder_fc(z))
        h = h.view(h.size(0), 256, self.feature_size[2], self.feature_size[3])
        return self.decoder(h)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for sampling from latent distribution"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x):
        """VAE forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def sample(self, num_samples, device):
        """Sample from the latent space"""
        z = torch.randn(num_samples, self.fc_mu.out_features, device=device)
        return self.decode(z)


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for feature fusion
    """
    def __init__(self, query_dim, key_dim, num_heads=8, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        assert self.head_dim * num_heads == query_dim, "query_dim must be divisible by num_heads"
        
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(key_dim, query_dim)
        self.to_v = nn.Linear(key_dim, query_dim)
        
        self.to_out = nn.Sequential(
            nn.Linear(query_dim, query_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, q, k, v, mask=None):
        """
        Apply cross-attention
        
        Args:
            q: Query tensor [batch, seq_len_q, query_dim]
            k: Key tensor [batch, seq_len_k, key_dim]
            v: Value tensor [batch, seq_len_k, key_dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch, seq_len_q, query_dim]
        """
        batch_size, seq_len_q, _ = q.shape
        _, seq_len_k, _ = k.shape
        
        # Project queries, keys, values
        q = self.to_q(q).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(k).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(v).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project to output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        output = self.to_out(output)
        
        return output


class FusionLayer(nn.Module):
    """
    Fusion layer that combines physics-aware and latent representations
    using cross-attention mechanisms
    """
    def __init__(self, physics_dim, latent_dim, output_dim, num_heads=8):
        super(FusionLayer, self).__init__()
        
        # Ensure dimensions are compatible with multi-head attention
        hidden_dim = max(physics_dim, latent_dim)
        if hidden_dim % num_heads != 0:
            hidden_dim = ((hidden_dim // num_heads) + 1) * num_heads
        
        # Project physics and latent features to common dimension
        self.physics_proj = nn.Conv2d(physics_dim, hidden_dim, kernel_size=1)
        self.latent_proj = nn.Conv2d(latent_dim, hidden_dim, kernel_size=1)
        
        # Cross-attention mechanism
        self.cross_attn = CrossAttention(
            query_dim=hidden_dim,
            key_dim=hidden_dim,
            num_heads=num_heads
        )
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        
    def forward(self, physics_features, latent_features):
        """
        Fuse physics-aware and latent representations
        
        Args:
            physics_features: Features from physics-aware module [batch, physics_dim, H, W]
            latent_features: Features from latent module [batch, latent_dim, H, W]
            
        Returns:
            Fused representation [batch, output_dim, H, W]
        """
        batch_size, _, H, W = physics_features.shape
        
        # Project to common dimension
        physics_proj = self.physics_proj(physics_features)  # [batch, hidden_dim, H, W]
        latent_proj = self.latent_proj(latent_features)     # [batch, hidden_dim, H, W]
        
        # Reshape for attention
        physics_flat = physics_proj.flatten(2).transpose(1, 2)  # [batch, H*W, hidden_dim]
        latent_flat = latent_proj.flatten(2).transpose(1, 2)    # [batch, H*W, hidden_dim]
        
        # Apply cross-attention (physics as query, latent as key/value)
        fused_flat = self.cross_attn(physics_flat, latent_flat, latent_flat)  # [batch, H*W, hidden_dim]
        
        # Reshape back to spatial dimensions
        fused = fused_flat.transpose(1, 2).reshape(batch_size, -1, H, W)  # [batch, hidden_dim, H, W]
        
        # Project to output dimension
        output = self.output_proj(fused)  # [batch, output_dim, H, W]
        
        # Residual connection
        if physics_features.shape[1] == output.shape[1]:
            output = output + physics_features
            
        return output


class HybridPhysicsInformedHailModel(nn.Module):
    """
    Hybrid Transformer-Physics-GenAI Framework for physics-informed hailstorm prediction
    
    This model combines:
    1. DGMR as a base spatiotemporal predictor
    2. FNO for physics-informed neural operator
    3. Latent variable module for uncertainty quantification
    4. Cross-attention based fusion layer
    """
    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        forecast_steps=18,
        base_model_config=None,
        fno_config=None,
        latent_dim=128,
        use_dgmr=True
    ):
        super(HybridPhysicsInformedHailModel, self).__init__()
        
        # Default configurations
        if base_model_config is None:
            base_model_config = {
                'forecast_steps': forecast_steps,
                'input_channels': input_channels,
                'output_shape': 256,  # Default spatial resolution
                'latent_channels': 768,
                'context_channels': 384
            }
        
        if fno_config is None:
            fno_config = {
                'hidden_dim': 64,
                'num_layers': 4,
                'modes': 32
            }
        
        # 1. Base spatiotemporal model (DGMR or components)
        if use_dgmr:
            self.base_model = DGMR(**base_model_config)
        else:
            # Use DGMR components directly for more control
            self.conditioning_stack = ContextConditioningStack(
                input_channels=input_channels,
                output_channels=base_model_config['context_channels']
            )
            self.latent_stack = LatentConditioningStack(
                shape=(8 * input_channels, 
                       base_model_config['output_shape'] // 32, 
                       base_model_config['output_shape'] // 32),
                output_channels=base_model_config['latent_channels']
            )
            self.sampler = Sampler(
                forecast_steps=forecast_steps,
                latent_channels=base_model_config['latent_channels'],
                context_channels=base_model_config['context_channels']
            )
            self.base_model = Generator(
                conditioning_stack=self.conditioning_stack,
                latent_stack=self.latent_stack,
                sampler=self.sampler
            )
        
        # 2. Physics-aware module (FNO)
        self.physics_module = FNO(
            in_channels=input_channels,
            out_channels=output_channels,
            hidden_dim=fno_config['hidden_dim'],
            num_layers=fno_config['num_layers'],
            modes=fno_config['modes']
        )
        
        # 3. Latent variable module for uncertainty
        self.latent_module = LatentVariableModule(
            input_dim=input_channels,
            latent_dim=latent_dim
        )
        
        # 4. Fusion layer
        self.fusion_layer = FusionLayer(
            physics_dim=output_channels,
            latent_dim=output_channels,
            output_dim=output_channels
        )
        
        # 5. Final refinement
        self.refinement = nn.Sequential(
            nn.Conv2d(output_channels * 2, output_channels * 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_channels * 4, output_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_channels * 2, output_channels, kernel_size=3, padding=1)
        )
        
        # Save config
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.forecast_steps = forecast_steps
        self.latent_dim = latent_dim
        
    def _extract_last_timestep(self, x):
        """Extract the last timestep from a sequence"""
        return x[:, -1] if len(x.shape) > 4 else x
    
    def forward(self, x, num_samples=1):
        """
        Forward pass of the hybrid model
        
        Args:
            x: Input tensor [batch, time, channels, height, width]
            num_samples: Number of samples to generate for uncertainty estimation
            
        Returns:
            Dictionary containing:
                - prediction: Main prediction [batch, forecast_steps, channels, height, width]
                - uncertainty: Uncertainty map [batch, forecast_steps, channels, height, width]
                - samples: List of sampled predictions
                - physics_output: Output of physics-aware module
                - latent_params: Mean and log variance of latent distribution
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 1. Generate base prediction using DGMR
        base_output = self.base_model(x)  # [batch, forecast_steps, channels, height, width]
        
        # 2. Process through physics-aware module
        # Extract features for FNO (last input frame)
        physics_input = x[:, -1]  # [batch, channels, height, width]
        physics_output = self.physics_module(physics_input)  # [batch, channels, height, width]
        
        # 3. Process through latent variable module and generate samples
        samples = []
        latent_outputs = []
        mu, logvar = None, None
        
        for _ in range(num_samples):
            # Generate latent sample from input
            if mu is None and logvar is None:
                latent_output, mu, logvar = self.latent_module(physics_input)
            else:
                # Resample from the same distribution for subsequent samples
                z = self.latent_module.reparameterize(mu, logvar)
                latent_output = self.latent_module.decode(z)
            
            latent_outputs.append(latent_output)
        
        # Stack latent outputs
        latent_output_mean = torch.stack(latent_outputs).mean(dim=0)
        
        # 4. Fuse physics-aware and latent representations
        fused_representation = self.fusion_layer(physics_output, latent_output_mean)
        
        # 5. Combine with base prediction and refine
        # Extract the first forecast step from base output
        base_first_step = base_output[:, 0]  # [batch, channels, height, width]
        
        # Concatenate base and fused representations
        combined = torch.cat([base_first_step, fused_representation], dim=1)
        refined = self.refinement(combined)
        
        # Generate full forecast sequence
        # For simplicity, we'll use the same refinement for all timesteps
        # In practice, you might want to have a more sophisticated approach
        prediction = base_output.clone()
        for t in range(self.forecast_steps):
            base_step = base_output[:, t]
            combined = torch.cat([base_step, fused_representation], dim=1)
            prediction[:, t] = self.refinement(combined)
        
        # Generate uncertainty map (standard deviation across samples)
        if num_samples > 1:
            # Generate individual predictions for each latent sample
            sample_predictions = []
            for latent_output in latent_outputs:
                fused = self.fusion_layer(physics_output, latent_output)
                
                sample_pred = base_output.clone()
                for t in range(self.forecast_steps):
                    base_step = base_output[:, t]
                    combined = torch.cat([base_step, fused], dim=1)
                    sample_pred[:, t] = self.refinement(combined)
                
                sample_predictions.append(sample_pred)
                samples.append(sample_pred)
            
            # Compute uncertainty as standard deviation across samples
            samples_stack = torch.stack(sample_predictions, dim=0)
            uncertainty = torch.std(samples_stack, dim=0)
        else:
            # If only one sample, use a simple uncertainty estimate
            uncertainty = torch.abs(prediction - base_output) * 0.1
        
        return {
            'prediction': prediction,
            'uncertainty': uncertainty,
            'samples': samples if num_samples > 1 else [prediction],
            'physics_output': physics_output,
            'latent_params': (mu, logvar)
        }
    
    def compute_kl_loss(self, mu, logvar):
        """
        Compute KL divergence loss for the latent variable module
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            KL divergence loss
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())