"""
conditional_timegrad.py

Custom TimeGrad model with conditional generation based on:
- Dynamic features (market data, macro indicators over time)
- Static features (regime labels)

Author: Wooseok Lee
"""

import torch
import torch.nn as nn
from typing import Optional

# Import from your PTS library
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'TimeGrad'))

from pts.modules.gaussian_diffusion import GaussianDiffusion
from pts.model.time_grad.epsilon_theta import EpsilonTheta


class ConditionalTimeGrad(nn.Module):
    """
    Custom TimeGrad that accepts:
    - x_hist: historical target values (batch, seq_len)
    - cond_dynamic: time-varying features (batch, seq_len, n_features)
    - cond_static: static regime labels (batch, n_regime_features)
    """
    def __init__(
        self,
        target_dim: int = 1,
        cond_dynamic_dim: int = 50,
        cond_static_dim: int = 10,
        diff_steps: int = 100,
        beta_end: float = 0.1,
        beta_schedule: str = 'linear',
        residual_layers: int = 8,
        residual_channels: int = 8,
        dilation_cycle_length: int = 2,
        time_emb_dim: int = 16,
        residual_hidden: int = 64,
    ):
        """
        Args:
            target_dim: Dimension of target variable (1 for univariate)
            cond_dynamic_dim: Number of dynamic conditioning features
            cond_static_dim: Number of static conditioning features
            diff_steps: Number of diffusion steps
            beta_end: Final beta value for diffusion schedule
            beta_schedule: Schedule type ('linear', 'cosine', etc.)
            residual_layers: Number of residual blocks in epsilon_theta
            residual_channels: Number of channels in residual blocks
            dilation_cycle_length: Dilation cycle for convolutional layers
            time_emb_dim: Dimension of time embedding
            residual_hidden: Hidden dimension in residual blocks
        """
        super().__init__()
        
        self.target_dim = target_dim
        self.cond_dynamic_dim = cond_dynamic_dim
        self.cond_static_dim = cond_static_dim
        
        # Conditioning length for epsilon_theta
        # Combine dynamic (averaged over time) + static
        self.cond_length = cond_dynamic_dim + cond_static_dim
        
        # Projection layer for dynamic features (seq_len, n_features) -> (n_features,)
        # This aggregates temporal information while preserving feature richness
        self.dynamic_aggregator = nn.Sequential(
            nn.Linear(cond_dynamic_dim, cond_dynamic_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(cond_dynamic_dim, cond_dynamic_dim),
            nn.ReLU(),
        )
        
        # Static feature projection (optional enhancement)
        self.static_projector = nn.Sequential(
            nn.Linear(cond_static_dim, cond_static_dim),
            nn.ReLU(),
        )
        
        # Denoising network (epsilon_theta)
        self.epsilon_theta = EpsilonTheta(
            target_dim=target_dim,
            cond_length=self.cond_length,
            time_emb_dim=time_emb_dim,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
            residual_hidden=residual_hidden,
        )
        
        # Diffusion process
        self.diffusion = GaussianDiffusion(
            denoise_fn=self._denoise_wrapper,
            input_size=target_dim,
            diff_steps=diff_steps,
            beta_end=beta_end,
            loss_type='l2',
            beta_schedule=beta_schedule,
        )
        
    def _denoise_wrapper(self, x, t, cond):
        """
        Wrapper to match epsilon_theta interface.
        
        Args:
            x: Noisy input (batch, 1, target_dim)
            t: Diffusion timestep (batch,)
            cond: Conditioning vector (batch, 1, cond_length)
        
        Returns:
            Denoised output (batch, 1, target_dim)
        """
        return self.epsilon_theta(x, t, cond)
    
    def prepare_conditioning(self, cond_dynamic, cond_static):
        """
        Combine dynamic and static conditioning into a single vector.
        
        Args:
            cond_dynamic: (batch, seq_len, n_dynamic_features)
            cond_static: (batch, n_static_features)
        
        Returns:
            cond: (batch, 1, cond_length)
        """
        # Aggregate dynamic features over time (mean pooling)
        cond_dynamic_agg = cond_dynamic.mean(dim=1)  # (batch, n_dynamic_features)
        cond_dynamic_agg = self.dynamic_aggregator(cond_dynamic_agg)
        
        # Project static features
        cond_static_proj = self.static_projector(cond_static)
        
        # Concatenate dynamic and static
        cond = torch.cat([cond_dynamic_agg, cond_static_proj], dim=-1)  # (batch, cond_length)
        
        return cond.unsqueeze(1)  # (batch, 1, cond_length)
    
    def forward(self, x_future, cond_dynamic, cond_static):
        """
        Training forward pass - computes diffusion loss.
        
        Args:
            x_future: Target to predict (batch, horizon) or (batch, horizon, 1)
            cond_dynamic: Dynamic features (batch, seq_len, n_features)
            cond_static: Static features (batch, n_features)
        
        Returns:
            loss: Scalar diffusion loss (higher = better fit)
        """
        # Prepare conditioning
        cond = self.prepare_conditioning(cond_dynamic, cond_static)
        
        # Ensure x_future has correct shape for diffusion: (batch, horizon, target_dim)
        if x_future.dim() == 2:
            x_future = x_future.unsqueeze(-1)  # (batch, horizon, 1)

        # Transpose for diffusion model: (batch, horizon, 1) -> (batch, 1, horizon)
        x_future = x_future.transpose(1, 2)

        # Expand conditioning to match time dimension of x_future
        # GaussianDiffusion.log_prob expects cond to have same time dim as x
        batch_size, _, horizon = x_future.shape
        cond = cond.expand(batch_size, horizon, -1)  # (batch, horizon, cond_length) -> This is handled inside log_prob
        
        # Compute diffusion loss (log probability)
        loss = self.diffusion.log_prob(x_future, cond)
        
        return -loss.mean()  # Return positive loss for minimization
    
    @torch.no_grad()
    def sample(self, cond_dynamic, cond_static, num_samples=100, horizon=5):
        """
        Generate samples from the model (inference).
        
        Args:
            cond_dynamic: Dynamic features (batch, seq_len, n_features)
            cond_static: Static features (batch, n_features)
            num_samples: Number of Monte Carlo samples to generate
            horizon: Forecast horizon (should match training horizon)
        
        Returns:
            samples: (batch, num_samples, horizon, 1)
        """
        # Prepare conditioning
        cond = self.prepare_conditioning(cond_dynamic, cond_static)
        
        # Expand conditioning vector to generate num_samples in parallel
        # Shape: (batch * num_samples, 1, cond_length)
        cond = cond.repeat_interleave(num_samples, dim=0)
        
        # Generate samples from the diffusion model
        # The diffusion.sample method can generate multiple samples per batch item
        # if the conditioning tensor is properly expanded.
        # It will return shape (batch * num_samples, 1, horizon)
        samples = self.diffusion.sample(cond=cond, horizon=horizon)
        
        # Reshape to (batch, num_samples, horizon, 1)
        batch_size = cond_dynamic.shape[0]
        # Reshape and transpose: (batch * num_samples, 1, horizon) -> (batch, num_samples, 1, horizon) -> (batch, num_samples, horizon, 1)
        samples = samples.view(batch_size, num_samples, self.target_dim, horizon)
        samples = samples.transpose(2, 3)
        return samples
    
    def predict(self, cond_dynamic, cond_static, num_samples=100, return_quantiles=True):
        """
        High-level prediction interface.
        
        Args:
            cond_dynamic: Dynamic features (batch, seq_len, n_features)
            cond_static: Static features (batch, n_features)
            num_samples: Number of Monte Carlo samples
            return_quantiles: If True, return quantiles; else return all samples
        
        Returns:
            If return_quantiles=True:
                Dictionary with keys: 'mean', 'median', 'q10', 'q90', 'std'
            Else:
                samples: (batch, num_samples, horizon, 1)
        """
        horizon = 5 # Default horizon, consider making this an argument
        samples = self.sample(cond_dynamic, cond_static, num_samples, horizon=horizon)
        
        if not return_quantiles:
            return samples
        
        # Compute statistics over samples
        samples_squeezed = samples.squeeze(-1)  # (batch, num_samples, horizon)
        
        predictions = {
            'mean': samples_squeezed.mean(dim=1),      # (batch, horizon)
            'median': samples_squeezed.median(dim=1)[0],  # (batch, horizon)
            'std': samples_squeezed.std(dim=1),        # (batch, horizon)
            'q10': samples_squeezed.quantile(0.1, dim=1),  # (batch, horizon)
            'q25': samples_squeezed.quantile(0.25, dim=1),
            'q75': samples_squeezed.quantile(0.75, dim=1),
            'q90': samples_squeezed.quantile(0.9, dim=1),
        }
        
        return predictions


# ===========================================
# Helper function to initialize model
# ===========================================
def create_conditional_timegrad(
    cond_dynamic_dim: int,
    cond_static_dim: int,
    **kwargs
) -> ConditionalTimeGrad:
    """
    Factory function to create ConditionalTimeGrad with sensible defaults.
    
    Args:
        cond_dynamic_dim: Number of dynamic conditioning features
        cond_static_dim: Number of static conditioning features
        **kwargs: Additional arguments to pass to ConditionalTimeGrad
    
    Returns:
        Initialized ConditionalTimeGrad model
    """
    default_config = {
        'target_dim': 1,
        'diff_steps': 100,
        'beta_end': 0.1,
        'beta_schedule': 'linear',
        'residual_layers': 8,
        'residual_channels': 8,
        'dilation_cycle_length': 2,
    }
    
    # Override defaults with provided kwargs
    config = {**default_config, **kwargs}
    
    model = ConditionalTimeGrad(
        cond_dynamic_dim=cond_dynamic_dim,
        cond_static_dim=cond_static_dim,
        **config
    )
    
    return model