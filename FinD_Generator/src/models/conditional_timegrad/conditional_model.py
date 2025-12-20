"""Conditional TimeGrad model that mirrors the vanilla architecture."""
from __future__ import annotations

import torch
import torch.nn as nn

from ..timegrad_core.gaussian_diffusion import GaussianDiffusion
from ..timegrad_core.epsilon_theta import EpsilonTheta
from .conditioned_epsilon_theta import ConditionedEpsilonTheta


class ConditionalTimeGrad(nn.Module):
    """TimeGrad variant with dynamic + static conditioning."""

    def __init__(
        self,
        target_dim: int,
        prediction_length: int,
        seq_len: int,
        cond_dynamic_dim: int,
        cond_static_dim: int,
        diff_steps: int = 100,
        beta_end: float = 0.1,
        beta_schedule: str = "linear",
        residual_layers: int = 6,
        residual_channels: int = 32,
        cond_embed_dim: int = 64,
        cond_attn_heads: int = 4,
        cond_attn_dropout: float = 0.1,
        cond_strategy: str = "fast",
        rnn_type: str = "lstm"
    ) -> None:
        super().__init__()

        base_eps = EpsilonTheta(
            input_size=target_dim,
            prediction_length=prediction_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            cond_channels=cond_embed_dim,
        )

        self.epsilon_theta = ConditionedEpsilonTheta(
            base_epsilon=base_eps,
            cond_dynamic_dim=cond_dynamic_dim,
            cond_static_dim=cond_static_dim,
            seq_len=seq_len,
            prediction_length=prediction_length,
            embed_dim=cond_embed_dim,
            attn_heads=cond_attn_heads,
            attn_dropout=cond_attn_dropout,
            cond_strategy=cond_strategy,
            rnn_type=rnn_type,
        )

        self.diffusion = GaussianDiffusion(
            denoise_fn=self.epsilon_theta,
            input_size=target_dim,
            diff_steps=diff_steps,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )

        self.prediction_length = prediction_length
        self.target_dim = target_dim

    def forward(
        self,
        x_future: torch.Tensor,
        cond_dynamic: torch.Tensor,
        cond_static: torch.Tensor,
    ) -> torch.Tensor:
        """Compute diffusion loss with conditioning.

        Args:
            x_future: Target future window ``[B, horizon, target_dim]``.
            cond_dynamic: Dynamic conditioning ``[B, seq_len, cond_dynamic_dim]``.
            cond_static: Static conditioning ``[B, cond_static_dim]``.
        """
        x = x_future.transpose(1, 2)  # [B, target_dim, horizon]

        cond = {
            "dynamic": cond_dynamic,
            "static": cond_static,
        }

        loss = self.diffusion.p_losses(x_start=x, cond=cond)

        return loss