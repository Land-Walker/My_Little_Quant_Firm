"""Training-ready conditional TimeGrad network.

This module wraps ``ConditionalTimeGrad`` with the pieces typically handled by
the GluonTS/PyTorchTS estimator: scale normalization, conditioning prep, and
forecast sampling utilities. It is intentionally self contained so it can be
plugged into a standard PyTorch training loop without any external estimator
machinery.

Design goals for this training network:
- Full compatibility with the cross-attention conditioning stack
- Safe normalization of both targets and conditioning features
- Fast sampling utilities for validation/inference
- Minimal assumptions about the upstream DataLoader (accepts tensors directly)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.models.conditional_timegrad import ConditionalTimeGrad


class ConditionalTimeGradTrainingNetwork(nn.Module):
    """High-level training wrapper for the conditional TimeGrad model."""

    def __init__(
        self,
        target_dim: int,
        context_length: int,
        prediction_length: int,
        cond_dynamic_dim: int,
        cond_static_dim: int,
        *,
        diff_steps: int = 100,
        beta_end: float = 0.1,
        beta_schedule: str = "linear",
        residual_layers: int = 6,
        residual_channels: int = 32,
        cond_embed_dim: int = 64,
        cond_attn_heads: int = 4,
        cond_attn_dropout: float = 0.1,
        cond_strategy: str = "fast",
        rnn_type: str = "lstm",
        scale_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.scale_eps = scale_eps

        # Keep track of inbound conditioning dimensions before augmentation.
        self.input_cond_dynamic_dim = cond_dynamic_dim
        self.input_cond_static_dim = cond_static_dim
        self.cond_embed_dim = cond_embed_dim

        # History encoder turns past targets into conditioning tokens so the
        # denoiser can leverage cross-attention, relative position bias, causal
        # masking, FiLM modulation, and learned alignment modules downstream.
        self.history_encoder = nn.Sequential(
            nn.Conv1d(
                target_dim,
                cond_embed_dim,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
            ),
            nn.LeakyReLU(0.4, inplace=True),
            nn.Conv1d(
                cond_embed_dim,
                cond_embed_dim,
                kernel_size=3,
                padding=1,
                padding_mode="circular",
            ),
            nn.LeakyReLU(0.4, inplace=True),
        )

        # Summary pooling feeds into static FiLM modulation alongside provided
        # static features.
        self.history_pool = nn.AdaptiveAvgPool1d(1)

        # Augment conditioning dims with learned history tokens so the model's
        # cross-attention stack can consume both exogenous signals and encoded
        # past targets.
        combined_cond_dynamic_dim = cond_dynamic_dim + cond_embed_dim
        combined_cond_static_dim = cond_static_dim + cond_embed_dim

        self.model = ConditionalTimeGrad(
            target_dim=target_dim,
            prediction_length=prediction_length,
            seq_len=context_length,
            cond_dynamic_dim=combined_cond_dynamic_dim,
            cond_static_dim=combined_cond_static_dim,
            diff_steps=diff_steps,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            cond_embed_dim=cond_embed_dim,
            cond_attn_heads=cond_attn_heads,
            cond_attn_dropout=cond_attn_dropout,
            cond_strategy=cond_strategy,
            rnn_type=rnn_type,
        )

    def _normalize_target(
        self, x_hist: torch.Tensor, x_future: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute per-series scaling statistics and normalize the target windows."""

        combined = torch.cat([x_hist, x_future], dim=1)  # [B, context+horizon, C]
        loc = combined.mean(dim=1, keepdim=True)
        scale = combined.std(dim=1, keepdim=True).clamp_min(self.scale_eps)

        x_hist_norm = (x_hist - loc) / scale
        x_future_norm = (x_future - loc) / scale
        return x_hist_norm, x_future_norm, loc, scale

    def _normalize_cond(
        self, cond_dynamic: torch.Tensor, cond_static: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize conditioning signals to stabilize training."""

        dyn_loc = cond_dynamic.mean(dim=1, keepdim=True)
        dyn_scale = cond_dynamic.std(dim=1, keepdim=True).clamp_min(self.scale_eps)
        cond_dynamic_norm = (cond_dynamic - dyn_loc) / dyn_scale

        static_loc = cond_static.mean(dim=1, keepdim=True)
        static_scale = cond_static.std(dim=1, keepdim=True).clamp_min(self.scale_eps)
        cond_static_norm = (cond_static - static_loc) / static_scale

        return cond_dynamic_norm, cond_static_norm

    def _prepare_conditioning(
        self,
        x_hist_norm: torch.Tensor,
        cond_dynamic_norm: torch.Tensor,
        cond_static_norm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse history-derived tokens with provided conditioning signals."""

        if cond_dynamic_norm.shape[-1] != self.input_cond_dynamic_dim:
            raise ValueError(
                f"Expected cond_dynamic dim {self.input_cond_dynamic_dim}, got {cond_dynamic_norm.shape[-1]}"
            )
        if cond_static_norm.shape[-1] != self.input_cond_static_dim:
            raise ValueError(
                f"Expected cond_static dim {self.input_cond_static_dim}, got {cond_static_norm.shape[-1]}"
            )

        # Encode history as dynamic tokens so downstream cross-attention and
        # causal masking can exploit temporal structure and relative biases.
        hist_tokens = self.history_encoder(x_hist_norm.transpose(1, 2))  # [B, E, T]
        hist_tokens = hist_tokens.transpose(1, 2)  # [B, T, E]

        # Concatenate learned history tokens with exogenous dynamic features.
        cond_dynamic_aug = torch.cat([cond_dynamic_norm, hist_tokens], dim=-1)

        # Static channel incorporates pooled history for FiLM modulation.
        hist_summary = self.history_pool(hist_tokens.transpose(1, 2)).squeeze(-1)
        cond_static_aug = torch.cat([cond_static_norm, hist_summary], dim=-1)

        return cond_dynamic_aug, cond_static_aug

    def forward(
        self,
        x_hist: torch.Tensor,
        x_future: torch.Tensor,
        cond_dynamic: torch.Tensor,
        cond_static: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the diffusion training loss for a batch.

        Args:
            x_hist: Historical target window ``[B, context_length, target_dim]``.
            x_future: Forecast target window ``[B, prediction_length, target_dim]``.
            cond_dynamic: Dynamic conditioning features aligned to history
                ``[B, context_length, cond_dynamic_dim]``.
            cond_static: Static conditioning features ``[B, cond_static_dim]``.
        Returns:
            Scalar diffusion training loss.
        """

        if x_hist.shape[1] != self.context_length:
            raise ValueError(
                f"Expected x_hist length {self.context_length}, got {x_hist.shape[1]}"
            )
        if x_future.shape[1] != self.prediction_length:
            raise ValueError(
                f"Expected x_future length {self.prediction_length}, got {x_future.shape[1]}"
            )
        if cond_dynamic.shape[1] != self.context_length:
            raise ValueError(
                f"Expected cond_dynamic length {self.context_length}, got {cond_dynamic.shape[1]}"
            )

        x_hist_norm, x_future_norm, _, _ = self._normalize_target(x_hist, x_future)
        cond_dynamic_norm, cond_static_norm = self._normalize_cond(
            cond_dynamic, cond_static
        )

        cond_dynamic_aug, cond_static_aug = self._prepare_conditioning(
            x_hist_norm, cond_dynamic_norm, cond_static_norm
        )

        loss = self.model(
            x_future=x_future_norm,
            cond_dynamic=cond_dynamic_aug,
            cond_static=cond_static_aug,
        )
        return loss

    @torch.no_grad()
    def sample_forecast(
        self,
        x_hist: torch.Tensor,
        cond_dynamic: torch.Tensor,
        cond_static: torch.Tensor,
        *,
        num_samples: int = 1,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Generate forecast samples using the trained diffusion model.

        Args:
            x_hist: Historical target window ``[B, context_length, target_dim]``.
            cond_dynamic: Dynamic conditioning features ``[B, context_length, cond_dynamic_dim]``.
            cond_static: Static conditioning features ``[B, cond_static_dim]``.
            num_samples: Number of forecast samples per series.
        Returns:
            Forecast samples with shape ``[num_samples, B, prediction_length, target_dim]``.
        """

        batch_size = x_hist.size(0)

        # Use history statistics for rescaling generated samples back to data space.
        dummy_future = torch.zeros(
            batch_size, self.prediction_length, self.target_dim, device=x_hist.device
        )
        x_hist_norm, _, loc, scale = self._normalize_target(x_hist, dummy_future)
        cond_dynamic_norm, cond_static_norm = self._normalize_cond(cond_dynamic, cond_static)

        cond_dynamic_aug, cond_static_aug = self._prepare_conditioning(
            x_hist_norm, cond_dynamic_norm, cond_static_norm
        )

        # Repeat conditioning for multiple samples.
        cond_dynamic_rep = cond_dynamic_aug.repeat(num_samples, 1, 1)
        cond_static_rep = cond_static_aug.repeat(num_samples, 1)

        cond = {"dynamic": cond_dynamic_rep, "static": cond_static_rep}
        samples = self.model.diffusion.sample(
            batch_size=batch_size * num_samples,
            horizon=self.prediction_length,
            cond=cond,
            clip_denoised=clip_denoised,
        )

        samples = samples.view(num_samples, batch_size, self.target_dim, self.prediction_length)
        samples = samples.permute(0, 1, 3, 2)  # -> [S, B, horizon, target_dim]

        # Rescale back to original space.
        samples = samples * scale + loc
        return samples


__all__ = ["ConditionalTimeGradTrainingNetwork"]