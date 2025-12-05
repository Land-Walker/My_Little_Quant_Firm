"""Prediction-ready network for conditional TimeGrad.

This module mirrors the training wrapper architecture but performs strictly
autoregressive sampling with careful conditioning handling:
- The history encoder is recomputed at every forecast step so cross-attention
  and FiLM modules see the latest generated targets.
- Dynamic conditioning is only applied to the historical window; generated
  steps receive zeroed dynamic tokens to avoid leaking future information.
- Loc/scale statistics are frozen from the initial history window and reused
  for the entire forecast horizon.
- Causal mask shapes remain consistent by maintaining a fixed-length
  conditioning window that slides alongside the autoregressive loop.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from src.models.conditional_timegrad import ConditionalTimeGrad


class ConditionalTimeGradPredictionNetwork(nn.Module):
    """Autoregressive inference wrapper for the conditional TimeGrad model."""

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
        scale_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.scale_eps = scale_eps

        self.input_cond_dynamic_dim = cond_dynamic_dim
        self.input_cond_static_dim = cond_static_dim
        self.cond_embed_dim = cond_embed_dim

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

        self.history_pool = nn.AdaptiveAvgPool1d(1)

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
        )

    def _normalize_history(
        self, x_hist: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Normalize the history window and return loc/scale for later reuse."""

        loc = x_hist.mean(dim=1, keepdim=True)
        scale = x_hist.std(dim=1, keepdim=True).clamp_min(self.scale_eps)
        x_hist_norm = (x_hist - loc) / scale
        return x_hist_norm, loc, scale

    def _normalize_cond(
        self, cond_dynamic: torch.Tensor, cond_static: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize conditioning signals to stabilize inference."""

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
        """Fuse history tokens with dynamic/static conditioning."""

        hist_tokens = self.history_encoder(x_hist_norm.transpose(1, 2))  # [B, E, T]
        hist_tokens = hist_tokens.transpose(1, 2)  # [B, T, E]

        # Ensure the historical token sequence matches the dynamic conditioning
        # length in case padding or circular convolutions alter the effective
        # width (e.g., very short contexts). This keeps cross-attention causal
        # masks aligned with the conditioning tokens used by the denoiser.
        if hist_tokens.size(1) != cond_dynamic_norm.size(1):
            hist_tokens = nn.functional.interpolate(
                hist_tokens.transpose(1, 2),
                size=cond_dynamic_norm.size(1),
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        cond_dynamic_aug = torch.cat([cond_dynamic_norm, hist_tokens], dim=-1)

        hist_summary = self.history_pool(hist_tokens.transpose(1, 2)).squeeze(-1)
        cond_static_aug = torch.cat([cond_static_norm, hist_summary], dim=-1)

        return cond_dynamic_aug, cond_static_aug

    @torch.no_grad()
    def sample_autoregressive(
        self,
        x_hist: torch.Tensor,
        cond_dynamic: torch.Tensor,
        cond_static: torch.Tensor,
        *,
        num_samples: int = 1,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Generate forecasts autoregressively with fixed normalization.

        Dynamic conditioning is only drawn from the provided history window.
        Each forecast step recomputes the history encoder to keep conditioning
        aligned with newly generated targets.
        """

        batch_size = x_hist.size(0)
        device = x_hist.device

        if x_hist.shape[1] != self.context_length:
            raise ValueError(
                f"Expected x_hist length {self.context_length}, got {x_hist.shape[1]}"
            )
        if cond_dynamic.shape[1] != self.context_length:
            raise ValueError(
                f"Expected cond_dynamic length {self.context_length}, got {cond_dynamic.shape[1]}"
            )
        if cond_dynamic.shape[-1] != self.input_cond_dynamic_dim:
            raise ValueError(
                f"Expected cond_dynamic dim {self.input_cond_dynamic_dim}, got {cond_dynamic.shape[-1]}"
            )
        if cond_static.shape[-1] != self.input_cond_static_dim:
            raise ValueError(
                f"Expected cond_static dim {self.input_cond_static_dim}, got {cond_static.shape[-1]}"
            )

        x_hist_norm, loc, scale = self._normalize_history(x_hist)
        cond_dynamic_norm, cond_static_norm = self._normalize_cond(
            cond_dynamic, cond_static
        )

        # Freeze loc/scale across samples and horizon.
        loc_rep = loc.repeat(num_samples, 1, 1)
        scale_rep = scale.repeat(num_samples, 1, 1)

        # Expand batch for multiple samples.
        hist_window = x_hist_norm.repeat(num_samples, 1, 1)
        cond_dynamic_window = cond_dynamic_norm.repeat(num_samples, 1, 1)
        cond_static_rep = cond_static_norm.repeat(num_samples, 1)

        forecasts_norm = []
        for _ in range(self.prediction_length):
            cond_dynamic_aug, cond_static_aug = self._prepare_conditioning(
                hist_window, cond_dynamic_window, cond_static_rep
            )

            cond = {"dynamic": cond_dynamic_aug, "static": cond_static_aug}
            # Sample the full prediction horizon so the denoiser can leverage
            # its learned alignment and causal masking, then take only the
            # first step for autoregressive rollout. The conditioning vectors
            # already include the most recent generated history, so each call
            # sees the updated context even though the internal denoiser
            # operates on the canonical prediction length.
            step = self.model.diffusion.sample(
                batch_size=batch_size * num_samples,
                horizon=self.prediction_length,
                cond=cond,
                clip_denoised=clip_denoised,
            )  # [SB, C, prediction_length]

            step = step[:, :, :1].permute(0, 2, 1)  # [SB, 1, C]
            forecasts_norm.append(step)

            # Slide history and zero-fill dynamic conditioning for generated steps.
            hist_window = torch.cat([hist_window[:, 1:, :], step], dim=1)
            zero_dyn = torch.zeros(
                batch_size * num_samples,
                1,
                self.input_cond_dynamic_dim,
                device=device,
                dtype=cond_dynamic.dtype,
            )
            cond_dynamic_window = torch.cat([cond_dynamic_window[:, 1:, :], zero_dyn], dim=1)

        forecasts_norm = torch.cat(forecasts_norm, dim=1)  # [SB, horizon, C]
        forecasts = forecasts_norm * scale_rep + loc_rep
        forecasts = forecasts.view(
            num_samples, batch_size, self.prediction_length, self.target_dim
        )
        return forecasts


__all__ = ["ConditionalTimeGradPredictionNetwork"]