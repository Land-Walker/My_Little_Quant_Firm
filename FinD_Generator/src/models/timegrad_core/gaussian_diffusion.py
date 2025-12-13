"""Gaussian diffusion implementation for TimeGrad-style models.

This module mirrors the functionality provided by the original TimeGrad
implementation in PyTorchTS. It supports both training-time loss computation
and ancestral sampling, and it keeps the conditioning interface compatible with
the denoising network so the conditional TimeGrad variant can reuse the same
core logic.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _linear_beta_schedule(diff_steps: int, beta_end: float) -> torch.Tensor:
    """Create a linear beta schedule from a small starting value to ``beta_end``.

    The starting value is fixed to 1e-4 to mirror the default DDPM schedule used
    by the original TimeGrad codebase, while the end value remains configurable
    for experimentation.
    """

    return torch.linspace(1e-4, beta_end, diff_steps, dtype=torch.float32)


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Extract coefficients for a batch of indices ``t`` and reshape for broadcast."""

    out = a.gather(-1, t.to(a.device))
    return out.reshape((t.shape[0],) + (1,) * (len(x_shape) - 1))


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn: nn.Module,
        input_size: int,
        diff_steps: int = 100,
        beta_end: float = 0.1,
        beta_schedule: str = "linear",
    ) -> None:
        super().__init__()
        self.denoise_fn = denoise_fn
        self.input_size = input_size
        self.diff_steps = diff_steps

        if beta_schedule != "linear":
            raise ValueError(f"Unsupported beta schedule: {beta_schedule}")

        betas = _linear_beta_schedule(diff_steps, beta_end)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32), alphas_cumprod[:-1]], dim=0
        )

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1),
        )

        # Posterior coefficients used for sampling
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # Numerically stable clip to avoid log(0)
        posterior_log_variance_clipped = torch.log(
            torch.clamp(posterior_variance, min=1e-20)
        )

        posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev)
            * torch.sqrt(alphas)
            / (1.0 - alphas_cumprod)
        )

        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped", posterior_log_variance_clipped
        )
        self.register_buffer("posterior_mean_coef1", posterior_mean_coef1)
        self.register_buffer("posterior_mean_coef2", posterior_mean_coef2)

    @torch.no_grad()
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Diffuse the data for a given number of timesteps ``t``.

        Args:
            x_start: Clean input of shape ``[B, C, T]``.
            t: Timesteps tensor of shape ``[B]``.
            noise: Optional pre-sampled noise; if ``None`` standard normal noise
                is drawn.
        """

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = _extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(
        self,
        x_start: torch.Tensor,
        cond: Optional[dict[str, Any]] = None,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the diffusion training loss.

        This follows the original DDPM objective used by TimeGrad: sample a
        random timestep ``t``, corrupt the input with Gaussian noise, and train
        the denoiser to predict that noise.
        """

        if x_start.dim() != 3:
            raise ValueError(
                f"Expected x_start with shape [B, C, T], got {tuple(x_start.shape)}"
            )

        batch_size = x_start.size(0)
        device = x_start.device

        if t is None:
            t = torch.randint(0, self.diff_steps, (batch_size,), device=device)
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Support conditioned denoisers by forwarding the conditioning dict when provided.
        if cond is None:
            pred_noise = self.denoise_fn(x_noisy, t)
        else:
            pred_noise = self.denoise_fn(x_noisy, t, cond)

        if pred_noise.shape != noise.shape:
            raise RuntimeError(
                "Denoiser output shape does not match noise shape: "
                f"pred={tuple(pred_noise.shape)} vs noise={tuple(noise.shape)}"
            )

        loss = nn.functional.mse_loss(pred_noise, noise)
        return loss

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct ``x_0`` from a noisy sample ``x_t`` and predicted noise."""

        return _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - _extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        ) * noise

    def p_mean_variance(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[dict[str, Any]] = None,
        clip_denoised: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the reverse diffusion mean/variance at timestep ``t``.

        Returns the model mean, variance, log-variance, and the reconstructed
        ``x_start`` prediction. This mirrors the utilities used by the original
        TimeGrad sampler and is needed for both training diagnostics and
        ancestral sampling.
        """

        if cond is None:
            pred_noise = self.denoise_fn(x, t)
        else:
            pred_noise = self.denoise_fn(x, t, cond)

        x_recon = self.predict_start_from_noise(x, t, pred_noise)
        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1.0, 1.0)

        model_mean = _extract(self.posterior_mean_coef1, t, x.shape) * x_recon + _extract(
            self.posterior_mean_coef2, t, x.shape
        ) * x
        model_variance = _extract(self.posterior_variance, t, x.shape)
        model_log_variance = _extract(self.posterior_log_variance_clipped, t, x.shape)
        return model_mean, model_variance, model_log_variance, x_recon

    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[dict[str, Any]] = None,
        clip_denoised: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Take one reverse diffusion step."""

        model_mean, _, model_log_variance, x_recon = self.p_mean_variance(
            x, t, cond=cond, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
        sample = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        return sample, x_recon

    @torch.no_grad()
    def p_sample_loop(
        self,
        shape: torch.Size,
        cond: Optional[dict[str, Any]] = None,
        clip_denoised: bool = True,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate samples by iterating the reverse diffusion process.

        Args:
            shape: Output shape ``[B, C, T]`` for the denoising trajectory.
            cond: Optional conditioning dictionary forwarded to the denoiser.
            clip_denoised: Whether to clamp reconstructed ``x_0`` predictions.
            spatial_mask: Optional mask of the same shape as ``shape`` (or
                broadcastable) indicating which positions to actively denoise.
                Unmasked locations remain zero throughout the reverse process,
                allowing targeted sampling of specific timesteps/channels
                without evolving the full horizon volume.
        """

        device = self.betas.device
        img = torch.randn(shape, device=device)
        background_noise = None

        if spatial_mask is not None:
            if spatial_mask.shape != shape:
                try:
                    spatial_mask = spatial_mask.expand(shape)
                except RuntimeError as exc:
                    raise ValueError(
                        "spatial_mask must be broadcastable to the sample shape"
                    ) from exc

            # Preserve a frozen background so masked-out positions remain random
            # rather than collapsing to zeros across repeated sampling calls.
            background_noise = img.detach().clone()

        for i in reversed(range(self.diff_steps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            img, _ = self.p_sample(img, t, cond=cond, clip_denoised=clip_denoised)

            if spatial_mask is not None:
                img = img * spatial_mask + background_noise * (1.0 - spatial_mask)

        return img

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        horizon: int,
        cond: Optional[dict[str, Any]] = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Convenience wrapper to sample forecasts with shape ``[B, C, horizon]``."""

        shape = (batch_size, self.input_size, horizon)
        return self.p_sample_loop(shape=shape, cond=cond, clip_denoised=clip_denoised)

    @torch.no_grad()
    def sample_step(
        self,
        batch_size: int,
        horizon: int,
        *,
        target_index: int,
        channel_slice: Optional[Sequence[int] | slice] = None,
        cond: Optional[dict[str, Any]] = None,
        clip_denoised: bool = True,
    ) -> torch.Tensor:
        """Sample a specific timestep (and optional channel subset) via DDPM.

        The denoiser still receives a full ``prediction_length``-sized context
        but a spatial mask constrains diffusion updates to the requested
        timestep/channels, avoiding unnecessary computation of the entire
        horizon during autoregressive rollout.
        """

        device = self.betas.device
        channel_indices = torch.arange(self.input_size, device=device)
        if channel_slice is not None:
            channel_indices = channel_indices[channel_slice]

        mask = torch.zeros((batch_size, self.input_size, horizon), device=device)
        mask[:, channel_indices, target_index : target_index + 1] = 1.0

        samples = self.p_sample_loop(
            shape=(batch_size, self.input_size, horizon),
            cond=cond,
            clip_denoised=clip_denoised,
            spatial_mask=mask,
        )

        return samples[:, channel_indices, target_index : target_index + 1]