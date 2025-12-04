"""Conditioned wrapper around the base TimeGrad epsilon network."""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..timegrad_core.epsilon_theta import EpsilonTheta


class ConditionedEpsilonTheta(nn.Module):
    """Injects dynamic and static conditioning into ``EpsilonTheta``.

    The design keeps the base network untouched, forwarding a conditioned
    tensor while preserving interface compatibility with ``GaussianDiffusion``.
    """

    def __init__(
        self,
        base_epsilon: EpsilonTheta,
        cond_dynamic_dim: int,
        cond_static_dim: int,
        seq_len: int,
        prediction_length: int,
        embed_dim: int = 64,
        attn_heads: int = 4,
    ) -> None:
        super().__init__()

        self.base = base_epsilon
        self.seq_len = seq_len
        self.prediction_length = prediction_length

        self.dynamic_encoder = nn.Linear(cond_dynamic_dim, embed_dim)
        self.static_encoder = nn.Linear(cond_static_dim, embed_dim)

        self.query_proj = nn.Conv1d(base_epsilon.input_size, embed_dim, kernel_size=1)
        self.context_proj = nn.Conv1d(embed_dim, base_epsilon.input_size, kernel_size=1)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=attn_heads, batch_first=False
        )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: Optional[Dict[str, torch.Tensor]] = None
    ):
        """Forward pass with conditioning.

        Args:
            x: Noisy input, shape ``[B, C, horizon]``.
            t: Diffusion timestep tensor ``[B]``.
            cond: Dict with keys ``"dynamic"`` -> ``[B, seq_len, cond_dim]`` and
                ``"static"`` -> ``[B, static_dim]``.
        """

        batch_size, _, horizon = x.shape

        if cond is None:
            # Unconditioned path for compatibility with the core diffusion APIs.
            return self.base(x, t, cond=None)

        if "dynamic" not in cond or "static" not in cond:
            raise ValueError("cond must contain 'dynamic' and 'static' keys")

        dyn = cond["dynamic"]  # [B, seq_len, cond_dim]
        static = cond["static"]  # [B, static_dim]

        if dyn.shape[0] != batch_size or static.shape[0] != batch_size:
            raise ValueError(
                "Batch size mismatch between noisy input and conditioning features"
            )

        if dyn.shape[1] != self.seq_len:
            raise ValueError(
                f"Expected dynamic conditioning length {self.seq_len}, got {dyn.shape[1]}"
            )

        dyn_tokens = self.dynamic_encoder(dyn)  # [B, seq_len, embed_dim]
        static_token = self.static_encoder(static).unsqueeze(1)  # [B, 1, embed_dim]

        cond_tokens = torch.cat([dyn_tokens, static_token], dim=1)  # [B, seq_len+1, embed_dim]
        cond_tokens = cond_tokens.transpose(0, 1)  # [cond_seq, B, embed_dim]

        query = self.query_proj(x)  # [B, embed_dim, horizon]
        query = query.permute(2, 0, 1)  # [horizon, B, embed_dim]

        attn_out, _ = self.cross_attention(query, cond_tokens, cond_tokens)
        attn_out = attn_out.permute(1, 2, 0)  # [B, embed_dim, horizon]

        # Align time dimension to the base denoiser's expected prediction length if needed.
        if horizon != self.prediction_length:
            attn_out = F.interpolate(attn_out, size=self.prediction_length, mode="nearest")
            x = F.interpolate(x, size=self.prediction_length, mode="nearest")

        cond_residual = self.context_proj(attn_out)  # [B, C, horizon]
        x_cond = x + cond_residual

        return self.base(x_cond, t, cond=attn_out)