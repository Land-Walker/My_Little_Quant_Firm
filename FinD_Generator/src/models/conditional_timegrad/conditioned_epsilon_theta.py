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
        attn_dropout: float = 0.1,
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
            embed_dim=embed_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=False,
        )

        self.rel_pos_bias = _RelativePositionBias(max_distance=max(seq_len, prediction_length))

        # Convolutional upsampling followed by optional interpolation for exact alignment.
        self.cond_upsampler = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
        )

        # Lightweight learned positional alignment to mitigate lossy temporal interpolation.
        self.pos_align = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.LeakyReLU(0.1),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
        )

        # FiLM modulation derived from static conditioning to rescale base inputs.
        self.static_film = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(embed_dim, base_epsilon.input_size),
            nn.Tanh(),
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
        static_film = self.static_film(static_token.squeeze(1))  # [B, C]

        cond_tokens = torch.cat([dyn_tokens, static_token], dim=1)  # [B, seq_len+1, embed_dim]
        cond_tokens = cond_tokens.transpose(0, 1)  # [cond_seq, B, embed_dim]

        query = self.query_proj(x)  # [B, embed_dim, horizon]
        query = query.permute(2, 0, 1)  # [horizon, B, embed_dim]

        rel_bias = self.rel_pos_bias(query_len=horizon, cond_len=cond_tokens.size(0))
        causal_mask = _causal_mask(
            query_len=horizon,
            cond_len=cond_tokens.size(0),
            dynamic_tokens=self.seq_len,
            device=query.device,
        )
        attn_mask = rel_bias + causal_mask
        attn_out, _ = self.cross_attention(query, cond_tokens, cond_tokens, attn_mask=attn_mask)
        attn_out = attn_out.permute(1, 2, 0)  # [B, embed_dim, horizon]

        # Align time dimension to the base denoiser's expected prediction length if needed.
        if horizon != self.prediction_length:
            attn_out = self.pos_align(attn_out)
            attn_out = self.cond_upsampler(attn_out)
            attn_out = F.interpolate(attn_out, size=self.prediction_length, mode="linear", align_corners=False)
            x = F.interpolate(x, size=self.prediction_length, mode="linear", align_corners=False)

        cond_residual = self.context_proj(attn_out)  # [B, C, horizon]

        # Apply FiLM multiplicative modulation from static conditioning.
        film_scale = (1.0 + static_film).unsqueeze(-1)  # [B, C, 1]
        x = x * film_scale
        cond_residual = cond_residual * film_scale

        x_cond = x + cond_residual

        return self.base(x_cond, t, cond=attn_out)


class _RelativePositionBias(nn.Module):
    """Learned relative positional bias for cross-attention."""

    def __init__(self, max_distance: int) -> None:
        super().__init__()
        self.max_distance = max_distance
        # Bias table indexed by clipped relative distance.
        self.bias_table = nn.Parameter(torch.zeros(2 * max_distance - 1))
        nn.init.normal_(self.bias_table, std=0.02)

    def forward(self, query_len: int, cond_len: int) -> torch.Tensor:
        device = self.bias_table.device
        q_ids = torch.arange(query_len, device=device).unsqueeze(1)
        k_ids = torch.arange(cond_len, device=device).unsqueeze(0)
        rel = q_ids - k_ids  # [L, S]
        rel = rel.clamp(-self.max_distance + 1, self.max_distance - 1)
        rel_index = rel + self.max_distance - 1
        bias = self.bias_table[rel_index]  # [L, S]
        return bias


def _causal_mask(
    query_len: int, cond_len: int, dynamic_tokens: int, device: torch.device
) -> torch.Tensor:
    """Create an additive causal mask for attention.

    Dynamic conditioning tokens are ordered in time and should not be attended to from
    queries that correspond to earlier or equal timesteps. The final conditioning token
    is static and remains fully visible.
    """

    mask = torch.zeros((query_len, cond_len), device=device)

    # Dynamic tokens are indices [0, dynamic_tokens - 1]; static token is the last column.
    static_idx = cond_len - 1

    q_ids = torch.arange(query_len, device=device).unsqueeze(1)
    dyn_ids = torch.arange(dynamic_tokens, device=device).unsqueeze(0)

    allowed_until = torch.clamp(q_ids, max=dynamic_tokens - 1)
    block = dyn_ids > allowed_until
    mask[:, :dynamic_tokens] = mask[:, :dynamic_tokens].masked_fill(block, float("-inf"))

    # Static column remains zero (no masking).
    if static_idx >= 0:
        mask[:, static_idx] = 0.0

    return mask