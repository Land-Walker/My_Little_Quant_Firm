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
        cond_strategy: str = "fast",
        rnn_type: str = "lstm",
    ) -> None:
        super().__init__()

        self.base = base_epsilon
        self.seq_len = seq_len
        self.prediction_length = prediction_length
        self.cond_strategy = cond_strategy.lower()

        self.dynamic_encoder = nn.Linear(cond_dynamic_dim, embed_dim)
        self.static_encoder = nn.Linear(cond_static_dim, embed_dim)

        if self.cond_strategy not in {"fast", "slow"}:
            raise ValueError("cond_strategy must be either 'fast' or 'slow'")

        rnn_type = rnn_type.lower()
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError("rnn_type must be either 'lstm' or 'gru'")
        self.rnn_type = rnn_type

        self.query_proj = nn.Conv1d(base_epsilon.input_size, embed_dim, kernel_size=1)
        self.context_proj = nn.Conv1d(embed_dim, base_epsilon.input_size, kernel_size=1)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=False,
        )

        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.history_encoder = nn.Linear(base_epsilon.input_size, embed_dim)
        self.history_rnn = rnn_cls(embed_dim, embed_dim, batch_first=True)

        # Use a generous distance budget so sliding-window conditioning in
        # autoregressive loops does not clip relative offsets when the
        # effective dynamic length varies slightly from initialization.
        rel_max = seq_len + prediction_length
        self.rel_pos_bias = _RelativePositionBias(max_distance=rel_max)

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

        dynamic_len = dyn.shape[1]
        if dynamic_len == 0:
            raise ValueError("Dynamic conditioning must have non-zero length")

        dyn_tokens = self.dynamic_encoder(dyn)  # [B, dynamic_len, embed_dim]
        static_token = self.static_encoder(static).unsqueeze(1)  # [B, 1, embed_dim]
        static_film = self.static_film(static_token.squeeze(1))  # [B, C]

        if self.cond_strategy == "fast":
            cond_context = self._fast_conditioning(
                x=x,
                horizon=horizon,
                dyn_tokens=dyn_tokens,
                dynamic_len=dynamic_len,
                static_token=static_token,
            )
        else:
            cond_context = self._slow_conditioning(
                x=x,
                horizon=horizon,
                dyn_tokens=dyn_tokens,
                static_token=static_token,
            )

        # Align time dimension to the base denoiser's expected prediction length if needed.
        if horizon != self.prediction_length:
            cond_context = self.pos_align(cond_context)
            cond_context = self.cond_upsampler(cond_context)
            cond_context = F.interpolate(
                cond_context, size=self.prediction_length, mode="linear", align_corners=False
            )
            x = F.interpolate(x, size=self.prediction_length, mode="linear", align_corners=False)

        cond_residual = self.context_proj(cond_context)  # [B, C, horizon]

        # Apply FiLM multiplicative modulation from static conditioning.
        film_scale = (1.0 + static_film).unsqueeze(-1)  # [B, C, 1]
        x = x * film_scale
        cond_residual = cond_residual * film_scale

        x_cond = x + cond_residual

        return self.base(x_cond, t, cond=cond_context)

    def _fast_conditioning(
        self,
        x: torch.Tensor,
        horizon: int,
        dyn_tokens: torch.Tensor,
        dynamic_len: int,
        static_token: torch.Tensor,
    ) -> torch.Tensor:

        cond_tokens = torch.cat([dyn_tokens, static_token], dim=1)  # [B, dynamic_len+1, embed_dim]
        cond_tokens = cond_tokens.transpose(0, 1)  # [cond_seq, B, embed_dim]

        query = self.query_proj(x)  # [B, embed_dim, horizon]
        query = query.permute(2, 0, 1)  # [horizon, B, embed_dim]

        cond_len = cond_tokens.size(0)
        rel_bias = self.rel_pos_bias(query_len=horizon, cond_len=cond_len)
        causal_mask = _causal_mask(
            query_len=horizon,
            cond_len=cond_len,
            dynamic_tokens=dynamic_len,
            dtype=query.dtype,
            device=query.device,
        )
        attn_mask = rel_bias.to(query.dtype) + causal_mask
        attn_out, _ = self.cross_attention(query, cond_tokens, cond_tokens, attn_mask=attn_mask)
        return attn_out.permute(1, 2, 0)  # [B, embed_dim, horizon]

    def _slow_conditioning(
        self,
        x: torch.Tensor,
        horizon: int,
        dyn_tokens: torch.Tensor,
        static_token: torch.Tensor,
    ) -> torch.Tensor:
        history_tokens = self.history_encoder(x.transpose(1, 2))  # [B, horizon, embed_dim]
        rnn_input = torch.cat([dyn_tokens, history_tokens, static_token], dim=1)  # [B, seq+horizon+1, embed]
        rnn_output, _ = self.history_rnn(rnn_input)

        # Take the slice aligned with the history portion so conditioning length matches horizon.
        cond_slice = rnn_output[:, dyn_tokens.size(1) : dyn_tokens.size(1) + horizon, :]
        return cond_slice.permute(0, 2, 1)  # [B, embed_dim, horizon]



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
    query_len: int,
    cond_len: int,
    dynamic_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create an additive causal mask for attention.

    Dynamic conditioning tokens are ordered in time and should not be attended to from
    queries that correspond to earlier or equal timesteps. The final conditioning token
    is static and remains fully visible.
    """

    if dynamic_tokens < 0:
        raise ValueError("dynamic_tokens must be non-negative")

    static_idx = cond_len - 1
    if dynamic_tokens > static_idx:
        raise ValueError(
            "dynamic_tokens exceeds available conditioning tokens (static token is assumed to be last)"
        )

    mask = torch.zeros((query_len, cond_len), device=device, dtype=dtype)

    # Dynamic tokens are indices [0, dynamic_tokens - 1]; static token is the last column.
    if dynamic_tokens > 0:
        q_ids = torch.arange(query_len, device=device).unsqueeze(1)
        dyn_ids = torch.arange(dynamic_tokens, device=device).unsqueeze(0)

        allowed_until = torch.clamp(q_ids, max=dynamic_tokens - 1)
        block = dyn_ids > allowed_until
        mask[:, :dynamic_tokens] = mask[:, :dynamic_tokens].masked_fill(block, float("-inf"))

    # Static column remains zero (no masking).
    if static_idx >= 0:
        mask[:, static_idx] = 0.0

    return mask
