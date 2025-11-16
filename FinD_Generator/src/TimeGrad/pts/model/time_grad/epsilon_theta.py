# pts/model/time_grad/epsilon_theta.py
import math

import torch
from torch import nn
import torch.nn.functional as F


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=500):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        if diffusion_step.dim() == 0:
            diffusion_step = diffusion_step.unsqueeze(0)
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, residual_channels, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
            padding_mode="circular",
        )
        self.diffusion_projection = nn.Linear(hidden_size, residual_channels)
        self.conditioner_projection = nn.Conv1d(
            1, 2 * residual_channels, 1
        )
        self.output_projection = nn.Conv1d(residual_channels, 2 * residual_channels, 1)

        nn.init.kaiming_normal_(self.conditioner_projection.weight)
        nn.init.kaiming_normal_(self.output_projection.weight)

    def forward(self, x, conditioner, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        # conditioner should have shape [B, 1, L]; the Conv1d expects 3D
        conditioner = self.conditioner_projection(conditioner)

        y = x + diffusion_step
        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        y = F.leaky_relu(y, 0.4)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / math.sqrt(2.0), skip


class CondUpsampler(nn.Module):
    def __init__(self, cond_length, target_dim):
        super().__init__()
        self.linear1 = nn.Linear(cond_length, target_dim // 2)
        self.linear2 = nn.Linear(target_dim // 2, target_dim)

    def forward(self, x):
        # x: [B, cond_length]
        x = self.linear1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.linear2(x)
        x = F.leaky_relu(x, 0.4)
        return x  # [B, target_dim]


class EpsilonTheta(nn.Module):
    def __init__(
    self,
    target_dim,
    cond_length,
    time_emb_dim=16,
    residual_layers=8,
    residual_channels=8,
    dilation_cycle_length=2,
    residual_hidden=64,
    *,
    # explicit-conditioning auxiliary args (optional)
    dyn_dim: int = 0,
    static_dim: int = 0,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.cond_length = cond_length
        self.dyn_dim = dyn_dim
        self.static_dim = static_dim

        # FIX: Remove padding=2, padding_mode="circular"
        self.input_projection = nn.Conv1d(
            1, residual_channels, 1
        )
        
        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=residual_hidden
        )
        self.cond_upsampler = CondUpsampler(
            cond_length=cond_length, target_dim=target_dim
        )


        # if explicit conditioning dims are provided and they don't match cond_length,
        # create an adapter to map concatenated [dyn_mean + static] --> cond_length
        cond_input_dim = 0
        if self.dyn_dim and self.dyn_dim > 0:
            cond_input_dim += self.dyn_dim
        if self.static_dim and self.static_dim > 0:
            cond_input_dim += self.static_dim

        if cond_input_dim > 0 and cond_input_dim != cond_length:
            self.cond_adapter = nn.Linear(cond_input_dim, cond_length)
        else:
            self.cond_adapter = None

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    residual_channels=residual_channels,
                    dilation=2 ** (i % dilation_cycle_length),
                    hidden_size=residual_hidden,
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3)
        self.output_projection = nn.Conv1d(residual_channels, 1, 3)

        nn.init.kaiming_normal_(self.input_projection.weight)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        nn.init.zeros_(self.output_projection.weight)

    def _build_cond_vec(self, cond_dynamic: torch.Tensor = None, cond_static: torch.Tensor = None):
        """
        Build a per-sample conditioning vector of shape [B, cond_length] from
        explicit cond_dynamic and cond_static.

        - cond_dynamic: [B, seq_len, dyn_dim] -> reduce across time (mean) -> [B, dyn_dim]
        - cond_static: [B, static_dim]
        - concatenate -> [B, cond_input_dim] -> optionally adapt to cond_length
        """
        parts = []
        if cond_dynamic is not None:
            # mean-pool across time dimension to get a per-sample vector
            # shape: [B, dyn_dim]
            dyn_mean = cond_dynamic.mean(dim=1)
            parts.append(dyn_mean)
        if cond_static is not None:
            # cond_static expected [B, static_dim]
            parts.append(cond_static)
        if not parts:
            return None
        cond_vec = torch.cat(parts, dim=-1)  # [B, cond_input_dim]

        if self.cond_adapter is not None:
            cond_vec = self.cond_adapter(cond_vec)  # map to cond_length

        return cond_vec  # [B, cond_length] if adapter used else intended shape

    def forward(self, inputs, time, cond=None, cond_dynamic: torch.Tensor = None, cond_static: torch.Tensor = None):
        """
        Extended forward:

        - inputs: [B, 1, L] or [B, L] depending on calling convention (we expect [B, 1, L])
        - time: tensor of step indices [B] (or scalar)
        - cond: original-style cond (legacy) shape [B, cond_length]; if provided, used as-is
        - cond_dynamic: explicit dynamic cond [B, seq_len, dyn_dim]
        - cond_static: explicit static cond [B, static_dim]

        Behavior:
          * If `cond` is provided (legacy), we use it (backwards compatibility).
          * Else, if `cond_dynamic`/`cond_static` provided, we build a per-sample cond vector,
            adapt it to cond_length (if needed) and use it.
        """
        # inputs expected shape for input_projection conv1d: [B, 1, L]
        # if user passed [B, L], make it [B, 1, L]
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)

        x = self.input_projection(inputs)
        x = F.leaky_relu(x, 0.4)

        diffusion_step = self.diffusion_embedding(time)

        # Determine cond_vector of shape [B, cond_length]
        if cond is not None:
            cond_vec = cond
        else:
            cond_vec = self._build_cond_vec(cond_dynamic, cond_static)

        if cond_vec is None:
            # fallback: zeros
            cond_vec = torch.zeros((x.size(0), self.cond_length), device=x.device, dtype=x.dtype)

        # cond_vec -> cond_up [B, target_dim]
        cond_up = self.cond_upsampler(cond_vec)
        # cond_up expected by conditioner_projection Conv1d: [B, 1, target_dim]
        cond_up = cond_up.unsqueeze(1)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_up, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.leaky_relu(x, 0.4)
        x = self.output_projection(x)
        return x
