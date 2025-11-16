# pts/model/time_grad/time_grad_network.py
from typing import List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn

from gluonts.core.component import validated

from src.TimeGrad.pts.model.utils import weighted_average
from src.TimeGrad.pts.modules import GaussianDiffusion, DiffusionOutput, MeanScaler, NOPScaler

from .epsilon_theta import EpsilonTheta


class TimeGradTrainingNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        history_length: int,
        context_length: int,
        prediction_length: int,
        dropout_rate: float,
        lags_seq: List[int],
        target_dim: int,
        conditioning_length: int,
        diff_steps: int,
        loss_type: str,
        beta_end: float,
        beta_schedule: str,
        residual_layers: int,
        residual_channels: int,
        dilation_cycle_length: int,
        cardinality: List[int] = [1],
        embedding_dimension: int = 1,
        scaling: bool = True,
        *,
        # explicit conditioning dims (new)
        dyn_dim: int = 0,
        static_dim: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_dim = target_dim
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.history_length = history_length
        self.scaling = scaling

        assert len(set(lags_seq)) == len(lags_seq), "no duplicated lags allowed!"
        lags_seq.sort()
        self.lags_seq = lags_seq

        self.cell_type = cell_type
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )

        # store conditioning dims
        self.cond_length = conditioning_length
        self.dyn_dim = dyn_dim
        self.static_dim = static_dim

        self.denoise_fn = EpsilonTheta(
            target_dim=target_dim,
            cond_length=conditioning_length,
            residual_layers=residual_layers,
            residual_channels=residual_channels,
            dilation_cycle_length=dilation_cycle_length,
            dyn_dim=self.dyn_dim,
            static_dim=self.static_dim,
        )

        self.diffusion = GaussianDiffusion(
            self.denoise_fn,
            input_size=target_dim,
            diff_steps=diff_steps,
            loss_type=loss_type,
            beta_end=beta_end,
            beta_schedule=beta_schedule,
        )

        self.distr_output = DiffusionOutput(
            self.diffusion, input_size=target_dim, cond_size=conditioning_length
        )

        self.proj_dist_args = self.distr_output.get_args_proj(num_cells)

        # NEW: project conditioning vector into rnn hidden dim and add to rnn outputs
        # so existing proj_dist_args (which expects num_cells-sized inputs) will
        # receive conditioning via additive injection.
        if self.cond_length and self.cond_length > 0:
            self.cond_proj_to_rnn = nn.Linear(self.cond_length, num_cells)
        else:
            self.cond_proj_to_rnn = None

        # If the concatenated conditioning vector from features doesn't match conditioning_length,
        # we need an adapter. This should be defined here to be part of the model's parameters.
        feature_cond_dim = self.dyn_dim + self.static_dim
        if feature_cond_dim > 0 and feature_cond_dim != self.cond_length:
            self.cond_adapter = nn.Linear(feature_cond_dim, self.cond_length)
        else:
            self.cond_adapter = None

        self.embed_dim = 1
        self.embed = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.embed_dim
        )

        if self.scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

        self.residual_layers = residual_layers
        self.residual_channels = residual_channels

    @staticmethod
    def get_lagged_subsequences(
        sequence: torch.Tensor,
        sequence_length: int,
        indices: List[int],
        subsequences_length: int = 1,
    ) -> torch.Tensor:
        assert max(indices) + subsequences_length <= sequence_length, (
            f"lags cannot go further than history length, found lag "
            f"{max(indices)} while history length is only {sequence_length}"
        )
        assert all(lag_index >= 0 for lag_index in indices)

        lagged_values = []
        for lag_index in indices:
            begin_index = -lag_index - subsequences_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1))
        return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)

    def unroll(
        self,
        lags: torch.Tensor,
        scale: torch.Tensor,
        time_feat: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        unroll_length: int,
        begin_state: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ):
        # (batch_size, sub_seq_len, target_dim, num_lags)
        lags_scaled = lags / scale.unsqueeze(-1)

        input_lags = lags_scaled.reshape(
            (-1, unroll_length, len(self.lags_seq) * self.target_dim)
        )

        index_embeddings = self.embed(target_dimension_indicator)

        repeated_index_embeddings = (
            index_embeddings.unsqueeze(1)
            .expand(-1, unroll_length, -1, -1)
            .reshape((-1, unroll_length, self.target_dim * self.embed_dim))
        )

        inputs = torch.cat((input_lags, repeated_index_embeddings, time_feat), dim=-1)

        outputs, state = self.rnn(inputs, begin_state)

        return outputs, state, lags_scaled, inputs

    def unroll_encoder(
        self,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: Optional[torch.Tensor],
        future_target_cdf: Optional[torch.Tensor],
        target_dimension_indicator: torch.Tensor,
    ):
        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        if future_time_feat is None or future_target_cdf is None:
            time_feat = past_time_feat[:, -self.context_length :, ...]
            sequence = past_target_cdf
            sequence_length = self.history_length
            subsequences_length = self.context_length
        else:
            time_feat = torch.cat(
                (past_time_feat[:, -self.context_length :, ...], future_time_feat),
                dim=1,
            )
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            sequence_length = self.history_length + self.prediction_length
            subsequences_length = self.context_length + self.prediction_length

        lags = self.get_lagged_subsequences(
            sequence=sequence,
            sequence_length=sequence_length,
            indices=self.lags_seq,
            subsequences_length=subsequences_length,
        )

        _, scale = self.scaler(
            past_target_cdf[:, -self.context_length :, ...],
            past_observed_values[:, -self.context_length :, ...],
        )

        outputs, states, lags_scaled, inputs = self.unroll(
            lags=lags,
            scale=scale,
            time_feat=time_feat,
            target_dimension_indicator=target_dimension_indicator,
            unroll_length=subsequences_length,
            begin_state=None,
        )

        return outputs, states, scale, lags_scaled, inputs

    def distr_args(self, rnn_outputs: torch.Tensor):
        (distr_args,) = self.proj_dist_args(rnn_outputs)
        return distr_args

    def _build_cond_vector_from_feats(
        self, past_time_feat: torch.Tensor, future_time_feat: Optional[torch.Tensor], past_feat_static_real: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Build a per-sample cond vector [B, cond_length] from stacked time features and static real features.
        - past_time_feat: (B, history_length, feat_dim)
        - future_time_feat: (B, prediction_length, feat_dim) or None
        NOTE: feat_dim is expected to be time_features_dim + dyn_dim (if dyn features were vstacked earlier).
        """
        # dynamic features are assumed to be the last self.dyn_dim channels of full_time
        dyn_vec = None
        if self.dyn_dim and self.dyn_dim > 0:
            if future_time_feat is None:
                full_time = past_time_feat[:, -self.context_length :, ...]
            else:
                full_time = torch.cat((past_time_feat[:, -self.context_length :, ...], future_time_feat), dim=1)
            if full_time.shape[-1] < self.dyn_dim:
                # defensive: nothing to extract
                dyn_vec = None
            else:
                dyn_part = full_time[..., -self.dyn_dim :]  # [B, seq_len, dyn_dim]
                dyn_vec = dyn_part.mean(dim=1)  # [B, dyn_dim]

        static_vec = None
        if self.static_dim and self.static_dim > 0 and past_feat_static_real is not None:
            # past_feat_static_real may be (B, static_dim) or (B, history_length, static_dim)
            if past_feat_static_real.dim() == 3:
                # reduce across time (mean)
                static_vec = past_feat_static_real.mean(dim=1)
            else:
                static_vec = past_feat_static_real

        parts = []
        if dyn_vec is not None:
            parts.append(dyn_vec)
        if static_vec is not None:
            parts.append(static_vec)
        if not parts:
            return None

        cond_vec = torch.cat(parts, dim=-1)  # [B, dyn_dim + static_dim]

        if self.cond_adapter is not None:
            cond_vec = self.cond_adapter(cond_vec)

        return cond_vec

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
        future_target_cdf: torch.Tensor,
        future_observed_values: torch.Tensor,
        # NEW optional static feature fields (these names match those produced by the InstanceSplitter)
        past_feat_static_real: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        seq_len = self.context_length + self.prediction_length

        rnn_outputs, _, scale, _, _ = self.unroll_encoder(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=future_time_feat,
            future_target_cdf=future_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
        )

        target = torch.cat(
            (past_target_cdf[:, -self.context_length :, ...], future_target_cdf),
            dim=1,
        )

        # Build explicit conditioning vector
        cond_vec = self._build_cond_vector_from_feats(past_time_feat, future_time_feat, past_feat_static_real)

        # If cond projection exists, map cond_vec to rnn hidden size and add to rnn_outputs
        if cond_vec is not None and self.cond_proj_to_rnn is not None:
            # cond_proj -> [B, num_cells]; expand to time dimension
            cond_proj = self.cond_proj_to_rnn(cond_vec)  # [B, num_cells]
            cond_proj = cond_proj.unsqueeze(1).expand(-1, rnn_outputs.shape[1], -1)
            rnn_outputs = rnn_outputs + cond_proj

        if self.scaling:
            self.diffusion.scale = scale

        distr_args = self.distr_args(rnn_outputs=rnn_outputs)

        likelihoods = self.diffusion.log_prob(target, distr_args).unsqueeze(-1)

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        observed_values = torch.cat(
            (
                past_observed_values[:, -self.context_length :, ...],
                future_observed_values,
            ),
            dim=1,
        )

        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)

        loss = weighted_average(likelihoods, weights=loss_weights, dim=1)

        return (loss.mean(), likelihoods, distr_args)


class TimeGradPredictionNetwork(TimeGradTrainingNetwork):
    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

        self.shifted_lags = [l - 1 for l in self.lags_seq]

    def sampling_decoder(
        self,
        past_target_cdf: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        time_feat: torch.Tensor,
        scale: torch.Tensor,
        begin_states: Union[List[torch.Tensor], torch.Tensor],
        past_feat_static_real: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        repeated_past_target_cdf = repeat(past_target_cdf)
        repeated_time_feat = repeat(time_feat)
        repeated_scale = repeat(scale)
        if self.scaling:
            self.diffusion.scale = repeated_scale
        repeated_target_dimension_indicator = repeat(target_dimension_indicator)

        if self.cell_type == "LSTM":
            repeated_states = [repeat(s, dim=1) for s in begin_states]
        else:
            repeated_states = repeat(begin_states, dim=1)

        # Pre-calculate conditioning vector
        # Note: future_time_feat is the time features for the prediction window.
        # We use it to build the conditioning vector once, which is then used for all prediction steps.
        cond_vec = self._build_cond_vector_from_feats(
            past_time_feat=repeated_past_target_cdf, future_time_feat=repeated_time_feat, past_feat_static_real=repeat(past_feat_static_real) if past_feat_static_real is not None else None
        )

        future_samples = []

        for k in range(self.prediction_length):
            lags = self.get_lagged_subsequences(
                sequence=repeated_past_target_cdf,
                sequence_length=self.history_length + k,
                indices=self.shifted_lags,
                subsequences_length=1,
            )

            rnn_outputs, repeated_states, _, _ = self.unroll(
                begin_state=repeated_states,
                lags=lags,
                scale=repeated_scale,
                time_feat=repeated_time_feat[:, k : k + 1, ...],
                target_dimension_indicator=repeated_target_dimension_indicator,
                unroll_length=1,
            )

            # add conditioning to rnn_outputs if needed
            if cond_vec is not None and self.cond_proj_to_rnn is not None:
                cond_proj = self.cond_proj_to_rnn(cond_vec)
                cond_proj = cond_proj.unsqueeze(1).expand(-1, rnn_outputs.shape[1], -1)
                rnn_outputs = rnn_outputs + cond_proj

            distr_args = self.distr_args(rnn_outputs=rnn_outputs)

            new_samples = self.diffusion.sample(cond=distr_args)

            future_samples.append(new_samples)
            repeated_past_target_cdf = torch.cat(
                (repeated_past_target_cdf, new_samples), dim=1
            )

        samples = torch.cat(future_samples, dim=1)

        return samples.reshape(
            (
                -1,
                self.num_parallel_samples,
                self.prediction_length,
                self.target_dim,
            )
        )

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_time_feat: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_time_feat: torch.Tensor,
        # NEW: optional static feature forwarded here
        past_feat_static_real: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        _, begin_states, scale, _, _ = self.unroll_encoder(
            past_time_feat=past_time_feat,
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_time_feat=None,
            future_target_cdf=None,
            target_dimension_indicator=target_dimension_indicator,
        )

        return self.sampling_decoder(
            past_target_cdf=past_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
            time_feat=future_time_feat,
            scale=scale,
            begin_states=begin_states,
            past_feat_static_real=past_feat_static_real,
        )
