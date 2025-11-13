import torch
import torch.nn as nn
from gluonts.torch.model.timegrad.network import TimeGrad

class ConditionalTimeGrad(TimeGrad):
    """
    A conditional version of the TimeGrad model that accepts scenario labels
    and macroeconomic features to guide the time series generation process.
    """
    def __init__(
        self,
        num_scenarios: int,
        scenario_embed_dim: int,
        num_macro_features: int,
        **kwargs, # Pass all other TimeGrad arguments here
    ):
        """
        Args:
            num_scenarios: The number of unique market scenarios (e.g., 3 for bull, bear, neutral).
            scenario_embed_dim: The desired dimension for the scenario embedding vector.
            num_macro_features: The number of macroeconomic features.
            **kwargs: Arguments for the parent TimeGrad class.
        """
        # Initialize the original TimeGrad model first
        super().__init__(**kwargs)
        
        # --- Our Custom Conditioning Layers ---
        self.scenario_embedding = nn.Embedding(num_scenarios, scenario_embed_dim)
        
        combined_cond_dim = scenario_embed_dim + num_macro_features
        
        # This linear layer projects our conditioning vector to the same
        # dimension as the time embedding, so we can add them.
        self.condition_projector = nn.Linear(combined_cond_dim, self.time_embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        scenario_labels: torch.Tensor,
        macro_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        The modified forward pass that injects conditioning information.
        
        Args:
            x: The noisy time series at step t (batch_size, seq_len).
            t: The current diffusion timestep (batch_size,).
            scenario_labels: The scenario labels for the batch (batch_size,).
            macro_features: Macroeconomic data for the batch (batch_size, seq_len, num_features).
        """
        # 1. Get the original time embedding
        time_emb = self.time_mlp(t)
        
        # 2. Create the conditioning vector
        scenario_emb = self.scenario_embedding(scenario_labels)
        
        # Aggregate macro features over the time dimension (e.g., by taking the mean)
        # to get a single vector representation for the window.
        macro_features_agg = torch.mean(macro_features, dim=1)
        
        cond_vec = torch.cat([scenario_emb, macro_features_agg], dim=-1)
        
        # 3. Project the conditioning vector and add it to the time embedding
        projected_cond = self.condition_projector(cond_vec)
        final_embedding = time_emb + projected_cond
        
        # 4. Pass the combined embedding into the original U-Net backbone
        output = self.denoise_fn(x, final_embedding)
        
        return output