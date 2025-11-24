"""
inference.py

Load a trained ConditionalTimeGrad model and generate forecasts.

Usage:
    python /workspaces/My_Little_Quant_Firm/FinD_Generator/inference.py --checkpoint checkpoints/best_model.pt --sample_idx 30
"""

import sys
sys.path.append('FinD_Generator')

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import properscoring as ps
import os

from src.data_loader import TimeGradDataModule
from src.data_collector import DataCollector
from src.models import create_conditional_timegrad
from src import config


def parse_args():
    parser = argparse.ArgumentParser(description='Inference for ConditionalTimeGrad')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint (.pt file)')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index of the sample to plot from the validation set')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of Monte Carlo samples for prediction')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--scenario', type=str, default=None,
                        help='Path to a JSON file defining a custom scenario for conditioning features')
    parser.add_argument('--output_dir', type=str, default='FinD_Generator/image/graph',
                        help='Directory to save output plots')
    return parser.parse_args()


def inverse_transform_single_feature(scaler, data_array, feature_idx=0):
    """
    Inverse transforms a single-feature array (1D or 2D) using a multi-feature scaler.

    Args:
        scaler: The scikit-learn scaler object (already fitted).
        data_array (np.ndarray): The array for the single feature. Can be 1D or 2D.
        feature_idx (int): The column index of this feature in the original data.

    Returns:
        np.ndarray: The inverse-transformed single-feature array.
    """
    original_shape = data_array.shape
    data_flat = data_array.flatten()

    # Create a dummy array with the shape the scaler expects
    num_features = scaler.n_features_in_
    dummy_array = np.zeros((len(data_flat), num_features))

    # Place the single-feature data into the correct column
    dummy_array[:, feature_idx] = data_flat

    # Inverse transform the full array
    transformed_full = scaler.inverse_transform(dummy_array)

    # Extract and return only the feature we care about
    return transformed_full[:, feature_idx].reshape(original_shape)


class TimeGradPredictor:
    """
    A modular class for running inference with a trained ConditionalTimeGrad model.
    """
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initializes the predictor by loading the data pipeline and the trained model.

        Args:
            checkpoint_path (str): Path to the model checkpoint (.pt file).
            device (str): Device to use ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {self.device}")

        self.dm = self._load_data_pipeline()
        self.model = self._load_model(checkpoint_path)

        print("\n✨ Available features for scenarios:")
        print(f"  Dynamic Features: {self.dm.cond_dynamic_cols}")
        print(f"  Static Features (Regimes): {self.dm.cond_static_cols}")

    def _load_data_pipeline(self):
        """Loads and prepares the data module."""
        print("\n📊 Loading data pipeline...")
        collector = DataCollector()
        dfs = collector.collect_all_data()
        dm = TimeGradDataModule(
            data_dict=dfs,
            seq_len=config.DEFAULT_SEQ_LEN,
            forecast_horizon=config.DEFAULT_HORIZON,
            batch_size=1,
            device=self.device
        )
        dm.preprocess_and_split()
        dm.build_datasets()
        print(f"✅ Data pipeline loaded. Validation set has {len(dm.val_set)} samples.")
        return dm

    def _load_model(self, checkpoint_path):
        """Loads the trained model from a checkpoint."""
        print(f"\n🏗️ Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        sample_batch = next(iter(self.dm.val_dataloader()))
        cond_dynamic_dim = sample_batch['cond_dynamic'].shape[-1]
        cond_static_dim = sample_batch['cond_static'].shape[-1]

        model = create_conditional_timegrad(
            cond_dynamic_dim=cond_dynamic_dim,
            forecast_horizon=self.dm.forecast_horizon,
            cond_static_dim=cond_static_dim,
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("✅ Model loaded successfully.")
        return model

    def _get_inference_batch(self, sample_idx):
        """Retrieves a specific sample from the validation set."""
        if sample_idx >= len(self.dm.val_set):
            raise ValueError(f"sample_idx {sample_idx} is out of bounds for validation set of size {len(self.dm.val_set)}")
        
        val_iterator = iter(self.dm.val_dataloader())
        for _ in range(sample_idx + 1):
            inference_batch = next(val_iterator)
        return inference_batch

    def _run_prediction(self, inference_batch, num_samples):
        """Generates forecasts for a given batch."""
        with torch.no_grad():
            predictions = self.model.predict(
                cond_dynamic=inference_batch['cond_dynamic'],
                cond_static=inference_batch['cond_static'],
                num_samples=num_samples,
                return_raw_samples=True
            )
        return predictions

    def _process_and_inverse_transform(self, predictions, inference_batch):
        """Converts predictions to numpy and inverse transforms them."""
        # Move to CPU and convert to numpy
        for key in ['mean', 'q10', 'q90', 'raw_samples']:
            predictions[key] = predictions[key].squeeze(0).cpu().numpy()

        x_future_true_scaled = inference_batch['x_future'].squeeze(0).cpu().numpy()
        x_hist_scaled = inference_batch['x_hist'].squeeze(0).cpu().numpy()

        scaler = list(self.dm.scalers.values())[0]
        plot_feature_idx = 0

        for key in ['mean', 'q10', 'q90', 'raw_samples']:
            predictions[key] = inverse_transform_single_feature(scaler, predictions[key], plot_feature_idx)

        x_future_true = inverse_transform_single_feature(scaler, x_future_true_scaled, plot_feature_idx)
        x_hist = inverse_transform_single_feature(scaler, x_hist_scaled, plot_feature_idx)

        return predictions, x_hist, x_future_true

    def predict(self, sample_idx, num_samples=200):
        """
        Generates and returns a forecast for a specific sample from the validation set.

        Args:
            sample_idx (int): The index of the sample in the validation set.
            num_samples (int): Number of Monte Carlo samples for prediction.

        Returns:
            tuple: (predictions, x_hist, x_future_true)
                   - predictions (dict): Dictionary of forecast results ('mean', 'q10', 'q90', 'raw_samples').
                   - x_hist (np.ndarray): Historical data.
                   - x_future_true (np.ndarray): Ground truth future data.
        """
        print(f"\n🔍 Running inference for ground truth (sample {sample_idx})...")
        inference_batch = self._get_inference_batch(sample_idx)
        predictions_scaled = self._run_prediction(inference_batch, num_samples)
        return self._process_and_inverse_transform(predictions_scaled, inference_batch)

    def predict_scenario(self, sample_idx, scenario_data, num_samples=200):
        """
        Generates and returns a forecast for a custom scenario.

        Args:
            sample_idx (int): The index of the base sample to use.
            scenario_data (dict): Dictionary defining the custom scenario.
            num_samples (int): Number of Monte Carlo samples.

        Returns:
            tuple: (predictions, x_hist, x_future_true)
        """
        scenario_name = scenario_data.get("name", "Custom Scenario")
        print(f"\n🔄 Running inference for scenario '{scenario_name}' (base sample {sample_idx})...")
        inference_batch = self._get_inference_batch(sample_idx)

        # Override Dynamic Features
        if "dynamic_overrides" in scenario_data:
            cond_dynamic = inference_batch['cond_dynamic']
            for feature, value in scenario_data["dynamic_overrides"].items():
                if feature in self.dm.cond_dynamic_cols:
                    idx = self.dm.cond_dynamic_cols.index(feature)
                    new_vals = torch.tensor(value, dtype=cond_dynamic.dtype, device=self.device) if isinstance(value, list) else torch.full((cond_dynamic.shape[1],), value, dtype=cond_dynamic.dtype, device=self.device)
                    cond_dynamic[0, :, idx] = new_vals
                else:
                    print(f"    - WARNING: Dynamic feature '{feature}' not found.")

        # Override Static Features
        if "static_overrides" in scenario_data:
            new_static = inference_batch['cond_static'].clone().squeeze(0).fill_(0)
            for feature, value in scenario_data["static_overrides"].items():
                if feature in self.dm.cond_static_cols:
                    idx = self.dm.cond_static_cols.index(feature)
                    new_static[idx] = float(value)
                else:
                    print(f"    - WARNING: Static feature '{feature}' not found.")
            inference_batch['cond_static'] = new_static.unsqueeze(0)

        predictions_scaled = self._run_prediction(inference_batch, num_samples)
        return self._process_and_inverse_transform(predictions_scaled, inference_batch)


def plot_forecast(predictions, x_hist, x_future_true, title, output_path, is_scenario=False):
    """Generates and saves a plot of the forecast results."""
    print("🎨 Generating plot...")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    hist_range = np.arange(-len(x_hist), 0)
    future_range = np.arange(0, len(x_future_true))

    ax.plot(hist_range, x_hist.flatten(), color='gray', label='Historical Data')
    if not is_scenario:
        ax.plot(future_range, x_future_true.flatten(), color='black', lw=2, label='Ground Truth')
        crps_score = ps.crps_ensemble(x_future_true.squeeze(), predictions['raw_samples']).mean()
        ax.text(0.02, 0.95, f'CRPS: {crps_score:.4f}', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    ax.plot(future_range, predictions['mean'].flatten(), color='blue', lw=2, label='Mean Forecast')
    ax.fill_between(future_range, predictions['q10'].flatten(), predictions['q90'].flatten(),
                    color='blue', alpha=0.2, label='10%-90% Quantile Range')

    all_values = np.concatenate([
        x_hist.flatten(), x_future_true.flatten(), predictions['mean'].flatten(),
        predictions['q10'].flatten(), predictions['q90'].flatten()
    ])
    min_val, max_val = np.min(all_values), np.max(all_values)
    padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.5
    ax.set_ylim(min_val - padding, max_val + padding)

    ax.set_title(title)
    ax.set_xlabel('Time Steps (from forecast point)')
    ax.set_ylabel('Price (Inverse Transformed)')
    ax.legend()
    ax.axvline(0, color='r', linestyle='--', lw=1)

    plt.savefig(output_path, dpi=300)
    print(f"✅ Plot saved to {output_path}")
    plt.close(fig)


def main():
    args = parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    predictor = TimeGradPredictor(args.checkpoint, device)

    if args.scenario:
        with open(args.scenario, 'r') as f:
            scenario_data = json.load(f)
        predictions, x_hist, x_future_true = predictor.predict_scenario(
            args.sample_idx, scenario_data, args.num_samples
        )
        scenario_name = scenario_data.get("name", "Custom Scenario")
        title = f"Forecast for '{scenario_name}' (Base Sample: {args.sample_idx})"
        fname = f'scenario_{os.path.basename(args.scenario).replace(".json", "")}.png'
        is_scenario = True
    else:
        predictions, x_hist, x_future_true = predictor.predict(args.sample_idx, args.num_samples)
        title = f"Forecast for Ground Truth (Sample: {args.sample_idx})"
        fname = f'forecast_sample_{args.sample_idx}.png'
        is_scenario = False

    output_path = os.path.join(args.output_dir, fname)
    plot_forecast(predictions, x_hist, x_future_true, title, output_path, is_scenario)


if __name__ == '__main__':
    main()

# In a Jupyter cell
import sys
sys.path.append('FinD_Generator')
from FinD_Generator.inference import TimeGradPredictor, plot_forecast
import json

# Initialize the predictor with the path to your best model
predictor = TimeGradPredictor(checkpoint_path='notebook_checkpoints/best_model.pt')

# --- Prediction for a ground truth sample ---
sample_idx_to_predict = 42
predictions, x_hist, x_future_true = predictor.predict(sample_idx_to_predict)

# Plot the results
plot_forecast(
    predictions, x_hist, x_future_true,
    title=f"Forecast for Ground Truth (Sample: {sample_idx_to_predict})",
    output_path="forecast.png",
    is_scenario=False
)


# --- Prediction for a custom scenario ---
scenario = {
    "name": "High Volatility Scenario",
    "dynamic_overrides": {
        "vix_close": 35.0  # Set VIX to a constant high value
    },
    "static_overrides": {
        "Regime_3": 1.0 # Activate a specific regime
    }
}

scenario_preds, scenario_hist, _ = predictor.predict_scenario(
    sample_idx=sample_idx_to_predict,
    scenario_data=scenario
)

# Plot the scenario results
plot_forecast(
    scenario_preds, scenario_hist, x_future_true, # Use original ground truth for comparison
    title=f"Forecast for '{scenario['name']}'",
    output_path="scenario_forecast.png",
    is_scenario=True
)
