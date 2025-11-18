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


def main():
    args = parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"🔧 Using device: {device}")

    # 1. Load Data Pipeline (to get data and inverse scalers)
    print("\n📊 Loading data pipeline...")
    collector = DataCollector()
    dfs = collector.collect_all_data()

    dm = TimeGradDataModule(
        data_dict=dfs,
        seq_len=config.DEFAULT_SEQ_LEN,
        forecast_horizon=config.DEFAULT_HORIZON,
        batch_size=1,  # We only need one sample
        device=device
    )
    dm.preprocess_and_split()
    dm.build_datasets()

    val_loader = dm.val_dataloader()
    print(f"✅ Data pipeline loaded. Validation set has {len(dm.val_set)} samples.")
    
    # Print available features for scenario creation
    print("\n✨ Available features for scenarios:")
    print(f"  Dynamic Features: {dm.cond_dynamic_cols}")
    print(f"  Static Features (Regimes): {dm.cond_static_cols}")

    # 2. Load Model
    print(f"\n🏗️ Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get dimensions from a sample batch to instantiate the model correctly
    sample_batch = next(iter(val_loader))
    cond_dynamic_dim = sample_batch['cond_dynamic'].shape[-1]
    cond_static_dim = sample_batch['cond_static'].shape[-1]

    model = create_conditional_timegrad(
        cond_dynamic_dim=cond_dynamic_dim,
        forecast_horizon=dm.forecast_horizon,
        cond_static_dim=cond_static_dim,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ Model loaded successfully.")

    # 3. Get a Sample for Inference
    if args.sample_idx >= len(dm.val_set):
        raise ValueError(f"sample_idx {args.sample_idx} is out of bounds for validation set of size {len(dm.val_set)}")

    # We need to iterate to the specific sample
    val_iterator = iter(val_loader)
    for _ in range(args.sample_idx + 1):
        inference_batch = next(val_iterator)

    # 4. Handle Scenario Override if provided
    scenario_name = "Ground Truth"
    if args.scenario:
        print(f"\n🔄 Loading custom scenario from {args.scenario}...")
        with open(args.scenario, 'r') as f:
            scenario_data = json.load(f)
        
        scenario_name = scenario_data.get("name", "Custom Scenario")
        
        # Override Dynamic Features
        if "dynamic_overrides" in scenario_data:
            print("  -> Applying dynamic feature overrides...")
            cond_dynamic_tensor = inference_batch['cond_dynamic']
            for feature_name, value in scenario_data["dynamic_overrides"].items():
                if feature_name in dm.cond_dynamic_cols:
                    idx = dm.cond_dynamic_cols.index(feature_name)
                    print(f"    - Overriding '{feature_name}' (index {idx}).")
                    if isinstance(value, list):
                        new_values = torch.tensor(value, dtype=cond_dynamic_tensor.dtype, device=device)
                    else:
                        new_values = torch.full((cond_dynamic_tensor.shape[1],), fill_value=value, dtype=cond_dynamic_tensor.dtype, device=device)
                    cond_dynamic_tensor[0, :, idx] = new_values
                else:
                    print(f"    - WARNING: Dynamic feature '{feature_name}' not found. Skipping.")

        # Override Static Features (Regimes)
        if "static_overrides" in scenario_data:
            print("  -> Applying static feature overrides...")
            cond_static_tensor = inference_batch['cond_static']
            # Create a mutable copy to modify
            new_static_values = cond_static_tensor.clone().squeeze(0)
            
            # First, set all regime values to 0 (off)
            new_static_values.fill_(0)
            
            for feature_name, value in scenario_data["static_overrides"].items():
                 if feature_name in dm.cond_static_cols:
                    idx = dm.cond_static_cols.index(feature_name)
                    print(f"    - Setting '{feature_name}' (index {idx}) to {value}.")
                    new_static_values[idx] = float(value) # Ensure it's a float
                 else:
                    print(f"    - WARNING: Static feature '{feature_name}' not found. Skipping.")
            # Assign the modified tensor back
            inference_batch['cond_static'] = new_static_values.unsqueeze(0)

        print("✅ Scenario applied.")

    print(f"\n🔍 Running inference for '{scenario_name}' (using base sample {args.sample_idx})...")

    # 5. Generate Forecast
    with torch.no_grad():
        predictions = model.predict(
            cond_dynamic=inference_batch['cond_dynamic'],
            cond_static=inference_batch['cond_static'],
            num_samples=args.num_samples,
            return_raw_samples=True  # Request the raw samples for CRPS
        )

    # Move predictions to CPU and convert to numpy
    for key in ['mean', 'q10', 'q90', 'raw_samples']:
        predictions[key] = predictions[key].squeeze(0).cpu().numpy()

    # Get ground truth and historical data
    x_future_true_scaled = inference_batch['x_future'].squeeze(0).cpu().numpy()
    x_hist_scaled = inference_batch['x_hist'].squeeze(0).cpu().numpy()

    # 6. Inverse Transform to Original Price Space
    print("🔄 Inverse transforming data for plotting...")
    # dm.scalers is a dict mapping asset names to scalers. We'll use the first one.
    scaler = list(dm.scalers.values())[0]

    # We only plot the first feature (e.g., 'close_den')
    plot_feature_idx = 0

    # Inverse transform predictions
    for key in ['mean', 'q10', 'q90']:
        predictions[key] = inverse_transform_single_feature(scaler, predictions[key], plot_feature_idx)
    predictions['raw_samples'] = inverse_transform_single_feature(scaler, predictions['raw_samples'], plot_feature_idx)

    # Inverse transform ground truth and history
    x_future_true = inverse_transform_single_feature(scaler, x_future_true_scaled, plot_feature_idx)
    x_hist = inverse_transform_single_feature(scaler, x_hist_scaled, plot_feature_idx)

    # 7. Plot Results
    print("🎨 Generating plot...")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    hist_range = np.arange(-len(x_hist), 0)
    future_range = np.arange(0, len(x_future_true))

    # Plot history
    ax.plot(hist_range, x_hist.flatten(), color='gray', label='Historical Data')

    # Only plot ground truth if not running a custom scenario
    if not args.scenario:
        ax.plot(future_range, x_future_true.flatten(), color='black', lw=2, label='Ground Truth')

    # Plot mean forecast
    ax.plot(future_range, predictions['mean'].flatten(), color='blue', lw=2, label='Mean Forecast')

    # 8. Calculate and Display CRPS
    # CRPS is calculated only if we are not in a hypothetical scenario
    if not args.scenario:
        # crps_ensemble expects observations and forecasts with shape (time, samples)
        # Our raw_samples are (horizon, num_samples), e.g., (5, 200).
        # Our x_future_true is (horizon, 1), e.g., (5, 1).
        # We must squeeze x_future_true to (horizon,) for compatibility.
        crps_score = ps.crps_ensemble(x_future_true.squeeze(), predictions['raw_samples']).mean()
        ax.text(0.02, 0.95, f'CRPS: {crps_score:.4f}', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    # Plot uncertainty interval (quantiles)
    ax.fill_between(future_range, predictions['q10'].flatten(), predictions['q90'].flatten(),
                    color='blue', alpha=0.2, label='10%-90% Quantile Range')

    # Calculate dynamic y-axis limits for better visibility
    all_plotted_values = np.concatenate([
        x_hist.flatten(),
        x_future_true.flatten(),
        predictions['mean'].flatten(),
        predictions['q10'].flatten(),
        predictions['q90'].flatten()
    ])

    min_val = np.min(all_plotted_values)
    max_val = np.max(all_plotted_values)
    # Add a small buffer for better visualization, ensuring it's not zero if min_val == max_val
    padding = (max_val - min_val) * 0.1 if (max_val - min_val) > 0 else 0.5
    ax.set_ylim(min_val - padding, max_val + padding)

    ax.set_title(f"Forecast for '{scenario_name}' (Base Sample: {args.sample_idx})")
    ax.set_xlabel('Time Steps (from forecast point)')
    ax.set_ylabel('Price (Inverse Transformed)')
    ax.legend()
    ax.axvline(0, color='r', linestyle='--', lw=1)

    fname = f'scenario_{os.path.basename(args.scenario).replace(".json", "")}.png' if args.scenario else f'forecast_sample_{args.sample_idx}.png'
    output_path = os.path.join(args.output_dir, fname)
    plt.savefig(output_path, dpi=300)
    print(f"✅ Plot saved to {output_path}")

if __name__ == '__main__':
    main()