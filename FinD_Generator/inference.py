"""
inference.py

Load a trained ConditionalTimeGrad model and generate forecasts.

Usage:
    python /workspaces/My_Little_Quant_Firm/FinD_Generator/inference.py --checkpoint checkpoints/best_model.pt --sample_idx 10
"""

import sys
sys.path.append('FinD_Generator')

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
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
    parser.add_argument('--output_dir', type=str, default='FinD_Generator/image/graph',
                        help='Directory to save output plots')
    return parser.parse_args()


def inverse_transform_single_feature(scaler, array_1d, feature_idx=0):
    """
    Inverse transforms a single-feature array using a multi-feature scaler.

    Args:
        scaler: The scikit-learn scaler object (already fitted).
        array_1d (np.ndarray): The 1D or (n, 1) array for the single feature.
        feature_idx (int): The column index of this feature in the original data.

    Returns:
        np.ndarray: The inverse-transformed single-feature array.
    """
    # Create a dummy array with the shape the scaler expects
    num_features = scaler.n_features_in_
    dummy_array = np.zeros((len(array_1d), num_features))

    # Place the single-feature data into the correct column
    dummy_array[:, feature_idx] = array_1d.flatten()

    # Inverse transform the full array
    transformed_full = scaler.inverse_transform(dummy_array)

    # Extract and return only the feature we care about
    return transformed_full[:, feature_idx].reshape(-1, 1)


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

    # 2. Load Model
    print(f"\n🏗️ Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Get dimensions from a sample batch to instantiate the model correctly
    sample_batch = next(iter(val_loader))
    cond_dynamic_dim = sample_batch['cond_dynamic'].shape[-1]
    cond_static_dim = sample_batch['cond_static'].shape[-1]

    model = create_conditional_timegrad(
        cond_dynamic_dim=cond_dynamic_dim,
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

    print(f"\n🔍 Running inference on sample index {args.sample_idx}...")

    # 4. Generate Forecast
    with torch.no_grad():
        predictions = model.predict(
            cond_dynamic=inference_batch['cond_dynamic'],
            cond_static=inference_batch['cond_static'],
            num_samples=args.num_samples
        )

    # Move predictions to CPU and convert to numpy
    for key, tensor in predictions.items():
        predictions[key] = tensor.squeeze(0).cpu().numpy()

    # Get ground truth and historical data
    x_future_true_scaled = inference_batch['x_future'].squeeze(0).cpu().numpy()
    x_hist_scaled = inference_batch['x_hist'].squeeze(0).cpu().numpy()

    # 5. Inverse Transform to Original Price Space
    print("🔄 Inverse transforming data for plotting...")
    # dm.scalers is a dict mapping asset names to scalers. We'll use the first one.
    scaler = list(dm.scalers.values())[0]

    # We only plot the first feature (e.g., 'close_den')
    plot_feature_idx = 0

    # Inverse transform predictions
    for key, arr in predictions.items():
        predictions[key] = inverse_transform_single_feature(scaler, arr, plot_feature_idx)

    # Inverse transform ground truth and history
    x_future_true = inverse_transform_single_feature(scaler, x_future_true_scaled, plot_feature_idx)
    x_hist = inverse_transform_single_feature(scaler, x_hist_scaled, plot_feature_idx)

    # 6. Plot Results
    print("🎨 Generating plot...")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))

    hist_range = np.arange(-len(x_hist), 0)
    future_range = np.arange(0, len(x_future_true))

    # Plot history
    ax.plot(hist_range, x_hist.flatten(), color='gray', label='Historical Data')

    # Plot ground truth
    ax.plot(future_range, x_future_true.flatten(), color='black', lw=2, label='Ground Truth')

    # Plot mean forecast
    ax.plot(future_range, predictions['mean'].flatten(), color='blue', lw=2, label='Mean Forecast')

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

    ax.set_title(f'Forecast vs. Ground Truth (Sample {args.sample_idx})')
    ax.set_xlabel('Time Steps (from forecast point)')
    ax.set_ylabel('Price (Inverse Transformed)')
    ax.legend()
    ax.axvline(0, color='r', linestyle='--', lw=1)

    output_path = os.path.join(args.output_dir, f'forecast_sample_{args.sample_idx}.png')
    plt.savefig(output_path, dpi=300)
    print(f"✅ Plot saved to {output_path}")

if __name__ == '__main__':
    main()