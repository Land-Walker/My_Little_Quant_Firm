"""Convenience script to train and run conditional TimeGrad forecasts.

The script wires together the existing data pipeline, the training
wrapper, and the autoregressive predictor so you can:

1) Load prepared data from ``data/raw`` (or optionally download fresh
   data with the collector).
2) Train the conditioning-aware TimeGrad model for a small number of
   epochs.
3) Generate autoregressive forecasts on the held-out test split while
   respecting causal masking, relative positional bias, FiLM modulation,
   and history-derived conditioning used in the model stack.

Example (fast smoke test on CPU):
    python run.py --device cpu --epochs 1 --batch-size 2 \
        --max-train-steps 1 --max-val-steps 1 --num-samples 1

The script keeps loc/scale fixed across the forecast horizon and
recomputes the history encoder every step to mirror the prediction
network’s causal alignment logic.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

from src import config
from src.data_collector import DataCollector
from src.data_loader import TimeGradDataModule
from src.predictor import ConditionalTimeGradPredictionNetwork
from src.training import ConditionalTimeGradTrainingNetwork

def _load_local_data() -> Dict[str, pd.DataFrame]:
    """Load pre-downloaded parquet data from ``data/raw``.

    Returns a dict matching the collector output keys.
    """

    candidates = [
        Path(config.RAW_DATA_DIR),
        Path(__file__).resolve().parent / "data" / "raw",
        Path("data") / "raw",
    ]

    for base in candidates:
        datasets = {
            "target": base / "target.parquet",
            "market": base / "market.parquet",
            "daily_macro": base / "daily_macro.parquet",
            "monthly_macro": base / "monthly_macro.parquet",
            "quarterly_macro": base / "quarterly_macro.parquet",
        }
        if all(path.exists() for path in datasets.values()):
            return {name: pd.read_parquet(path) for name, path in datasets.items()}

    raise FileNotFoundError(
        "Missing raw parquet files in all known locations. "
        "Either download them with --download or place them under data/raw/."
    )


def _load_data(use_local: bool) -> Dict[str, pd.DataFrame]:
    if use_local:
        return _load_local_data()

    collector = DataCollector()
    return collector.collect_all_data()


def _prepare_datamodule(args: argparse.Namespace, device: torch.device) -> TimeGradDataModule:
    data = _load_data(use_local=not args.download)
    dm = TimeGradDataModule(
        data_dict=data,
        seq_len=args.context_length,
        forecast_horizon=args.prediction_length,
        batch_size=args.batch_size,
        device=str(device),
    )
    dm.preprocess_and_split()
    dm.build_datasets()
    return dm


def _build_networks(
    dm: TimeGradDataModule, args: argparse.Namespace, device: torch.device
):
    feature_cols = dm.get_feature_columns_by_type()
    target_dim = len(feature_cols["target"])
    cond_dynamic_dim = len(feature_cols["daily"]) + len(feature_cols["monthly"])
    cond_static_dim = len(feature_cols["regime"])

    train_net = ConditionalTimeGradTrainingNetwork(
        target_dim=target_dim,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        cond_dynamic_dim=cond_dynamic_dim,
        cond_static_dim=cond_static_dim,
        diff_steps=args.diff_steps,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        residual_layers=args.residual_layers,
        residual_channels=args.residual_channels,
        cond_embed_dim=args.cond_embed_dim,
        cond_attn_heads=args.cond_attn_heads,
        cond_attn_dropout=args.cond_attn_dropout,
    ).to(device)

    predictor = ConditionalTimeGradPredictionNetwork(
        target_dim=target_dim,
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        cond_dynamic_dim=cond_dynamic_dim,
        cond_static_dim=cond_static_dim,
        diff_steps=args.diff_steps,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
        residual_layers=args.residual_layers,
        residual_channels=args.residual_channels,
        cond_embed_dim=args.cond_embed_dim,
        cond_attn_heads=args.cond_attn_heads,
        cond_attn_dropout=args.cond_attn_dropout,
    ).to(device)

    return train_net, predictor


def train_and_validate(
    model: ConditionalTimeGradTrainingNetwork,
    dm: TimeGradDataModule,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        train_loss = 0.0
        for step, batch in enumerate(dm.train_dataloader()):
            x_hist = batch["x_hist"].to(device)
            x_future = batch["x_future"].to(device)
            cond_dynamic = batch["cond_dynamic"].to(device)
            cond_static = batch["cond_static"].to(device)

            optimizer.zero_grad()
            loss = model(x_hist, x_future, cond_dynamic, cond_static)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            if args.max_train_steps and (step + 1) >= args.max_train_steps:
                break

        avg_train = train_loss / max(1, (step + 1))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for v_step, batch in enumerate(dm.val_dataloader()):
                x_hist = batch["x_hist"].to(device)
                x_future = batch["x_future"].to(device)
                cond_dynamic = batch["cond_dynamic"].to(device)
                cond_static = batch["cond_static"].to(device)
                val_loss += model(x_hist, x_future, cond_dynamic, cond_static).item()
                if args.max_val_steps and (v_step + 1) >= args.max_val_steps:
                    break

        avg_val = val_loss / max(1, (v_step + 1))
        print(f"Epoch {epoch + 1}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")
        model.train()

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


def run_inference(
    predictor: ConditionalTimeGradPredictionNetwork,
    dm: TimeGradDataModule,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_path: Path,
) -> None:
    state = torch.load(checkpoint_path, map_location=device)
    predictor.load_state_dict(state, strict=False)
    predictor.eval()

    test_batch = next(iter(dm.test_dataloader()))
    x_hist = test_batch["x_hist"].to(device)
    cond_dynamic = test_batch["cond_dynamic"].to(device)
    cond_static = test_batch["cond_static"].to(device)

    samples = predictor.sample_autoregressive(
        x_hist=x_hist,
        cond_dynamic=cond_dynamic,
        cond_static=cond_static,
        num_samples=args.num_samples,
    )
    print(f"Generated samples shape: {samples.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run conditional TimeGrad")
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to cuda if available")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH)
    parser.add_argument("--context-length", type=int, default=config.DEFAULT_SEQ_LEN)
    parser.add_argument("--prediction-length", type=int, default=config.DEFAULT_HORIZON)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--diff-steps", type=int, default=100)
    parser.add_argument("--beta-end", type=float, default=0.1)
    parser.add_argument("--beta-schedule", type=str, default="linear")
    parser.add_argument("--residual-layers", type=int, default=6)
    parser.add_argument("--residual-channels", type=int, default=32)
    parser.add_argument("--cond-embed-dim", type=int, default=64)
    parser.add_argument("--cond-attn-heads", type=int, default=4)
    parser.add_argument("--cond-attn-dropout", type=float, default=0.1)
    parser.add_argument("--num-samples", type=int, default=2, help="forecast samples per series")
    parser.add_argument("--max-train-steps", type=int, default=0, help="optional cap for train steps per epoch")
    parser.add_argument("--max-val-steps", type=int, default=0, help="optional cap for val steps per epoch")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download fresh data instead of loading local parquet files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    print(f"Using device: {device}")

    dm = _prepare_datamodule(args, device)
    train_net, predictor = _build_networks(dm, args, device)
    checkpoint_path = Path(config.PROCESSED_DATA_DIR) / "timegrad_checkpoint.pt"

    train_and_validate(train_net, dm, args, device, checkpoint_path)
    run_inference(predictor, dm, args, device, checkpoint_path)

if __name__ == "__main__":
    main()