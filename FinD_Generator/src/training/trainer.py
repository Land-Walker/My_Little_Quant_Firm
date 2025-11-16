"""
train.py

Complete training pipeline for ConditionalTimeGrad model.

Usage:
    python train.py --epochs 50 --batch_size 32 --device cuda
"""

import sys
sys.path.append('FinD_Generator')

import argparse
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np

from src.data_collector import DataCollector
from src.data_loader import TimeGradDataModule
from src.models import create_conditional_timegrad
from src.training import ConditionalTimeGradTrainer, CheckpointCallback
from src import config


def parse_args():
    parser = argparse.ArgumentParser(description='Train ConditionalTimeGrad')
    
    # Data parameters
    parser.add_argument('--seq_len', type=int, default=config.DEFAULT_SEQ_LEN,
                        help='Sequence length for historical window')
    parser.add_argument('--horizon', type=int, default=config.DEFAULT_HORIZON,
                        help='Forecast horizon')
    parser.add_argument('--batch_size', type=int, default=config.DEFAULT_BATCH,
                        help='Batch size for training')
    
    # Model parameters
    parser.add_argument('--diff_steps', type=int, default=100,
                        help='Number of diffusion steps')
    parser.add_argument('--beta_end', type=float, default=0.1,
                        help='Final beta value for diffusion')
    parser.add_argument('--residual_layers', type=int, default=8,
                        help='Number of residual layers')
    parser.add_argument('--residual_channels', type=int, default=8,
                        help='Number of residual channels')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-2,
                        help='Maximum learning rate for OneCycle')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay for optimizer')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping threshold')
    
    # System parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_epoch(model, loader, optimizer, scheduler, device, clip_grad):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    with tqdm(loader, desc="Training", leave=False) as pbar:
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # Forward pass
            loss = model(
                x_future=batch['x_future'],
                cond_dynamic=batch['cond_dynamic'],
                cond_static=batch['cond_static'],
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })
    
    return total_loss / len(loader)


def eval_epoch(model, loader, device):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            loss = model(
                x_future=batch['x_future'],
                cond_dynamic=batch['cond_dynamic'],
                cond_static=batch['cond_static'],
            )
            total_loss += loss.item()
    
    return total_loss / len(loader)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"💾 Checkpoint saved: {path}")


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # ===========================================
    # 1. Load and preprocess data
    # ===========================================
    print("\n📊 Loading data...")
    collector = DataCollector()
    dfs = collector.collect_all_data()
    
    print("🔄 Preprocessing data...")
    dm = TimeGradDataModule(
        data_dict=dfs,
        seq_len=args.seq_len,
        forecast_horizon=args.horizon,
        batch_size=args.batch_size,
        device=device
    )
    dm.preprocess_and_split()
    dm.build_datasets()
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")
    
    # ===========================================
    # 2. Initialize model
    # ===========================================
    print("\n🏗️ Building model...")
    
    # Get dimensions from data
    sample_batch = next(iter(train_loader))
    cond_dynamic_dim = sample_batch['cond_dynamic'].shape[-1]
    cond_static_dim = sample_batch['cond_static'].shape[-1]
    
    print(f"  - Dynamic conditioning dim: {cond_dynamic_dim}")
    print(f"  - Static conditioning dim: {cond_static_dim}")
    
    model = create_conditional_timegrad(
        cond_dynamic_dim=cond_dynamic_dim,
        cond_static_dim=cond_static_dim,
        diff_steps=args.diff_steps,
        beta_end=args.beta_end,
        residual_layers=args.residual_layers,
        residual_channels=args.residual_channels,
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {num_params:,}")
    
    # ===========================================
    # 3. Setup optimizer and scheduler
    # ===========================================
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
    )
    
    # ===========================================
    # 4. Training loop
    # ===========================================
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            device, args.clip_grad
        )
        
        # Validate
        val_loss = eval_epoch(model, val_loader, device)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  - Train Loss: {train_loss:.4f}")
        print(f"  - Val Loss:   {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                f"{args.save_dir}/best_model.pt"
            )
            print(f"  ⭐ New best validation loss!")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss,
                f"{args.save_dir}/checkpoint_epoch_{epoch+1}.pt"
            )
    
    print(f"\n✅ Training complete!")
    print(f"   Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()