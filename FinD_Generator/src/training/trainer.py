"""
train.py

Complete training pipeline for ConditionalTimeGrad model.

Usage:
    python train.py --epochs 50 --batch_size 32 --device cuda
"""

import sys
sys.path.append('FinD_Generator')

import argparse
import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np

from src.data_collector import DataCollector
from src.data_loader import TimeGradDataModule
from src.models import create_conditional_timegrad
from src import config


class TimeGradTrainer:
    """
    A modular class for training the ConditionalTimeGrad model.
    Encapsulates data loading, model creation, training, and checkpointing.
    """
    def __init__(self, **kwargs):
        """
        Initializes the trainer with a configuration.
        Args:
            **kwargs: Configuration parameters for training.
                      Overrides default values.
        """
        self.config = self._get_default_config()
        self.config.update(kwargs)

        self.device = torch.device(self.config['device'] if torch.cuda.is_available() else 'cpu')
        self._set_seed(self.config['seed'])
        print(f"🔧 Using device: {self.device}")

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.dm = None

    @staticmethod
    def _get_default_config():
        """Returns a dictionary of default configuration parameters."""
        return {
            'seq_len': config.DEFAULT_SEQ_LEN,
            'horizon': config.DEFAULT_HORIZON,
            'batch_size': config.DEFAULT_BATCH,
            'diff_steps': 100,
            'beta_end': 0.1,
            'residual_layers': 8,
            'residual_channels': 8,
            'epochs': 50,
            'lr': 1e-3,
            'max_lr': 1e-2,
            'weight_decay': 1e-6,
            'clip_grad': 1.0,
            'device': 'cuda',
            'seed': 42,
            'save_dir': 'checkpoints',
            'save_freq': 10,
        }

    def _set_seed(self, seed):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _prepare_data(self):
        """Loads and preprocesses data."""
        print("\n📊 Loading and preprocessing data...")
        collector = DataCollector()
        dfs = collector.collect_all_data()

        self.dm = TimeGradDataModule(
            data_dict=dfs,
            seq_len=self.config['seq_len'],
            forecast_horizon=self.config['horizon'],
            batch_size=self.config['batch_size'],
            device=self.device
        )
        self.dm.preprocess_and_split()
        self.dm.build_datasets()

        self.train_loader = self.dm.train_dataloader()
        self.val_loader = self.dm.val_dataloader()
        print(f"✅ Train batches: {len(self.train_loader)}")
        print(f"✅ Val batches: {len(self.val_loader)}")

    def _build_model(self):
        """Initializes the model, optimizer, and scheduler."""
        print("\n🏗️ Building model...")
        sample_batch = next(iter(self.train_loader))
        cond_dynamic_dim = sample_batch['cond_dynamic'].shape[-1]
        cond_static_dim = sample_batch['cond_static'].shape[-1]

        print(f"  - Dynamic conditioning dim: {cond_dynamic_dim}")
        print(f"  - Static conditioning dim: {cond_static_dim}")

        self.model = create_conditional_timegrad(
            cond_dynamic_dim=cond_dynamic_dim,
            cond_static_dim=cond_static_dim,
            diff_steps=self.config['diff_steps'],
            beta_end=self.config['beta_end'],
            residual_layers=self.config['residual_layers'],
            residual_channels=self.config['residual_channels'],
        ).to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  - Total parameters: {num_params:,}")

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.config['max_lr'],
            epochs=self.config['epochs'],
            steps_per_epoch=len(self.train_loader),
        )

    def run(self):
        """Executes the complete training pipeline."""
        self._prepare_data()
        self._build_model()

        print(f"\n🚀 Starting training for {self.config['epochs']} epochs...")
        os.makedirs(self.config['save_dir'], exist_ok=True)
        best_val_loss = float('inf')

        for epoch in range(self.config['epochs']):
            print(f"\n{'='*60}\nEpoch {epoch+1}/{self.config['epochs']}\n{'='*60}")

            train_loss = train_epoch(
                self.model, self.train_loader, self.optimizer, self.scheduler,
                self.device, self.config['clip_grad']
            )
            val_loss = eval_epoch(self.model, self.val_loader, self.device)

            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"  - Train Loss: {train_loss:.4f}")
            print(f"  - Val Loss:   {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = f"{self.config['save_dir']}/best_model.pt"
                self._save_checkpoint(epoch, val_loss, save_path)
                print(f"  ⭐ New best validation loss!")

            if (epoch + 1) % self.config['save_freq'] == 0:
                save_path = f"{self.config['save_dir']}/checkpoint_epoch_{epoch+1}.pt"
                self._save_checkpoint(epoch, val_loss, save_path)

        print(f"\n✅ Training complete!")
        print(f"   Best validation loss: {best_val_loss:.4f}")

    def _save_checkpoint(self, epoch, loss, path):
        """Saves a model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, path)
        print(f"💾 Checkpoint saved: {path}")


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
    """Main function to run training from the command line."""
    args = parse_args()
    print(f"{'='*60}")
    # Convert argparse namespace to a dictionary
    config_from_args = vars(args)
    trainer = TimeGradTrainer(**config_from_args)
    trainer.run()


if __name__ == "__main__":
    main()

# In a Jupyter cell
import sys
sys.path.append('FinD_Generator')
from src.training.trainer import TimeGradTrainer

# Customize training parameters if needed
config = {
    'epochs': 20,
    'batch_size': 64,
    'lr': 0.001,
    'save_dir': 'notebook_checkpoints'
}

# Initialize and run the trainer
trainer = TimeGradTrainer(**config)
trainer.run()
