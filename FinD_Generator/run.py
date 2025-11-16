
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_collector import DataCollector
from src.data_loader import TimeGradDataModule
from src.models import create_conditional_timegrad
from src import config

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# ===========================================
# 1. Collect or load data
# ===========================================
print("📥 Collecting data...")
collector = DataCollector()
dfs = collector.collect_all_data()

# ===========================================
# 2. Initialize DataModule
# ===========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔧 Using device: {device}")

dm = TimeGradDataModule(data_dict=dfs, device=device)

# ===========================================
# 3. Preprocess, split, and transform data
# ===========================================
print("\n🔄 Preprocessing data...")
dm.preprocess_and_split()

# ===========================================
# 4. Build Datasets and Dataloaders
# ===========================================
print("\n🏗️ Building PyTorch datasets...")
dm.build_datasets()

train_loader = dm.train_dataloader()
val_loader = dm.val_dataloader()
test_loader = dm.test_dataloader()

print(f"✅ Train loader: {len(train_loader)} batches")
print(f"✅ Val loader:   {len(val_loader)} batches")
print(f"✅ Test loader:  {len(test_loader)} batches")

# ===========================================
# 5. Inspect a sample batch
# ===========================================
print("\n🔍 Inspecting a sample batch:")
sample_batch = next(iter(train_loader))

for key, tensor in sample_batch.items():
    print(f"  - {key:15s}: dtype={str(tensor.dtype):12s}, shape={str(tuple(tensor.shape))}")

# Verify shapes match expectations
print(f"\n✅ Shape Verification:")
print(f"  - x_future:     univariate ✓" if sample_batch['x_future'].dim() == 2 else f"  - x_future:     WRONG SHAPE")
print(f"  - x_hist:       univariate ✓" if sample_batch['x_hist'].dim() == 2 else f"  - x_hist:       WRONG SHAPE")
print(f"  - Dynamic conditioning features: {sample_batch['cond_dynamic'].shape[-1]}")
print(f"  - Static conditioning features:  {sample_batch['cond_static'].shape[-1]}")

# ===========================================
# 6. (Optional) Save processed data to CSV
# ===========================================
print("\n💾 Saving processed data splits...")
train_path = os.path.join(config.PROCESSED_DATA_DIR, "train_processed.csv")
val_path = os.path.join(config.PROCESSED_DATA_DIR, "val_processed.csv")
test_path = os.path.join(config.PROCESSED_DATA_DIR, "test_processed.csv")

dm.train_transformed_full.to_csv(train_path)
dm.val_transformed_full.to_csv(val_path)
dm.test_transformed_full.to_csv(test_path)

print(f"✅ Saved train data to {train_path}")
print(f"✅ Saved validation data to {val_path}")
print(f"✅ Saved test data to {test_path}")

# ===========================================
# 7. Initialize ConditionalTimeGrad Model
# ===========================================
print("\n🚀 Initializing ConditionalTimeGrad model...")

cond_dynamic_dim = sample_batch['cond_dynamic'].shape[-1]
cond_static_dim = sample_batch['cond_static'].shape[-1]

model = create_conditional_timegrad(
    cond_dynamic_dim=cond_dynamic_dim,
    cond_static_dim=cond_static_dim,
    target_dim=1,  # Univariate
    diff_steps=100,
    beta_end=0.1,
    residual_layers=8,
    residual_channels=8,
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✅ Model initialized with {num_params:,} parameters")

# ===========================================
# 8. Quick Training Test (5 epochs)
# ===========================================
print("\n🧪 Running quick training test (5 epochs)...")

optimizer = Adam(model.parameters(), lr=1e-3)
scheduler = OneCycleLR(optimizer, max_lr=1e-2, epochs=5, steps_per_epoch=len(train_loader))

for epoch in range(5):
    model.train()
    total_loss = 0.0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/5") as pbar:
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            loss = model(
                x_future=batch['x_future'],
                cond_dynamic=batch['cond_dynamic'],
                cond_static=batch['cond_static'],
            )
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    print(f"  Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

print("\n✅ Training test complete!")

# ===========================================
# 9. Save model checkpoint
# ===========================================
checkpoint_path = os.path.join(config.MODEL_DIR, "test_checkpoint.pt")
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, checkpoint_path)
print(f"💾 Saved checkpoint to {checkpoint_path}")

print("\n" + "="*70)
print("🎉 All systems operational!")
print("="*70)
print("\n📝 Next steps:")
print("  1. For full training: python src/training/trainer.py --epochs 50")
print("  2. Monitor tensorboard: tensorboard --logdir logs/")
print("  3. For inference: python inference.py --checkpoint models/best_model.pt")