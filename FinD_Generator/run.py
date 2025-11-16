import sys
import os

sys.path.append('\workspaces\AI_FinSys\FinD_Generator')

from src.data_collector import DataCollector
from src.data_loader import TimeGradDataModule
from src import config

from gluonts.dataset.common import ListDataset
# Import TimeGrad components
from src.TimeGrad.pts.model.time_grad import TimeGradEstimator
from src.TimeGrad.pts import Trainer

# 1. Collect or load data
collector = DataCollector()
dfs = collector.collect_all_data()  # returns dict of DataFrames

# ===========================================
# 2. Initialize DataModule
# ===========================================
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

dm = TimeGradDataModule(data_dict=dfs, device=device)

# ===========================================
# 3. Preprocess, split, and transform data
# ===========================================
# This single method handles:
# - Building raw blocks (wavelet, returns, etc.)
# - Merging all data sources
# - Adding calendar features and regime labels
# - Splitting into train/val/test sets
# - Fitting scalers/PCA on the training set and transforming all sets
dm.preprocess_and_split()

# ===========================================
# 4. Build Datasets and Dataloaders
# ===========================================
# The TimeGradEstimator works with GluonTS Datasets, not PyTorch DataLoaders.
# We will create GluonTS ListDataset objects from the processed dataframes.

# The target columns are the PCA components of the target series.
target_cols = [c for c in dm.train_df.columns if c.startswith('target_pca_')]

# The dynamic real features are all other numeric columns.
feat_dynamic_real_cols = [
    c for c in dm.train_df.columns if c not in target_cols and dm.train_df.dtypes[c].name in ['float64', 'int64']
]

train_ds = ListDataset(
    [{"start": dm.train_df.index[0], "target": dm.train_df[target_cols].values.T, "feat_dynamic_real": dm.train_df[feat_dynamic_real_cols].values.T}],
    freq=config.FREQ
)

val_ds = ListDataset(
    [{"start": dm.val_df.index[0], "target": dm.val_df[target_cols].values.T, "feat_dynamic_real": dm.val_df[feat_dynamic_real_cols].values.T}],
    freq=config.FREQ
)

# ===========================================
# 5. Inspect a sample batch
# ===========================================
# This is now handled internally by the GluonTS estimator, so we can inspect the dataset directly.
print("\n🔍 Inspecting the training dataset:")
train_entry = next(iter(train_ds))
for key, value in train_entry.items():
    print(f"  - {key}: {type(value)}", end="")
    if hasattr(value, 'shape'):
        print(f", shape: {value.shape}")
    else:
        print("")

# ===========================================
# 6. (Optional) Save processed data to CSV
# ===========================================
print("\n💾 Saving processed data splits to CSV...")
train_path = os.path.join(config.PROCESSED_DATA_DIR, "train_processed.csv")
val_path = os.path.join(config.PROCESSED_DATA_DIR, "val_processed.csv")
test_path = os.path.join(config.PROCESSED_DATA_DIR, "test_processed.csv")

dm.train_df.to_csv(train_path)
dm.val_df.to_csv(val_path)
dm.test_df.to_csv(test_path)

print(f"✅ Saved train data to {train_path}")
print(f"✅ Saved validation data to {val_path}")
print(f"✅ Saved test data to {test_path}")

# ===========================================
# 7. Configure and Train the Model
# ===========================================
print("\n🚀 Configuring and training the TimeGrad model...")

# Configure the Trainer
trainer = Trainer(
    epochs=50,
    batch_size=32,
    num_batches_per_epoch=100,
    learning_rate=1e-3,
    weight_decay=1e-8,
    device=device,
)

# Calculate input_size for the RNN
# This is based on the features created by the data pipeline:
# lagged values + time features + target dimension indicator
input_size = (
    len(config.LAGS_SEQ) * config.TARGET_DIM
    + len(config.TIME_FEATURES)
    + config.TARGET_DIM  # for target_dimension_indicator embedding
)

# Instantiate the Estimator
estimator = TimeGradEstimator(
    # --- Model/data dimensions ---
    target_dim=config.TARGET_DIM,
    prediction_length=config.PREDICTION_LENGTH,
    context_length=config.CONTEXT_LENGTH,
    input_size=input_size,
    freq=config.FREQ,
    lags_seq=config.LAGS_SEQ,
    time_features=config.TIME_FEATURES,
    # --- Diffusion parameters ---
    diff_steps=100,
    loss_type="l2",
    beta_end=0.1,
    beta_schedule="linear",
    # --- RNN parameters ---
    num_layers=2,
    num_cells=40,
    # --- WaveNet (EpsilonTheta) parameters ---
    residual_layers=8,
    residual_channels=8,
    dilation_cycle_length=2,
    # --- Feature flags ---
    use_feat_dynamic_real=True, # We are providing dynamic features
    use_feat_static_real=False, # We are not providing static real features in this setup
    # --- Trainer ---
    trainer=trainer,
)

# Train the model and get a predictor
predictor = estimator.train(
    training_data=train_ds, validation_data=val_ds, num_workers=0 # Use 0 for ListDataset
)

# ===========================================
# 8. Save the trained model
# ===========================================
model_save_path = os.path.join(config.MODEL_DIR, "timegrad_predictor")
predictor.serialize(model_save_path)
print(f"\n✅ Predictor saved to {model_save_path}")