import data_collector
import data_loader 

# 1. Collect or load data
collector = DataCollector()
dfs = collector.collect_all_data()  # returns dict of DataFrames

# ===========================================
# 2. Initialize DataModule
# ===========================================
dm = TimeGradDataModule(data_dict=dfs)

# ===========================================
# 3. Build raw blocks and preprocess
# ===========================================
raw_blocks = dm.build_raw_blocks()
print("✅ Raw block keys:", list(raw_blocks.keys()))

# Merge, label regimes, add calendar features
merged_df = dm.preprocess_raw_merge()
print("✅ Merged raw shape:", merged_df.shape)
print(merged_df.head(3))

# ===========================================
# 4. Split into train/val/test
# ===========================================
train_df, val_df, test_df = dm.split_chronologically()
print(f"✅ Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# ===========================================
# 5. Fit scalers and PCA (train-only)
# ===========================================
dm.fit_transform_train()

# ===========================================
# 6. Inspect transformed data
# ===========================================
print("✅ Train transformed shape:", dm.train_transformed.shape)
print(dm.train_transformed.filter(like="pca").head(3))

# (Optional) Save outputs for model training
dm.train_transformed.to_csv("train_processed.csv")