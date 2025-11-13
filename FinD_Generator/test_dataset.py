# test_dataset.py (or in a notebook cell)
from torch.utils.data import DataLoader
from FinD_Generator.src.data_loader import FinanceDataset

# --- Configuration ---
STOCK_PATH = 'data/raw/stock_data.csv'
MACRO_PATH = 'data/raw/macro_data.csv'
LABEL_PATH = 'data/raw/scenario_labels.csv'
SEQUENCE_LENGTH = 60 # Use 60 days of data for each sample
BATCH_SIZE = 4

# --- 1. Instantiate the Dataset ---
try:
    dataset = FinanceDataset(
        stock_path=STOCK_PATH,
        macro_path=MACRO_PATH,
        label_path=LABEL_PATH,
        sequence_length=SEQUENCE_LENGTH
    )

    # --- 2. Create a DataLoader ---
    # The DataLoader handles batching, shuffling, etc.
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # --- 3. Fetch one batch of data ---
    first_batch = next(iter(data_loader))

    # --- 4. Inspect the batch ---
    print("\n--- Inspecting the first batch ---")
    target_batch = first_batch['target']
    macro_batch = first_batch['macro_features']
    label_batch = first_batch['scenario_label']

    print(f"Target shape: {target_batch.shape}")
    print(f"Expected target shape: (batch_size, sequence_length) -> ({BATCH_SIZE}, {SEQUENCE_LENGTH})")
    
    print(f"\nMacro features shape: {macro_batch.shape}")
    print(f"Expected macro shape: (batch_size, sequence_length, num_macro_features)")

    print(f"\nLabels shape: {label_batch.shape}")
    print(f"Expected labels shape: (batch_size) -> ({BATCH_SIZE})")
    print(f"Sample labels: {label_batch.numpy()}")

except FileNotFoundError:
    print("Error: Make sure your CSV files are in the 'data/raw/' directory.")
except Exception as e:
    print(f"An error occurred: {e}")