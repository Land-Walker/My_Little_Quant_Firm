
"""
run.py

A simple script to start the training process for the ConditionalTimeGrad model.
This script is a lightweight wrapper around the TimeGradTrainer class.

For detailed, step-by-step execution and inference, please see the notebook:
notebooks/example_run.ipynb
"""

import sys
import os
sys.path.append(os.path.dirname(__file__)) # Add FinD_Generator to path

from src.training.trainer import main as run_training

if __name__ == "__main__":
    print("🚀 Starting TimeGrad training process...")
    run_training()