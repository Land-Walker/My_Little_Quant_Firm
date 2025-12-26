# FinD_Generator: Financial Diffusion Forecast Generator

**FinD_Generator** is a robust, regime-aware probabilistic forecasting system based on Conditional TimeGrad. It adapts Denoising Diffusion Probabilistic Models (DDPM) for financial time series by integrating multi-frequency covariates, static regime embeddings, and a "Student-t Copula" normalization scheme to handle heavy-tailed market data.

This repository implements a complete pipeline: **Data Loading (Wavelet/PCA)**  **Conditional Training**  **Autoregressive Inference**  **Scenario Stress Testing**.

---

## Key Features

* **Regime-Aware Denoising**: Injects global market contexts (e.g., "Bull", "Bear", "High Vol") via **FiLM modulation** and time-varying covariates via **Cross-Attention**.
* **Student-t Marginal Normalization**: automatically handles fat-tailed distributions by fitting a Student-t distribution to historical windows and transforming data to a Gaussian space before diffusion.
* **Strict Causal Inference**: Implements a "Zero-Leakage" autoregressive loop where future dynamic tokens are explicitly zeroed out to prevent look-ahead bias.
* **Scenario Stress Testing**: A dedicated engine for counterfactual analysis, allowing researchers to force specific regimes (e.g., "Crash") during inference without retraining the model.
* **Dual Conditioning Strategies**:
  * **Fast Strategy**: Uses Cross-Attention between noisy targets and conditioning tokens.
  * **Slow Strategy**: Uses an LSTM/GRU to encode the joint history-covariate sequence.



---

## Architecture Overview

The system is composed of four distinct layers:

### 1. The Data Layer (`data_loader.py`)

Handles the complexity of financial data before it reaches the model.

* **Preprocessing**: Applies **Wavelet Denoising (db4)** to raw prices, followed by **PCA (95% variance)** to extract structural factors.
* **Alignment**: Merges multi-frequency data (Daily Price + Monthly Macro) using forward-filling to prevent leakage.
* **Output**: Produces `x_hist` (context), `x_future` (target), `cond_dynamic` (covariates), and `cond_static` (regime tags).

### 2. The Model Core (`epsilon_theta.py` & `gaussian_diffusion.py`)

* **Backbone**: A dilated convolutional network (WaveNet-style) using **circular padding** to handle time-series boundaries.
* **Diffusion**: Standard Gaussian Diffusion with a fixed **linear beta schedule**.

### 3. The Conditioning Wrapper (`conditioned_epsilon_theta.py`)

Acts as the bridge between raw metadata and the diffusion core.

* **Fusion**: Projects dynamic/static features into a shared embedding space.
* **Mechanism**: Uses **FiLM** to scale the noisy input  and **Cross-Attention** (in "Fast" mode) to inject temporal details.

### 4. The Execution Loops

* **Training (`training_network.py`)**: Computes diffusion loss on full batches using parallelized "teacher forcing" logic.
* **Prediction (`prediction_network.py`)**: Performs iterative **autoregressive sampling**. It re-encodes the history at every step to maintain temporal coherence.

---

## Usage Workflow

### 1. Training

The `TrainingNetwork` handles the forward diffusion process. It automatically computes the Student-t parameters () for the current batch.

```python
# Conceptual Example
trainer = TrainingNetwork(
    target_dim=...,
    prediction_length=...,
    context_length=...
)

# Returns MSE loss between predicted noise and actual noise
loss = trainer(x_hist, x_future, cond_dynamic, cond_static)

```

### 2. Inference (Forecasting)

The `PredictionNetwork` manages the sliding window generation.

* **Strategy**: Autoregressive sliding-window sampling. At each step, the model either:
  * samples a single timestep via masked diffusion, or
  * samples a full horizon and retains only the first step.
* **Normalization**: Student-t parameters are frozen at  to ensure consistent scaling across the horizon.

```python
predictor = PredictionNetwork(model=trainer.model, ...)
forecast = predictor.sample_forecast(
    x_hist, 
    cond_dynamic, 
    cond_static, 
    num_samples=100
)

```

### 3. Scenario Analysis

Use `ScenarioGenerator` to modify regimes at runtime.

```python
from scenario_generator import ScenarioGenerator, ScenarioSpec

# Force a "High Volatility" regime for the next 10 days
scenario = ScenarioSpec(
    vol_regime="high",
    start_t=0,
    duration=10,
    transition="hard"
)

mod_cond = scenario_generator.apply_scenario(
    cond_df,
    scenario,
    horizon=10
)

# Apply to conditioning data
mod_cond = ScenarioGenerator.apply_scenarios(cond_dynamic, [scenario])

```

---

## Research Notes & Implementation Details

For researchers replicating this architecture, note the following critical implementation details:

1. **The "Zero-Fill" Policy**: During inference, the `PredictionNetwork` explicitly zeros out `cond_dynamic` tokens for time steps . This forces the model to rely on the learnt dynamics rather than leaking future covariates.
2. **Residual Context Projection**: In the `ConditionedEpsilonTheta` "Fast" path, the output of the cross-attention is passed through a `context_proj` layer and added as a **residual** to the noisy input before FiLM scaling.
3. **Order of Operations**: In `data_loader.py`, Wavelet Denoising is applied **before** PCA. This ensures dimensionality reduction focuses on the clean signal structure rather than noise.
