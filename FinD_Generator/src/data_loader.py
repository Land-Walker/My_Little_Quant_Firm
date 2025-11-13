"""
timegrad_data_module.py

Unified DataModule (function-based + DataModule) for TimeGrad preprocessing.

Key design choices:
- Wavelet denoising: db4, level=3 (train-time config)
- PCA: explained_variance=0.95 (train-fit)
- Scalers/PCA are FIT ON TRAIN only and applied to val/test (no leakage)
- Regime labeling (market daily; macro monthly/quarterly)
- Dataset outputs x_hist, c_hist, x_future for TimeGrad

Author: Wooseok Lee (adapted)
"""

import numpy as np
import pandas as pd

import pywt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset, DataLoader

from typing import Dict, List, Optional, Tuple
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

import config

# =========================================================
# 0. Basic helpers (from your initial code)
# =========================================================
def log_return(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1))

def log_growth(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1))

def seasonal_difference(series: pd.Series, period: int = 12) -> pd.Series:
    return series - series.shift(period)

def wavelet_denoise_series(series: pd.Series, wavelet: str = WAVELET, level: int = WAVELET_LEVEL) -> pd.Series:
    """Denoise by applying wavelet thresholding and reconstructing the signal."""
    x = series.fillna(method="ffill").fillna(method="bfill").values
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
    # estimate noise sigma from last detail coeff
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if len(coeffs[-1]) > 0 else 0.0
    uthresh = sigma * np.sqrt(2 * np.log(len(x))) if sigma > 0 else 0.0
    denoised = coeffs[:]
    denoised[1:] = [pywt.threshold(c, value=uthresh, mode="soft") for c in denoised[1:]]
    rec = pywt.waverec(denoised, wavelet=wavelet)
    rec = rec[: len(x)]
    return pd.Series(rec, index=series.index)

def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Return arr as 2D array with shape (n_rows, n_cols). If 1D, reshape to (n_rows,1)."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def add_calendar_features(
    df: pd.DataFrame,
    drop_original_index: bool = False
) -> pd.DataFrame:
    """Add standard calendar/time features based on datetime index."""
    df_copy = df.copy()
    # Ensure the index is a DatetimeIndex
    if not isinstance(df_copy.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex to add calendar features")

    df_copy['day_of_week'] = df_copy.index.dayofweek
    df_copy['month'] = df_copy.index.month
    df_copy['quarter'] = df_copy.index.quarter
    df_copy['year'] = df_copy.index.year
    df_copy['is_month_end'] = df_copy.index.is_month_end.astype(int)
    df_copy['is_quarter_end'] = df_copy.index.is_quarter_end.astype(int)

    # cyclical encoding
    df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
    df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
    df_copy['dow_sin'] = np.sin(2 * np.pi * df_copy['day_of_week'] / 7)
    df_copy['dow_cos'] = np.cos(2 * np.pi * df_copy['day_of_week'] / 7)
    df_copy['quarter_sin'] = np.sin(2 * np.pi * df_copy['quarter'] / 4)
    df_copy['quarter_cos'] = np.cos(2 * np.pi * df_copy['quarter'] / 4)
    if drop_original_index:
        df_copy = df_copy.reset_index(drop=True)
    return df_copy

def align_and_handle_missing_values(
    quarterly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    date_col: str = 'Date'
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align macroeconomic dataframes to correct frequency, handle missing values in a leakage-safe manner.

    Forward-fill is applied only on past data (ffill). No transforms (diff, log-growth, seasonal diff)
    are computed here, to avoid using future information.
    """
    print(f"align_and_handle_missing_values: daily_df columns before processing: {daily_df.columns.tolist()}")

    def preprocess_df(df: pd.DataFrame, suffix: str = '', freq: Optional[str] = None) -> pd.DataFrame:
        df_copy = df.copy()
        if date_col in df_copy.columns:
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            df_copy = df_copy.set_index(date_col).sort_index()
        elif not isinstance(df_copy.index, pd.DatetimeIndex):
             raise ValueError(f"DataFrame must have '{date_col}' column or DatetimeIndex")


        # Resample structurally without computing any statistics across the whole series
        if freq is not None:
            df_copy = df_copy.resample(freq).ffill()  # forward-fill is safe (past-only)

        if suffix:
            df_copy = df_copy.add_suffix(suffix)

        return df_copy

    # Quarterly -> monthly frequency, ffill past values only
    quarterly_aligned = preprocess_df(quarterly_df, suffix='_quarterly', freq='MS')

    # Monthly -> keep original frequency, ffill past values
    monthly_df_sorted = preprocess_df(monthly_df)

    # Daily -> keep original frequency
    daily_aligned = preprocess_df(daily_df, suffix='_daily')

    print(f"align_and_handle_missing_values: daily_aligned columns after processing: {daily_aligned.columns.tolist()}")

    return quarterly_aligned, monthly_df_sorted, daily_aligned

# =========================================================
# 1. Block processors (no train-fit state here)
#    These produce raw transformed columns but do NOT fit scalers/PCA.
# =========================================================

def process_target_raw(target_df: pd.DataFrame, ohlc_cols: List[str] = ['open','high','low','close']) -> pd.DataFrame:
    """Apply wavelet denoising to each OHLC column and keep volume unchanged."""
    df = target_df.copy() # .set_index('Date') - Removed: Assuming Date is already index
    for c in ohlc_cols:
        if c not in df.columns:
            raise KeyError(f"{c} not in target_df")
        df[f"{c}_den"] = wavelet_denoise_series(df[c])
    # keep volume separately (unaltered)
    if 'volume' in df.columns:
        df['volume_raw'] = df['volume']
    return df[[f"{c}_den" for c in ohlc_cols] + (['volume_raw'] if 'volume' in df.columns else [])]


def process_market_raw(market_df: pd.DataFrame, ohlc_cols: List[str] = ['open','high','low','close']) -> pd.DataFrame:
    """Compute log-returns for OHLC and keep volume (log1p) raw column."""
    df = market_df.copy() # .set_index('Date') - Removed: Assuming Date is already index
    for c in ohlc_cols:
        if c not in df.columns:
            raise KeyError(f"{c} not in market_df")
        df[f"{c}_ret"] = log_return(df[c])
    if 'volume' in df.columns:
        df['volume_raw'] = np.log1p(df['volume'])
    # return only generated columns
    cols = [f"{c}_ret" for c in ohlc_cols] + ((['volume_raw'] if 'volume' in df.columns else []))
    return df[cols]


def process_daily_macro_raw(daily_macro_df: pd.DataFrame) -> pd.DataFrame:
    """Scale-ready daily macro fields (vix, yield_curve) returned raw (no scaler fit here)."""
    df = daily_macro_df.copy() # .set_index('Date') - Removed: Assuming Date is already index
    # ensure columns exist
    assert 'vix_daily' in df.columns, "daily_macro_df must contain 'vix_daily'"
    assert 'yield_curve_daily' in df.columns, "daily_macro_df must contain 'yield_curve_daily'"
    return df[['vix_daily','yield_curve_daily']]


def process_monthly_macro_raw(monthly_macro_df: pd.DataFrame) -> pd.DataFrame:
    """Apply transforms described: cpi pct-change, unemployment rolling detrend(12), interest rate diffs, seasonal diff trade_balance."""
    df = monthly_macro_df.copy() # .set_index('Date') - Removed: Assuming Date is already index
    out = pd.DataFrame(index=df.index)
    # CPI -> month-on-month (or pct change)
    if 'cpi' in df.columns:
        out['cpi_mom'] = df['cpi'].pct_change()
    # unemployment detrend (rolling 12 mean subtract)
    if 'unemployment' in df.columns:
        out['unemployment_detrend'] = df['unemployment'] - df['unemployment'].rolling(12, min_periods=1).mean()
    # interest rate -> rate of change (diff)
    if 'interest_rate' in df.columns:
        out['interest_rate_diff'] = df['interest_rate'].diff()
    # trade balance -> seasonal difference (default 12)
    if 'trade_balance' in df.columns:
        out['trade_balance_seasdiff'] = seasonal_difference(df['trade_balance'], period=12)
    return out


def process_quarterly_macro_raw(quarterly_macro_df: pd.DataFrame) -> pd.DataFrame:
    """Quarterly transforms: gdp log-growth, gov fiscal balance ratio-to-gdp, keep other series as-is for scaling later."""
    df = quarterly_macro_df.copy() # .set_index('Date') - Removed: Assuming Date is already index
    out = pd.DataFrame(index=df.index)
    if 'gdp' in df.columns:
        out['gdp_yoy'] = log_growth(df['gdp'])
    # gov_fiscal_balance as ratio to (real) gdp (avoid division by zero)
    if 'gov_fiscal_balance' in df.columns and 'gdp' in df.columns:
        out['gov_fiscal_balance_to_gdp'] = df['gov_fiscal_balance'] / df['gdp'].replace({0: np.nan})
    # passthrough others
    for col in ['gov_debt','tax_receipts','gov_spending']:
        if col in df.columns:
            out[col] = df[col]
    return out

# =========================================================
# 2. Regime labellers (use after merging, but before scaling)
# =========================================================
def label_market_regimes(df: pd.DataFrame,
                         price_col: str = "market_close",
                         vol_col: str = "vix",
                         window: int = 30,
                         r_thresh: float = 0.02,
                         v_thresh: float = 20) -> pd.DataFrame:
    """Add market + volatility regimes and one-hot encode them."""
    df = df.copy()
    # rolling returns (backward-looking)
    if price_col not in df.columns:
        raise KeyError(f"price_col {price_col} not found in df for regime labelling")
    df['roll_return'] = np.log(df[price_col]).diff(window)
    df['roll_vol'] = df[price_col].pct_change().rolling(window, min_periods=1).std()
    cond_bull = (df['roll_return'] > r_thresh) & (df['roll_vol'] < df['roll_vol'].median())
    cond_bear = (df['roll_return'] < -r_thresh) & (df['roll_vol'] > df['roll_vol'].median())
    df['market_regime'] = np.select([cond_bull, cond_bear], ['bull','bear'], default='sideways')
    # vol regime
    if vol_col in df.columns:
        df['vol_regime'] = np.where(df[vol_col] > v_thresh, 'high_vol', 'normal_vol')
    # one-hot encode (safely)
    to_encode = [c for c in ['market_regime','vol_regime'] if c in df.columns]
    if to_encode:
        df = pd.get_dummies(df, columns=to_encode, prefix_sep='_')
    return df


def label_macro_regimes(df: pd.DataFrame,
                        infl_col: str = "cpi_mom",
                        gdp_col: str = "gdp_yoy",
                        rolling_window: int = 12,
                        high_infl: float = 0.03,
                        low_growth: float = 0.0) -> pd.DataFrame:
    """Add macro regime categories (expansion/recession/high_inflation/stagflation) and one-hot encode."""
    df = df.copy()
    # ensure these columns exist
    if infl_col in df.columns:
        # use already computed yoy or mom; assume infl_col is percent-change series
        infl = df[infl_col]
    else:
        infl = None
    if gdp_col in df.columns:
        gdp = df[gdp_col]
    else:
        gdp = None
    df['macro_regime'] = 'normal'
    if infl is not None and gdp is not None:
        cond_stag = (infl > high_infl) & (gdp < low_growth)
        cond_highinf = (infl > high_infl) & (gdp >= low_growth)
        cond_recess = (infl <= high_infl) & (gdp < low_growth)
        cond_expand = (infl <= high_infl) & (gdp >= low_growth)
        df.loc[cond_stag, 'macro_regime'] = 'stagflation'
        df.loc[cond_highinf, 'macro_regime'] = 'high_inflation'
        df.loc[cond_recess, 'macro_regime'] = 'recession'
        df.loc[cond_expand, 'macro_regime'] = 'expansion'
    # one-hot encode
    df = pd.get_dummies(df, columns=['macro_regime'], prefix_sep='_')
    return df

# =========================================================
# 3. Merge & align utilities
# =========================================================
def merge_all_blocks_unified(
    target_raw: pd.DataFrame,
    market_raw: pd.DataFrame,
    daily_macro_raw: pd.DataFrame,
    monthly_macro_raw: pd.DataFrame,
    quarterly_macro_raw: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Merge all raw blocks (target, market, daily/monthly/quarterly macro) into a single
    daily-indexed DataFrame safely without data leakage.

    Forward-fill is applied only on past data (safe). Monthly/quarterly data are
    reindexed to daily frequency.
    """
    # --- Step 1: ensure datetime index and sort ---
    def prep(df: pd.DataFrame):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    t = prep(target_raw)
    mkt = prep(market_raw)
    daily_macro = prep(daily_macro_raw)
    monthly_macro = prep(monthly_macro_raw)
    quarterly_macro = prep(quarterly_macro_raw)

    # --- Step 2: define daily index ---
    daily_index_base = t.index
    if start_date is None:
        start_date = daily_index_base.min()
    if end_date is None:
        end_date = daily_index_base.max()

    daily_index = pd.date_range(start=start_date, end=end_date, freq="B")

    # --- Step 3: reindex daily and forward-fill ---
    t = t.reindex(daily_index).ffill().bfill()
    mkt = mkt.reindex(daily_index).ffill().bfill()
    daily_macro = daily_macro.reindex(daily_index).ffill()

    # --- Step 4: align monthly/quarterly data ---
    monthly_macro.index = monthly_macro.index.to_period("M").to_timestamp("M")
    quarterly_macro.index = quarterly_macro.index.to_period("M").to_timestamp("M")

    monthly_aligned_daily = monthly_macro.reindex(daily_index).ffill()
    quarterly_aligned_daily = quarterly_macro.reindex(daily_index).ffill()

    # --- Step 5: concatenate all blocks ---
    merged_df = pd.concat([daily_macro, monthly_aligned_daily, quarterly_aligned_daily, t, mkt], axis=1)

    # drop columns that are all NaN
    merged_df = merged_df.dropna(how="all")

    return merged_df

# =========================================================
# 4. Train/Val/Test safe fit-transform pipeline
#    - preprocess_raw -> split -> fit_on_train -> transform_all -> dataset setup
# =========================================================

class TimeGradDataModule:
    """
    Data module that:
      - Accepts raw dataframes (same names as your collector)
      - Builds raw transformed blocks (wavelet, log-returns, seasonal diffs)
      - Merges them daily
      - Labels regimes (market + macro)
      - Splits chronologically into train/val/test
      - Fits scalers + PCA on TRAIN only and applies to all splits
      - Exposes PyTorch DataLoaders
    """

    def __init__(self,
                 data_dict: Dict[str, pd.DataFrame],
                 seq_len: int = DEFAULT_SEQ_LEN,
                 forecast_horizon: int = DEFAULT_HORIZON,
                 batch_size: int = DEFAULT_BATCH,
                 pca_variance: float = PCA_VARIANCE,
                 device: str = "cpu"):
        """
        data_dict keys required: target, market, daily_macro, monthly_macro, quarterly_macro
        target & market expected to have columns ['Date','open','high','low','close','volume'] (Date col will be set as index)
        monthly_macro/quarterly_macro/daily_macro expected to have 'Date' col and relevant names:
           daily_macro: 'vix','yield_curve'
           monthly_macro: 'cpi','unemployment','interest_rate','trade_balance'
           quarterly_macro: 'gdp','gov_debt','gov_fiscal_balance','tax_receipts','gov_spending'
        """
        required = ["target", "market", "daily_macro", "monthly_macro", "quarterly_macro"]
        missing = [k for k in required if k not in data_dict]
        if missing:
            raise ValueError(f"Missing datasets in data_dict: {missing}")
        self.data = data_dict
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.batch_size = batch_size
        self.pca_variance = pca_variance
        self.device = device

        # placeholders for fitted objects
        self.scalers: Dict[str, StandardScaler] = {}
        self.pcas: Dict[str, PCA] = {}

        # storage for splits
        self.train_df = None
        self.val_df = None
        self.test_df = None

        # Align all macro data safely (avoids leakage)
        # Store the aligned macro data back into the data_dict
        print("🔧 [Init] Aligning and preparing macroeconomic data...")
        self.data["quarterly_macro_aligned"], self.data["monthly_macro_aligned"], self.data["daily_macro_aligned"] = align_and_handle_missing_values(
            data_dict["quarterly_macro"],
            data_dict["monthly_macro"],
            data_dict["daily_macro"])
        print("✅ [Init] Macro data alignment complete.\n")


    # ---------------------------
    # Raw block preprocessing (no fit)
    # ---------------------------
    def build_raw_blocks(self) -> Dict[str, pd.DataFrame]:
        # Ensure Date is set as index before passing to processing functions
        target_indexed = self._ensure_index(self.data["target"])
        market_indexed = self._ensure_index(self.data["market"])
        # Use the already aligned and indexed macro data
        daily_macro_indexed = self.data["daily_macro_aligned"]
        monthly_macro_indexed = self.data["monthly_macro_aligned"]
        quarterly_macro_indexed = self.data["quarterly_macro_aligned"]

        print(f"build_raw_blocks: daily_macro_indexed columns before calling process_daily_macro_raw: {daily_macro_indexed.columns.tolist()}")

        print("🔄 [build_raw_blocks] Processing raw data blocks...")
        t_raw = process_target_raw(target_indexed)
        print("✅ Target wavelet denoising complete.")
        mkt_raw = process_market_raw(market_indexed)
        print("✅ Market log-returns computed.")
        daily_macro_raw = process_daily_macro_raw(daily_macro_indexed)
        print("✅ Daily macro block processed.")
        monthly_macro_raw = process_monthly_macro_raw(monthly_macro_indexed)
        print("✅ Monthly macro transformations complete.")
        quarterly_macro_raw = process_quarterly_macro_raw(quarterly_macro_indexed)
        print("✅ Quarterly macro transformations complete.")
        print("🏗️ [build_raw_blocks] All raw data blocks prepared.\n")

        return {
            "target": t_raw,
            "market": mkt_raw,
            "daily_macro": daily_macro_raw,
            "monthly_macro": monthly_macro_raw,
            "quarterly_macro": quarterly_macro_raw
        }

    def _ensure_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'Date' in df.columns:
            df = df.set_index(pd.DatetimeIndex(df['Date']))
            df.index.name = 'Date'
            df = df.drop(columns=['Date'])
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have 'Date' column or DatetimeIndex")
        return df.sort_index()

    # ---------------------------
    # Merge raw blocks, add regimes & calendar (no scaling/pca yet)
    # ---------------------------
    def preprocess_raw_merge(self) -> pd.DataFrame:
        blocks = self.build_raw_blocks()
        print("🔄 [preprocess_raw_merge] Merging all blocks into unified DataFrame...")
        merged = merge_all_blocks_unified(
            blocks["target"],
            blocks["market"],
            blocks["daily_macro"],
            blocks["monthly_macro"],
            blocks["quarterly_macro"],)
        print("✅ All blocks merged successfully.")
        # add simple calendar features - Ensure the index is DatetimeIndex before calling
        if not isinstance(merged.index, pd.DatetimeIndex):
             raise ValueError("Merged DataFrame index must be a DatetimeIndex before adding calendar features")
        print("🔄 Adding calendar and regime features...")
        merged = add_calendar_features(merged)
        # add basic market regime labels (needs 'market_close' column — pick 'close' from market if available)
        # We'll create a 'market_close' column if not present: use target close as fallback.
        if "close_ret" in merged.columns:
            # Market ret columns are like 'open_ret','high_ret',...; we want a price-level close if available.
            # Use the original market data (which has the 'close' column) and ensure it's indexed
            original_market = self._ensure_index(self.data["market"])
            merged['market_close'] = original_market['close'].reindex(merged.index).ffill().bfill()
        else:
            # fallback: ensure market_close exists by using the original market data
            original_market = self._ensure_index(self.data["market"])
            merged['market_close'] = original_market['close'].reindex(merged.index).ffill().bfill()

        # vix column should be 'vix' (from daily_macro)
        merged = label_market_regimes(merged, price_col='market_close', vol_col='vix_daily') # Updated to use vix_daily
        # macro regime uses monthly/quarterly derived columns (inflation: 'cpi_mom', gdp: 'gdp_yoy')
        merged = label_macro_regimes(merged, infl_col='cpi_mom', gdp_col='gdp_yoy')
        print("✅ Calendar features and regime labels added.\n")
        # drop intermediate helper cols to avoid leakage (roll_return, roll_vol kept? they are past-looking; we can drop)
        for c in ['roll_return','roll_vol','infl_yoy','gdp_growth_yoy']:
            if c in merged.columns:
                merged = merged.drop(columns=[c])
        self.merged_raw = merged
        return merged

    # ---------------------------
    # Split into train/val/test chronologically (on merged_raw)
    # ---------------------------
    def split_chronologically(self, train_ratio: float = TRAIN_RATIO, val_ratio: float = VAL_RATIO) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        if not hasattr(self, "merged_raw"):
            self.preprocess_raw_merge()
        n = len(self.merged_raw)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        self.train_df = self.merged_raw.iloc[:train_end].copy()
        self.val_df = self.merged_raw.iloc[train_end:val_end].copy()
        self.test_df = self.merged_raw.iloc[val_end:].copy()
        print(f"📊 [Split] Dataset split into Train ({len(self.train_df)}), Val ({len(self.val_df)}), Test ({len(self.test_df)}).\n")
        return self.train_df, self.val_df, self.test_df

    # ---------------------------
    # Fit scalers and PCA on TRAIN, transform all splits
    # ---------------------------
    def fit_transform_train(self,
                        target_ohlc_den_cols: Optional[List[str]] = None,
                        market_ret_cols: Optional[List[str]] = None,
                        daily_macro_cols: Optional[List[str]] = None,
                        monthly_macro_cols: Optional[List[str]] = None,
                        quarterly_macro_cols: Optional[List[str]] = None):
        if self.train_df is None:
            self.split_chronologically()

        df_train = self.train_df
        df_all = self.merged_raw

        if target_ohlc_den_cols is None:
            target_ohlc_den_cols = [c for c in df_all.columns if c.endswith('_den')]
        if market_ret_cols is None:
            market_ret_cols = [c for c in df_all.columns if c.endswith('_ret')]
        if daily_macro_cols is None:
            daily_macro_cols = [c for c in df_all.columns if c in ['vix_daily','yield_curve_daily']]
        if monthly_macro_cols is None:
            monthly_macro_cols = [c for c in df_all.columns if c in ['cpi_mom','unemployment_detrend','interest_rate_diff','trade_balance_seasdiff']]
        if quarterly_macro_cols is None:
            quarterly_macro_cols = [c for c in df_all.columns if c in ['gdp_yoy','gov_debt','gov_fiscal_balance_to_gdp','tax_receipts','gov_spending']]

        # ---- Fit scalers on train (use to_numpy() to guarantee 2D) ----
        # target scaler + PCA
        print("🔄 [Scaling/PCA] Fitting target scaler and PCA...")
        if target_ohlc_den_cols:
            X_target_train = df_train[target_ohlc_den_cols].fillna(method='ffill').fillna(0).to_numpy()
            X_target_train = _ensure_2d(X_target_train)
            target_scaler = StandardScaler().fit(X_target_train)
            self.scalers['target_scaler'] = target_scaler
            target_scaled_train = target_scaler.transform(X_target_train)
            pca_target = PCA(n_components=self.pca_variance, svd_solver='full').fit(target_scaled_train)
            self.pcas['target_pca'] = pca_target
        else:
            pca_target = None
        print("✅ Done.")

        # market scaler + PCA
        print("🔄 [Scaling/PCA] Fitting market scaler and PCA...")
        if market_ret_cols:
            X_market_train = df_train[market_ret_cols].fillna(method='ffill').fillna(0).to_numpy()
            X_market_train = _ensure_2d(X_market_train)
            market_scaler = StandardScaler().fit(X_market_train)
            self.scalers['market_scaler'] = market_scaler
            market_scaled_train = market_scaler.transform(X_market_train)
            pca_market = PCA(n_components=self.pca_variance, svd_solver='full').fit(market_scaled_train)
            self.pcas['market_pca'] = pca_market
        else:
            pca_market = None
        print("✅ Done.")

        # daily macro scaler
        print("🔄 [Scaling/PCA] Fitting daily macro scaler...")
        if daily_macro_cols:
            X_daily_train = df_train[daily_macro_cols].fillna(method='ffill').fillna(0).to_numpy()
            X_daily_train = _ensure_2d(X_daily_train)
            daily_macro_scaler = StandardScaler().fit(X_daily_train)
            self.scalers['daily_macro_scaler'] = daily_macro_scaler
        print("✅ Done.")

        # monthly macro scaler + PCA
        print("🔄 [Scaling/PCA] Fitting monthly macro scaler and PCA...")
        if monthly_macro_cols:
            X_monthly_train = df_train[monthly_macro_cols].fillna(method='ffill').fillna(0).to_numpy()
            X_monthly_train = _ensure_2d(X_monthly_train)
            monthly_scaler = StandardScaler().fit(X_monthly_train)
            self.scalers['monthly_macro_scaler'] = monthly_scaler
            monthly_scaled_train = monthly_scaler.transform(X_monthly_train)
            pca_monthly = PCA(n_components=self.pca_variance, svd_solver='full').fit(monthly_scaled_train)
            self.pcas['monthly_pca'] = pca_monthly
        else:
            pca_monthly = None
        print("✅ Done.")

        # quarterly macro scaler + PCA
        print("🔄 [Scaling/PCA] Fitting quarterly macro scaler and PCA...")
        if quarterly_macro_cols:
            X_quarterly_train = df_train[quarterly_macro_cols].fillna(method='ffill').fillna(0).to_numpy()
            X_quarterly_train = _ensure_2d(X_quarterly_train)
            quarterly_scaler = StandardScaler().fit(X_quarterly_train)
            self.scalers['quarterly_macro_scaler'] = quarterly_scaler
            quarterly_scaled_train = quarterly_scaler.transform(X_quarterly_train)
            pca_quarterly = PCA(n_components=self.pca_variance, svd_solver='full').fit(quarterly_scaled_train)
            self.pcas['quarterly_pca'] = pca_quarterly
        else:
            pca_quarterly = None
        print("✅ Done.")

        # store selected columns for later
        self._col_sets = dict(
            target_ohlc_den_cols=target_ohlc_den_cols,
            market_ret_cols=market_ret_cols,
            daily_macro_cols=daily_macro_cols,
            monthly_macro_cols=monthly_macro_cols,
            quarterly_macro_cols=quarterly_macro_cols
        )

        # Debug prints
        print(f"[fit_transform_train] train_df rows: {len(self.train_df)}")
        print(f"[fit_transform_train] target cols: {target_ohlc_den_cols} -> train shape {X_target_train.shape if target_ohlc_den_cols else None}")
        print(f"[fit_transform_train] market cols: {market_ret_cols} -> train shape {X_market_train.shape if market_ret_cols else None}")
        if 'volume_raw' in self.train_df.columns:
            print(f"[fit_transform_train] example volume_raw len: {len(self.train_df['volume_raw'].fillna(0))}")

        # ---- Transform train/val/test using fitted scalers + pcas
        self._transform_all_splits()

    def _transform_all_splits(self):
        print("🔄 [Transform] Applying fitted scalers and PCA to all splits...")
        def transform_df(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            # ----- target PCA -----
            tcols = self._col_sets['target_ohlc_den_cols']
            if tcols and 'target_scaler' in self.scalers and 'target_pca' in self.pcas:
                X_t = df[tcols].fillna(method='ffill').fillna(0).to_numpy()
                X_t = _ensure_2d(X_t)
                t_scaled = self.scalers['target_scaler'].transform(X_t)
                t_pca = self.pcas['target_pca'].transform(t_scaled)
                t_pca = _ensure_2d(t_pca)
                print(f"[target PCA] X_t {X_t.shape}, scaled {t_scaled.shape}, pca {t_pca.shape}, index {len(df.index)}")
                t_pca_cols = [f"target_pca_{i+1}" for i in range(t_pca.shape[1])]
                assert t_pca.shape[0] == len(df.index), f"target PCA row mismatch: {t_pca.shape[0]} vs {len(df.index)}"
                t_pca_df = pd.DataFrame(t_pca, index=df.index, columns=t_pca_cols)
                print(f"[target PCA] t_pca_df {t_pca_df.shape}")
                df = pd.concat([df, t_pca_df], axis=1)

            # ----- market PCA -----
            mcols = self._col_sets['market_ret_cols']
            if mcols and 'market_scaler' in self.scalers and 'market_pca' in self.pcas:
                X_m = df[mcols].fillna(method='ffill').fillna(0).to_numpy()
                X_m = _ensure_2d(X_m)
                m_scaled = self.scalers['market_scaler'].transform(X_m)
                m_pca = self.pcas['market_pca'].transform(m_scaled)
                m_pca = _ensure_2d(m_pca)
                m_pca_cols = [f"market_pca_{i+1}" for i in range(m_pca.shape[1])]
                m_pca_df = pd.DataFrame(m_pca, index=df.index, columns=m_pca_cols)

                # ✅ FIX: always concatenate using DataFrame, never assign .values
                df = pd.concat([df, m_pca_df], axis=1)

                # volume handling: use separate scaler if available
                if 'volume_raw' in df.columns:
                    # Always fill NaNs to avoid scaler errors
                    vol_vals = df['volume_raw'].fillna(0).to_numpy().reshape(-1, 1)

                    # Fit scaler only once, during training
                    if 'volume_scaler' not in self.scalers:
                        print("[fit_volume_scaler] fitting StandardScaler on volume_raw (train only)")
                        self.scalers['volume_scaler'] = StandardScaler().fit(vol_vals)

                    # Always transform using the fitted scaler
                    vol_scaled = self.scalers['volume_scaler'].transform(vol_vals)
                    print(
                        f"[volume debug] len(df)={len(df)}, "
                        f"vol_vals.shape={vol_vals.shape}, "
                        f"vol_scaled.shape={vol_scaled.shape}"
                    )
                    vol_scaled = vol_scaled[:len(df)]
                    df['volume_scaled'] = vol_scaled.reshape(-1)

                    print(f"[volume_scaled] df len={len(df)}, vol_scaled shape={vol_scaled.shape}")
                else:
                    print("[volume_scaled] no volume_raw column found — skipping")

            # ----- daily macro scaled -----
            dcols = self._col_sets['daily_macro_cols']
            if dcols and 'daily_macro_scaler' in self.scalers:
                X_d = df[dcols].fillna(method='ffill').fillna(0).to_numpy()
                X_d = _ensure_2d(X_d)
                d_scaled = self.scalers['daily_macro_scaler'].transform(X_d)
                dcols_names = [f"daily_{c}_scaled" for c in dcols]
                ddf = pd.DataFrame(d_scaled, index=df.index, columns=dcols_names)
                df = pd.concat([df, ddf], axis=1)

            # ----- monthly PCA -----
            mcols2 = self._col_sets['monthly_macro_cols']
            if mcols2 and 'monthly_macro_scaler' in self.scalers and 'monthly_pca' in self.pcas:
                X_mm = df[mcols2].fillna(method='ffill').fillna(0).to_numpy()
                X_mm = _ensure_2d(X_mm)
                mm_scaled = self.scalers['monthly_macro_scaler'].transform(X_mm)
                mm_pca = self.pcas['monthly_pca'].transform(mm_scaled)
                mm_pca = _ensure_2d(mm_pca)
                mm_cols = [f"monthly_pca_{i+1}" for i in range(mm_pca.shape[1])]
                mm_df = pd.DataFrame(mm_pca, index=df.index, columns=mm_cols)
                df = pd.concat([df, mm_df], axis=1)

            # ----- quarterly PCA -----
            qcols = self._col_sets['quarterly_macro_cols']
            if qcols and 'quarterly_macro_scaler' in self.scalers and 'quarterly_pca' in self.pcas:
                X_qq = df[qcols].fillna(method='ffill').fillna(0).to_numpy()
                X_qq = _ensure_2d(X_qq)
                qq_scaled = self.scalers['quarterly_macro_scaler'].transform(X_qq)
                qq_pca = self.pcas['quarterly_pca'].transform(qq_scaled)
                qq_pca = _ensure_2d(qq_pca)
                qq_cols = [f"quarterly_pca_{i+1}" for i in range(qq_pca.shape[1])]
                qq_df = pd.DataFrame(qq_pca, index=df.index, columns=qq_cols)
                df = pd.concat([df, qq_df], axis=1)

            print(f"[transform_df end] final df shape: {df.shape}")

            print("✅ [Transform] All datasets transformed successfully.\n")
            return df

        # transform each split and keep the results
        self.train_transformed = transform_df(self.train_df)
        self.val_transformed = transform_df(self.val_df)
        self.test_transformed = transform_df(self.test_df)

        # Debug: shapes after transform
        print(f"[ _transform_all_splits ] train_transformed rows: {len(self.train_transformed)} cols: {self.train_transformed.shape[1]}")
        print(f"[ _transform_all_splits ] val_transformed rows:   {len(self.val_transformed)} cols: {self.val_transformed.shape[1]}")
        print(f"[ _transform_all_splits ] test_transformed rows:  {len(self.test_transformed)} cols: {self.test_transformed.shape[1]}")

    # ---------------------------
    # Datasets / Dataloaders
    # ---------------------------
    def get_feature_columns(self) -> Tuple[List[str], List[str]]:
        """Return target columns (x) and conditioning columns (c) to be used by TimeGradDataset.
           Prioritize PCA-generated columns. """
        df = self.train_transformed if hasattr(self, 'train_transformed') else self.merged_raw
        # target PCA columns
        target_cols = [c for c in df.columns if c.startswith('target_pca_')]
        # conditioning: all other numeric columns except x_future placeholders
        # Also include scaled volume if it exists
        cond_cols = [c for c in df.columns if c not in target_cols and (df[c].dtype in [np.float32,np.float64,np.int64,np.int32])]
        # also drop raw den/ret columns to avoid duplication (we prefer PCA/scaled versions)
        cond_cols = [c for c in cond_cols if not (c.endswith('_den') or c.endswith('_ret') or c.endswith('_raw'))] # Added _raw to drop raw volume
        # Exclude volume_scaled if it exists but was not successfully created due to mismatch
        if 'volume_scaled' in cond_cols and 'volume_scaled' not in df.columns:
             cond_cols.remove('volume_scaled')

        return target_cols, cond_cols

    def build_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        if not hasattr(self, 'train_transformed'):
            raise RuntimeError("Call fit_transform_train(...) before build_datasets()")
        target_cols, cond_cols = self.get_feature_columns()
        self.train_set = TimeGradDataset(self.train_transformed, target_cols, cond_cols,
                                         seq_len=self.seq_len, forecast_horizon=self.forecast_horizon, device=self.device)
        self.val_set = TimeGradDataset(self.val_transformed, target_cols, cond_cols,
                                       seq_len=self.seq_len, forecast_horizon=self.forecast_horizon, device=self.device)
        self.test_set = TimeGradDataset(self.test_transformed, target_cols, cond_cols,
                                        seq_len=self.seq_len, forecast_horizon=self.forecast_horizon, device=self.device)
        return self.train_set, self.val_set, self.test_set

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

# =========================================================
# 5. TimeGradDataset
# =========================================================
class TimeGradDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_cols: List[str], cond_cols: List[str],
                 seq_len: int = DEFAULT_SEQ_LEN, forecast_horizon: int = DEFAULT_HORIZON, device: str = "cpu"):
        self.df = df.dropna(subset=target_cols + cond_cols)
        self.target_cols = target_cols
        self.cond_cols = cond_cols
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.device = device

        self.x = self.df[target_cols].values.astype(np.float32)
        self.c = self.df[cond_cols].values.astype(np.float32)
        self.time_index = self.df.index
        self.n_samples = len(self.df) - seq_len - forecast_horizon + 1
        if self.n_samples < 1:
            raise ValueError("Not enough data to build any sequence with given seq_len and forecast_horizon.")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_hist = self.x[idx: idx + self.seq_len]
        c_hist = self.c[idx: idx + self.seq_len]
        x_future = self.x[idx + self.seq_len: idx + self.seq_len + self.forecast_horizon]
        return {
            "x_hist": torch.tensor(x_hist, device=self.device),
            "c_hist": torch.tensor(c_hist, device=self.device),
            "x_future": torch.tensor(x_future, device=self.device),
            "time": self.time_index[idx: idx + self.seq_len + self.forecast_horizon]
        }