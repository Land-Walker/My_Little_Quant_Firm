import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

from typing import Dict
import os
import warnings
warnings.filterwarnings('ignore')

from .. import config

class DataCollector: 
    def __init__(self, data_dir: str = config.RAW_DATA_DIR):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    # ------------------------------
    # Helper Methods
    # ------------------------------
    def save(self, df: pd.DataFrame, filename: str, method: str="parquet"):
        """Save DataFrame to parquet or csv with logging."""
        if method not in ["csv", "parquet"]:
          raise ValueError("method must be 'standard' or 'minmax'")
        path = os.path.join(self.data_dir, filename+"."+method)
        if method == "csv":
          df.to_csv(path, index=False)
          print(f"✅ Saved {filename} ({len(df)} rows) as csv")
        elif method == "parquet":
          df.to_parquet(path, index=False)
          print(f"✅ Saved {filename} ({len(df)} rows) as parquet")

    @staticmethod
    def rename_price_to_dfname(df: pd.DataFrame) -> pd.DataFrame:
        """Flatten MultiIndex columns and rename 'Price_' prefix to df.name."""
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(map(str, col)).strip() for col in df.columns]

        if getattr(df, "name", None):
            df.columns = [col.replace("Price_", f"{df.name}_") for col in df.columns]

        return df

    # ------------------------------
    # Yahoo Finance Data
    # ------------------------------
    @staticmethod
    def fetch_yf_data(ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance."""
        data = yf.download(ticker, start=start, end=end)
        data.columns = data.columns.get_level_values(0)
        if data.empty:
            raise ValueError(f"No data returned for ticker {ticker}")
        data.columns = data.columns.get_level_values(0)
        data = data.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }).reset_index()
        return data[['Date', 'open', 'high', 'low', 'close', 'volume']]

    # ------------------------------
    # FRED Macro Data
    # ------------------------------
    @staticmethod
    def fetch_macro_data(ids: Dict[str, str], start: str, end: str) -> pd.DataFrame:
        """Fetch macroeconomic data from FRED."""
        fred_df = web.DataReader(list(ids.keys()), "fred", start, end)
        fred_df = fred_df.rename(columns=ids).reset_index()
        fred_df = fred_df.rename(columns={"DATE": "Date"})
        return fred_df

    # ------------------------------
    # VIX Data
    # ------------------------------
    @staticmethod
    def fetch_vix(start: str, end: str) -> pd.DataFrame:
        """Fetch VIX daily close prices from Yahoo Finance."""
        vix_df = yf.download("^VIX", interval='1d', start=start, end=end)
        vix_df.columns = vix_df.columns.get_level_values(0)
        vix_df = vix_df.reset_index()
        vix_df = vix_df.rename(columns={'Datetime': 'Date', 'Close': 'vix'})
        return vix_df[['Date', 'vix']]

    # ------------------------------
    # Collect All Data
    # ------------------------------
    def collect_all_data(self, flattening: bool=True, save: bool=True):
        """Fetch and preprocess all market, target, and macro data."""
        # Market and target
        market_df = self.fetch_yf_data(config.MARKET_TICKER, config.START_DATE, config.END_DATE)
        target_df = self.fetch_yf_data(config.TARGET_TICKER, config.START_DATE, config.END_DATE)
        market_df.name = "market"
        target_df.name = "target"
        self.rename_price_to_dfname(market_df)
        self.rename_price_to_dfname(target_df)

        # Macro
        daily_macro_df = self.fetch_macro_data(config.FRED_DAILY_IDS, config.START_DATE, config.END_DATE)
        monthly_macro_df = self.fetch_macro_data(config.FRED_MONTHLY_IDS, config.START_DATE, config.END_DATE)
        quarterly_macro_df = self.fetch_macro_data(config.FRED_QUARTERLY_IDS, config.START_DATE, config.END_DATE)

        # Add VIX to daily macro
        vix_df = self.fetch_vix(config.START_DATE, config.END_DATE)
        # Merge VIX data into daily_macro_df
        daily_macro_df = pd.merge(daily_macro_df, vix_df, on='Date', how='left')


        # Name macro dataframes
        daily_macro_df.name = "daily_macro"
        monthly_macro_df.name = "monthly_macro"
        quarterly_macro_df.name = "quarterly_macro"
        macro_df = pd.DataFrame()
        macro_df.name = "macro"

        self.save(target_df, f"{target_df.name}")
        self.save(market_df, f"{market_df.name}")
        self.save(daily_macro_df, f"{daily_macro_df.name}")
        self.save(monthly_macro_df, f"{monthly_macro_df.name}")
        self.save(quarterly_macro_df, f"{quarterly_macro_df.name}")

        # Flatten df column names if required
        if flattening == True:
          self.rename_price_to_dfname(target_df)
          self.rename_price_to_dfname(market_df)

        return {
            "market": market_df,
            "target": target_df,
            "daily_macro": daily_macro_df,
            "monthly_macro": monthly_macro_df,
            "quarterly_macro": quarterly_macro_df,
            "macro": macro_df
        }