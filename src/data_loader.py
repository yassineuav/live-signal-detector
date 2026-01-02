import yfinance as yf
import pandas as pd
import numpy as np
import os
import yaml
from datetime import datetime, timedelta

class DataLoader:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.symbols = self.config["market"]["symbols"]
        self.period = f"{self.config['data']['history_years']}y"
        
        # dynamic interval from loop_interval_minutes
        loop_min = self.config.get("loop_interval_minutes", 60)
        if loop_min == 60:
            self.interval = "1h"
        elif loop_min >= 1440:
            self.interval = "1d" 
        else:
            self.interval = f"{loop_min}m"
        
        # Create data directory if not exists
        os.makedirs(self.config["paths"]["data_dir"], exist_ok=True)

    def fetch_data(self, symbol: str, interval: str = None, period: str = None) -> pd.DataFrame:
        """
        Fetches historical OHLCV data using yfinance.
        """
        _interval = interval if interval else self.interval
        _period = period if period else self.period
        
        # Adjust period for limitations of yfinance
        if _interval in ['1m']:
            _period = '7d' 
        elif _interval in ['2m', '5m', '15m', '30m']:
            # yfinance max for <60m is 60 days
            if 'y' in _period: _period = '59d'
        
        print(f"Fetching data for {symbol} (Interval: {_interval}, Period: {_period})...")
        
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=_period, interval=_interval)
            
            if df.empty:
                print(f"Warning: No data found for {symbol}")
                return pd.DataFrame()
            
            # Standardization
            df.reset_index(inplace=True)
            
            # Identify the date column
            date_col = None
            for col in ['Date', 'Datetime', 'index']:
                if col in df.columns:
                    date_col = col
                    break
            
            if not date_col:
                print(f"Error: Could not find Date/Datetime column for {symbol}. Columns: {df.columns}")
                return pd.DataFrame()

            # Ensure it is datetime and UTC
            df[date_col] = pd.to_datetime(df[date_col], utc=True)

            # Rename columns
            rename_map = {
                date_col: "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            }
            df = df.rename(columns=rename_map)
            
            # Drop unnecessary columns like dividends/splits
            cols = ["timestamp", "open", "high", "low", "close", "volume"]
            df = df[[c for c in cols if c in df.columns]]
            
            # Save raw data
            save_path = os.path.join(self.config["paths"]["data_dir"], f"{symbol}_{_interval}.csv")
            df.to_csv(save_path, index=False)
            print(f"Saved {symbol} data to {save_path} ({len(df)} rows)")
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def get_options_chain(self, symbol: str):
        """
        Placeholder for fetching real options chain.
        If config 'use_mock_options' is True, returns None or simulated structure.
        """
        if self.config["data"]["use_mock_options"]:
            print(f"Mocking options data for {symbol}...")
            return None
        else:
            # Implement Real API (Polygon/Tradier) here
            pass

if __name__ == "__main__":
    loader = DataLoader()
    for sym in loader.symbols:
        loader.fetch_data(sym)
