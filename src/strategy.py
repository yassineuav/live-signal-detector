from datetime import datetime
import pandas as pd
from typing import Dict, Any

try:
    from .data_loader import DataLoader
    from .features import FeatureEngineer
    from .model import MLPredictor
    from .risk import RiskManager
except ImportError:
    # Allow running as script
    from data_loader import DataLoader
    from features import FeatureEngineer
    from model import MLPredictor
    from risk import RiskManager

class StrategyEngine:
    def __init__(self, config_path="config.yaml"):
        self.loader = DataLoader(config_path)
        self.fe = FeatureEngineer()
        self.ml = MLPredictor(config_path)
        self.risk = RiskManager(config_path)
        
        # Load models on init (assumes they are trained)
        self.available_models = []
        for sym in self.loader.symbols:
            if self.ml.load_model(sym):
                self.available_models.append(sym)
            else:
                print(f"Warning: No model for {sym}, signal generation will be skipped.")

    def generate_signal(self, symbol: str) -> Dict[str, Any]:
        """
        Full pipeline: Fetch Data -> Feature Eng -> Predict -> Risk Check -> Signal.
        """
        if symbol not in self.available_models:
             return {"action": "NO_TRADE", "reason": "Model Not Loaded", "symbol": symbol}
        
        # 1. Fetch live/latest data
        # Note: In a real live loop, we'd fetch just the latest bars efficiently.
        # Here we fetch a chunk to ensure we have enough for indicators.
        df = self.loader.fetch_data(symbol)
        
        if df.empty:
            return {"action": "NO_TRADE", "reason": "No Data"}

        # 2. Add features
        df = self.fe.add_indicators(df)
        
        # 3. Predict
        # LSTM needs the last `sequence_length` rows.
        # We pass the full tail; predict() handles taking the last N.
        min_len = self.ml.seq_length
        if len(df) < min_len:
             return {"action": "NO_TRADE", "reason": "Insufficient Data for LSTM", "symbol": symbol}

        # Pass last `min_len` rows (or more, predict handles it)
        latest_sequence = df.iloc[-min_len:]
        prob = self.ml.predict(latest_sequence)[0]
        
        timestamp = latest_sequence.iloc[-1]["timestamp"]
        close_price = latest_sequence.iloc[-1]["close"]

        # 4. Filter by Risk/Confidence
        # Action Logic
        confidence_threshold = self.risk.config["model"]["confidence_threshold"]
        action = "NO_TRADE"
        
        if prob > confidence_threshold:
            action = "BUY_CALL"
            trade_prob = prob
        elif prob < (1 - confidence_threshold):
            action = "BUY_PUT"
            trade_prob = 1 - prob # Confidence in Down
        else:
            trade_prob = prob # Neutral

        is_allowed = action != "NO_TRADE" and self.risk.check_trade_allowed(trade_prob)
        
        if is_allowed:
            # Simple TP/SL Calculation
            sl_pct = self.risk.risk_config["stop_loss_pct"]
            tp_pct = self.risk.risk_config["take_profit_pct_min"]
            
            return {
                "timestamp": str(timestamp),
                "symbol": symbol,
                "action": action,
                "price": close_price,
                "probability": float(prob),
                "confidence": float(trade_prob),
                "sl_pct": sl_pct,
                "tp_pct": tp_pct
            }
        else:
            return {
                "timestamp": str(timestamp),
                "symbol": symbol,
                "action": "NO_TRADE",
                "probability": float(prob),
                "reason": "Low Confidence or Risk Filter"
            }

if __name__ == "__main__":
    strategy = StrategyEngine()
    sig = strategy.generate_signal("SPY")
    print("Signal Generated:", sig)
