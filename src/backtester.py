import pandas as pd
import numpy as np
import yaml
import os
try:
    from .data_loader import DataLoader
    from .features import FeatureEngineer
    from .model import MLPredictor
    from .risk import RiskManager
except ImportError:
    from data_loader import DataLoader
    from features import FeatureEngineer
    from model import MLPredictor
    from risk import RiskManager

class Backtester:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.loader = DataLoader(config_path)
        self.fe = FeatureEngineer()
        self.ml = MLPredictor(config_path)
        self.risk = RiskManager(config_path)
        
        self.trade_log = []
        self.leverage_factor = 20  # Simulated Option Leverage (e.g. 1% move in stock = 20% move in option)
        self.theta_decay_per_hour = 0.05 # 5% decay per hour for 0DTE/1DTE simulation

    def run(self, symbol: str):
        print(f"Starting Backtest for {symbol}...")
        
        # 1. Prepare Data
        df = self.loader.fetch_data(symbol)
        df = self.fe.add_indicators(df)
        df = self.fe.create_target(df) # Useful for comparing with ground truth
        
        # Ensure models are loaded
        if not self.ml.load_model(symbol):
            print("Model not found, training new one...")
            self.ml.train(df, symbol)
            
        # 2. Iterate
        # Start after enough data for features
        start_idx = 100 
        
        active_position = None
        
        for i in range(start_idx, len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i+1] # Look ahead for PnL calculation
            
            # Reset daily limits if new day
            self.risk.reset_daily_limits()
            
            # --- Management of Active Position ---
            if active_position:
                # Update Position
                # Estimate Option Price Movement
                # High leverage for 0DTE cheap options ($0.20 range).
                # Leverage = (Delta * StockPrice) / OptionPrice
                # (0.10 * 500) / 0.20 = 250x.
                # Let's use 200x to be realistic.
                leverage = 200 
                
                underlying_change_pct = (next_row["close"] - active_position["entry_underlying_price"]) / active_position["entry_underlying_price"]
                
                if active_position["type"] == "PUT":
                    underlying_change_pct = -underlying_change_pct
                
                # Apply Leverage (capped at -100% loss)
                opt_change_pct = underlying_change_pct * leverage
                
                # Apply Theta (Time decay)
                # 0DTE decay is rapid. -10% per hour?
                opt_change_pct -= 0.10 

                current_opt_price = active_position["entry_opt_price"] * (1 + opt_change_pct)
                current_opt_price = max(0, current_opt_price) # Can't go below 0

                # Exit Logic
                pnl_pct = (current_opt_price - active_position["entry_opt_price"]) / active_position["entry_opt_price"]
                realized_pnl = active_position["risk_amount"] * pnl_pct

                # Target Price Check ($1.00 - $2.00)
                # The user said "take profit from $1 to 2$". 
                # If current price hits this range, we take profit.
                # We can simulate a "Limit Order" exit if price crossed into this zone or above.
                
                exit_reason = None
                
                if current_opt_price >= 1.0: # Minimum Target Hit
                    exit_reason = "TAKE_PROFIT_Target_$1"
                    # Assume we filled at average of current close and 1.0 if it gapped? 
                    # Or just take current price.
                    # Cap realized PnL if it blew past $2?
                elif pnl_pct <= -active_position["sl_pct"]:
                    exit_reason = "STOP_LOSS"
                    # Cap loss at -100%
                    if pnl_pct < -1.0: realized_pnl = -active_position["risk_amount"]
                
                # Time Exit (End of Day)
                elif row["timestamp"].hour >= 15 and row["timestamp"].minute >= 45:
                     exit_reason = "EOD_EXIT"
                
                if exit_reason:
                    self.risk.update_capital(realized_pnl)
                    self.trade_log.append({
                        "entry_time": str(active_position["entry_time"]),
                        "exit_time": str(next_row["timestamp"]),
                        "symbol": symbol,
                        "type": active_position["type"],
                        "entry_price": active_position["entry_underlying_price"],
                        "exit_price": next_row["close"],
                        "entry_opt": f"${active_position['entry_opt_price']:.2f}",
                        "exit_opt": f"${current_opt_price:.2f}",
                        "pnl": realized_pnl,
                        "exit_reason": exit_reason,
                        "prob": active_position["prob"]
                    })
                    active_position = None
                    continue
                else:
                    # Update 'active_position' for next loop? No, it's just state.
                    # But we are recalculating from ENTRY every time. 
                    # This is correct for "Underlying Change from Entry".
                    pass # Hold
                
                continue

            # --- Entry Logic ---
            # For LSTM: We need the window [i - seq_len + 1 : i + 1]
            seq_len = self.ml.seq_length
            if i < seq_len:
                continue

            # Pass the historical window ending at current row `i`
            # iloc slice is exclusive on upper bound, so [start : i+1]
            current_slice = df.iloc[i - seq_len + 1 : i + 1]
            
            prob = self.ml.predict(current_slice)[0]
            
            # Risk Check
            # Check for CALL (Prob > Threshold)
            # Check for PUT  (Prob < 1 - Threshold)
            confidence_threshold = self.config["model"]["confidence_threshold"]
            
            action = "NO_TRADE"
            if prob > confidence_threshold:
                action = "BUY_CALL"
            elif prob < (1 - confidence_threshold):
                action = "BUY_PUT"
            
            if action != "NO_TRADE" and self.risk.check_trade_allowed(prob if action=="BUY_CALL" else 1-prob):
                # Enter Position
                
                # SIMULATION: Lotto Strategy "Buy Cheap Option"
                # User wants to buy options priced $0.10 - $0.30
                # We simulate finding a contract in this range.
                import random
                min_price = self.config["risk"].get("option_min_price", 0.1)
                max_price = self.config["risk"].get("option_max_price", 0.3)
                
                # Assume we get filled at a random price in this range (or Average)
                entry_opt_price = random.uniform(min_price, max_price)
                
                sl_pct = self.config["risk"]["stop_loss_pct"]
                risk_amt = self.risk.current_capital * self.config["risk"]["risk_per_trade_pct"]
                
                # Position Sizing
                # Contracts = Risk Amount / (Entry Price * 100) ? 
                # OR does "use 20% of account" mean Cost Basis = 20%?
                # Usually "Use 20% of account" means Total Exposure.
                # Cost = Contracts * Price * 100
                total_cost = risk_amt # Allocating this much cash
                contracts = int(total_cost / (entry_opt_price * 100))
                
                if contracts < 1: 
                    continue

                active_position = {
                    "entry_time": row["timestamp"],
                    "entry_underlying_price": row["close"],
                    "entry_opt_price": entry_opt_price,
                    "contracts": contracts,
                    "risk_amount": total_cost, # Total capital commited
                    "sl_pct": sl_pct,
                    "tp_pct": self.config["risk"]["take_profit_pct_min"], # This is acting as Multiplier target (4.0 = 400%)
                    "prob": prob,
                    "type": "CALL" if action == "BUY_CALL" else "PUT"
                }
        
        # 3. Report
        self.generate_report(symbol, df)

    def generate_report(self, symbol: str, price_df: pd.DataFrame):
        print(f"--- Backtest Results: {symbol} ---")
        print(f"Final Capital: ${self.risk.current_capital:.2f}")
        print(f"Total Trades: {len(self.trade_log)}")
        
        if not self.trade_log:
            return

        df_log = pd.DataFrame(self.trade_log)
        wins = df_log[df_log["pnl"] > 0]
        losses = df_log[df_log["pnl"] <= 0]
        
        win_rate = len(wins) / len(df_log) if len(df_log) > 0 else 0
        total_pnl = df_log["pnl"].sum()
        
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Net PnL: ${total_pnl:.2f}")
        
        # Save results
        os.makedirs(self.config["paths"]["results_dir"], exist_ok=True)
        df_log.to_csv(os.path.join(self.config["paths"]["results_dir"], f"backtest_{symbol}.csv"), index=False)
        
        # Plots
        try:
            try:
                from .visualizer import Visualizer
            except ImportError:
                from visualizer import Visualizer
                
            viz = Visualizer()
            viz.plot_backtest_results(df_log, symbol)
            viz.plot_trades_on_chart(price_df, df_log, symbol)
        except Exception as e:
            print(f"Visualization failed: {e}")
            
        print("Detailed log and charts saved.")

if __name__ == "__main__":
    bt = Backtester()
    bt.run("SPY")
