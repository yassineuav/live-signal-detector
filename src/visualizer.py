import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml

class Visualizer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = self.config["paths"]["results_dir"]
        os.makedirs(self.results_dir, exist_ok=True)

    def plot_backtest_results(self, trade_log_df: pd.DataFrame, symbol: str):
        """
        Generates and saves:
        1. Equity Curve
        2. Trade P&L Distribution
        """
        if trade_log_df.empty:
            print(f"No trades to plot for {symbol}.")
            return

        # Prepare Data
        df = trade_log_df.copy()
        df['cumulative_pnl'] = df['pnl'].cumsum()
        starting_capital = self.config["risk"]["starting_capital"]
        df['equity'] = starting_capital + df['cumulative_pnl']
        df['trade_num'] = range(1, len(df) + 1)

        # 1. Equity Curve
        plt.figure(figsize=(12, 6))
        plt.plot(df['trade_num'], df['equity'], marker='o', linestyle='-', color='b')
        plt.title(f"Equity Curve - {symbol} (Starting: ${starting_capital})")
        plt.xlabel("Trade Number")
        plt.ylabel("Account Balance ($)")
        plt.grid(True)
        plt.axhline(y=starting_capital, color='r', linestyle='--', alpha=0.5)
        
        save_path_equity = os.path.join(self.results_dir, f"{symbol}_equity_curve.png")
        plt.savefig(save_path_equity)
        print(f"Saved Equity Curve: {save_path_equity}")
        plt.close()

        # 2. P&L Bar Chart
        plt.figure(figsize=(12, 6))
        colors = ['g' if p > 0 else 'r' for p in df['pnl']]
        plt.bar(df['trade_num'], df['pnl'], color=colors)
        plt.title(f"Trade P&L - {symbol}")
        plt.xlabel("Trade Number")
        plt.ylabel("Profit/Loss ($)")
        plt.grid(True, axis='y')
        plt.axhline(0, color='black', linewidth=0.8)
        
        save_path_pnl = os.path.join(self.results_dir, f"{symbol}_pnl_dist.png")
        plt.savefig(save_path_pnl)
        print(f"Saved P&L Chart: {save_path_pnl}")
        plt.close()

    def plot_trades_on_chart(self, price_df: pd.DataFrame, trade_log_df: pd.DataFrame, symbol: str):
        """
        Plots the price history and overlays entries/exits.
        """
        if price_df.empty: 
            return

        plt.figure(figsize=(14, 7))
        
        # Plot Price
        # Ensure correct type
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        plt.plot(price_df['timestamp'], price_df['close'], label='Close Price', color='gray', linewidth=1, alpha=0.7)
        
        # Plot Entries & Exits
        if not trade_log_df.empty:
            # CALL Entries
            calls = trade_log_df[trade_log_df['type'] == 'CALL']
            if not calls.empty:
                plt.scatter(pd.to_datetime(calls['entry_time']), calls['entry_price'], 
                           marker='^', color='green', s=100, label='Buy Call', zorder=5)
            
            # PUT Entries
            puts = trade_log_df[trade_log_df['type'] == 'PUT']
            if not puts.empty:
                plt.scatter(pd.to_datetime(puts['entry_time']), puts['entry_price'], 
                           marker='v', color='red', s=100, label='Buy Put', zorder=5)
            
            # Exits (Winners vs Losers)
            winners = trade_log_df[trade_log_df['pnl'] > 0]
            losers = trade_log_df[trade_log_df['pnl'] <= 0]
            
            if not winners.empty:
                plt.scatter(pd.to_datetime(winners['exit_time']), winners['exit_price'], 
                           marker='o', color='gold', s=80, label='Exit (Win)', zorder=4)
            if not losers.empty:
                plt.scatter(pd.to_datetime(losers['exit_time']), losers['exit_price'], 
                           marker='x', color='black', s=80, label='Exit (Loss)', zorder=4)

        plt.title(f"Trading Signals - {symbol}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        save_path = os.path.join(self.results_dir, f"{symbol}_chart.png")
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved Signal Chart: {save_path}")
        plt.close()
