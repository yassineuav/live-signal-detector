import argparse
import time
import schedule
from src.strategy import StrategyEngine
from src.backtester import Backtester
from src.notifier import Notifier
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.model import MLPredictor
import yaml

def run_live(config_path="config.yaml"):
    # Initialize components
    strat = StrategyEngine(config_path)
    notifier = Notifier()

    while True:
        try:
            # 1. Read dynamic interval from config
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
            interval = cfg.get("loop_interval_minutes", 1)

            print(f"--- Running Analysis (Interval: {interval}m) ---")

            for symbol in strat.available_models:
                print(f"Analyzing {symbol}...")
                signal = strat.generate_signal(symbol)
                
                if signal["action"] != "NO_TRADE":
                    notifier.send_alert(signal)
                else:
                    print(f"No trade for {symbol}. Prob: {signal.get('probability', 0):.2f}")

            # Sleep seconds
            time.sleep(interval * 60)

        except KeyboardInterrupt:
            print("Stopping live loop...")
            break
        except Exception as e:
            print(f"Error in live loop: {e}")
            time.sleep(60)

def train_models(config_path="config.yaml"):
    loader = DataLoader(config_path)
    fe = FeatureEngineer()
    ml = MLPredictor(config_path)
    
    for symbol in loader.symbols:
        df = loader.fetch_data(symbol)
        df = fe.add_indicators(df)
        df = fe.create_target(df)
        ml.train(df, symbol)

def backtest(config_path="config.yaml"):
    bt = Backtester(config_path)
    for symbol in bt.loader.symbols:
        bt.run(symbol)

def main():
    parser = argparse.ArgumentParser(description="Python ML Options Trading Bot")
    parser.add_argument("--mode", type=str, choices=["train", "backtest", "live"], required=True, help="Mode to run")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_models(args.config)
    elif args.mode == "backtest":
        backtest(args.config)
    elif args.mode == "live":
        run_live(args.config)

if __name__ == "__main__":
    main()
