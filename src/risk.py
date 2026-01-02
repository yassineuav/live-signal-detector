import yaml
from datetime import datetime

class RiskManager:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.risk_config = self.config["risk"]
        
        # State tracking
        self.current_capital = self.risk_config["starting_capital"]
        self.daily_trades_count = 0
        self.last_trade_date = None
        self.equity_curve = [self.current_capital]

    def reset_daily_limits(self):
        """
        Resets daily trade counters if the date has changed.
        """
        today = datetime.now().date()
        if self.last_trade_date != today:
            self.daily_trades_count = 0
            self.last_trade_date = today

    def check_trade_allowed(self, probability: float) -> bool:
        """
        Checks if a trade is allowed based on risk rules.
        """
        self.reset_daily_limits()
        
        # 1. Max trades limit
        if self.daily_trades_count >= self.risk_config["max_trades_per_day"]:
            return False
        
        # 2. Probability Threshold
        if probability < self.config["model"]["confidence_threshold"]:
            return False
            
        # 3. Drawdown Limit (Simplified check)
        # If current capital is < (1 - max_drawdown) * starting, stop.
        drawdown_limit = self.risk_config["starting_capital"] * (1 - self.risk_config["max_drawdown_limit_pct"])
        if self.current_capital < drawdown_limit:
            return False
            
        return True

    def calculate_position_size(self, stop_loss_amount_per_contract: float) -> int:
        """
        Calculates position size (number of contracts).
        
        Strategy: Risk a fixed % of account per trade.
        Risk Amount = Account * Risk% (e.g., $1000 * 0.10 = $100)
        Contracts = Risk Amount / Stop Loss $ Amount per contract
        """
        risk_amount = self.current_capital * self.risk_config["risk_per_trade_pct"]
        
        if stop_loss_amount_per_contract <= 0:
            return 0
            
        contracts = int(risk_amount // stop_loss_amount_per_contract)
        return max(1, contracts) # Ensure at least 1 if risk allows (or return 0 if strictly strict)

    def update_capital(self, pnl: float):
        """
        Updates capital after a trade close.
        """
        self.current_capital += pnl
        self.equity_curve.append(self.current_capital)
        self.daily_trades_count += 1

if __name__ == "__main__":
    rm = RiskManager()
    print(f"Starting Capital: {rm.current_capital}")
    allowed = rm.check_trade_allowed(0.70)
    print(f"Trade Allowed (0.70 prob): {allowed}")
    
    # Simulate a trade
    pos_size = rm.calculate_position_size(stop_loss_amount_per_contract=20) # Risking $20 per contract loss
    print(f"Contracts to buy (Risking 10% of 1000 = 100, SL=20): {pos_size}")
