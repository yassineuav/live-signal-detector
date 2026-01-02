import requests
import json
import os
from .storage import Storage

class Notifier:
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
        self.storage = Storage()

    def send_alert(self, signal: dict):
        """
        Sends a trading signal to Discord AND saves to DB.
        """
        # 1. Save to DB
        try:
            self.storage.save_signal(signal)
        except Exception as e:
            print(f"Failed to save signal to DB: {e}")

        # 2. Discord Alert
        if not self.webhook_url:
            print("Alert (No Webhook):", signal)
            return

        embed = {
            "title": f"🚀 Trade Signal: {signal.get('action')} {signal.get('symbol')}",
            "color": 5763719 if signal.get('action') == "BUY_CALL" else 15548997,
            "fields": [
                {"name": "Price", "value": f"${signal.get('price', 0):.2f}", "inline": True},
                {"name": "Probability", "value": f"{signal.get('probability', 0)*100:.1f}%", "inline": True},
                {"name": "Confidence", "value": f"{signal.get('confidence', 0)*100:.1f}%", "inline": True},
                {"name": "Stop Loss", "value": f"{signal.get('sl_pct', 0)*100:.1f}%", "inline": True},
                {"name": "Take Profit", "value": f"{signal.get('tp_pct', 0)*100:.1f}%", "inline": True},
                {"name": "Time", "value": str(signal.get('timestamp')), "inline": False}
            ]
        }
        
        payload = {
            "embeds": [embed]
        }

        try:
            requests.post(self.webhook_url, json=payload)
            print(f"Alert sent for {signal.get('symbol')}")
        except Exception as e:
            print(f"Failed to send alert: {e}")

if __name__ == "__main__":
    notifier = Notifier()
    notifier.send_alert({
        "action": "BUY_CALL",
        "symbol": "SPY",
        "price": 450.50,
        "probability": 0.85, 
        "confidence": 0.85,
        "sl_pct": 0.1,
        "tp_pct": 1.0,
        "timestamp": "2024-01-01 10:00:00"
    })
