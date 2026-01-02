import sqlite3
import json
import os
from datetime import datetime

class Storage:
    def __init__(self, db_path="signals.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                action TEXT,
                price REAL,
                probability REAL,
                confidence REAL,
                sl_pct REAL,
                tp_pct REAL,
                metadata TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def save_signal(self, signal: dict):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Serialize extra fields to metadata if needed
        metadata = json.dumps({k:v for k,v in signal.items() if k not in 
            ['timestamp', 'symbol', 'action', 'price', 'probability', 'confidence', 'sl_pct', 'tp_pct']})
        
        c.execute('''
            INSERT INTO signals (timestamp, symbol, action, price, probability, confidence, sl_pct, tp_pct, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(signal.get('timestamp')),
            signal.get('symbol'),
            signal.get('action'),
            signal.get('price'),
            signal.get('probability'),
            signal.get('confidence'),
            signal.get('sl_pct'),
            signal.get('tp_pct'),
            metadata
        ))
        conn.commit()
        conn.close()
        print(f"Signal saved to DB: {signal.get('symbol')}")

    def get_recent_signals(self, limit=50):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM signals ORDER BY id DESC LIMIT ?', (limit,))
        rows = c.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            d = dict(row)
            # Parse metadata back if needed, but for now flat return is fine
            results.append(d)
        return results

if __name__ == "__main__":
    db = Storage()
    db.save_signal({"symbol": "TEST", "action": "BUY_TEST", "price": 100})
    print(db.get_recent_signals(1))
