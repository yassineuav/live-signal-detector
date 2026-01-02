from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .storage import Storage
import uvicorn
import os
import yaml
import subprocess
import signal

app = FastAPI(title="Trading Bot API")

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Results for images
os.makedirs("results", exist_ok=True)
app.mount("/results_static", StaticFiles(directory="results"), name="results_static")

db = Storage()
CONFIG_PATH = "config.yaml"

# --- Data Models ---
class RiskSettings(BaseModel):
    starting_capital: float
    stop_loss_pct: float
    risk_per_trade_pct: float
    take_profit_pct_min: float
    take_profit_pct_max: float

class AppSettings(BaseModel):
    loop_interval_minutes: int
    training_interval: str # 5, 15, 30, 60, 1h etc (string for yfinance valid intervals if needed, or int)

class FullConfig(BaseModel):
    risk: RiskSettings
    loop_interval_minutes: int
    # Add other fields as needed, but we focus on these for UI

# --- Helper ---
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def save_config(new_config: dict):
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(new_config, f, sort_keys=False)

# --- Routes ---

@app.get("/")
def read_root():
    return {"status": "ok", "service": "Trading Bot API"}

@app.get("/signals")
def get_signals(limit: int = 50):
    return {"signals": db.get_recent_signals(limit)}

@app.get("/settings")
def get_settings():
    cfg = load_config()
    return {
        "risk": cfg.get("risk", {}),
        "loop_interval_minutes": cfg.get("loop_interval_minutes", 1),
        "model_config": cfg.get("model", {})
    }

@app.post("/settings")
def update_settings(payload: dict):
    """
    Accepts partial updates to config.
    Expects nested structure matching config.yaml
    """
    cfg = load_config()
    
    # Update Risk
    if "risk" in payload:
        cfg["risk"].update(payload["risk"])
        
    # Update Loop and Model settings
    if "loop_interval_minutes" in payload:
        cfg["loop_interval_minutes"] = payload["loop_interval_minutes"]
        
    if "model" in payload:
        cfg["model"].update(payload["model"])

    save_config(cfg)
    return {"status": "updated", "config": cfg}

import pandas as pd

@app.post("/control/{action}")
def control_process(action: str):
    """
    Action: 'train', 'backtest'
    """
    cmd = []
    if action == "train":
        cmd = ["python", "main.py", "--mode", "train"]
    elif action == "backtest":
        cmd = ["python", "main.py", "--mode", "backtest"]
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    try:
        # Run in separate process, don't wait
        subprocess.Popen(cmd, cwd=os.getcwd(), shell=True)
        return {"status": "started", "action": action}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{symbol}")
def get_history(symbol: str, limit: int = 100):
    """
    Returns historical OHLC data for the symbol.
    """
    try:
        # Try finding the file in data/ or root based on previous context
        file_path = f"data/{symbol}_1h.csv" # Assuming 1h timeframe is what we have
        if not os.path.exists(file_path):
             # Fallback to try finding it via data loader logic or just return empty
             return {"error": "File not found", "data": []}
        
        df = pd.read_csv(file_path)
        
        # Rename if needed to standardized lower case locally if stored differently
        # But our CSVs seem to be lower case cols: timestamp, open, high, low, close, volume (based on recent edits)
        
        # Sort and take last 'limit'
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').tail(limit)
        
        data = df.to_dict(orient="records")
        return {"symbol": symbol, "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chart-gallery")
def get_chart_gallery():
    """
    Returns list of generated images in results/.
    """
    files = os.listdir("results")
    images = [f for f in files if f.endswith(".png")]
    return {"images": images}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
