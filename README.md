# Python ML Options Trading Bot - Walkthrough

I have successfully built the **Python ML Options Trading Bot**. The system is modular and now uses a **Deep Learning LSTM** model (via PyTorch) to predict short-term price movements and generate options signals. The project now includes a **Real-Time Web Dashboard**.

## 📂 Project Structure
- `config.yaml`: Central configuration for symbols, risk parameters, and model settings.
- `main.py`: Entry point for Training, Backtesting, and Live Trading.
- `web-app/`: **[NEW] Next.js Frontend Dashboard**.
- `src/`
    - `server.py`: **[NEW] FastAPI Backend** for the dashboard.
    - `storage.py`: **[NEW] SQLite Storage** for trade signals.
    - `notifier.py`: Saves signals to DB and alerts Discord.
    - `data_loader.py`: Fetches 2-year historical data.
    - `model.py`: **LSTM Classifier (PyTorch)**.
    - `backtester.py`: Simulates trades and generates charts.

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (Important!)
Train the models for all symbols (SPY, QQQ, IWM) first.
```bash
python main.py --mode train
```

### 3. Run Web Dashboard (Backend + Frontend)
In one terminal, start the **API**:
```bash
uvicorn src.server:app --reload --host 0.0.0.0 --port 8000
```
In another terminal, start the **Frontend**:
```bash
cd web-app
npm run dev
```
👉 Open **http://localhost:3000** to view the interface.

### 4. Start Live Forecasting
Run the live prediction loop.
```bash
python main.py --mode live
```
*As signals are generated, they will instantly appear on the Web Dashboard!*

## 📊 Features & capabilities
- **Web Dashboard**: Modern, dark-mode Next.js UI showing live signals, confidence, and trade parameters.
- **Chart Overlays**: Visualizes exact Entry/Exit points on generated charts.
- **Dual Direction**: Trades **CALLs** (>60%) and **PUTs** (<40%).
- **Visual Analytics**: Automatically generates equity curves and P&L charts.

## ⚠️ Notes
- **Dependencies**: Ensure you have Node.js installed for the frontend.
