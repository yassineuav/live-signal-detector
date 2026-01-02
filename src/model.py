import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score
import joblib
import os
import yaml

# Define the PyTorch LSTM Module
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # Out shape: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)

class MLPredictor:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.models_dir = os.path.abspath(self.config["paths"]["models_dir"])
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.features = self.config["model"]["features"]
        self.target_col = "Target"
        self.seq_length = self.config["model"]["sequence_length"]
        self.hidden_size = self.config["model"]["lstm_units"]
        self.epochs = self.config["model"]["epochs"]
        self.batch_size = self.config["model"]["batch_size"]
        
        # We need to persist the scaler
        self.scaler = StandardScaler()
        self.model = None

    def create_sequences(self, X, y=None):
        """
        Converts 2D array (samples, features) into 3D keys (samples, seq_len, features)
        """
        Xs, ys = [], []
        if len(X) < self.seq_length:
            return np.array([]), np.array([])
            
        for i in range(len(X) - self.seq_length + 1):
            Xs.append(X[i:(i + self.seq_length)])
            if y is not None:
                ys.append(y[i + self.seq_length - 1]) # Target is aligned with the last step of sequence
                
        return np.array(Xs), np.array(ys)

    def train(self, df: pd.DataFrame, symbol: str):
        print(f"Training LSTM for {symbol}...")
        
        # 1. Prepare Data
        data = df.copy().dropna(subset=self.features + [self.target_col])
        X_raw = data[self.features].values
        y_raw = data[self.target_col].values
        
        # Normalize
        X_scaled = self.scaler.fit_transform(X_raw)
        
        # Create Sequences
        X, y = self.create_sequences(X_scaled, y_raw)
        
        if len(X) == 0:
            print("Not enough data to create sequences.")
            return

        # Convert to Tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        # TimeSeriesSplit Validation
        tscv = TimeSeriesSplit(n_splits=3)
        input_size = X.shape[2]
        
        for fold, (train_index, test_index) in enumerate(tscv.split(X)):
            X_train, X_test = X_tensor[train_index], X_tensor[test_index]
            y_train, y_test = y_tensor[train_index], y_tensor[test_index]
            
            # Init Model
            model = LSTMClassifier(input_size, self.hidden_size, num_layers=2)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Train Loop
            model.train()
            print(f"  Fold {fold+1} Training...")
            for epoch in range(5): # Short epochs for validation
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                
            # Evaluate
            model.eval()
            with torch.no_grad():
                preds = model(X_test)
                y_pred_cls = (preds > 0.5).float()
                acc = accuracy_score(y_test, y_pred_cls)
                print(f"  Fold {fold+1} Accuracy: {acc:.4f}")

        # Final Training
        print("Final Training on all data...")
        self.model = LSTMClassifier(input_size, self.hidden_size, num_layers=2)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        for epoch in range(self.epochs):
            total_loss = 0
            for xb, yb in loader:
                optimizer.zero_grad()
                outputs = self.model(xb)
                loss = criterion(outputs, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch+1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(loader):.4f}")
                
        self.save_model(symbol)

    def predict(self, df: pd.DataFrame):
        """
        Takes a DataFrame. Must contain at least 'seq_length' rows.
        Returns probability of UP for the last sequence.
        """
        if self.model is None or not hasattr(self.scaler, 'mean_'):
            # Fallback if uninitialized
            return np.array([0.5])

        X_raw = df[self.features].values
        
        # Handle normalization
        # Note: In prediction, we transform using the saved scaler
        try:
            X_scaled = self.scaler.transform(X_raw)
        except Exception as e:
            # If scaling fails (e.g. feature mismatch), return neutral
            print(f"Scaling error: {e}")
            return np.array([0.5])
            
        # We only need the last sequence
        if len(X_scaled) < self.seq_length:
            return np.array([0.5])
            
        last_seq = X_scaled[-self.seq_length:] # Shape (seq_len, features)
        input_seq = np.array([last_seq]) # Shape (1, seq_len, features)
        
        input_tensor = torch.tensor(input_seq, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            prob = self.model(input_tensor)
            return prob.numpy().flatten() # Returns array like [0.75]

    def save_model(self, symbol: str):
        models_path = os.path.abspath(self.models_dir)
        os.makedirs(models_path, exist_ok=True)
        
        path_model = os.path.join(models_path, f"{symbol}_lstm.pth")
        path_scaler = os.path.join(models_path, f"{symbol}_scaler.pkl")
        
        try:
            torch.save(self.model.state_dict(), path_model)
            joblib.dump(self.scaler, path_scaler)
            print(f"Successfully saved LSTM model to: {path_model}")
        except Exception as e:
            print(f"Failed to save models for {symbol}: {e}")

    def load_model(self, symbol: str):
        models_path = os.path.abspath(self.models_dir)
        path_model = os.path.join(models_path, f"{symbol}_lstm.pth")
        path_scaler = os.path.join(models_path, f"{symbol}_scaler.pkl")
        
        if os.path.exists(path_model) and os.path.exists(path_scaler):
            try:
                # Load Scaler
                self.scaler = joblib.load(path_scaler)
                
                # Load Model Structure (Need dims from config/scaler)
                # Assumes features didn't change. len(scaler.mean_) gives input dim
                input_size = len(self.scaler.mean_)
                self.model = LSTMClassifier(input_size, self.hidden_size, num_layers=2)
                self.model.load_state_dict(torch.load(path_model))
                self.model.eval()
                
                print(f"LSTM Model loaded for {symbol}")
                return True
            except Exception as e:
                print(f"Error loading models for {symbol}: {e}")
                return False
        else:
            print(f"No existing LSTM model found for {symbol}")
            return False

if __name__ == "__main__":
    from data_loader import DataLoader
    from features import FeatureEngineer
    
    loader = DataLoader()
    fe = FeatureEngineer()
    ml = MLPredictor()
    
    symbol = "SPY"
    df = loader.fetch_data(symbol)
    df = fe.add_indicators(df)
    df = fe.create_target(df)
    
    ml.train(df, symbol)
