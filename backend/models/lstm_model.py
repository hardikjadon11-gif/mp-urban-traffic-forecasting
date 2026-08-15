"""
LSTM Traffic Forecasting Model — Deep Learning Temporal Baseline.

2-layer LSTM network that captures temporal dependencies in traffic data.
Learns patterns like recent conditions, recurring patterns, and
longer-term sequential dependencies.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backend.models.base_model import BaseTrafficModel
from backend.config import (
    LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS
)

logger = logging.getLogger(__name__)

# Device selection
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMNetwork(nn.Module):
    """2-layer LSTM with fully connected output layer."""

    def __init__(self, input_size: int, hidden_size: int = LSTM_HIDDEN_SIZE,
                 num_layers: int = LSTM_NUM_LAYERS, dropout: float = LSTM_DROPOUT,
                 output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            out: (batch, output_size)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use the last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        out = self.fc(last_hidden)
        return out


class LSTMTrafficModel(BaseTrafficModel):
    """LSTM-based traffic speed forecasting model."""

    def __init__(self, forecast_horizon: int = 2,
                 hidden_size: int = LSTM_HIDDEN_SIZE,
                 num_layers: int = LSTM_NUM_LAYERS,
                 dropout: float = LSTM_DROPOUT):
        super().__init__("lstm", forecast_horizon)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.network = None
        self.input_size = None
        self.config = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "device": str(DEVICE),
        }

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> Dict[str, Any]:
        """
        Train LSTM model.

        Args:
            X_train: (n_samples, seq_length, n_features) numpy array
            y_train: (n_samples, forecast_horizon) numpy array
            X_val, y_val: Optional validation data
        """
        epochs = kwargs.get("epochs", DEFAULT_EPOCHS)
        batch_size = kwargs.get("batch_size", DEFAULT_BATCH_SIZE)
        lr = kwargs.get("learning_rate", DEFAULT_LEARNING_RATE)

        self.input_size = X_train.shape[2]
        output_size = y_train.shape[1] if y_train.ndim > 1 else 1

        logger.info(f"Training LSTM: input_size={self.input_size}, "
                     f"output_size={output_size}, device={DEVICE}")

        # Build network
        self.network = LSTMNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=output_size,
        ).to(DEVICE)

        # Data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train if y_train.ndim > 1 else y_train.reshape(-1, 1))
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val if y_val.ndim > 1 else y_val.reshape(-1, 1))
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        start = time.time()
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Training phase
            self.network.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                optimizer.zero_grad()
                output = self.network(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # Validation phase
            avg_val_loss = None
            if val_loader:
                self.network.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(DEVICE)
                        y_batch = y_batch.to(DEVICE)
                        output = self.network(X_batch)
                        val_loss += criterion(output, y_batch).item()
                avg_val_loss = val_loss / len(val_loader)
                history["val_loss"].append(avg_val_loss)
                scheduler.step(avg_val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"  Epoch {epoch+1}/{epochs} — Train Loss: {avg_train_loss:.6f}"
                if avg_val_loss is not None:
                    msg += f" — Val Loss: {avg_val_loss:.6f}"
                logger.info(msg)

        self.training_time = round(time.time() - start, 2)
        self.is_trained = True
        self.training_history = history

        logger.info(f"LSTM training complete in {self.training_time}s")
        return {
            "training_time_seconds": self.training_time,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "epochs": epochs,
            "history": history,
        }

    def predict(self, X) -> np.ndarray:
        """
        Generate predictions.

        Args:
            X: (n_samples, seq_length, n_features) numpy array
        Returns:
            predictions: (n_samples, output_size) numpy array
        """
        if self.network is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            output = self.network(X_tensor)
            return output.cpu().numpy()

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model state dict and config."""
        path = path or self.get_default_save_path()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.network.state_dict() if self.network else None,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "forecast_horizon": self.forecast_horizon,
            "training_time": self.training_time,
            "config": self.config,
        }
        torch.save(state, path)
        logger.info(f"LSTM model saved to {path}")
        return path

    def load(self, path: Optional[Path] = None) -> None:
        """Load model state dict."""
        path = path or self.get_default_save_path()
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        state = torch.load(path, map_location=DEVICE, weights_only=False)

        self.input_size = state["input_size"]
        self.hidden_size = state.get("hidden_size", LSTM_HIDDEN_SIZE)
        self.num_layers = state.get("num_layers", LSTM_NUM_LAYERS)
        self.dropout = state.get("dropout", LSTM_DROPOUT)
        self.forecast_horizon = state.get("forecast_horizon", 2)
        self.training_time = state.get("training_time")
        self.config = state.get("config", {})

        # Rebuild and load network
        output_size = self.forecast_horizon
        self.network = LSTMNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            output_size=output_size,
        ).to(DEVICE)

        if state["model_state_dict"]:
            self.network.load_state_dict(state["model_state_dict"])
        self.is_trained = True

        logger.info(f"LSTM model loaded from {path}")
