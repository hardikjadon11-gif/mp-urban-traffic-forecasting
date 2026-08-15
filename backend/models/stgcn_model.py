"""
STGCN — Spatio-Temporal Graph Convolutional Network.

PRIMARY / FINAL MODEL for traffic forecasting.

Architecture:
  ST-Conv Block × 2 → Output Layer

Each ST-Conv Block:
  Temporal Conv (1D causal conv with GLU gating)
  → Spatial Graph Conv (Chebyshev polynomial approximation)
  → Temporal Conv (1D causal conv with GLU gating)
  → Layer Norm

The model simultaneously captures:
  - Spatial relationships: Road sensors as graph nodes, edges = road connections
  - Temporal dependencies: Historical readings as time-series sequences
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from backend.models.base_model import BaseTrafficModel
from backend.config import (
    STGCN_NUM_LAYERS, CHEB_K,
    DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_EPOCHS,
)

logger = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Building Blocks ──────────────────────────────────────────────────────

class TemporalConvLayer(nn.Module):
    """
    Temporal convolution with GLU (Gated Linear Unit) gating.

    Input:  (batch, nodes, channels_in, time)
    Output: (batch, nodes, channels_out, time)
    """

    def __init__(self, c_in: int, c_out: int, kernel_size: int = 3):
        super().__init__()
        # GLU needs 2× output channels (one half for gate)
        self.conv = nn.Conv2d(c_in, 2 * c_out, kernel_size=(1, kernel_size),
                              padding=(0, kernel_size // 2))
        self.c_out = c_out

    def forward(self, x):
        """x: (batch, nodes, c_in, time) → (batch, nodes, c_out, time)"""
        # Reshape to (batch, c_in, nodes, time) for Conv2d
        b, n, c, t = x.shape
        x = x.permute(0, 2, 1, 3)  # (batch, c_in, nodes, time)
        x = self.conv(x)  # (batch, 2*c_out, nodes, time)

        # GLU gating
        x_p, x_q = x.split(self.c_out, dim=1)
        x = x_p * torch.sigmoid(x_q)  # (batch, c_out, nodes, time)

        x = x.permute(0, 2, 1, 3)  # (batch, nodes, c_out, time)
        return x


class SpatialGraphConv(nn.Module):
    """
    Graph convolution using Chebyshev polynomial approximation.

    Approximates spectral graph convolution with K-order Chebyshev polynomials
    of the scaled graph Laplacian.

    Input:  (batch, nodes, c_in, time)
    Output: (batch, nodes, c_out, time)
    """

    def __init__(self, c_in: int, c_out: int, K: int = CHEB_K):
        super().__init__()
        self.K = K
        self.c_in = c_in
        self.c_out = c_out
        # Learnable weight for each Chebyshev order
        self.weight = nn.Parameter(torch.FloatTensor(K, c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, cheb_polynomials: List[torch.Tensor]):
        """
        Args:
            x: (batch, nodes, c_in, time)
            cheb_polynomials: List of K tensors, each (nodes, nodes)
        """
        b, n, c_in, t = x.shape

        # Compute graph convolution for each Chebyshev order
        outputs = []
        for k in range(min(self.K, len(cheb_polynomials))):
            T_k = cheb_polynomials[k]  # (N, N)
            # Apply graph conv: T_k @ x for each batch and time step
            # x reshaped: (b*t, N, c_in)
            x_flat = x.permute(0, 3, 1, 2).reshape(b * t, n, c_in)
            gconv = torch.matmul(T_k, x_flat)  # (b*t, N, c_in)
            gconv = gconv.reshape(b, t, n, c_in).permute(0, 2, 3, 1)  # (b, N, c_in, t)

            # Apply learnable weight
            gconv = torch.einsum('bnct,co->bnot', gconv, self.weight[k])
            outputs.append(gconv)

        # Sum over Chebyshev orders
        out = sum(outputs) + self.bias.view(1, 1, -1, 1)
        return out


class STConvBlock(nn.Module):
    """
    Spatio-Temporal Convolutional Block.

    Structure: Temporal Conv → Spatial Graph Conv → Temporal Conv → LayerNorm
    """

    def __init__(self, c_in: int, c_spatial: int, c_out: int,
                 K: int = CHEB_K, kernel_size: int = 3):
        super().__init__()
        self.temporal1 = TemporalConvLayer(c_in, c_spatial, kernel_size)
        self.spatial = SpatialGraphConv(c_spatial, c_spatial, K)
        self.temporal2 = TemporalConvLayer(c_spatial, c_out, kernel_size)
        self.layer_norm = nn.LayerNorm(c_out)
        self.dropout = nn.Dropout(0.1)
        # Residual connection
        self.residual = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

    def forward(self, x, cheb_polynomials):
        """
        Args:
            x: (batch, nodes, c_in, time)
            cheb_polynomials: List of Chebyshev polynomial tensors
        """
        residual = x.permute(0, 2, 1, 3)  # (batch, c_in, nodes, time)
        residual = self.residual(residual).permute(0, 2, 1, 3)  # (batch, nodes, c_out, time)

        out = self.temporal1(x)
        out = self.spatial(out, cheb_polynomials)
        out = F.relu(out)
        out = self.temporal2(out)
        out = self.dropout(out)

        # Residual connection (with shape matching)
        min_t = min(out.shape[3], residual.shape[3])
        out = out[:, :, :, :min_t] + residual[:, :, :, :min_t]

        # Layer norm over channel dimension
        out = self.layer_norm(out.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return out


class STGCNNetwork(nn.Module):
    """
    Complete STGCN Network.

    Architecture:
      Input → ST-Conv Block 1 → ST-Conv Block 2 → Output FC Layer
    """

    def __init__(self, n_nodes: int, n_features: int = 1,
                 n_output: int = 1, seq_length: int = 12,
                 channels: list = None, K: int = CHEB_K):
        super().__init__()

        if channels is None:
            channels = [n_features, 32, 32, 64]

        self.n_nodes = n_nodes
        self.n_output = n_output

        # ST-Conv Blocks
        self.st_blocks = nn.ModuleList()
        for i in range(len(channels) - 2):
            self.st_blocks.append(
                STConvBlock(
                    c_in=channels[i] if i == 0 else channels[i + 1],
                    c_spatial=channels[i + 1],
                    c_out=channels[i + 1],
                    K=K,
                )
            )

        # Final output layer
        self.output_conv = nn.Conv2d(channels[-2], channels[-1], kernel_size=1)
        self.fc = nn.Linear(channels[-1] * seq_length, n_output)

    def forward(self, x, cheb_polynomials):
        """
        Args:
            x: (batch, nodes, features, seq_length)
            cheb_polynomials: List of Chebyshev polynomial tensors

        Returns:
            out: (batch, nodes, n_output) — predictions for each node
        """
        out = x
        for block in self.st_blocks:
            out = block(out, cheb_polynomials)

        # Output projection
        b, n, c, t = out.shape
        out = out.permute(0, 2, 1, 3)  # (b, c, n, t)
        out = self.output_conv(out)  # (b, c_out, n, t)
        out = F.relu(out)
        out = out.permute(0, 2, 1, 3)  # (b, n, c_out, t)

        # Flatten time and channel dimensions
        out = out.reshape(b, n, -1)  # (b, n, c_out * t)
        out = self.fc(out)  # (b, n, n_output)

        return out


class STGCNTrafficModel(BaseTrafficModel):
    """STGCN-based traffic forecasting — Primary Model."""

    def __init__(self, forecast_horizon: int = 2, K: int = CHEB_K,
                 channels: list = None):
        super().__init__("stgcn", forecast_horizon)
        self.K = K
        self.channels = channels
        self.network = None
        self.cheb_polynomials = None
        self.n_nodes = None
        self.seq_length = None
        self.config = {
            "K": K,
            "channels": channels,
            "device": str(DEVICE),
        }

    def _build_cheb_polynomials(self, adj_matrix: np.ndarray) -> List[torch.Tensor]:
        """Compute Chebyshev polynomials of the scaled Laplacian."""
        from backend.utils.graph import compute_scaled_laplacian, compute_cheb_polynomials

        L_scaled = compute_scaled_laplacian(adj_matrix)
        cheb_np = compute_cheb_polynomials(L_scaled, self.K)
        cheb_tensors = [torch.FloatTensor(c).to(DEVICE) for c in cheb_np]
        return cheb_tensors

    def train(self, X_train, y_train, X_val=None, y_val=None,
              adj_matrix=None, **kwargs) -> Dict[str, Any]:
        """
        Train STGCN.

        Args:
            X_train: (n_samples, n_nodes, features, seq_length) numpy array
            y_train: (n_samples, n_nodes, forecast_horizon) numpy array
            adj_matrix: (n_nodes, n_nodes) adjacency matrix
        """
        epochs = kwargs.get("epochs", DEFAULT_EPOCHS)
        batch_size = kwargs.get("batch_size", DEFAULT_BATCH_SIZE)
        lr = kwargs.get("learning_rate", DEFAULT_LEARNING_RATE)

        self.n_nodes = X_train.shape[1]
        n_features = X_train.shape[2]
        self.seq_length = X_train.shape[3]

        logger.info(f"Training STGCN: {self.n_nodes} nodes, seq_len={self.seq_length}, device={DEVICE}")

        # Build graph polynomials
        if adj_matrix is None:
            adj_matrix = np.eye(self.n_nodes)
        self.cheb_polynomials = self._build_cheb_polynomials(adj_matrix)

        # Build network
        channels = self.channels or [n_features, 32, 32, 64]
        self.network = STGCNNetwork(
            n_nodes=self.n_nodes,
            n_features=n_features,
            n_output=self.forecast_horizon,
            seq_length=self.seq_length,
            channels=channels,
            K=self.K,
        ).to(DEVICE)

        # Data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        start = time.time()
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Training
            self.network.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                optimizer.zero_grad()
                output = self.network(X_batch, self.cheb_polynomials)
                loss = criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item()

            avg_train = train_loss / len(train_loader)
            history["train_loss"].append(avg_train)

            # Validation
            avg_val = None
            if val_loader:
                self.network.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(DEVICE)
                        y_batch = y_batch.to(DEVICE)
                        output = self.network(X_batch, self.cheb_polynomials)
                        val_loss += criterion(output, y_batch).item()
                avg_val = val_loss / len(val_loader)
                history["val_loss"].append(avg_val)
                scheduler.step(avg_val)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                msg = f"  Epoch {epoch+1}/{epochs} — Train Loss: {avg_train:.6f}"
                if avg_val is not None:
                    msg += f" — Val Loss: {avg_val:.6f}"
                logger.info(msg)

        self.training_time = round(time.time() - start, 2)
        self.is_trained = True
        self.training_history = history

        logger.info(f"STGCN training complete in {self.training_time}s")
        return {
            "training_time_seconds": self.training_time,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
            "epochs": epochs,
            "n_nodes": self.n_nodes,
            "history": history,
        }

    def predict(self, X, adj_matrix=None) -> np.ndarray:
        """
        Generate predictions for all nodes.

        Args:
            X: (n_samples, n_nodes, features, seq_length)
        Returns:
            (n_samples, n_nodes, forecast_horizon)
        """
        if self.network is None:
            raise RuntimeError("STGCN not trained. Call train() first.")

        if self.cheb_polynomials is None and adj_matrix is not None:
            self.cheb_polynomials = self._build_cheb_polynomials(adj_matrix)

        self.network.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(DEVICE)
            output = self.network(X_tensor, self.cheb_polynomials)
            return output.cpu().numpy()

    def save(self, path: Optional[Path] = None) -> Path:
        """Save STGCN model."""
        path = path or self.get_default_save_path()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "model_state_dict": self.network.state_dict() if self.network else None,
            "n_nodes": self.n_nodes,
            "seq_length": self.seq_length,
            "K": self.K,
            "channels": self.channels,
            "forecast_horizon": self.forecast_horizon,
            "training_time": self.training_time,
            "config": self.config,
        }
        torch.save(state, path)
        logger.info(f"STGCN model saved to {path}")
        return path

    def load(self, path: Optional[Path] = None, adj_matrix=None) -> None:
        """Load STGCN model."""
        path = path or self.get_default_save_path()
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        state = torch.load(path, map_location=DEVICE, weights_only=False)

        self.n_nodes = state["n_nodes"]
        self.seq_length = state.get("seq_length", 12)
        self.K = state.get("K", CHEB_K)
        self.channels = state.get("channels")
        self.forecast_horizon = state.get("forecast_horizon", 2)
        self.training_time = state.get("training_time")
        self.config = state.get("config", {})

        # Rebuild network
        channels = self.channels or [1, 32, 32, 64]
        self.network = STGCNNetwork(
            n_nodes=self.n_nodes,
            n_features=channels[0],
            n_output=self.forecast_horizon,
            seq_length=self.seq_length,
            channels=channels,
            K=self.K,
        ).to(DEVICE)

        if state["model_state_dict"]:
            self.network.load_state_dict(state["model_state_dict"])

        # Rebuild Chebyshev polynomials if adjacency provided
        if adj_matrix is not None:
            self.cheb_polynomials = self._build_cheb_polynomials(adj_matrix)

        self.is_trained = True
        logger.info(f"STGCN model loaded from {path}")
