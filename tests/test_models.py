"""Tests for model implementations."""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestXGBoostModel:
    def test_train_and_predict(self):
        from backend.models.xgboost_model import XGBoostTrafficModel
        model = XGBoostTrafficModel(forecast_horizon=2)

        np.random.seed(42)
        X_train = np.random.rand(100, 10)
        y_train = np.random.rand(100)
        X_test = np.random.rand(20, 10)

        model.train(X_train, y_train)
        preds = model.predict(X_test)

        assert model.is_trained
        assert preds.shape == (20,)
        assert model.training_time is not None

    def test_save_load(self, tmp_path):
        from backend.models.xgboost_model import XGBoostTrafficModel
        model = XGBoostTrafficModel()
        X = np.random.rand(50, 5)
        y = np.random.rand(50)
        model.train(X, y)

        path = tmp_path / "xgb_test.pkl"
        model.save(path)

        model2 = XGBoostTrafficModel()
        model2.load(path)
        assert model2.is_trained

    def test_feature_importance(self):
        from backend.models.xgboost_model import XGBoostTrafficModel
        import pandas as pd
        model = XGBoostTrafficModel()
        X = pd.DataFrame(np.random.rand(100, 5), columns=["a", "b", "c", "d", "e"])
        y = np.random.rand(100)
        model.train(X, y)
        imp = model.get_feature_importance()
        assert len(imp) > 0


class TestLSTMModel:
    def test_train_and_predict(self):
        from backend.models.lstm_model import LSTMTrafficModel
        model = LSTMTrafficModel(forecast_horizon=2)

        np.random.seed(42)
        X_train = np.random.rand(50, 12, 1).astype(np.float32)
        y_train = np.random.rand(50, 2).astype(np.float32)

        result = model.train(X_train, y_train, epochs=3, batch_size=16)
        preds = model.predict(X_train[:5])

        assert model.is_trained
        assert preds.shape == (5, 2)
        assert "training_time_seconds" in result


class TestSTGCNModel:
    def test_train_and_predict(self):
        from backend.models.stgcn_model import STGCNTrafficModel
        model = STGCNTrafficModel(forecast_horizon=2, K=2)

        np.random.seed(42)
        n_nodes = 5
        seq_len = 12
        X_train = np.random.rand(30, n_nodes, 1, seq_len).astype(np.float32)
        y_train = np.random.rand(30, n_nodes, 2).astype(np.float32)
        adj = np.random.rand(n_nodes, n_nodes)
        adj = (adj + adj.T) / 2
        np.fill_diagonal(adj, 0)

        result = model.train(X_train, y_train, adj_matrix=adj, epochs=3, batch_size=16)
        preds = model.predict(X_train[:5])

        assert model.is_trained
        assert preds.shape == (5, n_nodes, 2)
        assert "training_time_seconds" in result
