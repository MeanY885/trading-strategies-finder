"""
LSTM/GRU Neural Network Predictors (PyTorch)
=============================================
Recurrent neural networks for sequence-based price prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from .base_predictor import BasePredictor, PredictionResult, TrainingResult
from .feature_engineering import FeatureEngineer, FeatureConfig

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class LSTMModel(nn.Module):
        """LSTM model for sequence classification/regression."""

        def __init__(self, input_size: int, hidden_size: int = 128,
                     num_layers: int = 2, num_classes: int = 3,
                     dropout: float = 0.2, bidirectional: bool = True):
            super().__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )

            # Attention layer
            direction_mult = 2 if bidirectional else 1
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * direction_mult, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )

            # Output layers
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * direction_mult, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes)
            )

        def forward(self, x):
            # x: (batch, seq_len, features)
            lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*direction)

            # Attention mechanism
            attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch, hidden*direction)

            # Classification
            output = self.fc(context)
            return output

    class GRUModel(nn.Module):
        """GRU model - lighter alternative to LSTM."""

        def __init__(self, input_size: int, hidden_size: int = 128,
                     num_layers: int = 2, num_classes: int = 3,
                     dropout: float = 0.2, bidirectional: bool = True):
            super().__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.bidirectional = bidirectional

            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )

            direction_mult = 2 if bidirectional else 1

            self.fc = nn.Sequential(
                nn.Linear(hidden_size * direction_mult, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes)
            )

        def forward(self, x):
            gru_out, _ = self.gru(x)
            # Use last output
            last_out = gru_out[:, -1, :]
            return self.fc(last_out)


class LSTMPredictor(BasePredictor):
    """
    LSTM-based predictor for trading signals.

    Uses bidirectional LSTM with attention mechanism to predict
    price direction from sequences of features.
    """

    def __init__(self, sequence_length: int = 30, hidden_size: int = 128,
                 num_layers: int = 2, feature_config: Optional[FeatureConfig] = None):
        """
        Initialize LSTM predictor.

        Args:
            sequence_length: Number of timesteps in each sequence
            hidden_size: LSTM hidden state size
            num_layers: Number of LSTM layers
            feature_config: Configuration for feature engineering
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed. Run: pip install torch")

        super().__init__(
            name="LSTM Neural Network",
            model_type="ml_lstm"
        )

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_engineer = FeatureEngineer(feature_config or FeatureConfig())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.confidence_threshold = 0.5

        self.training_config = {
            'sequence_length': sequence_length,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'device': str(self.device)
        }

    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32,
              learning_rate: float = 0.001, target_horizon: int = 1,
              target_threshold: float = 0.001, **kwargs) -> TrainingResult:
        """
        Train the LSTM model.

        Args:
            df: DataFrame with OHLCV data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            target_horizon: Candles ahead to predict
            target_threshold: Minimum move for signal

        Returns:
            TrainingResult with metrics
        """
        try:
            # Create features
            features = self.feature_engineer.create_features(df, fit=True)
            self.feature_names = features.columns.tolist()

            # Create target (ternary: -1, 0, 1)
            target = self.feature_engineer.create_target(
                df, target_type='direction',
                horizon=target_horizon,
                threshold=target_threshold
            )

            # Create sequences
            X, y = self.feature_engineer.create_sequences(
                features, target, self.sequence_length
            )

            # Convert to tensors
            # Shift target classes from (-1, 0, 1) to (0, 1, 2) for CrossEntropy
            y_shifted = y + 1

            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y_shifted)

            # Train/val split
            split_idx = int(len(X_tensor) * 0.8)
            X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
            y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

            # Create dataloaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # Initialize model
            input_size = X.shape[2]
            self.model = LSTMModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_classes=3  # -1, 0, 1 (shifted to 0, 1, 2)
            ).to(self.device)

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=5, factor=0.5
            )

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            training_history = []

            for epoch in range(epochs):
                # Train
                self.model.train()
                train_loss = 0
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimizer.zero_grad()
                    output = self.model(X_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                train_loss /= len(train_loader)

                # Validate
                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        output = self.model(X_batch)
                        loss = criterion(output, y_batch)
                        val_loss += loss.item()

                        _, predicted = torch.max(output, 1)
                        total += y_batch.size(0)
                        correct += (predicted == y_batch).sum().item()

                val_loss /= len(val_loader)
                val_acc = correct / total

                # Learning rate scheduling
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

                training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                })

            # Load best model
            if best_state:
                self.model.load_state_dict(best_state)

            self.is_trained = True
            self.metadata['epochs_trained'] = len(training_history)
            self.metadata['train_samples'] = len(X_train)
            self.metadata['val_samples'] = len(X_val)

            final_acc = training_history[-1]['val_accuracy'] if training_history else 0

            return TrainingResult(
                success=True,
                train_accuracy=1 - training_history[-1]['train_loss'] if training_history else 0,
                val_accuracy=final_acc,
                train_loss=training_history[-1]['train_loss'] if training_history else 0,
                val_loss=best_val_loss,
                training_history=training_history,
                message=f"Training complete. Val accuracy: {final_acc:.4f}"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return TrainingResult(
                success=False,
                message=f"Training failed: {str(e)}"
            )

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """Generate trading signals from the trained LSTM."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create features
        features = self.feature_engineer.create_features(df, fit=False)

        # Create sequences for prediction
        X_list = []
        indices = []

        for i in range(self.sequence_length, len(features)):
            seq = features.iloc[i - self.sequence_length:i].values
            X_list.append(seq)
            indices.append(features.index[i])

        if len(X_list) == 0:
            return PredictionResult(
                signals=pd.Series(0, index=df.index),
                confidence=None
            )

        X = np.array(X_list)
        X_tensor = torch.FloatTensor(X).to(self.device)

        # Predict
        self.model.eval()
        with torch.no_grad():
            output = self.model(X_tensor)
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

        # Convert back to signals (-1, 0, 1)
        signals = predicted.cpu().numpy() - 1  # Shift from (0,1,2) to (-1,0,1)
        confidence = probs.max(dim=1).values.cpu().numpy()

        # Create series with proper index
        signals_series = pd.Series(0, index=df.index)
        confidence_series = pd.Series(0.0, index=df.index)

        for idx, sig, conf in zip(indices, signals, confidence):
            signals_series.loc[idx] = sig
            confidence_series.loc[idx] = conf

        # Apply confidence threshold
        signals_series[confidence_series < self.confidence_threshold] = 0

        return PredictionResult(
            signals=signals_series.astype(int),
            confidence=confidence_series
        )

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(self.model.state_dict(), path / 'model.pt')

        # Save config
        config = {
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'input_size': len(self.feature_names)
        }
        import json
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f)

        # Save scaler
        self.feature_engineer.save_scaler(str(path / 'scaler.pkl'))

        # Save metadata
        self._save_metadata(str(path))

    def load(self, path: str) -> None:
        """Load model from disk."""
        path = Path(path)

        # Load config
        import json
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)

        self.sequence_length = config['sequence_length']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']

        # Initialize model
        self.model = LSTMModel(
            input_size=config['input_size'],
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=3
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(torch.load(path / 'model.pt', map_location=self.device))

        # Load scaler
        self.feature_engineer.load_scaler(str(path / 'scaler.pkl'))

        # Load metadata
        self._load_metadata(str(path))
        self.is_trained = True


class GRUPredictor(BasePredictor):
    """
    GRU-based predictor - lighter alternative to LSTM.

    Uses GRU (Gated Recurrent Unit) which has fewer parameters
    than LSTM but often achieves similar performance.
    """

    def __init__(self, sequence_length: int = 30, hidden_size: int = 128,
                 num_layers: int = 2, feature_config: Optional[FeatureConfig] = None):
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed. Run: pip install torch")

        super().__init__(
            name="GRU Neural Network",
            model_type="ml_gru"
        )

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_engineer = FeatureEngineer(feature_config or FeatureConfig())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.confidence_threshold = 0.5

    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32,
              learning_rate: float = 0.001, **kwargs) -> TrainingResult:
        """Train the GRU model (similar to LSTM)."""
        try:
            # Create features
            features = self.feature_engineer.create_features(df, fit=True)
            self.feature_names = features.columns.tolist()

            target = self.feature_engineer.create_target(
                df, target_type='direction', horizon=1, threshold=0.001
            )

            X, y = self.feature_engineer.create_sequences(
                features, target, self.sequence_length
            )

            y_shifted = y + 1
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y_shifted)

            split_idx = int(len(X_tensor) * 0.8)
            X_train, X_val = X_tensor[:split_idx], X_tensor[split_idx:]
            y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            input_size = X.shape[2]
            self.model = GRUModel(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_classes=3
            ).to(self.device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

            best_val_acc = 0
            for epoch in range(epochs):
                self.model.train()
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    optimizer.zero_grad()
                    output = self.model(X_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()
                    optimizer.step()

                # Validate
                self.model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        output = self.model(X_batch)
                        _, predicted = torch.max(output, 1)
                        total += y_batch.size(0)
                        correct += (predicted == y_batch).sum().item()

                val_acc = correct / total
                best_val_acc = max(best_val_acc, val_acc)

            self.is_trained = True

            return TrainingResult(
                success=True,
                val_accuracy=best_val_acc,
                message=f"Training complete. Val accuracy: {best_val_acc:.4f}"
            )

        except Exception as e:
            return TrainingResult(success=False, message=f"Training failed: {str(e)}")

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """Generate signals (same as LSTM)."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained.")

        features = self.feature_engineer.create_features(df, fit=False)

        X_list = []
        indices = []
        for i in range(self.sequence_length, len(features)):
            seq = features.iloc[i - self.sequence_length:i].values
            X_list.append(seq)
            indices.append(features.index[i])

        if len(X_list) == 0:
            return PredictionResult(signals=pd.Series(0, index=df.index))

        X_tensor = torch.FloatTensor(np.array(X_list)).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(X_tensor)
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)

        signals = predicted.cpu().numpy() - 1
        confidence = probs.max(dim=1).values.cpu().numpy()

        signals_series = pd.Series(0, index=df.index)
        confidence_series = pd.Series(0.0, index=df.index)

        for idx, sig, conf in zip(indices, signals, confidence):
            signals_series.loc[idx] = sig
            confidence_series.loc[idx] = conf

        signals_series[confidence_series < self.confidence_threshold] = 0

        return PredictionResult(
            signals=signals_series.astype(int),
            confidence=confidence_series
        )

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path / 'model.pt')
        self.feature_engineer.save_scaler(str(path / 'scaler.pkl'))
        self._save_metadata(str(path))

    def load(self, path: str) -> None:
        path = Path(path)
        self._load_metadata(str(path))
        self.feature_engineer.load_scaler(str(path / 'scaler.pkl'))
        self.is_trained = True
