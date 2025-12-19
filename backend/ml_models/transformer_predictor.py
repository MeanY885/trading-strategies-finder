"""
Time-Series Transformer Predictor (PyTorch)
============================================
Attention-based model for sequence prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import math

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
    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding for sequences."""

        def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)  # (1, max_len, d_model)

            self.register_buffer('pe', pe)

        def forward(self, x):
            # x: (batch, seq_len, d_model)
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)

    class TimeSeriesTransformer(nn.Module):
        """
        Transformer encoder for time-series classification.

        Architecture:
        - Linear embedding layer
        - Positional encoding
        - Transformer encoder layers
        - Classification head
        """

        def __init__(self, input_size: int, d_model: int = 64, nhead: int = 4,
                     num_layers: int = 3, dim_feedforward: int = 256,
                     num_classes: int = 3, dropout: float = 0.1):
            super().__init__()

            self.d_model = d_model

            # Input embedding
            self.embedding = nn.Linear(input_size, d_model)

            # Positional encoding
            self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

            # Transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes)
            )

        def forward(self, x):
            # x: (batch, seq_len, features)

            # Embed
            x = self.embedding(x) * math.sqrt(self.d_model)

            # Add positional encoding
            x = self.pos_encoder(x)

            # Transformer encode
            x = self.transformer_encoder(x)

            # Use last timestep for classification
            x = x[:, -1, :]  # (batch, d_model)

            # Classify
            return self.classifier(x)


class TransformerPredictor(BasePredictor):
    """
    Transformer-based predictor for trading signals.

    Uses attention mechanism to capture long-range dependencies
    in time-series data for price direction prediction.
    """

    def __init__(self, sequence_length: int = 50, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 3,
                 feature_config: Optional[FeatureConfig] = None):
        """
        Initialize Transformer predictor.

        Args:
            sequence_length: Number of timesteps in each sequence
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            feature_config: Configuration for feature engineering
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch not installed. Run: pip install torch")

        super().__init__(
            name="Transformer Time-Series",
            model_type="ml_transformer"
        )

        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.feature_engineer = FeatureEngineer(feature_config or FeatureConfig())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.confidence_threshold = 0.5

        self.training_config = {
            'sequence_length': sequence_length,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'device': str(self.device)
        }

    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32,
              learning_rate: float = 0.0001, target_horizon: int = 1,
              target_threshold: float = 0.001, **kwargs) -> TrainingResult:
        """
        Train the Transformer model.

        Args:
            df: DataFrame with OHLCV data
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate (smaller for transformers)
            target_horizon: Candles ahead to predict
            target_threshold: Minimum move for signal

        Returns:
            TrainingResult with metrics
        """
        try:
            # Create features
            features = self.feature_engineer.create_features(df, fit=True)
            self.feature_names = features.columns.tolist()

            # Create target
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
            y_shifted = y + 1  # Shift from (-1,0,1) to (0,1,2)

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
            self.model = TimeSeriesTransformer(
                input_size=input_size,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                num_classes=3
            ).to(self.device)

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )

            # Cosine annealing scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )

            # Training loop
            best_val_loss = float('inf')
            best_state = None
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

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)
                scheduler.step()

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

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = self.model.state_dict().copy()

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
            self.metadata['epochs_trained'] = epochs
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
        """Generate trading signals from the trained Transformer."""
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
        signals = predicted.cpu().numpy() - 1
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

    def get_attention_weights(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Get attention weights for visualization.

        Returns attention weights showing which timesteps
        the model focuses on for prediction.
        """
        # This would require modifying the model to return attention weights
        # Left as placeholder for future implementation
        return None

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        torch.save(self.model.state_dict(), path / 'model.pt')

        # Save config
        config = {
            'sequence_length': self.sequence_length,
            'd_model': self.d_model,
            'nhead': self.nhead,
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
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_layers = config['num_layers']

        # Initialize model
        self.model = TimeSeriesTransformer(
            input_size=config['input_size'],
            d_model=self.d_model,
            nhead=self.nhead,
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
