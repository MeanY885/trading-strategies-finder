"""
Base Predictor Abstract Class
=============================
Defines the interface that all ML models must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle


@dataclass
class PredictionResult:
    """Result of a prediction with metadata"""
    signals: pd.Series  # 1 = long, -1 = short, 0 = hold
    confidence: Optional[pd.Series] = None  # Probability/confidence of signal
    raw_output: Optional[Any] = None  # Raw model output (for debugging)


@dataclass
class TrainingResult:
    """Result of model training with metrics"""
    success: bool
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    train_loss: float = 0.0
    val_loss: float = 0.0
    feature_importance: Optional[Dict[str, float]] = None
    training_history: Optional[List[Dict]] = None
    best_params: Optional[Dict[str, Any]] = None
    message: str = ""


class BasePredictor(ABC):
    """
    Abstract base class for all ML predictors.

    All ML models (XGBoost, LSTM, Transformer, RL) must implement this interface
    to integrate with the unified optimizer.
    """

    def __init__(self, name: str, model_type: str):
        """
        Initialize the predictor.

        Args:
            name: Human-readable name for the model
            model_type: Type identifier (e.g., 'xgboost', 'lstm', 'rl_ppo')
        """
        self.name = name
        self.model_type = model_type
        self.is_trained = False
        self.model = None
        self.feature_names: List[str] = []
        self.training_config: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def train(self, df: pd.DataFrame, **kwargs) -> TrainingResult:
        """
        Train the model on the provided data.

        Args:
            df: DataFrame with OHLCV data and indicators
            **kwargs: Model-specific training parameters

        Returns:
            TrainingResult with metrics and status
        """
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """
        Generate trading signals from the model.

        Args:
            df: DataFrame with OHLCV data and indicators

        Returns:
            PredictionResult with signals (1=long, -1=short, 0=hold)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Directory path to save model files
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            path: Directory path containing model files
        """
        pass

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores (if available).

        Returns:
            Dictionary mapping feature names to importance scores
        """
        return {}

    def get_decision_rules(self) -> List[Dict[str, Any]]:
        """
        Extract interpretable decision rules from the model.
        Used for Pine Script generation.

        Returns:
            List of rule dictionaries with:
            - indicator: Feature/indicator name
            - operator: Comparison operator (<, >, ==, etc.)
            - threshold: Threshold value
            - importance: Rule importance score
        """
        return []

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model for display.

        Returns:
            Dictionary with model info
        """
        return {
            'name': self.name,
            'type': self.model_type,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names),
            'config': self.training_config,
            'metadata': self.metadata
        }

    def _save_metadata(self, path: str) -> None:
        """Save model metadata to JSON file."""
        metadata_path = Path(path) / 'metadata.json'
        metadata = {
            'name': self.name,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'training_config': self.training_config,
            'metadata': self.metadata
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def _load_metadata(self, path: str) -> None:
        """Load model metadata from JSON file."""
        metadata_path = Path(path) / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.name = metadata.get('name', self.name)
            self.model_type = metadata.get('model_type', self.model_type)
            self.is_trained = metadata.get('is_trained', False)
            self.feature_names = metadata.get('feature_names', [])
            self.training_config = metadata.get('training_config', {})
            self.metadata = metadata.get('metadata', {})

    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return f"{self.__class__.__name__}(name='{self.name}', type='{self.model_type}', status={status})"
