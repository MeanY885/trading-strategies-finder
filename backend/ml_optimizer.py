"""
ML Optimizer - Training Orchestrator
=====================================
Coordinates training of all ML models and provides a unified interface.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from datetime import datetime


@dataclass
class MLTrainingStatus:
    """Status of ML training process"""
    running: bool = False
    progress: int = 0
    message: str = "Ready"
    current_model: Optional[str] = None
    models_trained: List[str] = field(default_factory=list)
    models_failed: List[str] = field(default_factory=list)
    results: Dict[str, Dict] = field(default_factory=dict)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'running': self.running,
            'progress': self.progress,
            'message': self.message,
            'current_model': self.current_model,
            'models_trained': self.models_trained,
            'models_failed': self.models_failed,
            'results': self.results,
            'elapsed_time': (time.time() - self.start_time) if self.start_time else 0
        }


# Global status and trained models (shared with main.py)
ml_status = MLTrainingStatus()
trained_ml_models: Dict[str, Any] = {}


class MLOptimizer:
    """
    Orchestrates training of all ML models.

    Supports:
    - Gradient Boosting: XGBoost, LightGBM, CatBoost
    - Neural Networks: LSTM, GRU (when available)
    - Transformers: Time-series transformer (when available)
    - RL Agents: PPO, DQN (when available)
    """

    AVAILABLE_MODELS = {
        'ml_xgboost': 'XGBoost Classifier',
        'ml_lightgbm': 'LightGBM Classifier',
        'ml_catboost': 'CatBoost Classifier',
        'ml_lstm': 'LSTM Neural Network',
        'ml_gru': 'GRU Neural Network',
        'ml_transformer': 'Transformer Time-Series',
        'ml_rl_ppo': 'RL Agent (PPO)',
        'ml_rl_dqn': 'RL Agent (DQN)',
    }

    def __init__(self, df: pd.DataFrame, model_save_path: str = './ml_models_saved'):
        """
        Initialize the ML optimizer.

        Args:
            df: DataFrame with OHLCV data
            model_save_path: Directory to save trained models
        """
        self.df = df
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, Any] = {}

    def train_models(self, models_to_train: List[str],
                     status_callback: Optional[Callable] = None,
                     **training_params) -> Dict[str, Dict]:
        """
        Train selected ML models.

        Args:
            models_to_train: List of model identifiers to train
            status_callback: Optional callback to update status
            **training_params: Parameters passed to each model's train method

        Returns:
            Dictionary of model results
        """
        global ml_status, trained_ml_models

        ml_status.running = True
        ml_status.progress = 0
        ml_status.models_trained = []
        ml_status.models_failed = []
        ml_status.results = {}
        ml_status.start_time = time.time()

        results = {}
        total_models = len(models_to_train)

        for idx, model_id in enumerate(models_to_train):
            ml_status.current_model = model_id
            ml_status.message = f"Training {self.AVAILABLE_MODELS.get(model_id, model_id)}..."
            ml_status.progress = int((idx / total_models) * 100)

            if status_callback:
                status_callback(ml_status.to_dict())

            try:
                result = self._train_single_model(model_id, **training_params)
                results[model_id] = result

                if result.get('success', False):
                    ml_status.models_trained.append(model_id)
                    trained_ml_models[model_id] = self.models[model_id]
                else:
                    ml_status.models_failed.append(model_id)

            except Exception as e:
                results[model_id] = {
                    'success': False,
                    'error': str(e)
                }
                ml_status.models_failed.append(model_id)

            ml_status.results[model_id] = results[model_id]

        ml_status.running = False
        ml_status.progress = 100
        ml_status.end_time = time.time()
        ml_status.current_model = None
        ml_status.message = f"Complete. {len(ml_status.models_trained)}/{total_models} models trained."

        return results

    def _train_single_model(self, model_id: str, **params) -> Dict:
        """Train a single model by ID."""

        if model_id == 'ml_xgboost':
            return self._train_gradient_boosting('xgboost', **params)

        elif model_id == 'ml_lightgbm':
            return self._train_gradient_boosting('lightgbm', **params)

        elif model_id == 'ml_catboost':
            return self._train_gradient_boosting('catboost', **params)

        elif model_id == 'ml_lstm':
            return self._train_lstm(**params)

        elif model_id == 'ml_gru':
            return self._train_gru(**params)

        elif model_id == 'ml_transformer':
            return self._train_transformer(**params)

        elif model_id == 'ml_rl_ppo':
            return self._train_rl('PPO', **params)

        elif model_id == 'ml_rl_dqn':
            return self._train_rl('DQN', **params)

        else:
            return {'success': False, 'error': f'Unknown model: {model_id}'}

    def _train_gradient_boosting(self, model_type: str, **params) -> Dict:
        """Train a gradient boosting model."""
        try:
            from ml_models.gradient_boosting import GradientBoostingPredictor

            model = GradientBoostingPredictor(
                model_type=model_type,
                task='classification'
            )

            result = model.train(
                self.df,
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 6),
                learning_rate=params.get('learning_rate', 0.1),
                target_horizon=params.get('target_horizon', 1),
                target_threshold=params.get('target_threshold', 0.001)
            )

            if result.success:
                self.models[f'ml_{model_type}'] = model
                # Save model
                model.save(str(self.model_save_path / model_type))

            return {
                'success': result.success,
                'train_accuracy': result.train_accuracy,
                'val_accuracy': result.val_accuracy,
                'feature_importance': result.feature_importance,
                'message': result.message
            }

        except ImportError as e:
            return {'success': False, 'error': f'{model_type} not installed: {str(e)}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _train_lstm(self, **params) -> Dict:
        """Train LSTM model."""
        try:
            from ml_models.lstm_predictor import LSTMPredictor

            model = LSTMPredictor(
                sequence_length=params.get('sequence_length', 30),
                hidden_size=params.get('hidden_size', 128)
            )

            result = model.train(
                self.df,
                epochs=params.get('epochs', 50),
                batch_size=params.get('batch_size', 32)
            )

            if result.success:
                self.models['ml_lstm'] = model
                model.save(str(self.model_save_path / 'lstm'))

            return {
                'success': result.success,
                'train_accuracy': result.train_accuracy,
                'val_accuracy': result.val_accuracy,
                'message': result.message
            }

        except ImportError as e:
            return {'success': False, 'error': f'PyTorch not installed: {str(e)}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _train_gru(self, **params) -> Dict:
        """Train GRU model."""
        try:
            from ml_models.lstm_predictor import GRUPredictor

            model = GRUPredictor(
                sequence_length=params.get('sequence_length', 30),
                hidden_size=params.get('hidden_size', 128)
            )

            result = model.train(
                self.df,
                epochs=params.get('epochs', 50),
                batch_size=params.get('batch_size', 32)
            )

            if result.success:
                self.models['ml_gru'] = model
                model.save(str(self.model_save_path / 'gru'))

            return {
                'success': result.success,
                'train_accuracy': result.train_accuracy,
                'val_accuracy': result.val_accuracy,
                'message': result.message
            }

        except ImportError as e:
            return {'success': False, 'error': f'PyTorch not installed: {str(e)}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _train_transformer(self, **params) -> Dict:
        """Train Transformer model."""
        try:
            from ml_models.transformer_predictor import TransformerPredictor

            model = TransformerPredictor(
                sequence_length=params.get('sequence_length', 50),
                d_model=params.get('d_model', 64)
            )

            result = model.train(
                self.df,
                epochs=params.get('epochs', 50),
                batch_size=params.get('batch_size', 32)
            )

            if result.success:
                self.models['ml_transformer'] = model
                model.save(str(self.model_save_path / 'transformer'))

            return {
                'success': result.success,
                'train_accuracy': result.train_accuracy,
                'val_accuracy': result.val_accuracy,
                'message': result.message
            }

        except ImportError as e:
            return {'success': False, 'error': f'PyTorch not installed: {str(e)}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _train_rl(self, algorithm: str, **params) -> Dict:
        """Train RL agent."""
        try:
            from ml_models.rl_trader import RLTrader

            model = RLTrader(
                algorithm=algorithm,
                policy='MlpPolicy'
            )

            result = model.train(
                self.df,
                total_timesteps=params.get('total_timesteps', 50000)
            )

            if result.success:
                model_id = f'ml_rl_{algorithm.lower()}'
                self.models[model_id] = model
                model.save(str(self.model_save_path / f'rl_{algorithm.lower()}'))

            return {
                'success': result.success,
                'message': result.message
            }

        except ImportError as e:
            return {'success': False, 'error': f'stable-baselines3 not installed: {str(e)}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def get_trained_models(self) -> Dict[str, Any]:
        """Get dictionary of trained models."""
        return self.models.copy()

    def predict_all(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Get predictions from all trained models.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary mapping model_id to signal Series
        """
        predictions = {}

        for model_id, model in self.models.items():
            try:
                result = model.predict(df)
                predictions[model_id] = result.signals
            except Exception as e:
                print(f"Prediction failed for {model_id}: {e}")

        return predictions

    def get_model_summary(self) -> Dict[str, Dict]:
        """Get summary of all trained models."""
        summary = {}
        for model_id, model in self.models.items():
            summary[model_id] = model.get_model_summary()
        return summary


def run_ml_training(df: pd.DataFrame, models_to_train: List[str],
                    status_callback: Optional[Callable] = None,
                    **params) -> Dict:
    """
    Main entry point for ML training.

    Args:
        df: DataFrame with OHLCV data
        models_to_train: List of model IDs to train
        status_callback: Optional callback for status updates
        **params: Training parameters

    Returns:
        Training results dictionary
    """
    optimizer = MLOptimizer(df)
    results = optimizer.train_models(models_to_train, status_callback, **params)

    # Store models globally
    global trained_ml_models
    trained_ml_models.update(optimizer.get_trained_models())

    return results


def get_available_models() -> Dict[str, str]:
    """Get list of available ML models."""
    return MLOptimizer.AVAILABLE_MODELS.copy()


def get_ml_status() -> Dict:
    """Get current ML training status."""
    global ml_status
    return ml_status.to_dict()
