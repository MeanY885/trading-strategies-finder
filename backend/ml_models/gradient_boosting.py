"""
Gradient Boosting Models for Trading
=====================================
Implements XGBoost, LightGBM, and CatBoost for price direction prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pickle
import json

from .base_predictor import BasePredictor, PredictionResult, TrainingResult
from .feature_engineering import FeatureEngineer, FeatureConfig

# Try to import gradient boosting libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report


class GradientBoostingPredictor(BasePredictor):
    """
    Gradient Boosting predictor supporting XGBoost, LightGBM, and CatBoost.

    Can be used for:
    - Classification: Predict direction (up/down/neutral)
    - Regression: Predict actual return

    Features are automatically generated from OHLCV data using FeatureEngineer.
    """

    SUPPORTED_MODELS = {
        'xgboost': HAS_XGBOOST,
        'lightgbm': HAS_LIGHTGBM,
        'catboost': HAS_CATBOOST
    }

    def __init__(self, model_type: str = 'lightgbm', task: str = 'classification',
                 feature_config: Optional[FeatureConfig] = None):
        """
        Initialize the gradient boosting predictor.

        Args:
            model_type: 'xgboost', 'lightgbm', or 'catboost'
            task: 'classification' or 'regression'
            feature_config: Configuration for feature engineering
        """
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {list(self.SUPPORTED_MODELS.keys())}")

        if not self.SUPPORTED_MODELS[model_type]:
            raise ImportError(f"{model_type} is not installed. Please install it first.")

        super().__init__(
            name=f"{model_type.upper()} {'Classifier' if task == 'classification' else 'Regressor'}",
            model_type=f"ml_{model_type}"
        )

        self.gb_type = model_type
        self.task = task
        self.feature_engineer = FeatureEngineer(feature_config or FeatureConfig())
        self.confidence_threshold = 0.55  # Min probability to generate signal
        self.training_config = {
            'model_type': model_type,
            'task': task
        }

    def train(self, df: pd.DataFrame, target_horizon: int = 1,
              target_threshold: float = 0.001,  # 0.1% move for signal
              test_size: float = 0.2, random_state: int = 42,
              n_estimators: int = 100, max_depth: int = 6,
              learning_rate: float = 0.1, **kwargs) -> TrainingResult:
        """
        Train the gradient boosting model.

        Args:
            df: DataFrame with OHLCV data
            target_horizon: Candles ahead to predict
            target_threshold: Minimum % move for signal (ternary classification)
            test_size: Fraction of data for validation
            random_state: Random seed
            n_estimators: Number of trees
            max_depth: Max tree depth
            learning_rate: Learning rate
            **kwargs: Additional model-specific parameters

        Returns:
            TrainingResult with metrics
        """
        try:
            # Create features
            features = self.feature_engineer.create_features(df, fit=True)

            # Create target
            target_type = 'direction' if self.task == 'classification' else 'return'
            target = self.feature_engineer.create_target(
                df, target_type=target_type,
                horizon=target_horizon,
                threshold=target_threshold
            )

            # Align features and target
            common_idx = features.index.intersection(target.dropna().index)
            X = features.loc[common_idx]
            y = target.loc[common_idx]

            # Drop remaining NaN
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]

            self.feature_names = X.columns.tolist()

            # Train/validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=random_state, shuffle=False
            )

            # Model parameters
            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'random_state': random_state
            }

            # Train based on model type
            if self.gb_type == 'xgboost':
                self.model = self._train_xgboost(X_train, y_train, X_val, y_val, params, kwargs)
            elif self.gb_type == 'lightgbm':
                self.model = self._train_lightgbm(X_train, y_train, X_val, y_val, params, kwargs)
            elif self.gb_type == 'catboost':
                self.model = self._train_catboost(X_train, y_train, X_val, y_val, params, kwargs)

            # Evaluate
            train_pred = self._predict_raw(X_train)
            val_pred = self._predict_raw(X_val)

            if self.task == 'classification':
                train_acc = accuracy_score(y_train, train_pred)
                val_acc = accuracy_score(y_val, val_pred)
                train_loss = 1 - train_acc
                val_loss = 1 - val_acc
            else:
                from sklearn.metrics import mean_squared_error
                train_loss = mean_squared_error(y_train, train_pred)
                val_loss = mean_squared_error(y_val, val_pred)
                train_acc = 1 - train_loss  # R² approximation
                val_acc = 1 - val_loss

            # Get feature importance
            feature_importance = self._get_feature_importance()

            self.is_trained = True
            self.training_config.update(params)
            self.metadata['train_samples'] = len(X_train)
            self.metadata['val_samples'] = len(X_val)
            self.metadata['target_horizon'] = target_horizon
            self.metadata['target_threshold'] = target_threshold

            return TrainingResult(
                success=True,
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                train_loss=train_loss,
                val_loss=val_loss,
                feature_importance=feature_importance,
                best_params=params,
                message=f"Training complete. Val accuracy: {val_acc:.4f}"
            )

        except Exception as e:
            return TrainingResult(
                success=False,
                message=f"Training failed: {str(e)}"
            )

    def _train_xgboost(self, X_train, y_train, X_val, y_val, params, kwargs):
        """Train XGBoost model."""
        objective = 'multi:softmax' if self.task == 'classification' else 'reg:squarederror'

        model_params = {
            'objective': objective,
            'max_depth': params['max_depth'],
            'learning_rate': params['learning_rate'],
            'n_estimators': params['n_estimators'],
            'random_state': params['random_state'],
            'use_label_encoder': False,
            'eval_metric': 'mlogloss' if self.task == 'classification' else 'rmse',
            **kwargs
        }

        if self.task == 'classification':
            # XGBoost requires labels starting from 0, so remap -1,0,1 to 0,1,2
            # Store the label mapping for prediction
            unique_labels = np.sort(np.unique(y_train))
            self._xgb_label_map = {orig: new for new, orig in enumerate(unique_labels)}
            self._xgb_label_map_inverse = {new: orig for orig, new in self._xgb_label_map.items()}

            y_train_mapped = pd.Series(y_train).map(self._xgb_label_map).values
            y_val_mapped = pd.Series(y_val).map(self._xgb_label_map).values

            n_classes = len(unique_labels)
            model_params['num_class'] = n_classes

            model = xgb.XGBClassifier(**model_params)
            model.fit(
                X_train, y_train_mapped,
                eval_set=[(X_val, y_val_mapped)],
                verbose=False
            )
        else:
            model = xgb.XGBRegressor(**model_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

        return model

    def _train_lightgbm(self, X_train, y_train, X_val, y_val, params, kwargs):
        """Train LightGBM model."""
        model_params = {
            'max_depth': params['max_depth'],
            'learning_rate': params['learning_rate'],
            'n_estimators': params['n_estimators'],
            'random_state': params['random_state'],
            'verbose': -1,
            **kwargs
        }

        if self.task == 'classification':
            model = lgb.LGBMClassifier(**model_params)
        else:
            model = lgb.LGBMRegressor(**model_params)

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        return model

    def _train_catboost(self, X_train, y_train, X_val, y_val, params, kwargs):
        """Train CatBoost model."""
        model_params = {
            'depth': params['max_depth'],
            'learning_rate': params['learning_rate'],
            'iterations': params['n_estimators'],
            'random_seed': params['random_state'],
            'verbose': False,
            **kwargs
        }

        if self.task == 'classification':
            model = cb.CatBoostClassifier(**model_params)
        else:
            model = cb.CatBoostRegressor(**model_params)

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )

        return model

    def _predict_raw(self, X):
        """Get raw predictions from model."""
        predictions = self.model.predict(X)

        # If XGBoost classifier, map predictions back to original labels
        if self.gb_type == 'xgboost' and self.task == 'classification':
            if hasattr(self, '_xgb_label_map_inverse'):
                predictions = np.array([self._xgb_label_map_inverse.get(p, p) for p in predictions])

        return predictions

    def _predict_proba(self, X):
        """Get probability predictions from model."""
        if self.task == 'classification':
            return self.model.predict_proba(X)
        else:
            return None

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """
        Generate trading signals from the trained model.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            PredictionResult with signals (1=long, -1=short, 0=hold)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")

        # Create features
        features = self.feature_engineer.create_features(df, fit=False)

        # Ensure all required features are present
        missing = set(self.feature_names) - set(features.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        X = features[self.feature_names]

        # Get predictions
        raw_pred = self._predict_raw(X)

        if self.task == 'classification':
            # Get probabilities for confidence filtering
            proba = self._predict_proba(X)

            if proba is not None:
                # Max probability as confidence
                confidence = np.max(proba, axis=1)
                confidence = pd.Series(confidence, index=X.index)

                # Only signal if confidence exceeds threshold
                signals = pd.Series(raw_pred, index=X.index)
                signals[confidence < self.confidence_threshold] = 0
            else:
                signals = pd.Series(raw_pred, index=X.index)
                confidence = None
        else:
            # Regression: convert to signals based on predicted return
            signals = pd.Series(0, index=X.index)
            signals[raw_pred > 0.001] = 1  # Long if predicted return > 0.1%
            signals[raw_pred < -0.001] = -1  # Short if predicted return < -0.1%
            confidence = pd.Series(np.abs(raw_pred), index=X.index)

        return PredictionResult(
            signals=signals.astype(int),
            confidence=confidence,
            raw_output=raw_pred
        )

    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained or self.model is None:
            return {}

        if self.gb_type == 'xgboost':
            importance = self.model.feature_importances_
        elif self.gb_type == 'lightgbm':
            importance = self.model.feature_importances_
        elif self.gb_type == 'catboost':
            importance = self.model.get_feature_importance()

        return dict(zip(self.feature_names, importance))

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance sorted by value."""
        importance = self._get_feature_importance()
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def get_decision_rules(self) -> List[Dict[str, Any]]:
        """
        Extract decision rules from the model for Pine Script generation.

        Returns top features and their split thresholds.
        """
        if not self.is_trained or self.model is None:
            return []

        rules = []
        importance = self.get_feature_importance()

        # Get top 10 most important features
        top_features = list(importance.keys())[:10]

        for feature in top_features:
            # For gradient boosting, we use feature importance
            # and typical thresholds based on feature type
            rule = {
                'indicator': feature,
                'importance': importance.get(feature, 0),
                'operator': '<' if 'rsi' in feature.lower() or 'stoch' in feature.lower() else '>',
                'threshold': 0.0  # Will be refined during Pine Script generation
            }
            rules.append(rule)

        return rules

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = path / 'model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Save feature engineer scaler
        scaler_path = path / 'scaler.pkl'
        self.feature_engineer.save_scaler(str(scaler_path))

        # Save metadata
        self._save_metadata(str(path))

    def load(self, path: str) -> None:
        """Load model from disk."""
        path = Path(path)

        # Load model
        model_path = path / 'model.pkl'
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load scaler
        scaler_path = path / 'scaler.pkl'
        if scaler_path.exists():
            self.feature_engineer.load_scaler(str(scaler_path))

        # Load metadata
        self._load_metadata(str(path))
        self.is_trained = True


# Convenience classes for each model type
class XGBoostPredictor(GradientBoostingPredictor):
    """XGBoost classifier/regressor."""

    def __init__(self, task: str = 'classification', **kwargs):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        super().__init__(model_type='xgboost', task=task, **kwargs)


class LightGBMPredictor(GradientBoostingPredictor):
    """LightGBM classifier/regressor."""

    def __init__(self, task: str = 'classification', **kwargs):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        super().__init__(model_type='lightgbm', task=task, **kwargs)


class CatBoostPredictor(GradientBoostingPredictor):
    """CatBoost classifier/regressor."""

    def __init__(self, task: str = 'classification', **kwargs):
        if not HAS_CATBOOST:
            raise ImportError("CatBoost not installed. Run: pip install catboost")
        super().__init__(model_type='catboost', task=task, **kwargs)
