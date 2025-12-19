"""
Reinforcement Learning Trader (stable-baselines3)
=================================================
PPO and DQN agents for automated trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

from .base_predictor import BasePredictor, PredictionResult, TrainingResult
from .feature_engineering import FeatureEngineer, FeatureConfig

# Try to import stable-baselines3
try:
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.callbacks import BaseCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

# Try to import gymnasium
try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


class TrainingCallback(BaseCallback):
    """Callback for tracking training progress."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = []

    def _on_step(self) -> bool:
        # Track rewards
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
        return True


class RLTrader(BasePredictor):
    """
    Reinforcement Learning trader using stable-baselines3.

    Supports:
    - PPO (Proximal Policy Optimization) - stable, good for continuous tasks
    - DQN (Deep Q-Network) - good for discrete action spaces
    - A2C (Advantage Actor-Critic) - faster but less stable than PPO
    """

    SUPPORTED_ALGORITHMS = ['PPO', 'DQN', 'A2C']

    def __init__(self, algorithm: str = 'PPO', policy: str = 'MlpPolicy',
                 feature_config: Optional[FeatureConfig] = None):
        """
        Initialize the RL trader.

        Args:
            algorithm: RL algorithm ('PPO', 'DQN', 'A2C')
            policy: Policy network type ('MlpPolicy')
            feature_config: Configuration for feature engineering
        """
        if not HAS_SB3:
            raise ImportError("stable-baselines3 not installed. Run: pip install stable-baselines3")
        if not HAS_GYM:
            raise ImportError("gymnasium not installed. Run: pip install gymnasium")

        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"algorithm must be one of {self.SUPPORTED_ALGORITHMS}")

        super().__init__(
            name=f"RL Agent ({algorithm})",
            model_type=f"ml_rl_{algorithm.lower()}"
        )

        self.algorithm = algorithm
        self.policy = policy
        self.feature_config = feature_config or FeatureConfig()
        self.model = None
        self.env = None

        self.training_config = {
            'algorithm': algorithm,
            'policy': policy
        }

    def train(self, df: pd.DataFrame, total_timesteps: int = 50000,
              learning_rate: float = 0.0003, **kwargs) -> TrainingResult:
        """
        Train the RL agent.

        Args:
            df: DataFrame with OHLCV data
            total_timesteps: Total training timesteps
            learning_rate: Learning rate
            **kwargs: Additional algorithm-specific parameters

        Returns:
            TrainingResult with metrics
        """
        try:
            from .trading_env import SimpleTradingEnvironment

            # Create environment
            self.env = SimpleTradingEnvironment(df, feature_config=self.feature_config)

            # Create callback for tracking
            callback = TrainingCallback()

            # Initialize model based on algorithm
            if self.algorithm == 'PPO':
                self.model = PPO(
                    self.policy,
                    self.env,
                    learning_rate=learning_rate,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    verbose=0,
                    **kwargs
                )
            elif self.algorithm == 'DQN':
                self.model = DQN(
                    self.policy,
                    self.env,
                    learning_rate=learning_rate,
                    buffer_size=10000,
                    learning_starts=1000,
                    batch_size=32,
                    gamma=0.99,
                    exploration_fraction=0.1,
                    exploration_final_eps=0.05,
                    verbose=0,
                    **kwargs
                )
            elif self.algorithm == 'A2C':
                self.model = A2C(
                    self.policy,
                    self.env,
                    learning_rate=learning_rate,
                    n_steps=5,
                    gamma=0.99,
                    gae_lambda=0.95,
                    verbose=0,
                    **kwargs
                )

            # Train
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=False
            )

            # Evaluate
            mean_reward = self._evaluate_policy(df)

            self.is_trained = True
            self.metadata['total_timesteps'] = total_timesteps
            self.metadata['mean_reward'] = mean_reward

            return TrainingResult(
                success=True,
                message=f"Training complete. Mean reward: {mean_reward:.2f}"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return TrainingResult(
                success=False,
                message=f"Training failed: {str(e)}"
            )

    def _evaluate_policy(self, df: pd.DataFrame, n_episodes: int = 5) -> float:
        """Evaluate the trained policy."""
        from .trading_env import SimpleTradingEnvironment

        eval_env = SimpleTradingEnvironment(df, feature_config=self.feature_config)
        total_rewards = []

        for _ in range(n_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                done = terminated or truncated

            total_rewards.append(episode_reward)

        return np.mean(total_rewards)

    def predict(self, df: pd.DataFrame) -> PredictionResult:
        """
        Generate trading signals from the trained RL agent.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            PredictionResult with signals (1=long, -1=short, 0=hold)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        from .trading_env import SimpleTradingEnvironment

        # Create evaluation environment
        eval_env = SimpleTradingEnvironment(df, feature_config=self.feature_config)

        # Run through environment to get actions
        obs, _ = eval_env.reset()
        signals = []

        for i in range(len(df)):
            if i < len(eval_env.features):
                action, _ = self.model.predict(obs, deterministic=True)
                signals.append(action)
                obs, _, terminated, truncated, _ = eval_env.step(action)
                if terminated or truncated:
                    break
            else:
                signals.append(0)

        # Pad signals to match df length
        while len(signals) < len(df):
            signals.append(0)

        # Convert actions to signals:
        # 0 (hold) -> 0
        # 1 (long) -> 1
        # 2 (short) -> -1
        signal_map = {0: 0, 1: 1, 2: -1}
        signals = [signal_map.get(s, 0) for s in signals]

        return PredictionResult(
            signals=pd.Series(signals, index=df.index).astype(int),
            confidence=None
        )

    def get_decision_rules(self) -> List[Dict[str, Any]]:
        """
        Extract policy rules from the RL agent.

        For neural network policies, this is difficult to interpret.
        Returns a placeholder description.
        """
        return [{
            'indicator': 'rl_policy',
            'operator': 'neural_network',
            'threshold': 0,
            'importance': 1.0,
            'description': f'{self.algorithm} policy network'
        }]

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save(str(path / 'model'))

        # Save metadata
        self._save_metadata(str(path))

    def load(self, path: str) -> None:
        """Load model from disk."""
        path = Path(path)

        # Load metadata first to get algorithm type
        self._load_metadata(str(path))

        # Load model based on algorithm
        model_path = str(path / 'model')

        if self.algorithm == 'PPO':
            self.model = PPO.load(model_path)
        elif self.algorithm == 'DQN':
            self.model = DQN.load(model_path)
        elif self.algorithm == 'A2C':
            self.model = A2C.load(model_path)

        self.is_trained = True


# Convenience classes for specific algorithms
class PPOTrader(RLTrader):
    """PPO-based trading agent."""

    def __init__(self, **kwargs):
        super().__init__(algorithm='PPO', **kwargs)


class DQNTrader(RLTrader):
    """DQN-based trading agent."""

    def __init__(self, **kwargs):
        super().__init__(algorithm='DQN', **kwargs)


class A2CTrader(RLTrader):
    """A2C-based trading agent."""

    def __init__(self, **kwargs):
        super().__init__(algorithm='A2C', **kwargs)
