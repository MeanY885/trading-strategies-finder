"""
Trading Environment for Reinforcement Learning
===============================================
Gymnasium-compatible environment for training RL trading agents.
Inspired by FreqAI's Base5ActionRLEnvironment.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional, List

from .feature_engineering import FeatureEngineer, FeatureConfig

# Try to import gymnasium
try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


if HAS_GYM:
    class TradingEnvironment(gym.Env):
        """
        Custom Gymnasium environment for trading.

        Actions:
        - 0: Hold (do nothing)
        - 1: Enter Long
        - 2: Enter Short
        - 3: Exit Long (close long position)
        - 4: Exit Short (close short position)

        Observations:
        - Normalized technical indicators
        - Current position (-1, 0, 1)
        - Unrealized PnL
        - Bars in current position

        Rewards:
        - Based on realized and unrealized PnL
        - Penalties for invalid actions and long holding times
        """

        metadata = {'render_modes': ['human']}

        def __init__(self, df: pd.DataFrame, initial_balance: float = 10000,
                     risk_per_trade: float = 0.02, max_position_bars: int = 50,
                     feature_config: Optional[FeatureConfig] = None):
            """
            Initialize the trading environment.

            Args:
                df: DataFrame with OHLCV data
                initial_balance: Starting capital
                risk_per_trade: Fraction of capital to risk per trade
                max_position_bars: Max bars to hold a position (penalty after)
                feature_config: Configuration for feature engineering
            """
            super().__init__()

            self.df = df.copy()
            self.initial_balance = initial_balance
            self.risk_per_trade = risk_per_trade
            self.max_position_bars = max_position_bars

            # Create features
            self.feature_engineer = FeatureEngineer(feature_config or FeatureConfig())
            self.features = self.feature_engineer.create_features(df, fit=True)

            # Align features with original dataframe
            common_idx = self.features.index.intersection(df.index)
            self.features = self.features.loc[common_idx]
            self.df = self.df.loc[common_idx].reset_index(drop=True)
            self.features = self.features.reset_index(drop=True)

            self.n_features = len(self.features.columns)
            self.n_steps = len(self.df)

            # Action space: 5 discrete actions
            self.action_space = spaces.Discrete(5)

            # Observation space: features + position info
            # Features are normalized, position is one-hot, pnl is normalized
            obs_dim = self.n_features + 3 + 2  # features + position_one_hot + pnl_info
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )

            # Initialize state
            self.reset()

        def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
            """Reset the environment to initial state."""
            super().reset(seed=seed)

            self.current_step = 0
            self.balance = self.initial_balance
            self.position = 0  # -1 = short, 0 = flat, 1 = long
            self.entry_price = 0.0
            self.position_size = 0.0
            self.bars_in_position = 0
            self.total_trades = 0
            self.winning_trades = 0
            self.total_pnl = 0.0

            # Trade history
            self.trades = []

            return self._get_observation(), {}

        def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
            """
            Execute one step in the environment.

            Args:
                action: Action to take (0-4)

            Returns:
                observation, reward, terminated, truncated, info
            """
            # Get current price
            current_price = self.df.iloc[self.current_step]['close']

            # Track previous values for reward calculation
            prev_pnl = self._calculate_unrealized_pnl(current_price)

            # Execute action
            reward = 0.0
            invalid_action = False

            if action == 0:  # Hold
                pass

            elif action == 1:  # Enter Long
                if self.position == 0:  # Only if flat
                    self._enter_position(1, current_price)
                else:
                    invalid_action = True

            elif action == 2:  # Enter Short
                if self.position == 0:  # Only if flat
                    self._enter_position(-1, current_price)
                else:
                    invalid_action = True

            elif action == 3:  # Exit Long
                if self.position == 1:  # Only if long
                    reward += self._exit_position(current_price)
                else:
                    invalid_action = True

            elif action == 4:  # Exit Short
                if self.position == -1:  # Only if short
                    reward += self._exit_position(current_price)
                else:
                    invalid_action = True

            # Calculate reward
            reward += self._calculate_reward(current_price, prev_pnl, invalid_action)

            # Update position tracking
            if self.position != 0:
                self.bars_in_position += 1

            # Move to next step
            self.current_step += 1

            # Check if done
            terminated = self.current_step >= self.n_steps - 1
            truncated = False

            # Force close position at end
            if terminated and self.position != 0:
                self._exit_position(current_price)

            obs = self._get_observation()
            info = self._get_info()

            return obs, reward, terminated, truncated, info

        def _enter_position(self, direction: int, price: float) -> None:
            """Enter a new position."""
            self.position = direction
            self.entry_price = price
            self.position_size = (self.balance * self.risk_per_trade) / price
            self.bars_in_position = 0

        def _exit_position(self, price: float) -> float:
            """Exit current position and return PnL."""
            if self.position == 0:
                return 0.0

            # Calculate PnL
            if self.position == 1:  # Long
                pnl = (price - self.entry_price) * self.position_size
            else:  # Short
                pnl = (self.entry_price - price) * self.position_size

            # Update stats
            self.balance += pnl
            self.total_pnl += pnl
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1

            # Record trade
            self.trades.append({
                'entry_price': self.entry_price,
                'exit_price': price,
                'direction': self.position,
                'pnl': pnl,
                'bars_held': self.bars_in_position
            })

            # Reset position
            self.position = 0
            self.entry_price = 0.0
            self.position_size = 0.0
            self.bars_in_position = 0

            return pnl

        def _calculate_unrealized_pnl(self, current_price: float) -> float:
            """Calculate unrealized PnL for current position."""
            if self.position == 0:
                return 0.0

            if self.position == 1:  # Long
                return (current_price - self.entry_price) * self.position_size
            else:  # Short
                return (self.entry_price - current_price) * self.position_size

        def _calculate_reward(self, current_price: float, prev_pnl: float,
                             invalid_action: bool) -> float:
            """
            Calculate reward for the current step.

            Reward design inspired by FreqAI:
            - Small continuous rewards based on PnL change
            - Penalties for invalid actions
            - Penalties for holding too long
            """
            reward = 0.0

            # Penalty for invalid action
            if invalid_action:
                reward -= 1.0
                return reward

            # Reward based on unrealized PnL change
            current_pnl = self._calculate_unrealized_pnl(current_price)
            pnl_change = current_pnl - prev_pnl

            # Normalize reward to be small
            reward += pnl_change / self.initial_balance * 100

            # Penalty for holding position too long
            if self.position != 0 and self.bars_in_position > self.max_position_bars:
                reward -= 0.01 * (self.bars_in_position - self.max_position_bars)

            return reward

        def _get_observation(self) -> np.ndarray:
            """Get current observation."""
            if self.current_step >= len(self.features):
                # Return zeros if past the end
                return np.zeros(self.observation_space.shape[0], dtype=np.float32)

            # Feature values
            features = self.features.iloc[self.current_step].values

            # Position one-hot encoding
            position_one_hot = np.zeros(3)
            position_one_hot[self.position + 1] = 1  # -1->0, 0->1, 1->2

            # PnL info
            current_price = self.df.iloc[self.current_step]['close']
            unrealized_pnl = self._calculate_unrealized_pnl(current_price)
            normalized_pnl = unrealized_pnl / self.initial_balance
            normalized_bars = self.bars_in_position / self.max_position_bars

            # Concatenate all
            obs = np.concatenate([
                features,
                position_one_hot,
                [normalized_pnl, normalized_bars]
            ]).astype(np.float32)

            return obs

        def _get_info(self) -> Dict[str, Any]:
            """Get info dictionary."""
            return {
                'balance': self.balance,
                'position': self.position,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': self.winning_trades / max(1, self.total_trades),
                'total_pnl': self.total_pnl,
                'current_step': self.current_step
            }

        def render(self) -> None:
            """Render the environment (print current state)."""
            info = self._get_info()
            print(f"Step: {info['current_step']}/{self.n_steps} | "
                  f"Balance: ${info['balance']:.2f} | "
                  f"Position: {info['position']} | "
                  f"Trades: {info['total_trades']} | "
                  f"Win Rate: {info['win_rate']:.1%}")


class SimpleTradingEnvironment(gym.Env):
    """
    Simplified 3-action trading environment.

    Actions:
    - 0: Hold/Flat
    - 1: Long
    - 2: Short

    Simpler than TradingEnvironment, better for initial training.
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance: float = 10000,
                 feature_config: Optional[FeatureConfig] = None):
        super().__init__()

        self.df = df.copy()
        self.initial_balance = initial_balance

        # Create features
        self.feature_engineer = FeatureEngineer(feature_config or FeatureConfig())
        self.features = self.feature_engineer.create_features(df, fit=True)

        # Align
        common_idx = self.features.index.intersection(df.index)
        self.features = self.features.loc[common_idx].reset_index(drop=True)
        self.df = self.df.loc[common_idx].reset_index(drop=True)

        self.n_features = len(self.features.columns)
        self.n_steps = len(self.df)

        # 3 actions: hold, long, short
        self.action_space = spaces.Discrete(3)

        # Observation: features + current position
        obs_dim = self.n_features + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0  # 0=flat, 1=long, 2=short
        self.balance = self.initial_balance
        self.entry_price = 0.0
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        current_price = self.df.iloc[self.current_step]['close']
        next_price = self.df.iloc[min(self.current_step + 1, self.n_steps - 1)]['close']

        reward = 0.0

        # Calculate reward based on position and price change
        price_change = (next_price - current_price) / current_price

        if action == 1:  # Long
            reward = price_change * 100  # Scale reward
        elif action == 2:  # Short
            reward = -price_change * 100

        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1

        return self._get_observation(), reward, terminated, False, {}

    def _get_observation(self) -> np.ndarray:
        if self.current_step >= len(self.features):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        features = self.features.iloc[self.current_step].values
        obs = np.concatenate([features, [self.position]]).astype(np.float32)
        return obs
