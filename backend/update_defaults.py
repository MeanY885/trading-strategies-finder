"""
Update pinescript_generator.py with optimized defaults from JSON
"""
import json
import re
from pathlib import Path

# Load optimized defaults
defaults_path = Path(__file__).parent / "optimized_defaults.json"
with open(defaults_path) as f:
    data = json.load(f)

strategies = data['strategies']

# Map strategy names to their generator function prefixes
# (only strategies with positive scores)
strategy_generators = {
    'cvd_divergence': '_generate_cvd_divergence',
    'nadaraya_watson_reversion': '_generate_nadaraya_watson',
    'divergence_3wave': '_generate_divergence_3wave',
    'ema_crossover': '_generate_ema_crossover',
    'supertrend_follow': '_generate_supertrend',
    'macd_trend': '_generate_macd_trend',
    'donchian_breakout': '_generate_donchian_breakout',
    'stiff_surge_v1': '_generate_stiff_surge',
    'stiff_surge_v2': '_generate_stiff_surge_v2',
    'range_filter_adx': '_generate_range_filter',
    'supertrend_confluence': '_generate_supertrend_confluence',
    'bb_rsi_classic': '_generate_bb_rsi_classic',
    'bb_rsi_tight': '_generate_bb_rsi_tight',
    'bb_stoch': '_generate_bb_stoch',
    'keltner_rsi': '_generate_keltner_rsi',
    'z_score_reversion': '_generate_z_score_reversion',
    'stoch_extreme': '_generate_stoch_extreme',
    'rsi_extreme': '_generate_rsi_extreme',
    'williams_r': '_generate_williams_r',
    'cci_extreme': '_generate_cci_extreme',
    'bb_squeeze_breakout': '_generate_bb_squeeze',
    'adx_trend': '_generate_adx_di_trend',
    'squeeze_momentum_breakout': '_generate_squeeze_momentum',
    'connors_rsi_extreme': '_generate_connors_rsi',
    'order_block_bounce': '_generate_order_block_bounce',
    'fvg_fill': '_generate_fvg_fill',
    'pct_drop_buy': '_generate_pct_drop_buy',
    'simple_sma_cross': '_generate_simple_sma_cross',
    'consecutive_candles_reversal': '_generate_consecutive_candles',
    'simple_rsi_extreme': '_generate_simple_rsi',
    'range_breakout_simple': '_generate_range_breakout',
    'engulfing_pattern': '_generate_engulfing_pattern',
}

print("=" * 70)
print("OPTIMIZED PARAMETER DEFAULTS")
print("=" * 70)
print("\nCopy these values to update pinescript_generator.py\n")

for strategy_name, func_name in sorted(strategy_generators.items()):
    if strategy_name in strategies:
        strat_data = strategies[strategy_name]
        params = strat_data['params']
        score = strat_data['score']

        if score > 0:  # Only show strategies that worked
            print(f"\n{func_name} ({strategy_name}) - Score: {score:.2f}")
            print("-" * 50)

            # Standard risk params
            sl = params.get('sl_atr_mult', 2.0)
            tp = params.get('tp_ratio', 1.5)
            print(f"  sl_atr_mult: {sl}")
            print(f"  tp_ratio: {tp}")

            # Strategy-specific params
            for key, val in params.items():
                if key not in ['sl_atr_mult', 'tp_ratio']:
                    print(f"  {key}: {val}")

# Generate the actual code updates needed
print("\n" + "=" * 70)
print("CODE UPDATES REQUIRED")
print("=" * 70)

code_updates = []
for strategy_name, func_name in strategy_generators.items():
    if strategy_name in strategies:
        params = strategies[strategy_name]['params']
        score = strategies[strategy_name]['score']

        if score > 0:
            sl = params.get('sl_atr_mult', 2.0)
            tp = params.get('tp_ratio', 1.5)
            code_updates.append(f"# {func_name}: sl_atr_mult={sl}, tp_ratio={tp}")

print("\nKey updates (sl_atr_mult, tp_ratio):")
for update in code_updates:
    print(update)
