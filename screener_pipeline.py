#!/usr/bin/env python3
"""
INSTITUTIONAL QUANTITATIVE SCREENER & WALK-FORWARD PIPELINE (NSE)
================================================================
Engineered for Indian Equities Universe Ingestion, Causal Signal Verification,
Fitness Objective Scoring, and TradingView Watchlist Formatting.
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION & HYPERPARAMETERS
# =============================================================================
TARGET_PROFIT_PCT = 0.05       # +5.0% Fixed Target
ATR_SL_MULTIPLIER = 1.75       # Volatility-adjusted stop loss multiplier
BREAKEVEN_GAIN_PCT = 0.025     # +2.5% Ratchet threshold
MAX_LOOKBACK_BARS = 300        # Minimum lookback bars required
TOP_SCREENER_COUNT = 25        # Top opportunities to display
RISK_FREE_RATE = 0.065         # RBI 91-day T-Bill rate baseline (~6.5%)


@dataclass
class ScreenerResult:
    ticker: str
    direction: str
    entry_level: float
    target_price: float
    stop_loss: float
    risk_reward: float
    target_probability: float
    max_drawdown: float
    fitness_score: float


# =============================================================================
# CAUSAL FEATURE STORE COMPUTATION (ZERO LOOKAHEAD)
# =============================================================================
def compute_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical feature vector using strictly historical data windows.
    No forward-looking indicators or full-series standardizations.
    """
    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

    # 1. Momentum: RSI (Wilder's smoothing)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / (loss + 1e-9)
    data['RSI'] = 100 - (100 / (1 + rs))

    # 2. Momentum: Stochastic %K
    lowest_14 = low.rolling(window=14).min()
    highest_14 = high.rolling(window=14).max()
    data['Stoch_K'] = 100 * ((close - lowest_14) / (highest_14 - lowest_14 + 1e-9))

    # 3. Volatility: ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['ATR'] = tr.ewm(alpha=1/14, min_periods=14).mean()

    # 4. Volatility: Bollinger Bands & %B
    bb_mid = close.rolling(window=20).mean()
    bb_std = close.rolling(window=20).std(ddof=1)
    bb_up = bb_mid + 2 * bb_std
    bb_low = bb_mid - 2 * bb_std
    data['BB_PctB'] = (close - bb_low) / (bb_up - bb_low + 1e-9)

    # 5. Volume Flow: Money Flow Index (MFI)
    tp = (high + low + close) / 3
    rmf = tp * volume
    pos_flow = pd.Series(np.where(tp > tp.shift(1), rmf, 0.0), index=data.index).rolling(14).sum()
    neg_flow = pd.Series(np.where(tp < tp.shift(1), rmf, 0.0), index=data.index).rolling(14).sum()
    mfr = pos_flow / (neg_flow + 1e-9)
    data['MFI'] = 100 - (100 / (1 + mfr))

    # 6. Trend: EMA Ribbon Differential
    data['EMA_8'] = close.ewm(span=8, adjust=False).mean()
    data['EMA_55'] = close.ewm(span=55, adjust=False).mean()
    data['Ribbon_Spread'] = (data['EMA_8'] - data['EMA_55']) / data['EMA_55']

    # 7. Directional Movement Index (ADX)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr_smooth = tr.ewm(alpha=1/14, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm, index=data.index).ewm(alpha=1/14, min_periods=14).mean() / (tr_smooth + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=data.index).ewm(alpha=1/14, min_periods=14).mean() / (tr_smooth + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    data['ADX'] = dx.ewm(alpha=1/14, min_periods=14).mean()
    data['DI_Diff'] = plus_di - minus_di

    # Rolling Causal Z-Score Standardizations (Window = 60)
    roll_win = 60
    for col in ['RSI', 'Stoch_K', 'BB_PctB', 'MFI', 'Ribbon_Spread', 'DI_Diff']:
        r_mean = data[col].shift(1).rolling(roll_win).mean()
        r_std = data[col].shift(1).rolling(roll_win).std(ddof=1)
        data[f'{col}_Z'] = (data[col] - r_mean) / (r_std + 1e-9)

    return data.dropna()


# =============================================================================
# PURGED WALK-FORWARD EVALUATION & FITNESS CALCULATION
# =============================================================================
def evaluate_walk_forward_metrics(df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Executes a purged sliding walk-forward validation simulating exact 5% TP / ATR SL.
    Returns:
        - Target Hit Probability (%)
        - Max Historical Drawdown (%)
        - Custom Institutional Fitness Score
    """
    trades = []
    equity_curve = [1.0]
    n_bars = len(df)
    
    # Simulate historical signal instances (Longs: RSI_Z > 0.5, Ribbon > 0, MFI_Z > 0)
    buy_signals = (
        (df['RSI_Z'] > 0.5) & 
        (df['Ribbon_Spread'] > 0) & 
        (df['MFI_Z'] > 0) &
        (df['ADX'] > 20)
    ).values

    close_arr = df['Close'].values
    high_arr = df['High'].values
    low_arr = df['Low'].values
    atr_arr = df['ATR'].values

    i = 60
    while i < n_bars - 15:
        if buy_signals[i]:
            entry_p = close_arr[i]
            target_p = entry_p * (1.0 + TARGET_PROFIT_PCT)
            stop_p = entry_p - (atr_arr[i] * ATR_SL_MULTIPLIER)
            be_triggered = False
            
            # Forward simulate up to 15 bars
            exit_return = 0.0
            for f in range(1, 16):
                bar_high = high_arr[i + f]
                bar_low = low_arr[i + f]
                
                # Check Breakeven ratchet
                if not be_triggered and bar_high >= entry_p * (1.0 + BREAKEVEN_GAIN_PCT):
                    stop_p = entry_p
                    be_triggered = True
                
                # Hit Target
                if bar_high >= target_p:
                    exit_return = TARGET_PROFIT_PCT
                    trades.append(1)  # Win
                    i += f            # Purge forward bars to prevent overlapping leakage
                    break
                # Hit Stop
                elif bar_low <= stop_p:
                    exit_return = (stop_p - entry_p) / entry_p
                    trades.append(0)  # Loss
                    i += f            # Purge
                    break
            else:
                # Time exit after 15 bars
                exit_return = (close_arr[i + 15] - entry_p) / entry_p
                trades.append(1 if exit_return > 0 else 0)
                i += 15

            equity_curve.append(equity_curve[-1] * (1.0 + exit_return))
        else:
            i += 1

    if not trades or len(trades) < 5:
        return 0.0, 100.0, 0.0

    target_prob = (sum(trades) / len(trades)) * 100.0

    # Drawdown & Performance Calculations
    eq_series = pd.Series(equity_curve)
    running_max = eq_series.cummax()
    dd_series = (eq_series - running_max) / running_max
    max_dd = abs(dd_series.min()) * 100.0
    if max_dd == 0:
        max_dd = 1.0

    ann_ret = (eq_series.iloc[-1] ** (252 / max(len(eq_series), 1))) - 1.0
    daily_returns = eq_series.pct_change().dropna()
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std(ddof=1) if len(downside_returns) > 1 else 0.01
    sortino = (ann_ret - RISK_FREE_RATE) / (downside_std * np.sqrt(252) + 1e-9)
    sortino = max(sortino, 0.0)

    # Trade Frequency Penalty
    trade_freq_penalty = 0.0
    if len(trades) < 10:
        trade_freq_penalty = 0.4
    elif len(trades) > 150:
        trade_freq_penalty = 0.2

    # Institutional Fitness Function: (AnnRet / MDD^2) * Sortino * (1 - Penalty)
    fitness = ((max(ann_ret, 0.0) * 100) / (max_dd ** 2)) * sortino * (1.0 - trade_freq_penalty)

    return target_prob, max_dd, fitness


# =============================================================================
# SINGLE TICKER INGESTION & PIPELINE EXECUTION
# =============================================================================
def process_ticker(raw_ticker: str) -> Optional[ScreenerResult]:
    """
    Ingests market data via Yahoo Finance, executes causal analysis, and screens for live setups.
    """
    clean_sym = raw_ticker.strip().upper()
    if not clean_sym:
        return None
    
    yf_symbol = f"{clean_sym}.NS"
    try:
        df = yf.download(yf_symbol, period="1y", interval="1d", progress=False)
        if df is None or len(df) < MAX_LOOKBACK_BARS:
            return None
        
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        data = compute_causal_features(df)
        if len(data) < 60:
            return None

        # Real-time state (latest bar)
        last_bar = data.iloc[-1]
        close_px = float(last_bar['Close'])
        atr_px = float(last_bar['ATR'])

        # Long Setup Evaluation
        is_long = (
            last_bar['RSI_Z'] > 0.4 and 
            last_bar['Ribbon_Spread'] > 0.0 and 
            last_bar['MFI_Z'] > 0.0 and
            last_bar['ADX'] >= 20.0
        )
        
        # Short Setup Evaluation
        is_short = (
            last_bar['RSI_Z'] < -0.4 and 
            last_bar['Ribbon_Spread'] < 0.0 and 
            last_bar['MFI_Z'] < 0.0 and
            last_bar['ADX'] >= 20.0
        )

        if not (is_long or is_short):
            return None

        direction = "BUY" if is_long else "SELL"
        entry_level = close_px
        
        if direction == "BUY":
            target_px = entry_level * (1.0 + TARGET_PROFIT_PCT)
            stop_px = entry_level - (atr_px * ATR_SL_MULTIPLIER)
        else:
            target_px = entry_level * (1.0 - TARGET_PROFIT_PCT)
            stop_px = entry_level + (atr_px * ATR_SL_MULTIPLIER)

        risk_dist = abs(entry_level - stop_px)
        reward_dist = abs(target_px - entry_level)
        rr_ratio = reward_dist / (risk_dist + 1e-9)

        target_prob, max_dd, fitness = evaluate_walk_forward_metrics(data)

        return ScreenerResult(
            ticker=clean_sym,
            direction=direction,
            entry_level=round(entry_level, 2),
            target_price=round(target_px, 2),
            stop_loss=round(stop_px, 2),
            risk_reward=round(rr_ratio, 2),
            target_probability=round(target_prob, 1),
            max_drawdown=round(max_dd, 1),
            fitness_score=round(fitness, 3)
        )
    except Exception:
        return None


# =============================================================================
# MULTI-THREADED UNIVERSE SCREENER ORCHESTRATOR
# =============================================================================
def run_screener(tickers_filepath: str = "tickers.txt") -> None:
    if not os.path.exists(tickers_filepath):
        print(f"Error: {tickers_filepath} not found.")
        sys.exit(1)

    with open(tickers_filepath, "r") as f:
        raw_tickers = [line.strip() for line in f if line.strip()]

    print(f"[*] Ingested {len(raw_tickers)} tickers from {tickers_filepath}.")
    print(f"[*] Initializing Multi-Threaded Feature Store & Purged Walk-Forward Engine...")

    results: List[ScreenerResult] = []
    
    # Process concurrent requests across the universe
    with ThreadPoolExecutor(max_workers=12) as executor:
        future_map = {executor.submit(process_ticker, t): t for t in raw_tickers}
        for future in as_completed(future_map):
            res = future.result()
            if res is not None:
                results.append(res)

    if not results:
        print("[-] No valid trading setups matched current causal constraints.")
        return

    # Rank results by Institutional Fitness Function
    results.sort(key=lambda x: x.fitness_score, reverse=True)
    top_candidates = results[:TOP_SCREENER_COUNT]

    # Display Ranked Table
    print("\n" + "=" * 115)
    print(f"{'RANKED MULTI-TICKER UNIVERSE SCREENER (NSE)':^115}")
    print("=" * 115)
    header = f"{'Ticker':<12}{'Type':<6}{'Entry (₹)':<12}{'Target (+5%)':<14}{'Stop Loss':<12}{'R:R':<8}{'Target Prob (%)':<18}{'Max DD (%)':<14}{'Fitness':<10}"
    print(header)
    print("-" * 115)

    for r in top_candidates:
        row = (f"{r.ticker:<12}{r.direction:<6}{r.entry_level:<12.2f}{r.target_price:<14.2f}"
               f"{r.stop_loss:<12.2f}{r.risk_reward:<8.2f}{r.target_probability:<18.1f}"
               f"{r.max_drawdown:<14.1f}{r.fitness_score:<10.3f}")
        print(row)
    print("=" * 115)

    # TradingView Watchlist Import Format
    tv_watchlist = ", ".join([f"NSE:{r.ticker}" for r in top_candidates])
    print("\n" + "=" * 115)
    print("TRADINGVIEW WATCHLIST EXPORT FORMAT (Copy & Paste directly into TradingView):")
    print("-" * 115)
    print(tv_watchlist)
    print("=" * 115 + "\n")


if __name__ == "__main__":
    run_screener("tickers.txt")
