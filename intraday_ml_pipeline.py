#!/usr/bin/env python3
"""
INSTITUTIONAL INTRADAY ML ENGINE FOR INDIAN EQUITIES (NSE)
===========================================================
- Data Frequency: 15-Minute Intraday Candlesticks
- Labeling: Forward Barrier (Target vs ATR Stop Loss within 24 bars)
- Validation: Purged Sliding Walk-Forward Cross-Validation
- Models: HistGradientBoosting + k-NN + L2-Logistic Ensemble
- Optimization: Custom Institutional Fitness Metric
"""

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings

# Scikit-Learn Machine Learning Stack
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score

warnings.filterwarnings("ignore")

# =============================================================================
# STRATEGY & MODEL HYPERPARAMETERS
# =============================================================================
TARGET_PROFIT_PCT = 0.05        # 5.0% Profit Target
ATR_MULTIPLIER    = 1.75        # ATR Stop Loss Multiplier
HORIZON_BARS      = 24          # Max forward bars to reach target (6 hours of 15m)
MIN_TRAIN_BARS    = 250         # Minimum bars required to train ML models
TEST_FOLD_BARS    = 60          # Out-of-sample test window per walk-forward fold
MIN_CONFIDENCE    = 0.65        # 65% Ensemble Probability Threshold
RISK_FREE_RATE    = 0.065       # RBI 91-day T-Bill rate baseline (~6.5%)


@dataclass
class ScreenerOpportunity:
    ticker: str
    direction: str
    entry_price: float
    target_price: float
    stop_loss: float
    risk_reward: float
    ensemble_prob: float
    historical_win_rate: float
    max_drawdown: float
    fitness_score: float


# =============================================================================
# 1. FEATURE STORE GENERATION (STRICTLY CAUSAL)
# =============================================================================
def build_feature_store(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a standardized technical feature vector on intraday bars.
    Calculations strictly rely on current and past bars.
    """
    data = df.copy()
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

    # 1. RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / (loss + 1e-9)
    data['RSI'] = 100.0 - (100.0 / (1.0 + rs))

    # 2. Stochastic %K & %D (14, 3)
    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    data['Stoch_K'] = 100.0 * ((close - low_14) / (high_14 - low_14 + 1e-9))
    data['Stoch_D'] = data['Stoch_K'].rolling(3).mean()

    # 3. ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data['ATR'] = tr.ewm(alpha=1/14, min_periods=14).mean()
    data['NATR'] = (data['ATR'] / close) * 100.0

    # 4. Bollinger Bands (%B & Bandwidth)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=1)
    bb_up = bb_mid + 2.0 * bb_std
    bb_low = bb_mid - 2.0 * bb_std
    data['BB_PctB'] = (close - bb_low) / (bb_up - bb_low + 1e-9)
    data['BB_Width'] = (bb_up - bb_low) / (bb_mid + 1e-9)

    # 5. Money Flow Index (MFI 14)
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    pos_flow = pd.Series(np.where(tp > tp.shift(1), rmf, 0.0), index=data.index).rolling(14).sum()
    neg_flow = pd.Series(np.where(tp < tp.shift(1), rmf, 0.0), index=data.index).rolling(14).sum()
    data['MFI'] = 100.0 - (100.0 / (1.0 + (pos_flow / (neg_flow + 1e-9))))

    # 6. Trend: EMA Ribbon Dispersion
    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    data['EMA_Ribbon_Spread'] = (ema8 - ema55) / (ema55 + 1e-9)
    data['EMA_Short_Spread'] = (ema8 - ema21) / (ema21 + 1e-9)

    # 7. ADX & Directional Differentials (14)
    up_m = high - high.shift(1)
    down_m = low.shift(1) - low
    p_dm = np.where((up_m > down_m) & (up_m > 0), up_m, 0.0)
    m_dm = np.where((down_m > up_m) & (down_m > 0), down_m, 0.0)
    tr_s = tr.ewm(alpha=1/14, min_periods=14).mean()
    p_di = 100.0 * pd.Series(p_dm, index=data.index).ewm(alpha=1/14, min_periods=14).mean() / (tr_s + 1e-9)
    m_di = 100.0 * pd.Series(m_dm, index=data.index).ewm(alpha=1/14, min_periods=14).mean() / (tr_s + 1e-9)
    dx = 100.0 * (p_di - m_di).abs() / (p_di + m_di + 1e-9)
    data['ADX'] = dx.ewm(alpha=1/14, min_periods=14).mean()
    data['DI_Diff'] = p_di - m_di

    # 8. Intraday Momentum Returns
    data['ROC_4'] = close.pct_change(4)
    data['ROC_12'] = close.pct_change(12)

    return data


# =============================================================================
# 2. TRIPLE-BARRIER LABELING ENGINE
# =============================================================================
def generate_triple_barrier_labels(df: pd.DataFrame) -> pd.Series:
    """
    Simulates real execution barriers for training labels:
    - Target: Entry + 5%
    - Stop-Loss: Entry - (1.75 * ATR)
    - Horizon: 24 bars (~6 hours of trading)
    
    Returns binary classification series:
    - 1: Hit target before stop loss within horizon (Profitable Long)
    - 0: Hit stop loss or timed out below target (Unprofitable / Risky)
    """
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    atr = df['ATR'].values
    n = len(df)
    labels = np.zeros(n, dtype=int)

    for i in range(n - HORIZON_BARS):
        entry_price = close[i]
        target_price = entry_price * (1.0 + TARGET_PROFIT_PCT)
        stop_price = entry_price - (atr[i] * ATR_MULTIPLIER)

        for h in range(1, HORIZON_BARS + 1):
            curr_high = high[i + h]
            curr_low = low[i + h]

            if curr_high >= target_price:
                labels[i] = 1
                break
            elif curr_low <= stop_price:
                labels[i] = 0
                break
        else:
            # Did not hit either barrier; check if trade closed positive
            labels[i] = 1 if close[i + HORIZON_BARS] > entry_price else 0

    return pd.Series(labels, index=df.index)


# =============================================================================
# 3. ENSEMBLE BUILDER & PURGED WALK-FORWARD ENGINE
# =============================================================================
def build_ml_ensemble() -> VotingClassifier:
    """
    Constructs a soft-voting ensemble of diverse, uncorrelated models.
    """
    clf_gbm = HistGradientBoostingClassifier(
        max_iter=60, 
        max_depth=4, 
        learning_rate=0.05, 
        random_state=42
    )
    clf_knn = KNeighborsClassifier(
        n_neighbors=9, 
        weights='distance', 
        metric='manhattan'
    )
    clf_lr = LogisticRegression(
        C=0.1, 
        max_iter=500, 
        random_state=42
    )

    ensemble = VotingClassifier(
        estimators=[
            ('gbm', clf_gbm),
            ('knn', clf_knn),
            ('lr', clf_lr)
        ],
        voting='soft'
    )
    return ensemble


def run_walk_forward_validation(X: pd.DataFrame, y: pd.Series, df_raw: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Executes Purged Walk-Forward Cross-Validation.
    Computes Out-Of-Sample (OOS) Win Rate, Max Drawdown, and Fitness Score.
    """
    n_samples = len(X)
    if n_samples < (MIN_TRAIN_BARS + TEST_FOLD_BARS):
        return 0.0, 100.0, 0.0

    oos_returns = []
    oos_trades = 0
    oos_wins = 0

    close_arr = df_raw['Close'].values
    high_arr = df_raw['High'].values
    low_arr = df_raw['Low'].values
    atr_arr = df_raw['ATR'].values

    # Walk-forward loop (expanding train window, fixed test fold)
    for start_test in range(MIN_TRAIN_BARS, n_samples - TEST_FOLD_BARS, TEST_FOLD_BARS):
        end_test = start_test + TEST_FOLD_BARS
        
        # Purge buffer: exclude HORIZON_BARS immediately prior to test split
        purge_idx = start_test - HORIZON_BARS
        X_train, y_train = X.iloc[:purge_idx], y.iloc[:purge_idx]
        X_test = X.iloc[start_test:end_test]

        # Check for both classes in training fold
        if len(np.unique(y_train)) < 2:
            continue

        # Fit Scaler strictly on Training fold (No Leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Ensemble
        model = build_ml_ensemble()
        model.fit(X_train_scaled, y_train)

        # Out-of-sample inference
        test_probs = model.predict_proba(X_test_scaled)

        for i_local, prob in enumerate(test_probs):
            if prob >= MIN_CONFIDENCE:
                bar_idx = start_test + i_local
                if bar_idx >= n_samples - HORIZON_BARS:
                    continue

                entry_p = close_arr[bar_idx]
                target_p = entry_p * (1.0 + TARGET_PROFIT_PCT)
                stop_p = entry_p - (atr_arr[bar_idx] * ATR_MULTIPLIER)

                oos_trades += 1
                trade_ret = 0.0
                
                # Simulate trade forward
                for step in range(1, HORIZON_BARS + 1):
                    if high_arr[bar_idx + step] >= target_p:
                        trade_ret = TARGET_PROFIT_PCT
                        oos_wins += 1
                        break
                    elif low_arr[bar_idx + step] <= stop_p:
                        trade_ret = (stop_p - entry_p) / entry_p
                        break
                else:
                    trade_ret = (close_arr[bar_idx + HORIZON_BARS] - entry_p) / entry_p
                    if trade_ret > 0:
                        oos_wins += 1

                oos_returns.append(trade_ret)

    if oos_trades < 4:
        return 0.0, 100.0, 0.0

    win_rate = (oos_wins / oos_trades) * 100.0

    # Drawdown & Fitness Calculation
    eq_curve = [1.0]
    for r in oos_returns:
        eq_curve.append(eq_curve[-1] * (1.0 + r))
    
    eq_series = pd.Series(eq_curve)
    drawdowns = (eq_series - eq_series.cummax()) / eq_series.cummax()
    max_dd = abs(drawdowns.min()) * 100.0
    if max_dd < 0.5:
        max_dd = 0.5

    # Annualized Return Estimate (15m bars: ~1500 bars/year)
    total_ret = eq_series.iloc[-1] - 1.0
    ann_ret = total_ret * (1500.0 / max(len(oos_returns), 1))

    # Sortino Calculation
    neg_rets = [r for r in oos_returns if r < 0]
    downside_std = np.std(neg_rets) if len(neg_rets) > 1 else 0.01
    sortino = (np.mean(oos_returns) * 1500.0 - RISK_FREE_RATE) / (downside_std * np.sqrt(1500) + 1e-9)
    sortino = max(sortino, 0.0)

    # Fitness Objective: (AnnRet / MDD^2) * Sortino * (1 - TradeFreqPenalty)
    penalty = 0.3 if oos_trades < 8 else 0.0
    fitness = ((max(ann_ret, 0.0) * 100.0) / (max_dd ** 2)) * sortino * (1.0 - penalty)

    return win_rate, max_dd, fitness


# =============================================================================
# 4. SINGLE TICKER WORKFLOW (INGESTION -> ML TRAIN -> LIVE PREDICT)
# =============================================================================
def evaluate_ticker_ml(raw_ticker: str) -> Optional[ScreenerOpportunity]:
    """
    Downloads intraday data, trains the ensemble model, runs walk-forward validation,
    and infers live probability on the latest bar.
    """
    clean_ticker = raw_ticker.strip().upper()
    if not clean_ticker:
        return None

    yf_symbol = f"{clean_ticker}.NS"
    try:
        # Download 60 days of 15m intraday candlesticks (~1500 bars)
        df = yf.download(yf_symbol, period="60d", interval="15m", progress=False)
        if df is None or len(df) < (MIN_TRAIN_BARS + 50):
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Compute Technical Features
        df_feat = build_feature_store(df).dropna()
        if len(df_feat) < (MIN_TRAIN_BARS + 30):
            return None

        # Generate Barrier Labels
        labels = generate_triple_barrier_labels(df_feat)

        feature_cols = [
            'RSI', 'Stoch_K', 'Stoch_D', 'NATR', 'BB_PctB', 'BB_Width', 
            'MFI', 'EMA_Ribbon_Spread', 'EMA_Short_Spread', 'ADX', 'DI_Diff', 
            'ROC_4', 'ROC_12'
        ]

        # Isolate training set (excluding unclosed forward-barrier bars)
        valid_indices = df_feat.index[:-HORIZON_BARS]
        X_train_full = df_feat.loc[valid_indices, feature_cols]
        y_train_full = labels.loc[valid_indices]

        # Run Walk-Forward Validation
        win_rate, max_dd, fitness = run_walk_forward_validation(X_train_full, y_train_full, df_feat)

        # Train Full Model for Live Bar Inference
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_full)
        
        ensemble = build_ml_ensemble()
        ensemble.fit(X_train_scaled, y_train_full)

        # Predict Live Bar Probability (Latest closed bar)
        latest_features = df_feat.iloc[[-1]][feature_cols]
        latest_scaled = scaler.transform(latest_features)
        live_prob = float(ensemble.predict_proba(latest_scaled))

        # Filter by Confidence Threshold
        if live_prob < MIN_CONFIDENCE:
            return None

        last_close = float(df_feat.iloc[-1]['Close'])
        last_atr = float(df_feat.iloc[-1]['ATR'])
        target_p = last_close * (1.0 + TARGET_PROFIT_PCT)
        stop_p = last_close - (last_atr * ATR_MULTIPLIER)
        rr_ratio = abs(target_p - last_close) / (abs(last_close - stop_p) + 1e-9)

        return ScreenerOpportunity(
            ticker=clean_ticker,
            direction="BUY",
            entry_price=round(last_close, 2),
            target_price=round(target_p, 2),
            stop_loss=round(stop_p, 2),
            risk_reward=round(rr_ratio, 2),
            ensemble_prob=round(live_prob * 100.0, 1),
            historical_win_rate=round(win_rate, 1),
            max_drawdown=round(max_dd, 1),
            fitness_score=round(fitness, 3)
        )
    except Exception:
        return None


# =============================================================================
# 5. BATCH SCREENER & TRADINGVIEW EXPORTER
# =============================================================================
def main():
    ticker_file = "tickers.txt"
    if not os.path.exists(ticker_file):
        print(f"[-] {ticker_file} not found. Creating sample universe.")
        sample_tickers = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", 
            "BHARTIARTL", "SBIN", "LICI", "ITC", "HINDUNILVR",
            "LT", "BAJFINANCE", "TATAMOTORS", "SUNPHARMA", "MARUTI"
        ]
        with open(ticker_file, "w") as f:
            f.write("\n".join(sample_tickers))

    with open(ticker_file, "r") as f:
        tickers = [line.strip().upper() for line in f if line.strip()]

    print("=" * 115)
    print(f"INSTITUTIONAL INTRADAY ML ENGINE: SCREENING {len(tickers)} NSE EQUITIES (15-MIN BARS)")
    print("=" * 115)

    opportunities: List[ScreenerOpportunity] = []
    
    for idx, t in enumerate(tickers, start=1):
        print(f"[{idx}/{len(tickers)}] Training Ensemble & Simulating WF-CV on {t}...", end="\r", flush=True)
        res = evaluate_ticker_ml(t)
        if res is not None:
            opportunities.append(res)

    print("\n" + "-" * 115)
    if not opportunities:
        print("[-] No stocks currently meet the 65% ML probability threshold on the 15-minute timeframe.")
        return

    # Rank by Institutional Fitness Score
    opportunities.sort(key=lambda x: x.fitness_score, reverse=True)

    # Print Table
    header = f"{'Ticker':<12}{'Signal':<6}{'Entry (₹)':<12}{'Target (+5%)':<14}{'Stop Loss':<12}{'R:R':<8}{'ML Prob (%)':<14}{'OOS Win %':<12}{'Max DD %':<10}{'Fitness':<10}"
    print(header)
    print("-" * 115)
    for opp in opportunities:
        print(f"{opp.ticker:<12}{opp.direction:<6}{opp.entry_price:<12.2f}{opp.target_price:<14.2f}"
              f"{opp.stop_loss:<12.2f}{opp.risk_reward:<8.2f}{opp.ensemble_prob:<14.1f}"
              f"{opp.historical_win_rate:<12.1f}{opp.max_drawdown:<10.1f}{opp.fitness_score:<10.3f}")

    print("=" * 115)
    tv_symbols = ", ".join([f"NSE:{o.ticker}" for o in opportunities])
    print("TRADINGVIEW WATCHLIST EXPORT:")
    print(tv_symbols)
    print("=" * 115)


if __name__ == "__main__":
    main()
